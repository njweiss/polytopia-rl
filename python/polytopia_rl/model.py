"""Policy and value network with factored, conditionally masked heads (docs/MILESTONE_10 "Policy").

`spec` is `Env.spec()`; the keys used are `channels`, `globals`, `height`, `width`, `num_actors`,
`arg_lens` and `action_types`. Actor index `fam * N + tile` (unit, city, site families), global
actor `3N`; `tile = row * width + col`.

Parameters are grouped by submodule: `actor_head`, `kind_head`, `arg_head`, `value_head`; everything
else is the trunk. The heads share no weights (the kind and argument heads each own their
global-actor vector and family embedding).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from polytopia_rl import categorical

TILE_KINDS = ("move", "attack")  # argument is a target tile; every other kind with an argument is an enum
EMBED = 16  # size of the family and kind embeddings


@dataclass
class Features:
    fmap: Tensor  # (B, width, H, W): the trunk's feature map F
    g: Tensor  # (B, D): context vector [global MLP, mean-pool F, max-pool F]


@dataclass
class ActOut:
    actors: Tensor  # (B,) int64
    kinds: Tensor  # (B,) int64
    args: Tensor  # (B,) int64; 0 for kinds without an argument
    logp: Tensor | None  # (B,); None from act(stats=False), as are entropy and value
    entropy: Tensor | None  # (B,)
    value: Tensor | None  # (B,)
    actor_mask: Tensor  # (B, 3N+1) bool, as fetched through masks_fn
    kind_mask: Tensor  # (B, 10) bool
    arg_mask: Tensor  # (B, N) bool


@dataclass
class EvalOut:
    logp: Tensor  # (B,)
    entropy: Tensor  # (B,)
    head_entropy: dict  # {"actor", "kind", "arg"} -> (B,); 0 where the stage did not fire
    value: Tensor  # (B,)


def _init(layer: nn.Module, gain: float) -> nn.Module:
    """CleanRL's init: orthogonal weights, zero bias."""
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


def _mlp(sizes: list[int], out_gain: float) -> nn.Sequential:
    """Linear-ReLU stack; hidden layers use gain sqrt(2), the last layer `out_gain`."""
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        last = i == len(sizes) - 2
        layers.append(_init(nn.Linear(sizes[i], sizes[i + 1]), out_gain if last else math.sqrt(2)))
        if not last:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    """conv, ReLU, conv, add the input, ReLU (no batch norm)."""

    def __init__(self, width: int):
        super().__init__()
        self.conv1 = _init(nn.Conv2d(width, width, 3, padding=1), math.sqrt(2))
        self.conv2 = _init(nn.Conv2d(width, width, 3, padding=1), math.sqrt(2))

    def forward(self, x: Tensor) -> Tensor:
        return torch.relu(x + self.conv2(torch.relu(self.conv1(x))))


class ActorContext(nn.Module):
    """`z = [F at the actor's tile (or a learned vector for the global actor), g, family embedding]`.

    The kind head and the argument head each own one of these, so they share no weights."""

    def __init__(self, width: int, num_tiles: int):
        super().__init__()
        self.num_tiles = num_tiles
        self.global_vec = nn.Parameter(torch.randn(width) * 0.1)
        self.family = nn.Embedding(4, EMBED)  # unit, city, site, global

    def forward(self, feats: Features, actors: Tensor) -> Tensor:
        n = self.num_tiles
        family = actors // n  # 0, 1, 2 for tile actors; the global actor 3N gives 3
        tile = torch.where(family < 3, actors % n, torch.zeros_like(actors))
        flat = feats.fmap.flatten(2)  # (B, width, N)
        at_tile = flat.gather(2, tile.view(-1, 1, 1).expand(-1, flat.shape[1], 1)).squeeze(2)
        is_global = (family == 3).unsqueeze(1)
        at_tile = torch.where(is_global, self.global_vec.expand_as(at_tile), at_tile)
        return torch.cat([at_tile, feats.g, self.family(family)], dim=1)


class ActorHead(nn.Module):
    """Stage 1: a 1x1 conv gives one logit map per family (unit, city, site); one more logit for the
    global actor comes from `g`."""

    def __init__(self, width: int, ctx_dim: int):
        super().__init__()
        self.tile = _init(nn.Conv2d(width, 3, 1), 0.01)
        self.global_logit = _init(nn.Linear(ctx_dim, 1), 0.01)

    def forward(self, feats: Features) -> Tensor:
        maps = self.tile(feats.fmap).flatten(1)  # (B, 3N), family-major: index fam * N + tile
        return torch.cat([maps, self.global_logit(feats.g)], dim=1)


class KindHead(nn.Module):
    """Stage 2: MLP(z) -> 10 kind logits."""

    def __init__(self, width: int, ctx_dim: int, num_tiles: int, num_kinds: int):
        super().__init__()
        self.context = ActorContext(width, num_tiles)
        self.mlp = _mlp([width + ctx_dim + EMBED, 2 * width, num_kinds], 0.01)

    def forward(self, feats: Features, actors: Tensor) -> Tensor:
        return self.mlp(self.context(feats, actors))


class ArgHead(nn.Module):
    """Stage 3: `c = MLP([z, kind embedding])`. Tile kinds score every tile as
    `(1x1 conv of F) . c / sqrt(width) + bias_kind`; enum kinds use their own linear layer on `c`."""

    def __init__(self, width: int, ctx_dim: int, num_tiles: int, kinds: list[str], arg_lens: list[int]):
        super().__init__()
        self.width = width
        self.num_tiles = num_tiles
        self.context = ActorContext(width, num_tiles)
        self.kind_embed = nn.Embedding(len(kinds), EMBED)
        self.c = _mlp([width + ctx_dim + 2 * EMBED, 2 * width, width], math.sqrt(2))
        self.tile_keys = _init(nn.Conv2d(width, width, 1), 0.01)
        self.tile_bias = nn.Parameter(torch.zeros(len(kinds)))
        self.tile_kinds = [k for k, name in enumerate(kinds) if name in TILE_KINDS]
        self.enum_kinds = [k for k, n in enumerate(arg_lens) if n > 0 and k not in self.tile_kinds]
        self.enum = nn.ModuleList(_init(nn.Linear(width, arg_lens[k]), 0.01) for k in self.enum_kinds)

    def forward(self, feats: Features, actors: Tensor, kinds: Tensor) -> Tensor:
        z = self.context(feats, actors)
        c = self.c(torch.cat([z, self.kind_embed(kinds)], dim=1))  # (B, width)
        keys = self.tile_keys(feats.fmap).flatten(2)  # (B, width, N)
        tile_logits = torch.einsum("bwn,bw->bn", keys, c) / math.sqrt(self.width)
        tile_logits = tile_logits + self.tile_bias[kinds].unsqueeze(1)
        # Each row keeps the logits of its own kind; enum logits are padded to N (masked anyway).
        out = torch.zeros_like(tile_logits)
        for k in self.tile_kinds:
            out = torch.where((kinds == k).unsqueeze(1), tile_logits, out)
        for k, head in zip(self.enum_kinds, self.enum):
            logits = head(c)
            padded = nn.functional.pad(logits, (0, self.num_tiles - logits.shape[1]))
            out = torch.where((kinds == k).unsqueeze(1), padded, out)
        return out


class PolicyNet(nn.Module):
    def __init__(self, spec: dict, width: int = 64, blocks: int = 4):
        super().__init__()
        channels, num_globals = spec["channels"], spec["globals"]
        self.num_tiles = spec["height"] * spec["width"]
        kinds = spec["action_types"]
        arg_lens = spec["arg_lens"]
        # Kinds without an argument (capture, recover, promote, end_turn) have no stage 3.
        self.register_buffer("has_arg", torch.tensor([n > 0 for n in arg_lens]), persistent=False)
        ctx_dim = 3 * width  # [global MLP, mean-pool F, max-pool F]
        # Trunk.
        self.stem = _init(nn.Conv2d(channels, width, 3, padding=1), math.sqrt(2))
        self.blocks = nn.Sequential(*[ResBlock(width) for _ in range(blocks)])
        self.global_mlp = nn.Sequential(_mlp([num_globals, width, width], math.sqrt(2)), nn.ReLU())
        # Heads.
        self.actor_head = ActorHead(width, ctx_dim)
        self.kind_head = KindHead(width, ctx_dim, self.num_tiles, len(kinds))
        self.arg_head = ArgHead(width, ctx_dim, self.num_tiles, kinds, arg_lens)
        self.value_head = _mlp([ctx_dim, width, 1], 1.0)

    def trunk(self, board: Tensor, globals: Tensor) -> Features:
        fmap = self.blocks(torch.relu(self.stem(board)))
        g = torch.cat([self.global_mlp(globals), fmap.mean(dim=(2, 3)), fmap.amax(dim=(2, 3))], dim=1)
        return Features(fmap=fmap, g=g)

    def actor_logits(self, feats: Features) -> Tensor:
        """(B, 3N+1)."""
        return self.actor_head(feats)

    def kind_logits(self, feats: Features, actors: Tensor) -> Tensor:
        """(B, 10), conditioned on the chosen actor."""
        return self.kind_head(feats, actors)

    def arg_logits(self, feats: Features, actors: Tensor, kinds: Tensor) -> Tensor:
        """(B, N): tile logits for Move/Attack, `arg_len(kind)` enum logits padded to N for
        Train/ChooseReward/Site/Research; rows of kinds without an argument are padding."""
        return self.arg_head(feats, actors, kinds)

    def value(self, feats: Features) -> Tensor:
        """(B,)."""
        return self.value_head(feats.g).squeeze(1)

    def _arg_stage(self, kinds: Tensor, arg_mask: Tensor) -> tuple[Tensor, Tensor]:
        """`(fired, safe_mask)`: which rows have an argument stage, and their arg mask with the
        unfired rows set all-true so the masked distribution is defined (its terms are dropped)."""
        fired = self.has_arg[kinds]
        return fired, arg_mask | ~fired.unsqueeze(1)

    def act(
        self,
        feats: Features,
        masks_fn: Callable,
        generator: torch.Generator,
        greedy: bool = False,
        stats: bool = True,
    ) -> ActOut:
        """Sample actor, then kind, then argument. Masks are fetched lazily:
        `masks_fn("actor")`, `masks_fn("kind", actors)`, `masks_fn("arg", actors, kinds)`, with
        `actors`/`kinds` as int64 numpy arrays; each returns a bool array (numpy or tensor).
        `stats=False` (play only) leaves `logp`, `entropy` and `value` as None and skips the mask
        check (env masks always have a legal option); the actions are the same."""
        device = feats.g.device

        def fetch(*a):
            return torch.as_tensor(np.asarray(masks_fn(*a)), dtype=torch.bool, device=device)

        actor_mask = fetch("actor")
        logits = self.actor_logits(feats)
        actors = categorical.sample(logits, actor_mask, generator, greedy, check=stats)
        if stats:
            logp = categorical.log_prob(logits, actor_mask, actors)
            ent = categorical.entropy(logits, actor_mask)
        actors_np = actors.cpu().numpy()

        kind_mask = fetch("kind", actors_np)
        logits = self.kind_logits(feats, actors)
        kinds = categorical.sample(logits, kind_mask, generator, greedy, check=stats)
        if stats:
            logp = logp + categorical.log_prob(logits, kind_mask, kinds)
            ent = ent + categorical.entropy(logits, kind_mask)

        arg_mask = fetch("arg", actors_np, kinds.cpu().numpy())
        fired, safe = self._arg_stage(kinds, arg_mask)
        logits = self.arg_logits(feats, actors, kinds)
        args = torch.where(fired, categorical.sample(logits, safe, generator, greedy, check=stats), 0)
        if not stats:
            return ActOut(actors, kinds, args, None, None, None, actor_mask, kind_mask, arg_mask)
        zero = torch.zeros_like(logp)
        logp = logp + torch.where(fired, categorical.log_prob(logits, safe, args), zero)
        ent = ent + torch.where(fired, categorical.entropy(logits, safe), zero)
        return ActOut(
            actors=actors,
            kinds=kinds,
            args=args,
            logp=logp,
            entropy=ent,
            value=self.value(feats),
            actor_mask=actor_mask,
            kind_mask=kind_mask,
            arg_mask=arg_mask,
        )

    def evaluate(
        self,
        board: Tensor,
        globals: Tensor,
        actors: Tensor,
        kinds: Tensor,
        args: Tensor,
        actor_mask: Tensor,
        kind_mask: Tensor,
        arg_mask: Tensor,
    ) -> EvalOut:
        """Recompute log-probability, entropy and value of stored actions: the sum over the stages
        that fired, each stage's distribution conditioned on the earlier choices."""
        feats = self.trunk(board, globals)
        logits = self.actor_logits(feats)
        logp = categorical.log_prob(logits, actor_mask, actors)
        h_actor = categorical.entropy(logits, actor_mask)
        logits = self.kind_logits(feats, actors)
        logp = logp + categorical.log_prob(logits, kind_mask, kinds)
        h_kind = categorical.entropy(logits, kind_mask)
        fired, safe = self._arg_stage(kinds, arg_mask)
        logits = self.arg_logits(feats, actors, kinds)
        zero = torch.zeros_like(logp)
        logp = logp + torch.where(fired, categorical.log_prob(logits, safe, args), zero)
        h_arg = torch.where(fired, categorical.entropy(logits, safe), zero)
        return EvalOut(
            logp=logp,
            entropy=h_actor + h_kind + h_arg,
            head_entropy={"actor": h_actor, "kind": h_kind, "arg": h_arg},
            value=self.value(feats),
        )
