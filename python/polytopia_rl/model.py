"""Policy and value network with factored, conditionally masked heads (docs/MILESTONE_10 "Policy").

`spec` is `Env.spec()`; the keys used are `channels`, `globals`, `height`, `width`, `num_actors`,
`arg_lens` and `action_types`. Actor index `fam * N + tile` (unit, city, site families), global
actor `3N`; `tile = row * width + col`.

Parameters are grouped by submodule: `actor_head`, `kind_head`, `arg_head`, `value_head`; everything
else is the trunk. The heads share no weights (the kind and argument heads each own their
global-actor vector and family embedding).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class Features:
    fmap: Tensor  # (B, width, H, W): the trunk's feature map F
    g: Tensor  # (B, D): context vector [global MLP, mean-pool F, max-pool F]


@dataclass
class ActOut:
    actors: Tensor  # (B,) int64
    kinds: Tensor  # (B,) int64
    args: Tensor  # (B,) int64; 0 for kinds without an argument
    logp: Tensor  # (B,)
    entropy: Tensor  # (B,)
    value: Tensor  # (B,)
    actor_mask: Tensor  # (B, 3N+1) bool, as fetched through masks_fn
    kind_mask: Tensor  # (B, 10) bool
    arg_mask: Tensor  # (B, N) bool


@dataclass
class EvalOut:
    logp: Tensor  # (B,)
    entropy: Tensor  # (B,)
    head_entropy: dict  # {"actor", "kind", "arg"} -> (B,); 0 where the stage did not fire
    value: Tensor  # (B,)


class PolicyNet(nn.Module):
    def __init__(self, spec: dict, width: int = 64, blocks: int = 4):
        super().__init__()
        raise NotImplementedError

    def trunk(self, board: Tensor, globals: Tensor) -> Features:
        raise NotImplementedError

    def actor_logits(self, feats: Features) -> Tensor:
        """(B, 3N+1)."""
        raise NotImplementedError

    def kind_logits(self, feats: Features, actors: Tensor) -> Tensor:
        """(B, 10), conditioned on the chosen actor."""
        raise NotImplementedError

    def arg_logits(self, feats: Features, actors: Tensor, kinds: Tensor) -> Tensor:
        """(B, N): tile logits for Move/Attack, `arg_len(kind)` enum logits padded to N for
        Train/ChooseReward/Site/Research; rows of kinds without an argument are padding."""
        raise NotImplementedError

    def value(self, feats: Features) -> Tensor:
        """(B,)."""
        raise NotImplementedError

    def act(
        self,
        feats: Features,
        masks_fn: Callable,
        generator: torch.Generator,
        greedy: bool = False,
    ) -> ActOut:
        """Sample actor, then kind, then argument. Masks are fetched lazily:
        `masks_fn("actor")`, `masks_fn("kind", actors)`, `masks_fn("arg", actors, kinds)`, with
        `actors`/`kinds` as int64 numpy arrays; each returns a bool array (numpy or tensor)."""
        raise NotImplementedError

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
        raise NotImplementedError
