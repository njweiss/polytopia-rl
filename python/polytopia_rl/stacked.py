"""One stacked forward pass for all snapshot opponents (docs/MILESTONE_13b).

`StackedSnapshots` holds the parameters of the pool's snapshots stacked along a leading dimension
(`torch.func.stack_module_state`); `stacked_batch_policy` evaluates every snapshot group of a tick
with one `torch.func.vmap(functional_call(...))` per stage (actor, kind, argument).
"""

from __future__ import annotations

import copy
from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from torch.func import functional_call, stack_module_state, vmap

from polytopia_rl import categorical
from polytopia_rl.model import Features, PolicyNet
from polytopia_rl.selfplay import Entry
from polytopia_rl.vec_env import env_masks_fn


class _Stages(nn.Module):
    """`forward(stage, ...)` dispatches to one of `PolicyNet`'s play stages, so `functional_call`
    can run each of them: "trunk" returns `(fmap, g)`; the others take `(fmap, g, ...)`."""

    def __init__(self, net: PolicyNet):
        super().__init__()
        self.net = net

    def forward(self, stage: str, *args):
        if stage == "trunk":
            feats = self.net.trunk(*args)
            return feats.fmap, feats.g
        fmap, g, *rest = args
        return getattr(self.net, stage)(Features(fmap=fmap, g=g), *rest)


class StackedSnapshots:
    """The stacked, frozen parameters of a list of pool entries on `device`; `base_net` gives the
    architecture (its own weights are never used nor changed)."""

    def __init__(self, base_net: PolicyNet, device: str):
        self.device = device
        # A private copy: `functional_call` runs on it with the stacked parameters swapped in.
        self.module = _Stages(copy.deepcopy(base_net).to(device)).requires_grad_(False)
        self.names: list[str] | None = None
        self.params: dict = {}
        self.buffers: dict = {}
        self._rows: dict[str, int] = {}

    def sync(self, entries: list[Entry]) -> None:
        """(Re)stack the parameters of `entries` on `device` when their names differ from the last
        sync; no-op otherwise. Rows are in the order of `entries`."""
        names = [e.name for e in entries]
        if names == self.names:
            return
        copies = []
        for e in entries:
            m = copy.deepcopy(self.module)
            m.net.load_state_dict(e.state)
            copies.append(m)
        # torch.stack copies, so the stack never aliases the entries' tensors.
        self.params, self.buffers = stack_module_state(copies)
        self.names = names
        self._rows = {name: r for r, name in enumerate(names)}

    def index(self, name: str) -> int:
        """The row of an entry in the current stack."""
        return self._rows[name]


def stacked_batch_policy(stack: StackedSnapshots, generator, greedy: bool = False) -> Callable:
    """`policy(groups)`, `groups` a list of `(entry_name, envs)`: one list of actions per group,
    aligned with that group's envs, from one vmapped call per stage over all groups. The policy has
    a `sync(entries)` attribute that forwards to `stack.sync`."""
    device = stack.device

    def run(stage, params, buffers, *inputs):
        """`stage` of every gathered snapshot on its own (B, ...) slice of `inputs` (G, B, ...)."""

        def one(p, b, *x):
            return functional_call(stack.module, (p, b), (stage, *x))

        return vmap(one)(params, buffers, *inputs)

    def policy(groups):
        groups = list(groups)
        sizes = [len(envs) for _, envs in groups]
        envs = [env for _, g_envs in groups for env in g_envs]
        G, B = len(groups), max(sizes)
        # Padded position (g, j) reads valid row offset_g + j, or the group's first row when j >= b_g.
        offsets = np.cumsum([0] + sizes[:-1])
        pad_src = torch.as_tensor([int(o) + (j if j < b else 0) for o, b in zip(offsets, sizes) for j in range(B)], device=device)
        # The valid rows, (group, slot) order, as positions in the flattened (G * B) padded batch.
        valid = torch.as_tensor([g * B + j for g, b in enumerate(sizes) for j in range(b)], device=device)

        def pad(x):
            """(R, ...) -> (G, B, ...)."""
            return x.index_select(0, pad_src).view(G, B, *x.shape[1:])

        def unpad(x):
            """(G, B, K) -> (R, K)."""
            return x.reshape(G * B, *x.shape[2:]).index_select(0, valid)

        obs = [env.observation() for env in envs]
        board = torch.from_numpy(np.stack([b for b, _ in obs])).to(device)
        glob = torch.from_numpy(np.stack([g for _, g in obs])).to(device)
        masks_fn = env_masks_fn(envs, board.shape[2] * board.shape[3])

        def fetch(*a):
            return torch.as_tensor(masks_fn(*a), dtype=torch.bool, device=device)

        with torch.inference_mode():
            idx = torch.as_tensor([stack.index(name) for name, _ in groups], device=device)
            params = {k: v.index_select(0, idx) for k, v in stack.params.items()}
            buffers = {k: v.index_select(0, idx) for k, v in stack.buffers.items()}
            fmap, g = run("trunk", params, buffers, pad(board), pad(glob))

            actor_mask = fetch("actor")
            logits = unpad(run("actor_logits", params, buffers, fmap, g))
            actors = categorical.sample(logits, actor_mask, generator, greedy, check=False)
            actors_np = actors.cpu().numpy()

            kind_mask = fetch("kind", actors_np)
            logits = unpad(run("kind_logits", params, buffers, fmap, g, pad(actors)))
            kinds = categorical.sample(logits, kind_mask, generator, greedy, check=False)
            kinds_np = kinds.cpu().numpy()

            arg_mask = fetch("arg", actors_np, kinds_np)
            # As `PolicyNet.act`: kinds without an argument sample from an all-true mask, then get 0.
            fired, safe = stack.module.net._arg_stage(kinds, arg_mask)
            logits = unpad(run("arg_logits", params, buffers, fmap, g, pad(actors), pad(kinds)))
            args = torch.where(fired, categorical.sample(logits, safe, generator, greedy, check=False), 0)

        flat = list(zip(actors.tolist(), kinds.tolist(), args.tolist()))
        return [flat[o : o + b] for o, b in zip(offsets, sizes)]

    policy.sync = stack.sync
    return policy
