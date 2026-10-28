"""One stacked forward pass for all snapshot opponents (docs/MILESTONE_13b).

`StackedSnapshots` holds the parameters of the pool's snapshots stacked along a leading dimension
(`torch.func.stack_module_state`); `stacked_batch_policy` evaluates every snapshot group of a tick
with one `torch.func.vmap(functional_call(...))` per stage (actor, kind, argument).
"""

from __future__ import annotations

from collections.abc import Callable

from polytopia_rl.model import PolicyNet
from polytopia_rl.selfplay import Entry


class StackedSnapshots:
    """The stacked, frozen parameters of a list of pool entries on `device`; `base_net` gives the
    architecture (its own weights are never used nor changed)."""

    def __init__(self, base_net: PolicyNet, device: str):
        raise NotImplementedError("milestone 13b")

    def sync(self, entries: list[Entry]) -> None:
        """(Re)stack the parameters of `entries` on `device` when their names differ from the last
        sync; no-op otherwise. Rows are in the order of `entries`."""
        raise NotImplementedError("milestone 13b")

    def index(self, name: str) -> int:
        """The row of an entry in the current stack."""
        raise NotImplementedError("milestone 13b")


def stacked_batch_policy(stack: StackedSnapshots, generator, greedy: bool = False) -> Callable:
    """`policy(groups)`, `groups` a list of `(entry_name, envs)`: one list of actions per group,
    aligned with that group's envs, from one vmapped call per stage over all groups. The policy has
    a `sync(entries)` attribute that forwards to `stack.sync`."""
    raise NotImplementedError("milestone 13b")
