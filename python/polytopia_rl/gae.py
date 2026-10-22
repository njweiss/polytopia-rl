"""Generalized advantage estimation (docs/MILESTONE_10 "PPO")."""

from __future__ import annotations

from torch import Tensor


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    next_values: Tensor,
    last_value: Tensor,
    gamma: float,
    lam: float,
) -> tuple[Tensor, Tensor]:
    """Inputs are (T, E) except `last_value` (E,). `next_values[t]` is the value of the final
    observation, used only where `truncated[t]`; elsewhere `V_next` is `values[t + 1]` (or
    `last_value` at `t = T - 1`). Returns `(advantages, returns)`, both (T, E)."""
    raise NotImplementedError
