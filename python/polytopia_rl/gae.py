"""Generalized advantage estimation (docs/MILESTONE_10 "PPO")."""

from __future__ import annotations

import torch
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
    num_steps = rewards.shape[0]
    terminated, truncated = terminated.bool(), truncated.bool()
    advantages = torch.zeros_like(rewards)
    a_next = torch.zeros_like(last_value)
    for t in reversed(range(num_steps)):
        # The value of the state after step t: the next stored value, the rollout-end bootstrap,
        # or, if the episode was cut off by truncation, the value of its final observation.
        v_next = last_value if t == num_steps - 1 else values[t + 1]
        v_next = torch.where(truncated[t], next_values[t], v_next)
        # A terminated step has no future: no bootstrap.
        not_term = (~terminated[t]).to(rewards.dtype)
        delta = rewards[t] + gamma * v_next * not_term - values[t]
        # Any episode end stops the advantage from flowing back across the reset.
        not_done = (~(terminated[t] | truncated[t])).to(rewards.dtype)
        a_next = delta + gamma * lam * not_done * a_next
        advantages[t] = a_next
    return advantages, advantages + values


def compute_gae_segments(
    rewards: Tensor,
    values: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    next_values: Tensor,
    segment_end: Tensor,
    gamma: float,
    lam: float,
) -> tuple[Tensor, Tensor]:
    """1D segmented GAE over packed per-(slot, seat) lanes of length `N`. `segment_end[i]` marks
    the last step of a lane (the final element `N - 1` is always treated as a segment end)."""
    n = rewards.shape[0]
    if n == 0:
        empty = torch.empty(0, dtype=rewards.dtype, device=rewards.device)
        return empty, empty.clone()
    term = terminated.bool()
    trunc = truncated.bool()
    seg_end = segment_end.bool().clone()
    seg_end[-1] = True
    shifted_values = torch.empty_like(values)
    if n > 1:
        shifted_values[:-1] = values[1:]
    shifted_values[-1] = 0.0
    v_next = torch.where(trunc | seg_end, next_values, shifted_values)
    not_term = (~term).to(rewards.dtype)
    deltas = rewards + gamma * v_next * not_term - values
    cont = (~(term | trunc | seg_end)).to(rewards.dtype)

    deltas_list = deltas.detach().cpu().tolist()
    cont_list = cont.detach().cpu().tolist()
    adv_list = [0.0] * n
    gl = gamma * lam
    a_next = 0.0
    for i in range(n - 1, -1, -1):
        a_next = deltas_list[i] + gl * cont_list[i] * a_next
        adv_list[i] = a_next
    advantages = torch.tensor(adv_list, dtype=rewards.dtype, device=rewards.device)
    return advantages, advantages + values
