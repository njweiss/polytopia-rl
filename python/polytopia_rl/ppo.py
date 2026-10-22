"""PPO loss, rollout storage and the update step (docs/MILESTONE_10 "PPO", CleanRL conventions)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class LossOut:
    pg: Tensor
    value: Tensor
    entropy: Tensor
    total: Tensor
    approx_kl: Tensor  # mean((ratio - 1) - log ratio), no gradient
    clipfrac: Tensor  # mean(|ratio - 1| > clip_eps), no gradient


def ppo_loss(new_logp, old_logp, adv, new_value, old_value, returns, entropy, cfg) -> LossOut:
    """All inputs (B,). With `cfg.norm_adv` the advantages are normalized over this batch as
    `(adv - adv.mean()) / (adv.std() + 1e-8)` (unbiased std). With `cfg.clip_vloss` the value loss is
    `0.5 * mean(max((V - R)^2, (old_V + clamp(V - old_V, -clip_eps, clip_eps) - R)^2))`."""
    raise NotImplementedError


def minibatch_indices(n: int, num_minibatches: int, generator: torch.Generator) -> list[Tensor]:
    """A random permutation of `range(n)` split into `num_minibatches` equal parts."""
    raise NotImplementedError


class RolloutBuffer:
    """`num_steps x num_envs` samples. Fields, all tensors indexed `[t, e]`: `board` (C,H,W),
    `globals` (G,), `actors`, `kinds`, `args` (int64), `actor_mask` (3N+1,), `kind_mask` (10,),
    `arg_mask` (N,) (bool), `logp`, `values`, `rewards`, `terminated`, `truncated` (bool),
    `next_values` (the value of the final observation where `truncated`); and `last_value` (E,),
    the value of the observations after the last step."""

    def __init__(self, num_steps: int, num_envs: int, spec: dict, device="cpu"):
        raise NotImplementedError


def collect_rollout(net, vec_env, buffer: RolloutBuffer, generator: torch.Generator) -> list:
    """Fill `buffer` with `num_steps` steps of `vec_env` (already reset) under `net`; return the
    `EpisodeStat`s of the episodes that finished."""
    raise NotImplementedError


def train_update(net, optimizer, buffer: RolloutBuffer, cfg, generator: torch.Generator) -> dict:
    """GAE on `buffer`, then `cfg.update_epochs` epochs of shuffled minibatches. Uses the
    optimizer's current learning rate. Returns the update's metrics (`pg_loss`, `v_loss`,
    `entropy`, `entropy_actor`, `entropy_kind`, `entropy_arg`, `approx_kl`, `clipfrac`,
    `explained_variance`, `adv_mean`, `adv_std`)."""
    raise NotImplementedError
