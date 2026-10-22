"""PPO loss, rollout storage and the update step (docs/MILESTONE_10 "PPO", CleanRL conventions)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from polytopia_rl.gae import compute_gae


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
    eps = cfg.clip_eps
    # Probability ratio r = pi_new(a|s) / pi_old(a|s). The old policy is a constant.
    logratio = new_logp - old_logp.detach()
    ratio = logratio.exp()
    if cfg.norm_adv:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    # Clipped surrogate: take the more pessimistic of the plain and the clipped objective, so a
    # step gains nothing from pushing r outside [1 - eps, 1 + eps].
    pg = torch.max(-adv * ratio, -adv * ratio.clamp(1 - eps, 1 + eps)).mean()
    # Value regression towards the GAE returns; the clipped variant (CleanRL) likewise limits how
    # far the value can move from the one recorded in the rollout.
    v_loss = (new_value - returns) ** 2
    if cfg.clip_vloss:
        v_clipped = old_value + (new_value - old_value).clamp(-eps, eps)
        v_loss = torch.max(v_loss, (v_clipped - returns) ** 2)
    v_loss = 0.5 * v_loss.mean()
    # Entropy bonus: subtracted, so higher entropy lowers the loss and keeps exploration alive.
    ent = entropy.mean()
    total = pg + cfg.vf_coef * v_loss - cfg.ent_coef * ent
    with torch.no_grad():
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1).abs() > eps).float().mean()
    return LossOut(pg=pg, value=v_loss, entropy=ent, total=total, approx_kl=approx_kl, clipfrac=clipfrac)


def minibatch_indices(n: int, num_minibatches: int, generator: torch.Generator) -> list[Tensor]:
    """A random permutation of `range(n)` split into `num_minibatches` equal parts."""
    size = n // num_minibatches
    perm = torch.randperm(n, generator=generator)
    return list(perm[: size * num_minibatches].view(num_minibatches, size))


class RolloutBuffer:
    """`num_steps x num_envs` samples. Fields, all tensors indexed `[t, e]`: `board` (C,H,W),
    `globals` (G,), `actors`, `kinds`, `args` (int64), `actor_mask` (3N+1,), `kind_mask` (10,),
    `arg_mask` (N,) (bool), `logp`, `values`, `rewards`, `terminated`, `truncated` (bool),
    `next_values` (the value of the final observation where `truncated`); and `last_value` (E,),
    the value of the observations after the last step."""

    def __init__(self, num_steps: int, num_envs: int, spec: dict, device="cpu"):
        n = spec["height"] * spec["width"]
        te = (num_steps, num_envs)

        def zeros(*shape, dtype=torch.float32):
            return torch.zeros(*te, *shape, dtype=dtype, device=device)

        self.num_steps, self.num_envs, self.device = num_steps, num_envs, device
        self.board = zeros(spec["channels"], spec["height"], spec["width"])
        self.globals = zeros(spec["globals"])
        self.actors = zeros(dtype=torch.int64)
        self.kinds = zeros(dtype=torch.int64)
        self.args = zeros(dtype=torch.int64)
        self.actor_mask = zeros(spec["num_actors"], dtype=torch.bool)
        self.kind_mask = zeros(len(spec["action_types"]), dtype=torch.bool)
        self.arg_mask = zeros(n, dtype=torch.bool)
        self.logp = zeros()
        self.values = zeros()
        self.rewards = zeros()
        self.terminated = zeros(dtype=torch.bool)
        self.truncated = zeros(dtype=torch.bool)
        self.next_values = zeros()
        self.last_value = torch.zeros(num_envs, device=device)
        # Wall-clock split of the last rollout (seconds): env stepping vs the policy forward pass.
        self.env_time = 0.0
        self.policy_time = 0.0


def collect_rollout(net, vec_env, buffer: RolloutBuffer, generator: torch.Generator) -> list:
    """Fill `buffer` with `num_steps` steps of `vec_env` (already reset) under `net`; return the
    `EpisodeStat`s of the episodes that finished."""
    device = buffer.device
    fetch = {"actor": vec_env.actor_masks, "kind": vec_env.kind_masks, "arg": vec_env.arg_masks}
    stats = []
    buffer.env_time = buffer.policy_time = 0.0

    def tensors(board, glob):
        return torch.from_numpy(board).to(device), torch.from_numpy(glob).to(device)

    board, glob = tensors(*vec_env.observations())
    for t in range(buffer.num_steps):
        t0 = time.perf_counter()
        with torch.no_grad():
            out = net.act(net.trunk(board, glob), lambda stage, *a: fetch[stage](*a), generator)
        t1 = time.perf_counter()
        step = vec_env.step(out.actors.cpu().numpy(), out.kinds.cpu().numpy(), out.args.cpu().numpy())
        stats += vec_env.episode_stats()
        t2 = time.perf_counter()
        buffer.board[t], buffer.globals[t] = board, glob
        buffer.actors[t], buffer.kinds[t], buffer.args[t] = out.actors, out.kinds, out.args
        buffer.actor_mask[t], buffer.kind_mask[t], buffer.arg_mask[t] = out.actor_mask, out.kind_mask, out.arg_mask
        buffer.logp[t], buffer.values[t] = out.logp, out.value
        buffer.rewards[t] = torch.from_numpy(step.reward).to(device)
        buffer.terminated[t] = torch.from_numpy(step.terminated).to(device)
        buffer.truncated[t] = torch.from_numpy(step.truncated).to(device)
        buffer.next_values[t] = 0.0
        # A truncated episode did not really end: bootstrap from the value of its final
        # observation (the one before the auto-reset), computed for those envs in one batch.
        cut = np.flatnonzero(step.truncated)
        with torch.no_grad():
            if cut.size:
                fb, fg = tensors(step.final_board[cut], step.final_globals[cut])
                buffer.next_values[t, torch.from_numpy(cut).to(device)] = net.value(net.trunk(fb, fg))
            board, glob = tensors(*vec_env.observations())
        buffer.env_time += t2 - t1
        buffer.policy_time += t1 - t0 + time.perf_counter() - t2
    with torch.no_grad():
        buffer.last_value[:] = net.value(net.trunk(board, glob))
    return stats


def train_update(net, optimizer, buffer: RolloutBuffer, cfg, generator: torch.Generator) -> dict:
    """GAE on `buffer`, then `cfg.update_epochs` epochs of shuffled minibatches. Uses the
    optimizer's current learning rate. Returns the update's metrics (`pg_loss`, `v_loss`,
    `entropy`, `entropy_actor`, `entropy_kind`, `entropy_arg`, `approx_kl`, `clipfrac`,
    `explained_variance`, `adv_mean`, `adv_std`)."""
    with torch.no_grad():
        adv, returns = compute_gae(
            buffer.rewards,
            buffer.values,
            buffer.terminated,
            buffer.truncated,
            buffer.next_values,
            buffer.last_value,
            cfg.gamma,
            cfg.gae_lambda,
        )
    # Flatten (T, E) into one batch of samples.
    n = buffer.num_steps * buffer.num_envs
    flat = {
        k: getattr(buffer, k).reshape(n, *getattr(buffer, k).shape[2:])
        for k in ("board", "globals", "actors", "kinds", "args", "actor_mask", "kind_mask", "arg_mask", "logp", "values")
    }
    adv, returns = adv.reshape(n), returns.reshape(n)

    sums: dict[str, float] = {}
    count = 0
    for _ in range(cfg.update_epochs):
        for idx in minibatch_indices(n, cfg.num_minibatches, generator):
            idx = idx.to(buffer.device)
            ev = net.evaluate(
                flat["board"][idx],
                flat["globals"][idx],
                flat["actors"][idx],
                flat["kinds"][idx],
                flat["args"][idx],
                flat["actor_mask"][idx],
                flat["kind_mask"][idx],
                flat["arg_mask"][idx],
            )
            out = ppo_loss(ev.logp, flat["logp"][idx], adv[idx], ev.value, flat["values"][idx], returns[idx], ev.entropy, cfg)
            optimizer.zero_grad()
            out.total.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            optimizer.step()
            step_metrics = {
                "pg_loss": out.pg,
                "v_loss": out.value,
                "entropy": out.entropy,
                "entropy_actor": ev.head_entropy["actor"].mean(),
                "entropy_kind": ev.head_entropy["kind"].mean(),
                "entropy_arg": ev.head_entropy["arg"].mean(),
                "approx_kl": out.approx_kl,
                "clipfrac": out.clipfrac,
            }
            for k, v in step_metrics.items():
                sums[k] = sums.get(k, 0.0) + v.item()
            count += 1
        # Optional early stop once the policy has moved too far from the rollout policy.
        if cfg.target_kl is not None and out.approx_kl.item() > cfg.target_kl:
            break

    metrics = {k: v / count for k, v in sums.items()}
    # Explained variance of the rollout values w.r.t. the returns: 1 is perfect, 0 is no better than
    # a constant; undefined (None) when the returns do not vary.
    var_r = returns.var().item()
    metrics["explained_variance"] = None if var_r == 0 else 1 - (returns - flat["values"]).var().item() / var_r
    metrics["adv_mean"] = adv.mean().item()
    metrics["adv_std"] = adv.std().item()
    return metrics
