"""PPO loss, rollout storage and the update step (docs/MILESTONE_10 "PPO", CleanRL conventions)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from polytopia_rl.gae import compute_gae, compute_gae_segments
from polytopia_rl.transfer import compute_teacher_logits, distill_kl
from polytopia_rl.vec_env import env_masks_fn


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
        self.segment_end: Tensor | None = None
        # Wall-clock split of the last rollout (seconds): env stepping vs the policy forward pass.
        self.env_time = 0.0
        self.policy_time = 0.0


def _obs_tensors(board: np.ndarray, glob: np.ndarray, device) -> tuple[Tensor, Tensor]:
    return torch.from_numpy(board).to(device), torch.from_numpy(glob).to(device)


def _copy_record(rec: dict) -> dict:
    return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in rec.items()}


def collect_rollout(net, vec_env, buffer: RolloutBuffer, generator: torch.Generator) -> list:
    """Fill `buffer` with `num_steps` steps of `vec_env` (already reset) under `net`; return the
    `EpisodeStat`s of the episodes that finished."""
    device = buffer.device
    fetch = {"actor": vec_env.actor_masks, "kind": vec_env.kind_masks, "arg": vec_env.arg_masks}
    stats = []
    buffer.segment_end = None
    buffer.env_time = buffer.policy_time = 0.0

    board, glob = _obs_tensors(*vec_env.observations(), device)
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
                fb, fg = _obs_tensors(step.final_board[cut], step.final_globals[cut], device)
                buffer.next_values[t, torch.from_numpy(cut).to(device)] = net.value(net.trunk(fb, fg))
            board, glob = _obs_tensors(*vec_env.observations(), device)
        buffer.env_time += t2 - t1
        buffer.policy_time += t1 - t0 + time.perf_counter() - t2
    with torch.no_grad():
        buffer.last_value[:] = net.value(net.trunk(board, glob))
    return stats


def collect_rollout_interleaved(net, vec_env, buffer: RolloutBuffer, generator: torch.Generator) -> list:
    """Fill `buffer` (`[t_i, i]` per slot) from the tick interface of a `SelfPlayVecEnv` (already
    reset) under `net`; return the `EpisodeStat`s of the episodes that finished, in completion order
    (docs/MILESTONE_13 "Collecting a rollout")."""
    device = buffer.device
    T, n = buffer.num_steps, buffer.num_envs
    t = [0] * n  # per slot: transitions written so far
    stats = []
    buffer.segment_end = None
    buffer.env_time = buffer.policy_time = 0.0

    while True:
        ready = vec_env.learner_slots()
        if len(ready) == n and all(x == T for x in t):
            break
        acting = [i for i in ready if t[i] < T]
        actions = {}
        t0 = time.perf_counter()
        if acting:
            # One learner forward pass for the slots on their learner's move with rows left.
            envs = [vec_env.envs[i] for i in acting]
            obs = [env.observation() for env in envs]
            board, glob = _obs_tensors(np.stack([b for b, _ in obs]), np.stack([g for _, g in obs]), device)
            with torch.no_grad():
                out = net.act(net.trunk(board, glob), env_masks_fn(envs, vec_env.num_tiles), generator)
            ti = torch.tensor([t[i] for i in acting], device=device)
            si = torch.tensor(acting, device=device)
            buffer.board[ti, si], buffer.globals[ti, si] = board, glob
            buffer.actors[ti, si], buffer.kinds[ti, si], buffer.args[ti, si] = out.actors, out.kinds, out.args
            buffer.actor_mask[ti, si], buffer.kind_mask[ti, si], buffer.arg_mask[ti, si] = out.actor_mask, out.kind_mask, out.arg_mask
            buffer.logp[ti, si], buffer.values[ti, si] = out.logp, out.value
            actors_list = out.actors.cpu().tolist()
            kinds_list = out.kinds.cpu().tolist()
            args_list = out.args.cpu().tolist()
            for i, a, k, x in zip(acting, actors_list, kinds_list, args_list):
                actions[i] = (a, k, x)
        t1 = time.perf_counter()
        done = vec_env.tick(actions)
        stats += vec_env.episode_stats()
        t2 = time.perf_counter()
        if done:
            slots = [tr.slot for tr in done]
            ti = torch.tensor([t[i] for i in slots], dtype=torch.long, device=device)
            si = torch.tensor(slots, dtype=torch.long, device=device)
            buffer.rewards[ti, si] = torch.tensor([tr.reward for tr in done], dtype=torch.float32, device=device)
            buffer.terminated[ti, si] = torch.tensor([bool(tr.terminated) for tr in done], dtype=torch.bool, device=device)
            buffer.truncated[ti, si] = torch.tensor([bool(tr.truncated) for tr in done], dtype=torch.bool, device=device)
            buffer.next_values[ti, si] = 0.0
            # A truncated episode did not really end: bootstrap from the value of its final
            # observation (the one before the auto-reset), computed for those slots in one batch.
            cut = [tr for tr in done if tr.truncated]
            if cut:
                fb, fg = _obs_tensors(np.stack([tr.final_board for tr in cut]), np.stack([tr.final_globals for tr in cut]), device)
                with torch.no_grad():
                    values = net.value(net.trunk(fb, fg))
                cut_slots = [tr.slot for tr in cut]
                cut_ti = torch.tensor([t[i] for i in cut_slots], dtype=torch.long, device=device)
                cut_si = torch.tensor(cut_slots, dtype=torch.long, device=device)
                buffer.next_values[cut_ti, cut_si] = values
            for i in slots:
                t[i] += 1
        buffer.env_time += t2 - t1
        buffer.policy_time += t1 - t0 + time.perf_counter() - t2
    with torch.no_grad():
        buffer.last_value[:] = net.value(net.trunk(*_obs_tensors(*vec_env.observations(), device)))
    return stats


class StreamState:
    """Pending in-flight learner steps and overflow completed transitions carried across
    `collect_rollout_stream` boundaries."""

    def __init__(self):
        self.pending: dict[tuple[int, int], dict] = {}
        self.carryover: list[tuple[tuple[int, int], dict]] = []

    def state_dict(self) -> dict:
        return {
            "pending": [[list(k), _copy_record(v)] for k, v in sorted(self.pending.items())],
            "carryover": [[list(k), _copy_record(v)] for k, v in self.carryover],
        }

    def load_state_dict(self, state: dict) -> None:
        self.pending = {(int(k[0]), int(k[1])): _copy_record(v) for k, v in state["pending"]}
        self.carryover = [((int(k[0]), int(k[1])), _copy_record(v)) for k, v in state["carryover"]]


def collect_rollout_stream(
    net,
    vec_env,
    buffer: RolloutBuffer,
    generator: torch.Generator,
    stream_state: StreamState | None = None,
) -> list:
    """Barrier-free collector: fills `buffer` with exactly `num_steps * num_envs` completed
    transitions grouped into contiguous `(slot, seat)` GAE lanes without draining mid-turn slots
    at rollout boundaries (docs/MILESTONE_23)."""
    device = buffer.device
    T, E = buffer.num_steps, buffer.num_envs
    batch = T * E
    state = StreamState() if stream_state is None else stream_state
    stats = []
    buffer.env_time = buffer.policy_time = 0.0
    lanes: dict[tuple[int, int], list[dict]] = {}
    collected = 0

    if state.carryover:
        take = min(batch, len(state.carryover))
        for lane_id, rec in state.carryover[:take]:
            lanes.setdefault(lane_id, []).append(rec)
        del state.carryover[:take]
        collected += take

    while collected < batch:
        acting = vec_env.learner_slots()
        actions = {}
        t0 = time.perf_counter()
        if acting:
            envs = [vec_env.envs[i] for i in acting]
            seats = [vec_env.learner_seat(i) for i in acting]
            obs = [env.observation() for env in envs]
            boards_np = [b.copy() for b, _ in obs]
            globs_np = [g.copy() for _, g in obs]
            board, glob = _obs_tensors(np.stack(boards_np), np.stack(globs_np), device)
            with torch.no_grad():
                out = net.act(net.trunk(board, glob), env_masks_fn(envs, vec_env.num_tiles), generator)
            actors_list = out.actors.cpu().tolist()
            kinds_list = out.kinds.cpu().tolist()
            args_list = out.args.cpu().tolist()
            am_np = out.actor_mask.cpu().numpy()
            km_np = out.kind_mask.cpu().numpy()
            xm_np = out.arg_mask.cpu().numpy()
            lp_list = out.logp.cpu().tolist()
            val_list = out.value.cpu().tolist()
            for idx, (slot, seat) in enumerate(zip(acting, seats)):
                act = (actors_list[idx], kinds_list[idx], args_list[idx])
                actions[slot] = act
                state.pending[(slot, seat)] = {
                    "board": boards_np[idx],
                    "globals": globs_np[idx],
                    "action": act,
                    "actor_mask": am_np[idx].copy(),
                    "kind_mask": km_np[idx].copy(),
                    "arg_mask": xm_np[idx].copy(),
                    "logp": lp_list[idx],
                    "value": val_list[idx],
                }
        t1 = time.perf_counter()
        done = vec_env.tick(actions)
        stats += vec_env.episode_stats()
        t2 = time.perf_counter()
        for tr in done:
            lane_id = (tr.slot, tr.seat)
            step_rec = state.pending.pop(lane_id)
            rec = {
                **step_rec,
                "reward": float(tr.reward),
                "terminated": bool(tr.terminated),
                "truncated": bool(tr.truncated),
                "final_board": None if tr.final_board is None else tr.final_board.copy(),
                "final_globals": None if tr.final_globals is None else tr.final_globals.copy(),
            }
            if collected < batch:
                lanes.setdefault(lane_id, []).append(rec)
                collected += 1
            else:
                state.carryover.append((lane_id, rec))
        buffer.env_time += t2 - t1
        buffer.policy_time += t1 - t0 + time.perf_counter() - t2

    first_carry_val: dict[tuple[int, int], float] = {}
    for c_lane, c_rec in state.carryover:
        if c_lane not in first_carry_val:
            first_carry_val[c_lane] = c_rec["value"]

    ordered: list[dict] = []
    seg_end_list: list[bool] = []
    next_val_list: list[float] = []
    trunc_indices: list[int] = []
    trunc_obs: list[tuple[np.ndarray, np.ndarray]] = []
    live_boot_indices: list[int] = []
    live_boot_slots: list[int] = []

    for lane_id in sorted(lanes):
        recs = lanes[lane_id]
        last_j = len(recs) - 1
        for j, rec in enumerate(recs):
            flat_idx = len(ordered)
            is_end = j == last_j
            ordered.append(rec)
            seg_end_list.append(is_end)
            if rec["truncated"]:
                next_val_list.append(0.0)
                trunc_indices.append(flat_idx)
                trunc_obs.append((rec["final_board"], rec["final_globals"]))
            elif is_end and not rec["terminated"]:
                if lane_id in first_carry_val:
                    next_val_list.append(first_carry_val[lane_id])
                elif lane_id in state.pending:
                    next_val_list.append(state.pending[lane_id]["value"])
                else:
                    next_val_list.append(0.0)
                    live_boot_indices.append(flat_idx)
                    live_boot_slots.append(lane_id[0])
            else:
                next_val_list.append(0.0)

    def fill_boot(indices: list[int], obs_list: list[tuple[np.ndarray, np.ndarray]]) -> None:
        if not indices:
            return
        b, g = _obs_tensors(np.stack([x for x, _ in obs_list]), np.stack([y for _, y in obs_list]), device)
        for idx, v in zip(indices, net.value(net.trunk(b, g)).cpu().tolist()):
            next_val_list[idx] = v

    t_boot = time.perf_counter()
    with torch.no_grad():
        fill_boot(trunc_indices, trunc_obs)
        if live_boot_indices:
            fill_boot(live_boot_indices, [vec_env.envs[s].observation() for s in live_boot_slots])
    buffer.policy_time += time.perf_counter() - t_boot

    actions_t = torch.tensor([r["action"] for r in ordered], dtype=torch.int64).reshape(T, E, 3)
    buffer.board.copy_(torch.from_numpy(np.stack([r["board"] for r in ordered])).reshape(T, E, *buffer.board.shape[2:]))
    buffer.globals.copy_(torch.from_numpy(np.stack([r["globals"] for r in ordered])).reshape(T, E, -1))
    buffer.actors.copy_(actions_t[..., 0])
    buffer.kinds.copy_(actions_t[..., 1])
    buffer.args.copy_(actions_t[..., 2])
    buffer.actor_mask.copy_(torch.from_numpy(np.stack([r["actor_mask"] for r in ordered])).reshape(T, E, -1))
    buffer.kind_mask.copy_(torch.from_numpy(np.stack([r["kind_mask"] for r in ordered])).reshape(T, E, -1))
    buffer.arg_mask.copy_(torch.from_numpy(np.stack([r["arg_mask"] for r in ordered])).reshape(T, E, -1))
    buffer.logp.copy_(torch.tensor([r["logp"] for r in ordered], dtype=torch.float32).reshape(T, E))
    buffer.values.copy_(torch.tensor([r["value"] for r in ordered], dtype=torch.float32).reshape(T, E))
    buffer.rewards.copy_(torch.tensor([r["reward"] for r in ordered], dtype=torch.float32).reshape(T, E))
    buffer.terminated.copy_(torch.tensor([r["terminated"] for r in ordered], dtype=torch.bool).reshape(T, E))
    buffer.truncated.copy_(torch.tensor([r["truncated"] for r in ordered], dtype=torch.bool).reshape(T, E))
    buffer.next_values.copy_(torch.tensor(next_val_list, dtype=torch.float32).reshape(T, E))
    buffer.last_value.zero_()
    buffer.segment_end = torch.tensor(seg_end_list, dtype=torch.bool, device=device)
    return stats


def train_update(
    net,
    optimizer,
    buffer: RolloutBuffer,
    cfg,
    generator: torch.Generator,
    *,
    teacher=None,
    teacher_spec: dict | None = None,
    student_spec: dict | None = None,
    distill_coef: float = 0.0,
) -> dict:
    """GAE on `buffer`, then `cfg.update_epochs` epochs of shuffled minibatches. Uses the
    optimizer's current learning rate. Returns the update's metrics (`pg_loss`, `v_loss`,
    `entropy`, `entropy_actor`, `entropy_kind`, `entropy_arg`, `approx_kl`, `clipfrac`,
    `explained_variance`, `adv_mean`, `adv_std`; plus `distill_kl` and `distill_coef` when
    `teacher` is provided)."""
    n = buffer.num_steps * buffer.num_envs
    with torch.no_grad():
        if buffer.segment_end is not None:
            adv, returns = compute_gae_segments(
                buffer.rewards.reshape(n),
                buffer.values.reshape(n),
                buffer.terminated.reshape(n),
                buffer.truncated.reshape(n),
                buffer.next_values.reshape(n),
                buffer.segment_end,
                cfg.gamma,
                cfg.gae_lambda,
            )
        else:
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
            adv, returns = adv.reshape(n), returns.reshape(n)
    # Flatten (T, E) into one batch of samples.
    flat = {
        k: getattr(buffer, k).reshape(n, *getattr(buffer, k).shape[2:])
        for k in ("board", "globals", "actors", "kinds", "args", "actor_mask", "kind_mask", "arg_mask", "logp", "values")
    }

    use_distill = teacher is not None and distill_coef > 0.0
    if use_distill:
        if teacher_spec is None or student_spec is None:
            raise ValueError("teacher_spec and student_spec are required when teacher is set")
        t_actor, t_kind, t_arg = compute_teacher_logits(
            teacher, teacher_spec, student_spec, flat["board"], flat["globals"], flat["actors"], flat["kinds"]
        )
        distill_temp = getattr(cfg, "distill_temp", 1.0)

    sums: dict[str, float] = {}
    count = 0
    for _ in range(cfg.update_epochs):
        epoch_kl = 0.0
        epoch_mbs = 0
        for idx in minibatch_indices(n, cfg.num_minibatches, generator):
            idx = idx.to(buffer.device)
            eval_kwargs = {"return_logits": True} if use_distill else {}
            ev = net.evaluate(
                flat["board"][idx],
                flat["globals"][idx],
                flat["actors"][idx],
                flat["kinds"][idx],
                flat["args"][idx],
                flat["actor_mask"][idx],
                flat["kind_mask"][idx],
                flat["arg_mask"][idx],
                **eval_kwargs,
            )
            out = ppo_loss(ev.logp, flat["logp"][idx], adv[idx], ev.value, flat["values"][idx], returns[idx], ev.entropy, cfg)
            total_loss = out.total
            if use_distill:
                d_kl, _ = distill_kl(
                    (t_actor[idx], t_kind[idx], t_arg[idx]),
                    ev.logits,
                    flat["kinds"][idx],
                    flat["actor_mask"][idx],
                    flat["kind_mask"][idx],
                    flat["arg_mask"][idx],
                    teacher_spec,
                    student_spec,
                    temp=distill_temp,
                )
                d_kl_mean = d_kl.mean()
                total_loss = total_loss + distill_coef * d_kl_mean
            optimizer.zero_grad()
            total_loss.backward()
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
            if use_distill:
                step_metrics["distill_kl"] = d_kl_mean
            for k, v in step_metrics.items():
                val = v.item()
                sums[k] = sums.get(k, 0.0) + val
                if k == "approx_kl":
                    epoch_kl += val
            count += 1
            epoch_mbs += 1
        # Optional early stop once the policy has moved too far from the rollout policy.
        if cfg.target_kl is not None and epoch_kl / epoch_mbs > cfg.target_kl:
            break

    metrics = {k: v / count for k, v in sums.items()}
    if teacher is not None:
        if not use_distill:
            metrics["distill_kl"] = 0.0
        metrics["distill_coef"] = float(distill_coef)
    # Explained variance of the rollout values w.r.t. the returns: 1 is perfect, 0 is no better than
    # a constant; undefined (None) when the returns do not vary.
    var_r = returns.var().item()
    metrics["explained_variance"] = None if var_r == 0 else 1 - (returns - flat["values"]).var().item() / var_r
    metrics["adv_mean"] = adv.mean().item()
    metrics["adv_std"] = adv.std().item()
    return metrics
