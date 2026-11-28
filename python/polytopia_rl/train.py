"""Training driver (docs/MILESTONE_10).

`python -m polytopia_rl.train --config CONFIG [--run-dir DIR] [--resume CKPT] [--init-checkpoint CKPT] [key=value ...]`

`RUNDIR` gets `metrics.jsonl` (one line per update), `eval.jsonl`, and the checkpoints
`ckpt_{global_step}.pt` and `latest.pt` (written at every evaluation, i.e. each time global_step
passes a multiple of `cfg.eval_every`, and at the end) and `best.pt`. A checkpoint
(`polytopia_rl.checkpoint`) is a dict with at least `model`, `optimizer`, `config` (JSON),
`global_step`, `stage`, `vec_env` (the vec env's `state_dict()`, which holds the next training
seeds), `rng` and `spec` (`Env.spec()`, with `obs_hash` and `rules_hash`); self-play checkpoints also
hold the pool. With `--resume`, `--init-checkpoint` is ignored.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import json
import os
import random
import time
from collections import Counter

import numpy as np
import polytopia_sim
import torch

from polytopia_rl.checkpoint import SpecMismatch, describe, load_checkpoint, load_net, save_checkpoint
from polytopia_rl.config import PPOConfig, load_config
from polytopia_rl.curriculum import StageHistory, next_stage
from polytopia_rl.evaluate import env_kwargs_of, evaluate, net_policy
from polytopia_rl.model import PolicyNet
from polytopia_rl.pfsp import WinRateTracker, parse_bot_specs, shaping_schedule
from polytopia_rl.ppo import (
    RolloutBuffer,
    StreamState,
    collect_rollout,
    collect_rollout_interleaved,
    collect_rollout_stream,
    train_update,
)
from polytopia_rl.rating import fit_ratings
from polytopia_rl.selfplay import (
    BOT_OPPONENTS,
    LIVE_SELF,
    Entry,
    SnapshotPool,
    make_selfplay_vec_env,
    net_batch_policy,
    regression,
)
from polytopia_rl.stacked import StackedSnapshots, stacked_batch_policy
from polytopia_rl.suite import DEFAULT_SUITE, run_suite
from polytopia_rl.tournament import BOTS, Player, anchor_of, play_pair, round_robin, write_tournament
from polytopia_rl.transfer import distill_schedule, load_warm_start
from polytopia_rl.vec_env import make_vec_env


TRAIN_SEEDS_PER_RUN = 10_000_000  # run seed s trains on map seeds [train_seed_base(s), train_seed_base(s + 1))


def train_seed_base(seed: int) -> int:
    """`(seed - 1) * TRAIN_SEEDS_PER_RUN`; `ValueError` unless `1 <= seed <= 100` (docs/MILESTONE_12
    "Seed-varied training maps")."""
    if not 1 <= seed <= 100:
        raise ValueError(f"run seed {seed} is outside 1..100")
    return (seed - 1) * TRAIN_SEEDS_PER_RUN


def _check_run_seeds(vec, limit: int) -> None:
    """Raise once an episode in progress uses a map seed of the next run seed's range."""
    seeds = vec.state_dict()["seeds"]
    if max(seeds) >= limit:
        raise ValueError(f"training seed {max(seeds)} reaches the next run seed's range (>= {limit})")


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def env_spec(cfg) -> dict:
    return polytopia_sim.Env(size=cfg.size, ruleset=getattr(cfg, "ruleset", "v1")).spec()


def stage_config(cfg, stage: int):
    """The config with the stage's overrides applied (in practice its `opponents`)."""
    return dataclasses.replace(cfg, **cfg.stages[stage])


def _rng_state(gen: torch.Generator) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "sample": gen.get_state(),
    }


def _set_rng_state(state: dict, gen: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    gen.set_state(state["sample"])


@contextlib.contextmanager
def _preserve_torch_rng():
    state = torch.get_rng_state()
    try:
        yield
    finally:
        torch.set_rng_state(state)


def _weighted_win_rate(tracker: WinRateTracker, names: list[str]) -> float | None:
    games = sum(tracker.games(n) for n in names)
    if games == 0:
        return None
    return sum(tracker.win_rate(n) * tracker.games(n) for n in names) / games


def _should_add_snapshot(cfg, pool: SnapshotPool, tracker: WinRateTracker, global_step: int, last_snapshot_step: int, batch: int) -> bool:
    snap_wr = getattr(cfg, "snapshot_win_rate", None)
    if snap_wr is None:
        return global_step // cfg.snapshot_every > (global_step - batch) // cfg.snapshot_every
    elapsed = global_step - last_snapshot_step
    if elapsed < cfg.snapshot_every:
        return False
    latest = pool.latest_snapshot()
    min_games = getattr(cfg, "snapshot_min_games", 20)
    max_steps = getattr(cfg, "snapshot_max_steps", None)
    gated_ok = latest is not None and tracker.games(latest.name) >= min_games and tracker.win_rate(latest.name) >= snap_wr
    forced = max_steps is not None and elapsed >= max_steps
    return gated_ok or forced


def _window_metrics(recent: list[dict]) -> dict:
    """Return, win rate and length over the finished episodes in the window (None if empty)."""
    if not recent:
        return {"mean_return": None, "train_win_rate": None, "mean_episode_len": None}
    return {
        "mean_return": float(np.mean([e["ret"] for e in recent])),
        "train_win_rate": float(np.mean([e["outcome"] == "win" for e in recent])),
        "mean_episode_len": float(np.mean([e["length"] for e in recent])),
    }


def run(cfg, run_dir: str, resume: str | None = None, init_checkpoint: str | None = None):
    """Train for `cfg.total_steps` agent steps and return the `PolicyNet`. With `resume`, model,
    optimizer, counters, curriculum state, RNG states and env state come from that checkpoint, and
    the run continues as if it had never stopped. With `init_checkpoint`, only the model weights
    come from that checkpoint (obs hash enforced); optimizer and LR schedule start fresh. Every run
    ends with `ckpt_final.pt` (the final weights, as `latest.pt`).

    `cfg.selfplay` (docs/MILESTONE_11 "Self-play"): the learner plays `SelfPlayVecEnv` against the
    pool and the bots instead of the curriculum. Snapshot 0 is the initial weights; then one every
    `snapshot_every` steps. `eval.jsonl` gets rows `{"type": "suite", "global_step", "suite": {name:
    {"win_rate", ...}}, "regression"}` every `eval_every` steps (conditions economy_e0, rush_e0,
    rush_e20, `eval_seeds` seeds) and `{"type": "rating", "global_step", "rating": {"learner",
    "snapshot_0"}}` every `rating_every` steps. Metrics rows also have `pool_size`. At the end:
    `pool/snap_{step}.pt` (checkpoints), `tournament.json`, `tournament.md`
    (`tournament_seeds` seeds) and `train.log`."""
    os.makedirs(run_dir, exist_ok=True)
    device = resolve_device(cfg.device)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)  # all action sampling and minibatch shuffling

    spec = env_spec(cfg)
    kinds = spec["action_types"]
    net = PolicyNet(spec, width=cfg.width, blocks=cfg.blocks, arch=getattr(cfg, "arch", "v1")).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)
    batch = cfg.num_envs * cfg.num_steps
    num_updates = cfg.total_steps // batch
    anneal_updates = (cfg.lr_anneal_steps or cfg.total_steps) // batch  # H (docs/MILESTONE_12c)
    distill_steps = getattr(cfg, "distill_steps", None)
    if distill_steps is None:
        distill_updates = None
    elif distill_steps == 0:
        distill_updates = 0
    else:
        distill_updates = max(1, distill_steps // batch)

    teacher = None
    teacher_spec = None
    if getattr(cfg, "teacher_checkpoint", None) is not None:
        # Preserve global torch RNG so teacher construction does not perturb fresh/resumed runs.
        with _preserve_torch_rng():
            teacher, teacher_ck = load_net(cfg.teacher_checkpoint, device=device, strict_obs=True)
        teacher.eval().requires_grad_(False)
        teacher_spec = teacher_ck.spec if teacher_ck.spec is not None else env_spec(teacher_ck.cfg)
        print(f"teacher {describe(cfg.teacher_checkpoint, teacher_ck)}", flush=True)

    # Self-play state: the pool, win-rate tracker, anchors, opponent-draw RNG and action generator.
    pool = SnapshotPool(cfg.pool_max, cfg.p_latest) if cfg.selfplay else None
    tracker = WinRateTracker(alpha=getattr(cfg, "pfsp_ema_alpha", 0.05)) if cfg.selfplay else None
    stream_state = StreamState() if (cfg.selfplay and cfg.rollout == "stream") else None
    anchors: list[tuple[str, PolicyNet, object]] = []
    anchor_templates: dict[str, PolicyNet] = {}
    anchor_cfgs: dict[str, object] = {}
    if cfg.selfplay and getattr(cfg, "anchor_checkpoints", None):
        with _preserve_torch_rng():
            for i, anchor_path in enumerate(cfg.anchor_checkpoints):
                anchor_net, anchor_ck = load_net(anchor_path, device=device, strict_obs=True)
                if anchor_ck.spec is None or anchor_ck.spec["obs_hash"] != spec["obs_hash"]:
                    saved_h = None if anchor_ck.spec is None else anchor_ck.spec["obs_hash"]
                    raise SpecMismatch(f"{anchor_path}: obs_hash {saved_h} differs from run spec {spec['obs_hash']}")
                anchor_net.eval().requires_grad_(False)
                name = f"anchor_{i}"
                anchors.append((name, anchor_net, anchor_ck.cfg))
                anchor_templates[name] = anchor_net
                anchor_cfgs[name] = anchor_ck.cfg
                print(f"{name} {describe(anchor_path, anchor_ck)}", flush=True)

    opp_rng = np.random.default_rng([cfg.seed, 11])
    opp_gen = torch.Generator(device=device).manual_seed(cfg.seed + 1)

    shaping_anneal_steps = getattr(cfg, "shaping_anneal_steps", None)
    shaping_anneal_updates = max(1, shaping_anneal_steps // batch) if shaping_anneal_steps is not None else None
    raw_sp_bots = getattr(cfg, "selfplay_bots", None)
    sp_bot_names = (
        [b.name for b in parse_bot_specs(raw_sp_bots, default_epsilon=cfg.opponent_epsilon)]
        if raw_sp_bots is not None
        else list(BOT_OPPONENTS)
    )

    def make_policy(entry):
        snap = _frozen(net, entry)  # no fresh init, so the global torch RNG is untouched
        return net_batch_policy(snap, opp_gen, greedy=cfg.opponent_greedy, device=device)

    def new_vec(seed_base):
        if cfg.selfplay:
            stacked_policy = None
            if cfg.opponent_forward in ("stacked", "auto"):
                stack = StackedSnapshots(copy.deepcopy(net), device)
                stacked_policy = stacked_batch_policy(stack, opp_gen, cfg.opponent_greedy)
            return make_selfplay_vec_env(
                cfg,
                seed_base,
                pool=pool,
                make_policy=make_policy,
                rng=opp_rng,
                stacked_policy=stacked_policy,
                tracker=tracker,
            )
        return make_vec_env(stage_config(cfg, stage), seed_base=seed_base)

    # Run state: everything a checkpoint has to carry for an exact resume.
    global_step, update, stage = 0, 0, 0
    history = StageHistory()
    recent: list[dict] = []  # the last `stage_window` finished episodes of this stage
    best_score = None
    last_snapshot_step = 0
    last_regression: bool | None = None
    if resume is None:
        warm_start = getattr(cfg, "warm_start_checkpoint", None)
        if init_checkpoint is not None and warm_start is not None:
            raise ValueError("init_checkpoint and warm_start_checkpoint are mutually exclusive")
        if init_checkpoint is not None:
            # Warm start: weights only (obs hash enforced); fresh optimizer and LR schedule.
            ck = load_checkpoint(init_checkpoint, spec=spec)
            print(describe(init_checkpoint, ck), flush=True)
            net.load_state_dict(ck.model)
        elif warm_start is not None:
            ck = load_warm_start(net, warm_start, spec)
            print(describe(warm_start, ck), flush=True)
        if cfg.selfplay:
            for name, anchor_net, _ in anchors:
                pool.add(anchor_net.state_dict(), 0, pinned=True, name=name, net_template=anchor_net)
            pool.add(net.state_dict(), 0)
        vec = new_vec(train_seed_base(cfg.seed))
        vec.reset_all()
    else:
        ck = load_checkpoint(resume, spec=spec)
        print(describe(resume, ck), flush=True)
        net.load_state_dict(ck.model)
        optimizer.load_state_dict(ck.optimizer)
        global_step, stage = ck.global_step, ck.stage
        update = ck.extra["update"]
        history = StageHistory(**ck.extra["history"])
        recent = ck.extra["recent"]
        best_score = ck.extra["best_score"]
        if cfg.selfplay:
            for e in ck.extra["pool"]:
                name = e.get("name", f"snap_{e['step']}")
                pinned = bool(e.get("pinned", False))
                tpl = anchor_templates.get(name) if pinned else None
                pool.entries.append(Entry(e["step"], e["state"], name, pinned=pinned, net_template=tpl))
            if ck.extra.get("win_rate_tracker") is not None:
                tracker.load_state_dict(ck.extra["win_rate_tracker"])
            last_snapshot_step = ck.extra.get("last_snapshot_step", 0)
            last_regression = ck.extra.get("last_regression", None)
            if stream_state is not None and ck.extra.get("stream_state") is not None:
                stream_state.load_state_dict(ck.extra["stream_state"])
            opp_rng.bit_generator.state = ck.rng_state["opponent_rng"]
            opp_gen.set_state(ck.rng_state["opponent_gen"])
        vec = new_vec(ck.seed_index["seed_base"])
        vec.load_state_dict(ck.seed_index)
        _set_rng_state(ck.rng_state, gen)
    seed_limit = train_seed_base(cfg.seed) + TRAIN_SEEDS_PER_RUN  # no overlap with the next run seed
    _check_run_seeds(vec, seed_limit)
    buffer = RolloutBuffer(cfg.num_steps, cfg.num_envs, spec, device=device)

    def save(names: list[str]) -> None:
        rng = _rng_state(gen)
        extra = {"update": update, "history": dataclasses.asdict(history), "recent": recent, "best_score": best_score}
        if cfg.selfplay:
            rng.update(opponent_rng=opp_rng.bit_generator.state, opponent_gen=opp_gen.get_state())
            extra["pool"] = [{"step": e.step, "state": e.state, "name": e.name, "pinned": e.pinned} for e in pool.entries]
            extra["win_rate_tracker"] = tracker.state_dict()
            extra["last_snapshot_step"] = last_snapshot_step
            extra["last_regression"] = last_regression
            if stream_state is not None:
                extra["stream_state"] = stream_state.state_dict()
        for name in names:
            save_checkpoint(
                os.path.join(run_dir, name),
                model=net,
                optimizer=optimizer,
                cfg=cfg,
                global_step=global_step,
                stage=stage,
                seed_index=vec.state_dict(),
                rng_state=rng,
                spec=spec,
                extra=extra,
            )

    def crossed(every: int) -> bool:
        return global_step // every > (global_step - batch) // every

    mode = "a" if resume else "w"
    with (
        open(os.path.join(run_dir, "metrics.jsonl"), mode) as mfile,
        open(os.path.join(run_dir, "eval.jsonl"), mode) as efile,
        open(os.path.join(run_dir, "train.log"), mode) as logfile,
    ):
        saved_at = None
        while update < num_updates:
            update += 1
            # Linear LR annealing (CleanRL) over H updates: full lr at update 1; off with `anneal_lr` false.
            lr = cfg.lr * (1 - (update - 1) / anneal_updates) if cfg.anneal_lr else cfg.lr
            for group in optimizer.param_groups:
                group["lr"] = lr

            cur_shaping = shaping_schedule(update, cfg.shaping_weight, getattr(cfg, "shaping_end", 0.0), shaping_anneal_updates)
            if cfg.selfplay and shaping_anneal_steps is not None:
                vec.set_shaping_weight(cur_shaping)

            t0 = time.perf_counter()
            if cfg.selfplay and cfg.rollout == "stream":
                stats = collect_rollout_stream(net, vec, buffer, gen, stream_state)
            elif cfg.selfplay and cfg.rollout == "interleaved":
                stats = collect_rollout_interleaved(net, vec, buffer, gen)
            else:
                stats = collect_rollout(net, vec, buffer, gen)
            _check_run_seeds(vec, seed_limit)
            t1 = time.perf_counter()
            d_coef = (
                distill_schedule(update, getattr(cfg, "distill_coef", 0.0), num_updates, distill_updates)
                if teacher is not None
                else 0.0
            )
            m = train_update(
                net,
                optimizer,
                buffer,
                cfg,
                gen,
                teacher=teacher,
                teacher_spec=teacher_spec,
                student_spec=spec,
                distill_coef=d_coef,
            )
            t2 = time.perf_counter()
            global_step += batch
            history.steps_in_stage += batch
            for s in stats:
                history.outcomes.append(s.outcome)
                recent.append({"outcome": s.outcome, "ret": s.ret, "length": s.length, "opponent": s.opponent})
                if cfg.selfplay and s.opponent != LIVE_SELF:
                    tracker.record(s.opponent, s.outcome)
            # Only the last window matters, for the curriculum and for the logged rates.
            history.outcomes = history.outcomes[-cfg.stage_window :]
            recent = recent[-cfg.stage_window :]
            if cfg.selfplay and _should_add_snapshot(cfg, pool, tracker, global_step, last_snapshot_step, batch):
                pool.add(net.state_dict(), global_step)
                last_snapshot_step = global_step

            counts = np.bincount(buffer.kinds.flatten().cpu().numpy(), minlength=len(kinds))
            kind_freq = {name: float(c) / batch for name, c in zip(kinds, counts)}
            row = {
                "global_step": global_step,
                "update": update,
                "sps": batch / (t2 - t0),
                "rollout_sps": batch / (t1 - t0),
                "update_sps": batch / (t2 - t1),
                "env_sps": batch / buffer.env_time if buffer.env_time > 0 else 0.0,
                "forward_sps": batch / buffer.policy_time if buffer.policy_time > 0 else 0.0,
                "rollout_time": t1 - t0,
                "update_time": t2 - t1,
                "stage": stage,
                "lr": lr,
                **m,
                **_window_metrics(recent),
                "kind_freq": kind_freq,
                "end_turn_freq": kind_freq["end_turn"],
            }
            if cfg.selfplay:
                row["pool_size"] = len(pool.entries)
                row["shaping_weight"] = float(cur_shaping)
                latest_entry = pool.latest_snapshot()
                anchor_names = [e.name for e in pool.entries if e.pinned]
                sp_candidates = list(dict.fromkeys(sp_bot_names + anchor_names + ([latest_entry.name] if latest_entry is not None else [])))
                row["sp_win_rate"] = {n: tracker.win_rate(n) for n in sp_candidates if tracker.games(n) > 0}
                row["win_rate_bots"] = _weighted_win_rate(tracker, sp_bot_names)
                row["win_rate_pool"] = _weighted_win_rate(tracker, [e.name for e in pool.entries])
                row["win_rate_latest"] = (
                    tracker.win_rate(latest_entry.name)
                    if (latest_entry is not None and tracker.games(latest_entry.name) > 0)
                    else None
                )
                mix = Counter(e["opponent"] for e in recent[-200:])
                logfile.write(f"step {global_step} pool {[e.name for e in pool.entries]} opponents (last {sum(mix.values())} episodes) {dict(sorted(mix.items()))}\n")
                logfile.flush()
            mfile.write(json.dumps(row) + "\n")
            mfile.flush()

            evaluated = crossed(cfg.eval_every)
            is_best = False
            if evaluated and cfg.selfplay:
                # The suite subset for the current weights; the regression flag does not stop the run.
                subset = [c for c in DEFAULT_SUITE if c.name in SELFPLAY_SUBSET]
                res = run_suite(
                    net_policy(net, device=device),
                    subset,
                    cfg.eval_seeds,
                    env_kwargs_of(cfg),
                    step=global_step,
                    make_batch_fn=lambda greedy: net_batch_policy(net, torch.Generator().manual_seed(0), greedy=greedy, device=device),
                )
                rates = {name: r.win_rate for name, r in res.conditions.items()}
                reg = regression(rates, cfg.regression_floor)
                last_regression = reg
                erow = {
                    "type": "suite",
                    "global_step": global_step,
                    "suite": {name: {k: v for k, v in dataclasses.asdict(r).items() if k != "per_game"} for name, r in res.conditions.items()},
                    "regression": reg,
                }
                efile.write(json.dumps(erow) + "\n")
                efile.flush()
            elif evaluated:
                # Periodic greedy evaluation against this stage's opponents, then a checkpoint.
                opponents = list(dict.fromkeys(cfg.stages[stage]["opponents"]))
                erow = {"global_step": global_step, "stage": stage}
                for opp in opponents:
                    res = evaluate(
                        net_policy(net, device=device),
                        opp,
                        cfg.eval_seeds,
                        env_kwargs_of(cfg),
                        greedy=True,
                        batch_fn=net_batch_policy(net, torch.Generator().manual_seed(0), greedy=True, device=device),
                    )
                    erow[opp] = {k: v for k, v in dataclasses.asdict(res).items() if k != "per_game"}
                efile.write(json.dumps(erow) + "\n")
                efile.flush()
                # The best checkpoint is ranked by the final stage's score only.
                if stage == len(cfg.stages) - 1:
                    score = float(np.mean([erow[o]["win_rate"] for o in opponents]))
                    is_best = best_score is None or score > best_score
                    best_score = score if is_best else best_score
            if cfg.selfplay and crossed(cfg.rating_every):
                ratings = learner_ratings(net, pool, make_policy=lambda e: net_batch_policy(_frozen(net, e), torch.Generator(), greedy=True, device=device), seeds=cfg.rating_seeds, env_kwargs=env_kwargs_of(cfg), device=device)
                efile.write(json.dumps({"type": "rating", "global_step": global_step, **ratings}) + "\n")
                efile.flush()
                if last_regression is False and (best_score is None or ratings["rating"]["learner"] > best_score):
                    best_score = ratings["rating"]["learner"]
                    is_best = True

            # Curriculum: on advance, new opponents and fresh seeds; policy, optimizer and LR
            # schedule carry on.
            new_stage = stage if cfg.selfplay else next_stage(stage, history, cfg)
            if new_stage != stage:
                state = vec.state_dict()
                seed_base = state["seed_base"] + max(state["next_k"]) * vec.num_envs * state["seed_stride"]
                stage = new_stage
                history = StageHistory()
                recent = []
                vec = new_vec(seed_base)
                vec.reset_all()

            if evaluated:
                save([f"ckpt_{global_step}.pt", "latest.pt"] + (["best.pt"] if is_best else []))
                saved_at = global_step
            elif is_best:
                save(["best.pt"])
            print(
                f"update {update}/{num_updates} step {global_step} stage {row['stage']} sps {row['sps']:.0f} "
                f"win {row['train_win_rate']} ent {m['entropy']:.3f} kl {m['approx_kl']:.4f}",
                flush=True,
            )
        if saved_at != global_step:
            save([f"ckpt_{global_step}.pt", "latest.pt"])
        save(["ckpt_final.pt"])  # every run, so the experiment harness can find finished runs
        if cfg.selfplay:
            finish_selfplay(net, pool, cfg, spec, run_dir, device, anchor_cfgs=anchor_cfgs)
    return net


SELFPLAY_SUBSET = ("economy_e0", "rush_e0", "rush_e20")  # the suite conditions monitored during self-play


def _frozen(net, entry):
    """A copy of `entry.net_template` (or `net`) with `entry`'s weights."""
    base = entry.net_template if entry.net_template is not None else net
    snap = copy.deepcopy(base)
    snap.load_state_dict(entry.state)
    return snap


def learner_ratings(net, pool, make_policy, seeds, env_kwargs, device) -> dict:
    """The current weights (greedy) against snapshot 0, the newest snapshot, three snapshots spread
    evenly over the pool and the three bots, `seeds` seeds each with seats swapped; Bradley-Terry
    ratings (random bot = 0) of the learner and every opponent from those games."""
    k = len(pool.entries)
    picks = [0, k - 1] + [int(round(x)) for x in np.linspace(0, k - 1, 5)[1:4]]
    entries = [pool.entries[i] for i in dict.fromkeys(picks)]
    snap0 = next((e for e in pool.entries if not e.pinned), pool.entries[0])
    if all(e.name != snap0.name for e in entries):
        entries.append(snap0)
    players = [Player("learner", batch_fn=net_batch_policy(net, torch.Generator(), greedy=True, device=device))]
    players += [Player(e.name, batch_fn=make_policy(e)) for e in entries]
    players += [Player(b, bot=b) for b in BOTS]
    n = len(players)
    wins, games = np.zeros((n, n)), np.zeros((n, n))
    for j in range(1, n):
        r = play_pair(players[0], players[j], seeds, env_kwargs)
        wins[0, j], wins[j, 0] = r.a_wins + 0.5 * r.draws, r.b_wins + 0.5 * r.draws
        games[0, j] = games[j, 0] = r.games
    names = [p.name for p in players]
    ratings = fit_ratings(wins, games, anchor=anchor_of(names))
    learner, snapshot_0 = float(ratings[0]), float(ratings[names.index(snap0.name)])
    return {
        "rating": {"learner": learner, "snapshot_0": snapshot_0},
        "gap_snapshot_0": learner - snapshot_0,
        "players": {name: float(r) for name, r in zip(names, ratings)},
        "learner_score": {names[j]: wins[0, j] / games[0, j] for j in range(1, n)},
    }


def finish_selfplay(net, pool, cfg, spec, run_dir, device, anchor_cfgs: dict | None = None) -> None:
    """Write the pool as `pool/{name}.pt` (snapshots and pinned anchors) and the round robin of
    the pool and the bots."""
    os.makedirs(os.path.join(run_dir, "pool"), exist_ok=True)
    players = [Player(b, bot=b) for b in BOTS]
    for e in pool.entries:
        snap = _frozen(net, e)
        entry_cfg = anchor_cfgs[e.name] if (anchor_cfgs and e.name in anchor_cfgs) else cfg
        save_checkpoint(
            os.path.join(run_dir, "pool", f"{e.name}.pt"),
            model=snap,
            optimizer=None,
            cfg=entry_cfg,
            global_step=e.step,
            stage=0,
            seed_index=None,
            rng_state=None,
            spec=spec,
        )
        players.append(Player(e.name, batch_fn=net_batch_policy(snap, torch.Generator(), greedy=True, device=device)))
    res = round_robin(players, cfg.tournament_seeds, env_kwargs_of(cfg))
    write_tournament(res, run_dir)


def load_policy(path: str, device: str = "cpu"):
    """`(PolicyNet, PPOConfig)` from a checkpoint (enforces `obs_hash`, warns on `rules_hash` or
    unstamped checkpoints)."""
    net, ck = load_net(path, device=device)
    return net, ck.cfg


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="PPO training against the bots.")
    p.add_argument("--config", required=True)
    p.add_argument("--run-dir", help="default: runs/<config name>_seed<seed>")
    p.add_argument("--resume", help="checkpoint to continue from")
    p.add_argument("--init-checkpoint", help="checkpoint whose weights start the run (fresh optimizer)")
    p.add_argument("overrides", nargs="*", help="key=value config overrides")
    a = p.parse_args(argv)
    cfg = load_config(a.config, a.overrides)
    run_dir = a.run_dir or os.path.join("runs", f"{os.path.splitext(os.path.basename(a.config))[0]}_seed{cfg.seed}")
    run(cfg, run_dir, resume=a.resume, init_checkpoint=a.init_checkpoint)


if __name__ == "__main__":
    main()
