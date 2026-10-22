"""Training driver (docs/MILESTONE_10).

`python -m polytopia_rl.train --config CONFIG [--run-dir DIR] [--resume CKPT] [key=value ...]`

`RUNDIR` gets `metrics.jsonl` (one line per update), `eval.jsonl`, and the checkpoints
`ckpt_{global_step}.pt` and `latest.pt` (written at every evaluation, i.e. each time global_step
passes a multiple of `cfg.eval_every`, and at the end) and `best.pt`. A checkpoint is a dict with
at least `model`, `optimizer`, `config` (JSON), `global_step`, `stage`, `vec_env`
(`SyncVecEnv.state_dict()`, which holds the next training seeds) and `rng`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import time

import numpy as np
import polytopia_sim
import torch

from polytopia_rl.config import PPOConfig, load_config
from polytopia_rl.curriculum import StageHistory, next_stage
from polytopia_rl.evaluate import env_kwargs_of, evaluate, net_policy
from polytopia_rl.model import PolicyNet
from polytopia_rl.ppo import RolloutBuffer, collect_rollout, train_update
from polytopia_rl.vec_env import make_vec_env


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def env_spec(cfg) -> dict:
    return polytopia_sim.Env(size=cfg.size).spec()


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


def _window_metrics(recent: list[dict]) -> dict:
    """Return, win rate and length over the finished episodes in the window (None if empty)."""
    if not recent:
        return {"mean_return": None, "train_win_rate": None, "mean_episode_len": None}
    return {
        "mean_return": float(np.mean([e["ret"] for e in recent])),
        "train_win_rate": float(np.mean([e["outcome"] == "win" for e in recent])),
        "mean_episode_len": float(np.mean([e["length"] for e in recent])),
    }


def run(cfg, run_dir: str, resume: str | None = None):
    """Train for `cfg.total_steps` agent steps and return the `PolicyNet`. With `resume`, model,
    optimizer, counters, curriculum state, RNG states and env state come from that checkpoint, and
    the run continues as if it had never stopped."""
    os.makedirs(run_dir, exist_ok=True)
    device = resolve_device(cfg.device)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)  # all action sampling and minibatch shuffling

    spec = env_spec(cfg)
    kinds = spec["action_types"]
    net = PolicyNet(spec, width=cfg.width, blocks=cfg.blocks).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)
    batch = cfg.num_envs * cfg.num_steps
    num_updates = cfg.total_steps // batch

    # Run state: everything a checkpoint has to carry for an exact resume.
    global_step, update, stage = 0, 0, 0
    history = StageHistory()
    recent: list[dict] = []  # the last `stage_window` finished episodes of this stage
    best_score = None
    if resume is None:
        vec = make_vec_env(stage_config(cfg, stage), seed_base=0)
        vec.reset_all()
    else:
        ck = torch.load(resume, map_location=device, weights_only=False)
        net.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        global_step, update, stage = ck["global_step"], ck["update"], ck["stage"]
        history = StageHistory(**ck["history"])
        recent = ck["recent"]
        best_score = ck["best_score"]
        vec = make_vec_env(stage_config(cfg, stage), seed_base=ck["vec_env"]["seed_base"])
        vec.load_state_dict(ck["vec_env"])
        _set_rng_state(ck["rng"], gen)
    buffer = RolloutBuffer(cfg.num_steps, cfg.num_envs, spec, device=device)

    def save(names: list[str]) -> None:
        ck = {
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg.to_json(),
            "global_step": global_step,
            "update": update,
            "stage": stage,
            "history": dataclasses.asdict(history),
            "recent": recent,
            "best_score": best_score,
            "vec_env": vec.state_dict(),
            "rng": _rng_state(gen),
        }
        for name in names:
            torch.save(ck, os.path.join(run_dir, name))

    mode = "a" if resume else "w"
    with open(os.path.join(run_dir, "metrics.jsonl"), mode) as mfile, open(os.path.join(run_dir, "eval.jsonl"), mode) as efile:
        saved_at = None
        while update < num_updates:
            update += 1
            # Linear LR annealing (CleanRL): full lr at update 1, lr / num_updates at the last.
            lr = cfg.lr * (1 - (update - 1) / num_updates)
            for group in optimizer.param_groups:
                group["lr"] = lr

            t0 = time.perf_counter()
            stats = collect_rollout(net, vec, buffer, gen)
            t1 = time.perf_counter()
            m = train_update(net, optimizer, buffer, cfg, gen)
            t2 = time.perf_counter()
            global_step += batch
            history.steps_in_stage += batch
            for s in stats:
                history.outcomes.append(s.outcome)
                recent.append({"outcome": s.outcome, "ret": s.ret, "length": s.length})
            # Only the last window matters, for the curriculum and for the logged rates.
            history.outcomes = history.outcomes[-cfg.stage_window :]
            recent = recent[-cfg.stage_window :]

            counts = np.bincount(buffer.kinds.flatten().cpu().numpy(), minlength=len(kinds))
            kind_freq = {name: float(c) / batch for name, c in zip(kinds, counts)}
            row = {
                "global_step": global_step,
                "update": update,
                "sps": batch / (t2 - t0),
                "env_sps": batch / buffer.env_time,
                "forward_sps": batch / buffer.policy_time,
                "rollout_time": t1 - t0,
                "update_time": t2 - t1,
                "stage": stage,
                "lr": lr,
                **m,
                **_window_metrics(recent),
                "kind_freq": kind_freq,
                "end_turn_freq": kind_freq["end_turn"],
            }
            mfile.write(json.dumps(row) + "\n")
            mfile.flush()

            # Periodic greedy evaluation against this stage's opponents, then a checkpoint.
            evaluated = global_step // cfg.eval_every > (global_step - batch) // cfg.eval_every
            if evaluated:
                opponents = list(dict.fromkeys(cfg.stages[stage]["opponents"]))
                erow = {"global_step": global_step, "stage": stage}
                for opp in opponents:
                    res = evaluate(net_policy(net, device=device), opp, cfg.eval_seeds, env_kwargs_of(cfg), greedy=True)
                    erow[opp] = {k: v for k, v in dataclasses.asdict(res).items() if k != "per_game"}
                efile.write(json.dumps(erow) + "\n")
                efile.flush()
                # The best checkpoint is ranked by the final stage's score only.
                is_best = False
                if stage == len(cfg.stages) - 1:
                    score = float(np.mean([erow[o]["win_rate"] for o in opponents]))
                    is_best = best_score is None or score > best_score
                    best_score = score if is_best else best_score

            # Curriculum: on advance, new opponents and fresh seeds; policy, optimizer and LR
            # schedule carry on.
            new_stage = next_stage(stage, history, cfg)
            if new_stage != stage:
                state = vec.state_dict()
                seed_base = state["seed_base"] + max(state["next_k"]) * vec.num_envs * state["seed_stride"]
                stage = new_stage
                history = StageHistory()
                recent = []
                vec = make_vec_env(stage_config(cfg, stage), seed_base=seed_base)
                vec.reset_all()

            if evaluated:
                save([f"ckpt_{global_step}.pt", "latest.pt"] + (["best.pt"] if is_best else []))
                saved_at = global_step
            print(
                f"update {update}/{num_updates} step {global_step} stage {row['stage']} sps {row['sps']:.0f} "
                f"win {row['train_win_rate']} ent {m['entropy']:.3f} kl {m['approx_kl']:.4f}",
                flush=True,
            )
        if saved_at != global_step:
            save([f"ckpt_{global_step}.pt", "latest.pt"])
    return net


def load_policy(path: str, device: str = "cpu"):
    """`(PolicyNet, PPOConfig)` from a checkpoint."""
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = PPOConfig.from_json(ck["config"])
    net = PolicyNet(env_spec(cfg), width=cfg.width, blocks=cfg.blocks).to(device)
    net.load_state_dict(ck["model"])
    return net, cfg


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="PPO training against the bots.")
    p.add_argument("--config", required=True)
    p.add_argument("--run-dir", help="default: runs/<config name>_seed<seed>")
    p.add_argument("--resume", help="checkpoint to continue from")
    p.add_argument("overrides", nargs="*", help="key=value config overrides")
    a = p.parse_args(argv)
    cfg = load_config(a.config, a.overrides)
    run_dir = a.run_dir or os.path.join("runs", f"{os.path.splitext(os.path.basename(a.config))[0]}_seed{cfg.seed}")
    run(cfg, run_dir, resume=a.resume)


if __name__ == "__main__":
    main()
