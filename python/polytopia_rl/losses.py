"""Loss review: replays of lost games (docs/MILESTONE_11 "Loss review").

`python -m polytopia_rl.losses --checkpoint CKPT --opponent rush --count 5 --out DIR`
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import polytopia_sim

from polytopia_rl.evaluate import _play
from polytopia_rl.suite import DEFAULT_TURN_CAP, SUITE_SEED_BASE, loss_type


@dataclass
class LossRecord:
    seed: int
    seat: int
    final_turn: int  # info()["turn"] at the end
    loss_type: str  # "capital_captured" (game ended before the turn cap) or "score"
    scores: tuple  # info()["scores"] at the end, (seat 0, seat 1)
    first_city_lost_turn: int | None  # see collect_losses
    path: str  # the replay file


def collect_losses(policy_fn, opponent, seeds, env_kwargs, out_dir, limit) -> list[LossRecord]:
    """Play seeds `SUITE_SEED_BASE + j` for `j in range(seeds)`, seat 0 then seat 1, greedy, against
    `opponent` (epsilon 0); keep the lost games in that order, stopping after `limit`. Each is
    written to `out_dir/{opponent}-{seed}-seat{s}.json` (`Env.replay_json()`).

    `first_city_lost_turn`: `info()["turn"]` at the first observation of the agent (a decision or
    the final observation) where a tile that held one of its cities at its previous observation no
    longer does; None if that never happens."""
    kwargs = {k: v for k, v in env_kwargs.items() if k not in ("opponent", "opponent_epsilon", "agent_seat")}
    turn_cap = env_kwargs.get("turn_cap", DEFAULT_TURN_CAP)
    os.makedirs(out_dir, exist_ok=True)
    out = []
    for j in range(seeds):
        for seat in (0, 1):
            if len(out) >= limit:
                return out
            seed = SUITE_SEED_BASE + j
            env = polytopia_sim.Env(**kwargs, opponent=opponent, opponent_epsilon=0.0, agent_seat=seat)
            env.reset(seed)
            watcher = _CityWatcher()
            game = _play(policy_fn, env, seat, True, watch=watcher)
            if game["outcome"] != "loss":
                continue
            info = env.info()
            path = os.path.join(out_dir, f"{opponent}-{seed}-seat{seat}.json")
            with open(path, "w") as f:
                f.write(env.replay_json())
            out.append(
                LossRecord(
                    seed=seed,
                    seat=seat,
                    final_turn=info["turn"],
                    loss_type=loss_type(game, turn_cap),
                    scores=tuple(info["scores"]),
                    first_city_lost_turn=watcher.first_lost,
                    path=path,
                )
            )
    return out


class _CityWatcher:
    """Records `info()["turn"]` the first time a tile showing `city.mine` stops showing it."""

    def __init__(self):
        self.prev = None
        self.first_lost = None

    def __call__(self, env):
        board, _ = env.observation()
        c = list(env.spec()["channel_names"]).index("city.mine")
        cur = set(zip(*np.nonzero(board[c] > 0.5)))
        if self.first_lost is None and self.prev is not None and self.prev - cur:
            self.first_lost = env.info()["turn"]
        self.prev = cur


def summary(records: list[LossRecord]) -> str:
    lines = [
        "| seed | seat | final turn | loss type | scores | first city lost |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in records:
        lines.append(f"| {r.seed} | {r.seat} | {r.final_turn} | {r.loss_type} | {r.scores[0]}-{r.scores[1]} | {r.first_city_lost_turn} |")
    n = len(records)
    shares = [sum(r.seat == s for r in records) / n if n else 0.0 for s in (0, 1)]
    lines.append("")
    lines.append(f"{n} losses; share by seat: seat 0 {shares[0]:.2f}, seat 1 {shares[1]:.2f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    from polytopia_rl.checkpoint import describe, load_net
    from polytopia_rl.evaluate import net_policy
    from polytopia_rl.train import resolve_device
    from polytopia_rl.vec_env import env_kwargs_of

    p = argparse.ArgumentParser(description="Replays of a checkpoint's lost games.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--opponent", default="rush")
    p.add_argument("--count", type=int, default=5)
    p.add_argument("--seeds", type=int, default=200, help="suite seeds to search")
    p.add_argument("--out", required=True)
    p.add_argument("--allow-obs-mismatch", action="store_true")
    a = p.parse_args(argv)
    device = resolve_device("auto")
    net, ck = load_net(a.checkpoint, device, strict_obs=not a.allow_obs_mismatch)
    print(describe(a.checkpoint, ck), flush=True)
    recs = collect_losses(net_policy(net, device=device), a.opponent, a.seeds, env_kwargs_of(ck.cfg), a.out, a.count)
    print(summary(recs))


if __name__ == "__main__":
    main()
