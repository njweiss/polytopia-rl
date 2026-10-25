"""Side-swapped evaluation (docs/MILESTONE_10 "Evaluation").

`python -m polytopia_rl.evaluate --checkpoint CKPT --opponents random,economy,rush --seeds 200`
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
import polytopia_sim
import torch

from polytopia_rl.vec_env import TRAIN_SEED_LIMIT as EVAL_SEED_BASE
from polytopia_rl.vec_env import agent_outcome, env_kwargs_of


@dataclass
class EvalResult:
    games: int
    wins: int
    losses: int
    draws: int
    win_rate: float  # wins / games
    wilson_lo: float
    wilson_hi: float
    by_seat: dict  # {0: win rate as seat 0, 1: win rate as seat 1}
    mean_length: float  # agent actions per game
    mean_score_diff: float  # agent score minus opponent score at the end
    per_game: list  # dicts {"seed", "seat", "outcome", "length", "score_diff", "turn"}, seed order, seat 0 first


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion `wins / n`."""
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _play(policy_fn, env, seat: int, greedy: bool, watch=None) -> dict:
    """One game to the end; the result from the agent seat's perspective. `watch(env)`, if given,
    sees every agent observation: the first one and the one after each agent action."""
    length = 0
    if watch:
        watch(env)
    while True:
        actor, kind, arg = policy_fn(env, greedy)
        _, _, terminated, truncated = env.step_factored(actor, kind, arg)
        length += 1
        if watch:
            watch(env)
        if terminated or truncated:
            break
    return _game_result(env, seat, length)


def _game_result(env, seat: int, length: int) -> dict:
    info = env.info()
    scores = info["scores"]
    return {"seed": info["seed"], "seat": seat, "outcome": agent_outcome(info["outcome"], seat), "length": length, "score_diff": scores[seat] - scores[1 - seat], "turn": info["turn"]}


def _play_batched(batch_fn, envs: list, seats: list) -> list:
    """All the games together in rounds: each round, one `batch_fn` call over the live games in game
    order, then one step of each; a game leaves when it terminates or truncates."""
    length = [0] * len(envs)
    live = list(range(len(envs)))
    while live:
        actions = batch_fn([envs[g] for g in live])
        still = []
        for g, (actor, kind, arg) in zip(live, actions):
            _, _, terminated, truncated = envs[g].step_factored(actor, kind, arg)
            length[g] += 1
            if not (terminated or truncated):
                still.append(g)
        live = still
    return [_game_result(env, seat, n) for env, seat, n in zip(envs, seats, length)]


def evaluate(policy_fn, opponent: str, seeds: int, env_kwargs: dict, greedy: bool, *, seed_base: int = EVAL_SEED_BASE, epsilon: float = 0.0, batch_fn=None) -> EvalResult:
    """Play seeds `seed_base + j` for `j in range(seeds)`, each as seat 0 then seat 1, against
    `opponent` with `epsilon`. `env_kwargs` go to `polytopia_sim.Env` (opponent, epsilon and seat are
    set here). `policy_fn(env, greedy) -> (actor, kind, arg)`. `turn` in a game's dict is the final
    `info()["turn"]`.

    `batch_fn(envs) -> one (actor, kind, arg) per env` (greediness bound in; docs/MILESTONE_12
    "Batched evaluation"): when given, `policy_fn` is ignored and all the games run together in
    rounds, one `batch_fn` call per round over the live games in game order."""
    kwargs = {k: v for k, v in env_kwargs.items() if k not in ("opponent", "opponent_epsilon", "agent_seat")}

    def make_env(j, seat):
        # Same seed, same map; only the side changes.
        env = polytopia_sim.Env(**kwargs, opponent=opponent, opponent_epsilon=epsilon, agent_seat=seat)
        env.reset(seed_base + j)
        return env

    if batch_fn is not None:
        games = [(j, seat) for j in range(seeds) for seat in (0, 1)]
        per_game = _play_batched(batch_fn, [make_env(j, seat) for j, seat in games], [seat for _, seat in games])
    else:
        per_game = [_play(policy_fn, make_env(j, seat), seat, greedy) for j in range(seeds) for seat in (0, 1)]
    games = len(per_game)
    wins = sum(g["outcome"] == "win" for g in per_game)
    losses = sum(g["outcome"] == "loss" for g in per_game)
    by_seat = {}
    for seat in (0, 1):
        mine = [g for g in per_game if g["seat"] == seat]
        by_seat[seat] = sum(g["outcome"] == "win" for g in mine) / len(mine) if mine else 0.0
    lo, hi = wilson_interval(wins, games)
    return EvalResult(
        games=games,
        wins=wins,
        losses=losses,
        draws=games - wins - losses,
        win_rate=wins / games if games else 0.0,
        wilson_lo=lo,
        wilson_hi=hi,
        by_seat=by_seat,
        mean_length=float(np.mean([g["length"] for g in per_game])) if games else 0.0,
        mean_score_diff=float(np.mean([g["score_diff"] for g in per_game])) if games else 0.0,
        per_game=per_game,
    )


def net_policy(net, generator=None, device="cpu"):
    """`policy_fn` for a `PolicyNet`: argmax of every masked head when greedy, else sampled with
    `generator`."""
    if generator is None:
        generator = torch.Generator().manual_seed(0)

    def policy_fn(env, greedy):
        board, glob = env.observation()
        n = board.shape[1] * board.shape[2]

        def masks_fn(stage, actors=None, kinds=None):
            if stage == "actor":
                return env.actor_mask()[None]
            if stage == "kind":
                return env.kind_mask(int(actors[0]))[None]
            m = np.zeros((1, n), dtype=bool)
            xm = env.arg_mask(int(actors[0]), int(kinds[0]))
            m[0, : xm.size] = xm
            return m

        with torch.inference_mode():
            feats = net.trunk(torch.from_numpy(board)[None].to(device), torch.from_numpy(glob)[None].to(device))
            out = net.act(feats, masks_fn, generator, greedy=greedy, stats=False)
        return int(out.actors[0]), int(out.kinds[0]), int(out.args[0])

    return policy_fn


def main(argv: list[str] | None = None) -> None:
    from polytopia_rl.checkpoint import describe, load_net
    from polytopia_rl.train import resolve_device

    p = argparse.ArgumentParser(description="Side-swapped evaluation of a checkpoint.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--opponents", default="random,economy,rush")
    p.add_argument("--seeds", type=int, default=200)
    a = p.parse_args(argv)
    device = resolve_device("auto")
    net, ck = load_net(a.checkpoint, device)
    cfg = ck.cfg
    print(describe(a.checkpoint, ck), flush=True)
    for opponent in a.opponents.split(","):
        res = evaluate(net_policy(net, device=device), opponent, a.seeds, env_kwargs_of(cfg), greedy=True)
        print(
            f"{opponent:8s} games {res.games}  W-L-D {res.wins}-{res.losses}-{res.draws}  "
            f"win rate {res.win_rate:.3f} [{res.wilson_lo:.3f}, {res.wilson_hi:.3f}]  "
            f"seat0 {res.by_seat[0]:.3f} seat1 {res.by_seat[1]:.3f}  "
            f"len {res.mean_length:.1f}  score diff {res.mean_score_diff:+.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
