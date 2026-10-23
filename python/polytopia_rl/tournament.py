"""Side-swapped matches and round robins (docs/MILESTONE_11 "Ratings", "Self-play").

`python -m polytopia_rl.tournament --run RUNDIR [--seeds 100]`: round robin of `RUNDIR/pool/*.pt`
plus the three bots, written to `RUNDIR/tournament.json` and `RUNDIR/tournament.md`.

`tournament.json` is `{"players", "wins", "draws", "games", "seat_win_rate", "ratings"}`: the
`TournamentResult` fields, matrices as nested lists, `seat_win_rate` keyed `"0"` and `"1"`.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polytopia_sim

from polytopia_rl.rating import fit_ratings

RATING_SEED_BASE = 2_000_000_000


@dataclass
class Player:
    name: str
    bot: str | None = None  # "random" | "economy" | "rush": the Rust bot at epsilon 0
    policy_fn: Callable | None = None  # policy_fn(env, greedy) -> (actor, kind, arg), as in evaluate
    greedy: bool = True


@dataclass
class PairResult:
    a: str
    b: str
    games: int
    a_wins: int
    b_wins: int
    draws: int
    seat_wins: dict  # {0: games won by seat 0, 1: games won by seat 1}
    per_game: list  # dicts {"seed", "a_seat", "winner": "a" | "b" | None, "length"}: seed order, a as seat 0 first;
    # "length" counts the actions chosen on the Python side (by policy players, or by bots in a bot-vs-bot game)


@dataclass
class TournamentResult:
    players: list  # names
    wins: np.ndarray  # wins[i, j]: decisive games i won against j
    draws: np.ndarray  # symmetric
    games: np.ndarray  # symmetric; wins[i, j] + wins[j, i] + draws[i, j] == games[i, j]
    seat_win_rate: dict  # {0: share of all games won by seat 0, 1: by seat 1}
    ratings: np.ndarray  # fit_ratings(wins + 0.5 * draws, games), anchored at "random" = 0 (else player 0)


def play_pair(a: Player, b: Player, seeds: int, env_kwargs: dict) -> PairResult:
    """Seeds `RATING_SEED_BASE + j` for `j in range(seeds)`, each played twice: `a` as seat 0, then
    `a` as seat 1. A bot against a policy plays in the bot-mode env (as `evaluate` does); any other
    pairing plays in a two-seat env, bots moving through `Env.bot_action`."""
    kwargs = {k: v for k, v in env_kwargs.items() if k not in ("opponent", "opponent_epsilon", "agent_seat")}
    per_game = []
    for j in range(seeds):
        seed = RATING_SEED_BASE + j
        for a_seat in (0, 1):
            seats = {a_seat: a, 1 - a_seat: b}
            bots = [s for s in (0, 1) if seats[s].bot is not None]
            if len(bots) == 1:
                agent = 1 - bots[0]
                env = polytopia_sim.Env(**kwargs, opponent=seats[bots[0]].bot, opponent_epsilon=0.0, agent_seat=agent)
            else:
                env = polytopia_sim.Env(**kwargs, opponent="none")
            env.reset(seed)
            length = 0
            done = False
            while not done:
                p = seats[env.info()["current_player"]]
                action = env.bot_action(p.bot) if p.bot is not None else p.policy_fn(env, p.greedy)
                _, _, terminated, truncated = env.step_factored(*action)
                length += 1
                done = terminated or truncated
            out = env.info()["outcome"]
            winner = {f"win_{a_seat}": "a", f"win_{1 - a_seat}": "b"}.get(out)
            per_game.append({"seed": seed, "a_seat": a_seat, "winner": winner, "length": length})
    a_wins = sum(g["winner"] == "a" for g in per_game)
    b_wins = sum(g["winner"] == "b" for g in per_game)
    seat0 = sum((g["winner"] == "a") == (g["a_seat"] == 0) for g in per_game if g["winner"] is not None)
    return PairResult(
        a=a.name,
        b=b.name,
        games=len(per_game),
        a_wins=a_wins,
        b_wins=b_wins,
        draws=len(per_game) - a_wins - b_wins,
        seat_wins={0: seat0, 1: a_wins + b_wins - seat0},
        per_game=per_game,
    )


def round_robin(players: list, seeds: int, env_kwargs: dict) -> TournamentResult:
    """`play_pair` for every pair `i < j`, in index order."""
    n = len(players)
    wins, draws, games = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
    seat_wins = [0, 0]
    for i in range(n):
        for j in range(i + 1, n):
            r = play_pair(players[i], players[j], seeds, env_kwargs)
            wins[i, j], wins[j, i] = r.a_wins, r.b_wins
            draws[i, j] = draws[j, i] = r.draws
            games[i, j] = games[j, i] = r.games
            seat_wins[0] += r.seat_wins[0]
            seat_wins[1] += r.seat_wins[1]
    names = [p.name for p in players]
    total = games.sum() / 2
    return TournamentResult(
        players=names,
        wins=wins,
        draws=draws,
        games=games,
        seat_win_rate={s: seat_wins[s] / total if total else 0.0 for s in (0, 1)},
        ratings=fit_ratings(wins + 0.5 * draws, games, anchor=anchor_of(names)),
    )


def anchor_of(names: list) -> int:
    """The `random` bot, else player 0."""
    return names.index("random") if "random" in names else 0


def tournament_markdown(result: TournamentResult) -> str:
    names = result.players
    lines = ["| player | Elo |", "| --- | --- |"]
    for i in np.argsort(-result.ratings, kind="stable"):
        lines.append(f"| {names[i]} | {result.ratings[i]:.0f} |")
    lines += [
        "",
        f"Seat win rate: seat 0 {result.seat_win_rate[0]:.3f}, seat 1 {result.seat_win_rate[1]:.3f}",
        "",
        "Score of row against column (wins + draws / 2, over games):",
        "",
        "| | " + " | ".join(names) + " |",
        "| --- |" + " --- |" * len(names),
    ]
    for i, name in enumerate(names):
        cells = []
        for j in range(len(names)):
            g = result.games[i, j]
            cells.append("" if g == 0 else f"{(result.wins[i, j] + 0.5 * result.draws[i, j]) / g:.2f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_tournament(result: TournamentResult, out_dir: str) -> None:
    """`out_dir/tournament.json` and `out_dir/tournament.md`."""
    d = {
        "players": result.players,
        "wins": result.wins.tolist(),
        "draws": result.draws.tolist(),
        "games": result.games.tolist(),
        "seat_win_rate": {str(k): v for k, v in result.seat_win_rate.items()},
        "ratings": result.ratings.tolist(),
    }
    with open(os.path.join(out_dir, "tournament.json"), "w") as f:
        json.dump(d, f, indent=1)
    with open(os.path.join(out_dir, "tournament.md"), "w") as f:
        f.write(tournament_markdown(result))


BOTS = ("random", "economy", "rush")


def main(argv: list[str] | None = None) -> None:
    from polytopia_rl.checkpoint import load_net
    from polytopia_rl.evaluate import net_policy
    from polytopia_rl.train import resolve_device
    from polytopia_rl.vec_env import env_kwargs_of

    p = argparse.ArgumentParser(description="Round robin of a run's pool snapshots and the bots.")
    p.add_argument("--run", required=True)
    p.add_argument("--seeds", type=int, default=100)
    a = p.parse_args(argv)
    device = resolve_device("auto")
    paths = sorted(glob.glob(os.path.join(a.run, "pool", "snap_*.pt")), key=lambda q: int(os.path.basename(q)[5:-3]))
    players = [Player(name, bot=name) for name in BOTS]
    cfg, spec = None, None
    for path in paths:
        net, ck = load_net(path, device)
        cfg, spec = ck.cfg, ck.spec
        players.append(Player(os.path.basename(path)[:-3], policy_fn=net_policy(net, device=device)))
    if spec:
        print(f"run {a.run}: obs_hash {spec['obs_hash']} rules_hash {spec['rules_hash']} snapshots {[p.name for p in players[3:]]}", flush=True)
    res = round_robin(players, a.seeds, env_kwargs_of(cfg) if cfg else {})
    write_tournament(res, a.run)
    print(tournament_markdown(res), end="")


if __name__ == "__main__":
    main()
