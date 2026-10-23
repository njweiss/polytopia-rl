"""Side-swapped matches and round robins (docs/MILESTONE_11 "Ratings", "Self-play").

`python -m polytopia_rl.tournament --run RUNDIR [--seeds 100]`: round robin of `RUNDIR/pool/*.pt`
plus the three bots, written to `RUNDIR/tournament.json` and `RUNDIR/tournament.md`.

`tournament.json` is `{"players", "wins", "draws", "games", "seat_win_rate", "ratings"}`: the
`TournamentResult` fields, matrices as nested lists, `seat_win_rate` keyed `"0"` and `"1"`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

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
    per_game: list  # dicts {"seed", "a_seat", "winner": "a" | "b" | None, "length"}: seed order, a as seat 0 first


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
    `a` as seat 1."""
    raise NotImplementedError


def round_robin(players: list, seeds: int, env_kwargs: dict) -> TournamentResult:
    """`play_pair` for every pair `i < j`, in index order."""
    raise NotImplementedError


def tournament_markdown(result: TournamentResult) -> str:
    raise NotImplementedError


def main(argv: list[str] | None = None) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
