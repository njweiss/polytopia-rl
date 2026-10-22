"""Side-swapped evaluation (docs/MILESTONE_10 "Evaluation").

`python -m polytopia_rl.evaluate --checkpoint CKPT --opponents random,economy,rush --seeds 200`
"""

from __future__ import annotations

from dataclasses import dataclass

EVAL_SEED_BASE = 1_000_000_000


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
    per_game: list  # dicts {"seed", "seat", "outcome", "length", "score_diff"}, seed order, seat 0 first


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    raise NotImplementedError


def evaluate(policy_fn, opponent: str, seeds: int, env_kwargs: dict, greedy: bool) -> EvalResult:
    """Play seeds `EVAL_SEED_BASE + j` for `j in range(seeds)`, each as seat 0 then seat 1, against
    `opponent` with epsilon 0. `env_kwargs` go to `polytopia_sim.Env` (opponent, epsilon and seat are
    set here). `policy_fn(env, greedy) -> (actor, kind, arg)`."""
    raise NotImplementedError


def net_policy(net, generator=None, device="cpu"):
    """`policy_fn` for a `PolicyNet`: argmax of every masked head when greedy, else sampled with
    `generator`."""
    raise NotImplementedError


def main(argv: list[str] | None = None) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
