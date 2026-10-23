"""The fixed evaluation suite (docs/MILESTONE_11 "Evaluation suite").

`python -m polytopia_rl.suite --checkpoint CKPT [--seeds 200] [--out DIR]` (writes `DIR/suite.json`
and `DIR/suite.md`), `python -m polytopia_rl.suite --compare A.json B.json`.
"""

from __future__ import annotations

from dataclasses import dataclass

SUITE_SEED_BASE = 1_000_000_000  # suite seeds: [SUITE_SEED_BASE, SUITE_SEED_LIMIT)
SUITE_SEED_LIMIT = 1_500_000_000


@dataclass
class Condition:
    name: str
    opponent: str  # "random" | "economy" | "rush"
    epsilon: float = 0.0
    full_observability: bool = False
    greedy: bool = True


DEFAULT_SUITE: list[Condition] = []  # the brief's table, in table order


@dataclass
class ConditionResult:
    name: str
    games: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    wilson_lo: float
    wilson_hi: float
    by_seat: dict  # {0: win rate as seat 0, 1: win rate as seat 1}
    mean_length: float
    mean_score_diff: float
    loss_capital_share: float  # share of the losses that were capital captures (0 without losses)
    loss_score_share: float  # share of the losses decided by score at the cap (0 without losses)
    per_game: list  # evaluate()'s per-game dicts plus "loss_type" (None unless lost)


@dataclass
class SuiteResult:
    conditions: dict  # name -> ConditionResult, in the order run
    obs_hash: str  # of the build that ran the suite (Env.spec())
    rules_hash: str
    step: int | None  # training step of the evaluated weights, if known


@dataclass
class Comparison:
    name: str
    win_rate_a: float
    win_rate_b: float
    diff: float  # win_rate_b - win_rate_a
    z: float  # two-proportion z statistic with the pooled proportion (0 when the pooled sd is 0)
    significant: bool  # |z| > 1.96


def run_suite(policy_fn, conditions, seeds, env_kwargs, *, step=None) -> SuiteResult:
    """Each condition plays seeds `SUITE_SEED_BASE + j` for `j in range(seeds)`, each seed as seat 0
    then seat 1, against the condition's opponent with its epsilon and full observability, calling
    `policy_fn(env, condition.greedy)`. `env_kwargs` give the map and turn cap. A seed range that
    reaches `SUITE_SEED_LIMIT` raises `ValueError`."""
    raise NotImplementedError


def compare(a: SuiteResult, b: SuiteResult) -> list[Comparison]:
    """One `Comparison` per condition present in both, in `a`'s order."""
    raise NotImplementedError


def suite_markdown(result) -> str:
    raise NotImplementedError


def main(argv: list[str] | None = None) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
