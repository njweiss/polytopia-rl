"""Multi-seed experiments (docs/MILESTONE_12 "Experiment runs", "Report").

`python -m polytopia_rl.experiment run SPEC [--parallel K] [--only ARM]`,
`python -m polytopia_rl.experiment report SPEC`.

Layout (paths relative to the working directory): run `(arm, seed)` lives in
`runs/exp/{name}/{arm}/seed{seed}` with `command.txt`, the trainer's files, `suite_{step}/suite.json`
(and `suite.md`) per evaluated checkpoint and, with an init checkpoint, `h2h_{step}.json`:
`dataclasses.asdict(play_pair(run, init, h2h_seeds, env_kwargs))` with the run's checkpoint as player
`a` and the init checkpoint as `b` (named `"init"`). The experiment round robin is
`runs/exp/{name}/tournament.json` and `.md` (`tournament.write_tournament`); its players are the bots
(`random`, `economy`, `rush`), `init` (if any) and each run's `ckpt_final.pt` as `{arm}_seed{seed}`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_TOTAL_STEPS = 10_000_000


@dataclass(kw_only=True)
class Spec:
    name: str
    base_config: str
    seeds: list
    arms: dict  # arm name -> config overrides (JSON values), in spec order
    total_steps: int = DEFAULT_TOTAL_STEPS
    init_checkpoint: str | None = None
    primary: list = field(default_factory=list)  # suite condition names and/or "elo_vs_init"
    suite_seeds: int
    tournament_seeds: int
    eval_checkpoints: int = 3
    h2h_seeds: int = 100


@dataclass
class ExperimentReport:
    name: str
    arms: list  # spec order
    seeds: list  # spec order
    # One dict per (arm, seed), arms outer and seeds inner: {"arm", "seed", "steps" (evaluated steps,
    # ascending), "suite": {condition: [win rate per step]}, "suite_mean": {condition: mean over the
    # steps}, "elo_vs_init": [Elo per step] or None, "elo_vs_init_mean": float or None, "obs_hash",
    # "rules_hash"}.
    runs: list
    # metric -> arm -> {"per_seed": [run value per seed, spec order], "mean", "sd" (ddof=1), "min",
    # "max"}; metrics are the suite conditions, then "elo_vs_init" when there is an init checkpoint.
    tables: dict
    # One dict per (pair of arms in spec order, metric): {"a", "b", "metric", "primary", "mean_a",
    # "mean_b", "diff", "t", "df", "p", "significant"} from `welch(per_seed[a], per_seed[b])`;
    # "significant" is a bool for primary metrics and None for exploratory ones.
    comparisons: list
    n_flagged: int  # comparisons that carry a significance flag (primary ones)
    elo: dict | None  # player -> Elo from the experiment round robin (None if not played yet)
    obs_hash: str


def eval_checkpoints(run_dir, k) -> list[str]:
    """Paths of the last `k` distinct steps among `run_dir/ckpt_{step}.pt` (numeric order, oldest
    first); fewer when the run has fewer."""
    raise NotImplementedError("milestone 12: eval_checkpoints")


def load_spec(path) -> Spec:
    """Read a spec JSON. Unknown keys, and `seed` or `total_steps` in an arm, raise `ValueError`."""
    raise NotImplementedError("milestone 12: load_spec")


def run_dir(spec, arm, seed) -> str:
    """`runs/exp/{name}/{arm}/seed{seed}`."""
    raise NotImplementedError("milestone 12: run_dir")


def run(spec, parallel=1) -> None:
    """Train every (arm, seed) not yet done (at most `parallel` subprocesses at once), evaluate the
    last `eval_checkpoints` checkpoints of each finished run, then play the experiment round robin."""
    raise NotImplementedError("milestone 12: run")


def report(spec) -> ExperimentReport:
    """Aggregate the experiment; also writes `runs/exp/{name}/report.md`, `report.json`
    (`dataclasses.asdict` of the result) and `report_curves.png`."""
    raise NotImplementedError("milestone 12: report")


def main(argv: list[str] | None = None) -> None:
    raise NotImplementedError("milestone 12: experiment CLI")


if __name__ == "__main__":
    main()
