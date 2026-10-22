"""Plots of a run's metrics. `python -m polytopia_rl.plot --run RUNDIR`"""

from __future__ import annotations


def plot_run(run_dir: str) -> list[str]:
    """Write PNG plots of `RUNDIR/metrics.jsonl` (and `eval.jsonl` if present) under `run_dir`;
    return their paths."""
    raise NotImplementedError


def main(argv: list[str] | None = None) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
