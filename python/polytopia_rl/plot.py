"""Plots of a run's metrics. `python -m polytopia_rl.plot --run RUNDIR`"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")  # files only, no display
import matplotlib.pyplot as plt  # noqa: E402

# (title, metric keys) per panel of the training plot.
PANELS = [
    ("train win rate", ["train_win_rate"]),
    ("mean return", ["mean_return"]),
    ("entropy", ["entropy", "entropy_actor", "entropy_kind", "entropy_arg"]),
    ("approx kl", ["approx_kl"]),
    ("clip fraction", ["clipfrac"]),
    ("explained variance", ["explained_variance"]),
    ("losses", ["pg_loss", "v_loss"]),
    ("end turn freq", ["end_turn_freq"]),
    ("episode length", ["mean_episode_len"]),
    ("steps per second", ["sps", "env_sps", "forward_sps"]),
    ("learning rate", ["lr"]),
    ("stage", ["stage"]),
]


def _read(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _series(rows: list[dict], key: str):
    pts = [(r["global_step"], r[key]) for r in rows if r.get(key) is not None]
    return [x for x, _ in pts], [y for _, y in pts]


def plot_run(run_dir: str) -> list[str]:
    """Write PNG plots of `RUNDIR/metrics.jsonl` (and `eval.jsonl` if present) under `run_dir`;
    return their paths."""
    paths = []
    rows = _read(os.path.join(run_dir, "metrics.jsonl"))
    fig, axes = plt.subplots(4, 3, figsize=(15, 14))
    for ax, (title, keys) in zip(axes.flat, PANELS):
        for key in keys:
            ax.plot(*_series(rows, key), label=key)
        ax.set_title(title)
        ax.set_xlabel("global step")
        if len(keys) > 1:
            ax.legend(fontsize=8)
    fig.tight_layout()
    paths.append(os.path.join(run_dir, "metrics.png"))
    fig.savefig(paths[-1])
    plt.close(fig)

    eval_path = os.path.join(run_dir, "eval.jsonl")
    evals = _read(eval_path) if os.path.exists(eval_path) else []
    if evals:
        fig, ax = plt.subplots(figsize=(8, 5))
        opponents = sorted({k for r in evals for k in r if isinstance(r[k], dict)})
        for opp in opponents:
            pts = [(r["global_step"], r[opp]) for r in evals if opp in r]
            xs = [x for x, _ in pts]
            ax.plot(xs, [e["win_rate"] for _, e in pts], marker="o", label=opp)
            ax.fill_between(xs, [e["wilson_lo"] for _, e in pts], [e["wilson_hi"] for _, e in pts], alpha=0.2)
        ax.set_title("greedy evaluation win rate (Wilson 95%)")
        ax.set_xlabel("global step")
        ax.set_ylim(0, 1)
        ax.legend()
        fig.tight_layout()
        paths.append(os.path.join(run_dir, "eval.png"))
        fig.savefig(paths[-1])
        plt.close(fig)
    return paths


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Plot a training run.")
    p.add_argument("--run", required=True)
    for path in plot_run(p.parse_args(argv).run):
        print(path)


if __name__ == "__main__":
    main()
