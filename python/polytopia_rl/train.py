"""Training driver (docs/MILESTONE_10).

`python -m polytopia_rl.train --config CONFIG [--run-dir DIR] [--resume CKPT] [key=value ...]`

`RUNDIR` gets `metrics.jsonl` (one line per update), `eval.jsonl`, and the checkpoints
`ckpt_{global_step}.pt` and `latest.pt` (written at every evaluation, i.e. each time global_step
passes a multiple of `cfg.eval_every`, and at the end) and `best.pt`. A checkpoint is a dict with
at least `model`, `optimizer`, `config` (JSON), `global_step`, `stage`, `vec_env`
(`SyncVecEnv.state_dict()`, which holds the next training seeds) and `rng`.
"""

from __future__ import annotations


def run(cfg, run_dir: str, resume: str | None = None):
    """Train for `cfg.total_steps` agent steps and return the `PolicyNet`. With `resume`, model,
    optimizer, counters, curriculum state, RNG states and env state come from that checkpoint, and
    the run continues as if it had never stopped."""
    raise NotImplementedError


def load_policy(path: str, device: str = "cpu"):
    """`(PolicyNet, PPOConfig)` from a checkpoint."""
    raise NotImplementedError


def main(argv: list[str] | None = None) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
