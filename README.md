# polytopia_rl

A deterministic, headless Rust simulator of a reduced 1v1 Battle of Polytopia, with Python bindings
and a PPO agent trained against scripted bots and through self-play.

It is also a learning project in RL and agentic workflows: Claude Code agents write every milestone
from a spec the owner approves, tests first.

## Status

- **Game v1** (the reduced ruleset) is complete. The PPO agent was trained on it (curriculum against
  bots, then snapshot-pool self-play); results are in `docs/RL_REPORT.md`.
- **Game v2** (more of the real game) is in progress, behind a `ruleset: v2` flag. Done so far: tech
  research for the v2 techs (M15), roads and city connections (M16). v1 stays byte-identical.
- The current state and next steps are in `docs/STATUS.md`.

## Layout

| Path | What |
| --- | --- |
| `crates/sim-core` | The rules: state, `legal_actions`, `apply`, map generation, combat, movement. Pure and deterministic. |
| `crates/bots` | Scripted opponents: `random`, `rush`, `economy`. |
| `crates/sim-tools` | Digests, replay files, a text renderer, an HTML viewer, and the `polytopia` CLI. |
| `crates/sim-env` | The RL environment: observations, the action space, rewards. |
| `crates/sim-py` | PyO3 bindings (`polytopia_sim`). |
| `data/` | Rules tables (units, techs, rules, rewards) as JSON; `data/v2/` for game v2. |
| `python/polytopia_rl` | PPO, self-play, evaluation, the evaluation suite, tournaments, experiments. |
| `python/configs`, `python/experiments` | Training configs and multi-seed experiment specs. |
| `scripts/` | The check gate and launchers for training and evaluation. |
| `docs/` | The spec, milestone briefs, decisions and status (see below). |

## Setup

Requirements: Rust (stable) and [pixi](https://pixi.sh), which provides Python 3.12, PyTorch (CUDA
12.9), maturin and pytest. The pixi environment is configured for linux-64.

```sh
pixi install
pixi run check        # fmt, clippy -D warnings, all Rust tests, Python tests, RL tests
```

`pixi run check` (`scripts/check.sh`) is the one gate: work is not done until it passes.

## Usage

The simulator CLI:

```sh
cargo run --release -p sim-tools --bin polytopia -- play --bots rush,economy --seed 1 --out game.json
cargo run --release -p sim-tools --bin polytopia -- verify game.json
cargo run --release -p sim-tools --bin polytopia -- view game.json --step 10
cargo run --release -p sim-tools --bin polytopia -- html game.json game.html
cargo run --release -p sim-tools --bin polytopia -- tournament --bots random,rush,economy --games 100
```

`scripts/demo.sh` builds, verifies and renders a sample replay.

Training and evaluation (run inside the pixi environment):

```sh
pixi run scripts/train.sh python/configs/stage1_random.json        # PPO against the bots
pixi run scripts/selfplay.sh python/configs/selfplay.json          # snapshot-pool self-play
pixi run scripts/suite.sh runs/<run>/latest.pt                     # fixed evaluation suite
pixi run scripts/tournament.sh runs/<run>                          # round robin of a run's pool
pixi run scripts/experiment.sh run python/experiments/smoke.json   # arms x seeds
pixi run scripts/experiment.sh report python/experiments/smoke.json
```

Runs are written to `runs/`.

## Documentation

- `docs/SPEC.md`: the rulebook (game v1, then "Game v2 rules"). The owner writes or approves every rule.
- `docs/MILESTONE_N.md`: one brief per milestone.
- `docs/WORKFLOW.md`: the per-milestone agent workflow (tests, implementation, review).
- `docs/ORCHESTRATOR.md`, `docs/STATUS.md`: how the orchestrating session works, and where it stands.
- `docs/QUESTIONS.md`, `docs/DECISIONS.md`: open questions from agents, and the owner's answers.
- `docs/RL_REPORT.md`: training results.
- `CLAUDE.md`: rules for coding agents (determinism, data-driven rules, locked tests).

## Determinism

The same seed and the same actions give the same game, byte for byte: all randomness comes from an RNG
stored in the game state, iteration order never depends on hashing, and every replay file carries a
digest per step that `polytopia verify` checks.
