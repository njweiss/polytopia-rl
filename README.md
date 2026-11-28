# polytopia_rl

A deterministic, headless Rust simulator of a reduced 1v1 Battle of Polytopia (v1 and v2 rulesets),
with Python bindings and a PPO agent trained against scripted bots and through self-play.

## Status

- **Game v1** (reduced 9-tech, 4-unit ruleset) is complete; PPO and self-play results are in
  [`docs/RL_REPORT.md`](docs/RL_REPORT.md).
- **Game v2** (`ruleset = v2`) is complete through Milestone 23: 7 additional land technologies,
  3 new units (`Swordsman`, `Catapult`, `Knight`) with `Stiff`/`Persist`, roads and city connections,
  economy buildings (`Sawmill`, `Windmill`, `Forge`, `Market`), `BurnForest`/`Disband`/`Destroy`,
  monuments, lighthouses, a 6-channel damage preview observation, an `arch="v2"` policy network,
  v1→v2 policy transfer (prefix warm-start & distillation), and revised self-play (PFSP, live
  two-sided self-play, stream rollouts, anchors, win-rate gating, shaping annealing, paired seeds).
  v1 remains byte-identical.
- Current state, test counts, hashes, and next steps are tracked in [`docs/STATUS.md`](docs/STATUS.md).

## Layout

| Path | What |
| --- | --- |
| `crates/sim-core` | Pure, deterministic game rules: state, `legal_actions`, `apply`, map gen, combat, movement, `PlayerView`. |
| `crates/bots` | Scripted opponents (`random`, `rush`, `economy`) supporting v1 and v2, plus match runner. |
| `crates/sim-tools` | State digests, replay logs, text renderer, HTML viewer, and the `polytopia` CLI. |
| `crates/sim-env` | RL environment: v1/v2 observations, factored action space and masks, rewards, spec hashes. |
| `crates/sim-py` | PyO3 bindings (`polytopia_sim`). |
| `data/`, `data/v2/` | JSON rules tables (units, techs, site actions, rewards, buildings, monuments, rules). |
| `python/polytopia_rl` | Policy networks (`arch="v1"`/`"v2"`), PPO, curriculum, transfer, self-play/PFSP, suite, Elo, tournaments, experiment harness. |
| `python/configs`, `python/experiments` | Training configs (`v1` & `v2`) and multi-seed experiment specs (`transfer22.json`, `selfplay23.json`, etc.). |
| `scripts/` | Check gate (`check.sh`) and training/evaluation launchers. |
| `docs/` | Authoritative rulebook (`SPEC.md`), status, decisions, workflow, milestone briefs, and RL report. |

## Setup

Requirements: Rust (stable) and [pixi](https://pixi.sh), which provides Python 3.12, PyTorch
(`linux-64` CUDA 12.9 and `osx-arm64` CPU), `maturin`, and `pytest`.

```sh
pixi install
pixi run check        # test-lock check, fmt, clippy -D warnings, Rust tests, Python & RL tests
```

`pixi run check` (`scripts/check.sh`) is the single quality gate. Inside a git worktree,
`scripts/py_check.sh` and `scripts/rl_check.sh` automatically build an isolated wheel into
`target/pyext` instead of touching the main checkout's Python environment.

## Usage

### Simulator CLI

Pass `--ruleset v2` to `play`, `play-random`, or `tournament` to use the v2 ruleset (defaults to `v1`):

```sh
cargo run --release -p sim-tools --bin polytopia -- play --bots rush,economy --seed 1 --out game.json
cargo run --release -p sim-tools --bin polytopia -- play --bots rush,economy --seed 1 --ruleset v2 --out game_v2.json
cargo run --release -p sim-tools --bin polytopia -- verify game.json
cargo run --release -p sim-tools --bin polytopia -- view game.json --step 10
cargo run --release -p sim-tools --bin polytopia -- html game.json game.html
cargo run --release -p sim-tools --bin polytopia -- tournament --bots random,rush,economy --games 100 --ruleset v2
```

`scripts/demo.sh` builds, verifies, and renders a sample replay.

### Training, Evaluation, and Experiments

Run inside the pixi environment (outputs go to `runs/`):

```sh
# Curriculum PPO against bots (v1 or v2 with arch="v2")
pixi run scripts/train.sh python/configs/curriculum.json
pixi run scripts/train.sh python/configs/v2_curriculum.json

# Self-play (v1 legacy pool, or v2 revised PFSP + live two-sided stream self-play)
pixi run scripts/selfplay.sh python/configs/selfplay.json
pixi run scripts/selfplay.sh python/configs/v2_selfplay_revised.json --init-checkpoint runs/<run>/ckpt_final.pt

# Evaluation suite and pool tournament
pixi run scripts/suite.sh runs/<run>/latest.pt
pixi run scripts/tournament.sh runs/<run>

# Multi-seed experiments (arms x seeds)
pixi run scripts/experiment.sh run python/experiments/smoke.json
pixi run scripts/experiment.sh run python/experiments/transfer22.json
pixi run scripts/experiment.sh run python/experiments/selfplay23.json
pixi run scripts/experiment.sh report python/experiments/transfer22.json
```

Any config field can be overridden on the command line as `key=value` (e.g., `arch=v2`,
`warm_start_checkpoint=...`, `teacher_checkpoint=... distill_coef=0.1`).

## Documentation

- [`docs/SPEC.md`](docs/SPEC.md): single authoritative rulebook for game v1 and v2.
- [`docs/STATUS.md`](docs/STATUS.md): current state, hashes, test counts, and next actions.
- [`docs/RL_REPORT.md`](docs/RL_REPORT.md): empirical training and experiment results.
- [`docs/DECISIONS.md`](docs/DECISIONS.md), [`docs/QUESTIONS.md`](docs/QUESTIONS.md): answered decisions (Q1–Q59) and open agent questions.
- [`docs/WORKFLOW.md`](docs/WORKFLOW.md), [`docs/ORCHESTRATOR.md`](docs/ORCHESTRATOR.md), [`CLAUDE.md`](CLAUDE.md): agent workflow and coding rules.
- `docs/MILESTONE_N.md`: per-milestone briefs (M1–M23).

## Determinism

The same seed and the same actions produce the same game byte for byte: all randomness comes from
a `pcg32` RNG stored inside the game state, iteration order never depends on hashing, and every
replay file carries a per-step state digest verified by `polytopia verify`.
