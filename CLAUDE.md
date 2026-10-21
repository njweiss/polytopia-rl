# Polytopia headless sim: instructions for coding agents

Goal: a deterministic, headless Rust simulator of a reduced 1v1 Battle of Polytopia, used to
train a PPO agent. The project owner is learning RL and agentic workflows; correctness and
clarity beat cleverness. Speed work comes later and only after profiling.

## Source of truth
- `docs/SPEC.md` is the rulebook. The owner writes or approves every rule.
- `docs/MILESTONE_N.md` is your current assignment. Do that milestone only.
- Tests in `crates/*/tests/` were written first and encode the spec. **Never edit an expected
  value, delete a test, or add `#[ignore]` to get green.** If a test looks wrong, stop and
  record it in `docs/QUESTIONS.md`.

## When the spec is silent or ambiguous
Do not guess. Append the question to `docs/QUESTIONS.md` (what you needed, which spec section,
the options you see, which you would pick and why), implement only what is unambiguous, and
say so in your final report. The owner answers; the answer goes back into the spec.

## Hard rules
- Determinism: no `HashMap`/`HashSet` iteration order in anything that affects game state
  (use `Vec`, `BTreeMap`, or sort); no wall-clock time; no thread-dependent behavior; all
  randomness from the game's own RNG whose state lives inside the game state.
- Rules data lives in `data/*.json`, loaded through `sim_core::data::Tables`. Do not hard-code
  unit stats, tech costs, etc. in Rust branches.
- Combat numerics are `f32`, evaluated in the order the spec writes them, half-up rounding.
- `apply(state, action)` will be a pure function of state and action. No hidden global state.
- `unsafe` is forbidden. No new dependencies without asking (record in QUESTIONS.md).
- Keep scope tight: no features, abstractions, or files the milestone does not call for.

## Definition of done
`./scripts/check.sh` exits 0 (fmt, clippy `-D warnings`, all tests). Then report: what you
built, anything you deferred or were unsure about (with QUESTIONS.md entry numbers), and the
exact test output summary. Do not claim done without running it.

## Layout
- `crates/sim-core`: the simulator library (no Python, no I/O beyond embedded data).
- `data/`: rules tables (JSON). `docs/`: spec, milestone briefs, open questions.
- Future crates (not yet): `sim-py` (PyO3), `bots`, `render`.
