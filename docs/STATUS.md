# Project status

Maintained by the orchestrator session (`docs/ORCHESTRATOR.md`). A new orchestrator session continues
from this file. Update it at the end of every working session.

## Current milestone and phase

**Milestone 11, gate 11b (the 20M self-play run), running on the owner's machine.**

- Branch: all M11 work is on `milestone-11` (head `c943c80`, not merged). `main` is at `e5ab483`
  (M10 merged) and has none of the M11 code, brief or report.
- Run: `runs/selfplay_seed1`, started 2026-10-24 00:42 with
  `pixi run scripts/selfplay.sh python/configs/selfplay.json` (log `runs/selfplay_seed1.out`). Config:
  20M steps, `num_envs 128`, `num_steps 32`, `lr 1e-4`, warm start from
  `runs/curriculum_seed1/latest.pt`. First 80k steps: ~500-660 sps overall, approx_kl ~0.06-0.08 (the
  3M smoke run at `lr 3e-4` had ~0.235 in its first 250k steps).
- Expected duration: ~11 h of training at 500 sps plus evals and the final round robin (~1.2 h). Not
  measured past 80k steps; the smoke run's throughput halved once the second snapshot joined the pool.
- The checkout was switched to `main` after the run started. The run is safe (all its modules are
  imported at startup), but a `--resume` must be launched from `milestone-11`; `main` lacks
  `checkpoint.py`, `selfplay.py`, `suite.py` and the rest.

## Done

| Milestone | Content | Result |
| --- | --- | --- |
| M1-M6 | data/rng/combat, map gen, state/movement/legality, economy, Explorer/Super Unit, capture/win/score | gated, merged |
| M7 | renderer, digest, replay, HTML viewer, CLI | gated, merged |
| M8 | PlayerView, Random/Rush/Economy bots, tournament | gated, merged (Q13: Rush beats Economy 84%, accepted) |
| M9 | observation, factored actions, masks, Env, PyO3 | gated, merged |
| M10 | CleanRL-style PPO, curriculum, 20M steps | `latest.pt`: 100% vs random and economy, 97.8% vs rush (seat 1 96.0%) |
| M11 11a | hashes, checkpoints, suite, losses, Elo, tournament, snapshot-pool self-play | pipeline passes (RL_REPORT); 3M smoke self-play **regressed** vs bots (rush_e0 0.978 -> 0.868) |

## Waiting on the owner

1. 11b finishes; then run the 11b gate: `scripts/suite.sh` on `runs/selfplay_seed1/ckpt_final.pt`,
   `--compare` against `runs/curriculum_seed1/suite_latest/`, and the 200-seed head-to-head of the
   final weights vs snapshot 0 (gate (i), >= 60%).
2. Review the spec merge in the Claude Doc (applied 2026-10-24, doc rev 394; version history shows
   every edit). `docs/SPEC.md` on `main` is an export of that revision; re-export after any edits.
3. Merge `milestone-11` into `main` after 11b (or now, since 11a passed; owner's call).

## Open questions

- `QUESTIONS.md`: Q1-Q30 all have owner answers. None open.
- Spec open questions: none. Turn cap closed at 30 (owner, 2026-10-24); RL library closed by M10.
- Spec merge: applied to the Claude Doc master spec
  (https://claude.ai/artifact/3yVekWM5SAoUVxjZVmomTm) on 2026-10-24, awaiting owner review. It covers
  M2 C1-C5, M3 C1-C12, M4 D1-D11 (D4 overrule), M5 E1-E9, M6 F1-F9, M7 G1/G2/G4, M8 H1-H6,
  M9 J2-J10, M10 K1-K9, M11 L1-L11 and the rule-level Q answers (Q1-Q4, Q10, Q11, Q13, Q16).
  Implementation-only answers (APIs, formats, pins) stay in the briefs and QUESTIONS.md.
  `docs/SPEC.md` on `main` is synced to the doc (rev 394).
- `sim_core::Config::default().turn_cap` changed 50 -> 30 on `main` (owner, 2026-10-24): test commit
  `f37fb61` (4 expected values in 3 locked test files), code commit `5e4f114`. `milestone-11` does not
  have it yet; it arrives when M11 merges into `main` (or merge `main` into `milestone-11` first).
- The test lock `.git/tests-lock` points at the M11 test commit `03014c8`, so `./scripts/check.sh`
  fails on `main` by design. On `main` the gate was run step by step (fmt, clippy, 520 Rust tests,
  26 + 106 Python tests, all passing).

## Next three actions

1. Orchestrator: after the owner's review of the Claude Doc, re-export `docs/SPEC.md` if it changed, and
   turn answered QUESTIONS.md entries into a decisions log (after M11 merges, since QUESTIONS.md Q22-Q30
   live on `milestone-11`).
2. Owner: let 11b run; check `runs/selfplay_seed1/eval.jsonl` at 1M/2M for `rush_e0` (the smoke run
   was at 0.79 by 1M) and approx_kl.
3. Orchestrator: review the 11b results against the gate when they are in, and update RL_REPORT
   through an implementer session.

## Change log

| Date | Change |
| --- | --- |
| 2026-10-24 | Default turn cap 30 in sim-core (main); `docs/SPEC.md` synced to the Claude Doc rev 394. 11b at 1.0M steps. |
| 2026-10-24 | Spec merge applied to the Claude Doc (rev 346 -> 394); turn cap closed at 30. 11b at 434k steps, ~700 sps. |
| 2026-10-24 | STATUS.md created by the orchestrator session. 11b run `runs/selfplay_seed1` in progress. |
