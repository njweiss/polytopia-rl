# Project status

Maintained by the orchestrator session (`docs/ORCHESTRATOR.md`). A new orchestrator session continues
from this file. Update it at the end of every working session.

## Current milestone and phase

**Milestone 11: gate 11b passed (2026-10-24, run `runs/selfplay_seed1`, 20M steps, ~18 h).** Waiting on
the owner's review and the merge of `milestone-11` (head `eab02fb`) into `main`.

- (i) final vs snapshot 0: 135-65 of 200, 0.675 [0.607, 0.736] (seat 0 0.74, seat 1 0.61).
- (ii) suite vs the M10 baseline: largest drop rush_e0 0.978 -> 0.948 (-3.0 pp, z -2.23); every other
  condition within 2.5 pp. Report: `docs/RL_REPORT.md` on `milestone-11`.
- Orchestrator caveats (not in the report):
  1. Late checkpoints swing against rush. In-training rush_e0 (the first 50 suite seeds, 100 games):
     0.80, 0.91, 0.83, 0.92, 0.86, 0.94 at 14M-19M. The round robin (200 games, rating seeds) gives
     0.875-0.975 for the 14M-19M snapshots. The final weights scored 0.95 on the same 50 seeds, so the
     -3.0 pp in gate (ii) partly depends on where the run stopped. One run, one stop point.
  2. The report credits the lower `lr` for fixing the warm-start collapse, but the same restart also
     changed the batch shape (32x128 -> 128x32, Q29 R3). The two are confounded.
  3. The in-training "learner" Elo is 921.278 at every rating step: in the star graph its rating is fixed
     by its 100-0 record against `random` plus the prior (400 log10(100.5/0.5)). Only the gap
     learner - snapshot_0 is informative there. The final round robin is unaffected.
- Elo of the pool (round robin): snap_0 1138, low point snap_3M 981, then rising from 10M to snap_19M
  1383; the last five snapshots are the top five.

## Roadmap (owner, 2026-10-24)

1. **Close M11:** gate 11b, RL_REPORT write-up, merge `milestone-11` into `main`, decisions log.
2. **M12 experiments:** a multi-seed harness (mean and spread per suite condition and Elo), then
   seed-replicated experiments (11b reproducibility, shaping weight, `p_bot`). Cheap, profiled speed
   wins that save experiment time are in scope.
3. **M13 architecture and scaling:** network width/depth ladder measured with the M12 harness.
4. **Game v2:** drafted after M13. The damage preview goes into v2, not into a separate v1
   milestone (owner: it is unlikely to be a significant information source).

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

0. Owner: review `docs/MILESTONE_12.md` (Clarifications M1-M6); pick the 12b experiments after 11b.

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
| 2026-10-24 | 11b finished and passed (0.675 vs snap_0; worst suite drop -3.0 pp). Orchestrator caveats recorded. |
| 2026-10-24 | M12 brief drafted (infrastructure + speed; experiments chosen after 11b). Owner: 10M steps per run, seed-varied maps, Welch t across seeds. |
| 2026-10-24 | Roadmap set: M12 experiments, M13 architecture/scaling, then game v2 (with the damage preview). |
| 2026-10-24 | WSL memory raised to 12 GB in config, pending a WSL restart after 11b. |
| 2026-10-24 | Implementer status: 11b resumed at 1.0M with OMP_NUM_THREADS=1; 1M eval clean (rush_e0 0.96). Shared extension clobbered by a main build, restored. |
| 2026-10-24 | Default turn cap 30 in sim-core (main); `docs/SPEC.md` synced to the Claude Doc rev 394. 11b at 1.0M steps. |
| 2026-10-24 | Spec merge applied to the Claude Doc (rev 346 -> 394); turn cap closed at 30. 11b at 434k steps, ~700 sps. |
| 2026-10-24 | STATUS.md created by the orchestrator session. 11b run `runs/selfplay_seed1` in progress. |
