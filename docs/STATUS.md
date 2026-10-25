# Project status

Maintained by the orchestrator session (`docs/ORCHESTRATOR.md`). A new orchestrator session continues
from this file. Update it at the end of every working session.

## Current milestone and phase

**Milestone 12, phase A done; phase B next.** Branch `milestone-12`: tests `8d866be` (76 new tests in
`python/tests/rl/`, 31 mutants all caught, reviewed by the orchestrator), owner answers Q31-Q33 (A) in
`a5a3cb6`. Tests approved by the owner (2026-10-25). Next: `/implement 12`.

Milestone 11 is closed: gate 11b passed and `milestone-11` is merged into `main` (`540a214`,
2026-10-24).
`pixi run check` passes on the merged `main` (534 Rust tests, 26 + 188 Python). `.git/tests-lock` now
points at `540a214` (M11's lock `03014c8` plus the owner-approved turn-cap test edits).

- (i) final vs snapshot 0: 135-65 of 200, 0.675 [0.607, 0.736] (seat 0 0.74, seat 1 0.61).
- (ii) suite vs the M10 baseline: largest drop rush_e0 0.978 -> 0.948 (-3.0 pp, z -2.23); every other
  condition within 2.5 pp. Report: `docs/RL_REPORT.md` on `milestone-11`.
- Orchestrator caveats (not in the report):
  1. Late checkpoints swing against rush. In-training rush_e0 (the first 50 suite seeds, 100 games):
     0.80, 0.91, 0.83, 0.92, 0.86, 0.94 at 14M-19M. The round robin (200 games, rating seeds) gives
     0.875-0.975 for the 14M-19M snapshots. Full 400-game suites on earlier checkpoints (rush_e0 /
     rush_fullobs / rush_e0_sampled): 16M 0.902 / 0.870 / 0.917 (gate (ii) would FAIL: -7.5, -9.5,
     -5.2 pp), 18M 0.930 / 0.922 / 0.938 (passes, worst -4.7), final 0.948 / 0.940 / 0.945. So the late
     trend is upward, and the pass holds from about 18M on; at 16M it would have failed. One run.
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

1. Start phase B (tests `8d866be` approved 2026-10-25): either your own session
   (`claude --disallowedTools "Edit(crates/*/tests/**)" "Write(crates/*/tests/**)" "Edit(python/tests/rl/**)" "Write(python/tests/rl/**)"`,
   then `/implement 12`), or ask the orchestrator to launch a background agent (guarded by the
   check.sh test lock only).

## Open questions

- `QUESTIONS.md`: Q31-Q33 answered (A, 2026-10-25); they stay there until M12 merges, then move to DECISIONS.md.
- Spec open questions: none. Turn cap closed at 30 (owner, 2026-10-24); RL library closed by M10.
- Spec merge: applied to the Claude Doc master spec
  (https://claude.ai/artifact/3yVekWM5SAoUVxjZVmomTm) on 2026-10-24. It covers
  M2 C1-C5, M3 C1-C12, M4 D1-D11 (D4 overrule), M5 E1-E9, M6 F1-F9, M7 G1/G2/G4, M8 H1-H6,
  M9 J2-J10, M10 K1-K9, M11 L1-L11 and the rule-level Q answers (Q1-Q4, Q10, Q11, Q13, Q16).
  Implementation-only answers (APIs, formats, pins) stay in the briefs and QUESTIONS.md.
  `docs/SPEC.md` on `main` is synced to the doc (rev 394).
- `sim_core::Config::default().turn_cap` is 30 since `5e4f114` (test commit `f37fb61`, owner-approved).
- Answered questions Q1-Q30 are in `docs/DECISIONS.md`; `docs/QUESTIONS.md` holds only open ones
  (next number Q31).

## Next three actions


1. Orchestrator: re-export `docs/SPEC.md` if the Claude Doc changed in the owner's review.
2. Owner: start M12 (`/write-tests 12`, then `/implement 12` with the test-dir deny rules, then `/review 12`).
3. Orchestrator: review the M12 phase A tests against the brief before phase B.

## Change log

| Date | Change |
| --- | --- |
| 2026-10-25 | M12 phase A committed (8d866be) and reviewed; Q31-Q33 answered A. WSL restarted with 12 GB. |
| 2026-10-24 | Q1-Q30 moved to `docs/DECISIONS.md`; QUESTIONS.md keeps only open items (next Q31). |
| 2026-10-24 | M11 merged into main (540a214), gate green, test lock moved. 16M/18M suites run (see caveat 1). M12 brief final for phase A (840dc63). |
| 2026-10-24 | 11b finished and passed (0.675 vs snap_0; worst suite drop -3.0 pp). Orchestrator caveats recorded. |
| 2026-10-24 | M12 brief drafted (infrastructure + speed; experiments chosen after 11b). Owner: 10M steps per run, seed-varied maps, Welch t across seeds. |
| 2026-10-24 | Roadmap set: M12 experiments, M13 architecture/scaling, then game v2 (with the damage preview). |
| 2026-10-24 | WSL memory raised to 12 GB in config, pending a WSL restart after 11b. |
| 2026-10-24 | Implementer status: 11b resumed at 1.0M with OMP_NUM_THREADS=1; 1M eval clean (rush_e0 0.96). Shared extension clobbered by a main build, restored. |
| 2026-10-24 | Default turn cap 30 in sim-core (main); `docs/SPEC.md` synced to the Claude Doc rev 394. 11b at 1.0M steps. |
| 2026-10-24 | Spec merge applied to the Claude Doc (rev 346 -> 394); turn cap closed at 30. 11b at 434k steps, ~700 sps. |
| 2026-10-24 | STATUS.md created by the orchestrator session. 11b run `runs/selfplay_seed1` in progress. |
