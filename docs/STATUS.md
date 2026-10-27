# Project status

Maintained by the orchestrator session (`docs/ORCHESTRATOR.md`). A new orchestrator session continues
from this file. Update it at the end of every working session.

## Current milestone and phase

**Milestone 12 is closed (2026-10-26); the next step is the owner's decision.** `milestone-12` is merged
into `main` (fast-forward, then `9e5fd5d` Q34 answer). Results: `docs/RL_REPORT.md` "Milestone 12a"
and "Milestone 12b"; experiment data in `runs/exp/selfplay12b/`.

- **12a:** batched suite 22 min -> 28 s (greedy W-L-D identical; Q34 TF32 length differences accepted);
  3 trainers at once give 1,570 sps total against 804 for one (`--parallel 3` default); smoke
  experiment ran end to end.
- **12b** (`base`, `pbot40` = `p_bot` 0.4, `ent003` = `ent_coef` 0.003; 3 seeds x 10M; mean of the
  8/9/10M checkpoints): rush_e0 0.913 +- 0.058 / 0.942 +- 0.021 / 0.884 +- 0.008; `elo_vs_init` 181 /
  168 / 190. Only pbot40 > ent003 on rush_e0 is flagged (p 0.029), which does not survive a Bonferroni
  correction for 6 tests. Readings:
  1. Seed variance in one config spans 0.848-0.958 against rush: single-run comparisons are unreliable.
  2. `p_bot` 0.4 is ahead of `base` seed by seed on every rush condition, at no Elo cost (consistent,
     not significant).
  3. A smaller entropy bonus hurts against rush; keep `ent_coef` 0.01.
  4. Hypothesis: the LR schedule matters. `base` seed 1 (11b config, 10M) scores 0.80 vs the M10
     checkpoint where 11b's 20M final scored 0.675, on the same rating seeds.
- Throughput note: GPU at 40-50% because each trainer is one Python thread making many tiny
  opponent forward passes (Q29/Q30); per-trainer sps falls from ~550 (pool 2) to ~200 (pool 10).

## Roadmap (owner, 2026-10-24)

1. **Close M11:** done (2026-10-24).
2. **M12 experiments:** done (2026-10-26).
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
| M11 | hashes, checkpoints, suite, losses, Elo, tournament, snapshot-pool self-play | 11b passed: 0.675 vs M10, worst suite drop -3.0 pp (passes from ~18M on) |
| M12 | batched suite, seed-varied maps, multi-seed experiment harness, Welch | 12b: see above; `pbot40` best against rush, not significant |

## Waiting on the owner

1. Read `docs/MILESTONE_12c.md` (LR schedule experiment: arms `decay10`, `decay20`, `const`; clarifications
   N1-N3, mainly whether to reuse 12b's `pbot40` runs as `decay10`), then say go for phase A.

## Open questions

- `QUESTIONS.md`: none open (next number Q35). Q1-Q34 are in `docs/DECISIONS.md`.
- Spec open questions: none. The Claude Doc master spec
  (https://claude.ai/artifact/3yVekWM5SAoUVxjZVmomTm) and `docs/SPEC.md` are in sync (rev 394).
- `.git/tests-lock` points at `6a8b487` (the owner-approved `p_bot` 0.4 test update).
- Self-play default `p_bot` is 0.4 since `4c6740f` (owner, 2026-10-27). 11b and 12b `base` used 0.2.

## Next three actions

1. Owner: review the 12c brief; then `/write-tests 12c` and `/implement 12c` (small; may run back to back).
2. Orchestrator: after 12c, draft the speed milestone (Q30 C CUDA graphs, Q29 R4) and M13.
3. Orchestrator: re-export `docs/SPEC.md` if the Claude Doc changes.

## Change log

| Date | Change |
| --- | --- |
| 2026-10-27 | `p_bot` 0.4 adopted (tests `6a8b487`, config `4c6740f`, gate green). M12c brief (LR schedule experiment) drafted. |
| 2026-10-27 | 12b done and reported (RL_REPORT "Milestone 12b"); M12 closed; Q31-Q34 moved to DECISIONS.md. |
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
