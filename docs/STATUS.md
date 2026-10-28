# Project status

Maintained by the orchestrator session (`docs/ORCHESTRATOR.md`). A new orchestrator session continues
from this file. Update it at the end of every working session.

## Current milestone and phase

**Milestone 13 (interleaved self-play rollout): merged into `main` (`c8a0fcf`, 2026-10-28), `pixi run
check` green (300 RL + 26 Python tests); gate 13a done (`13f6a5d`).** At pool 10 the interleaved
rollout is 1.66x faster (350 -> 582 rollout sps; total 311 -> 478), below the orchestrator's "several
times" estimate: snapshot groups per tick rose to 4.4, and learner calls tripled. Smoke run (2M, 41 min,
~820 sps) inside the 12b spread: learning not broken. `--parallel 4` default (+7% over 3; ~1.7 GiB RAM
margin estimated, watch it). With a pool of 1 under 3-4 trainers, interleaved is ~8% slower than
lockstep. Re-profile: opponent forward 51%, Python loop and per-element GPU buffer writes 27% (~29k
host-device syncs per rollout), learner forward 17%. Tests `b243472` (21 tests,
equivalence with the lockstep path on the real env), implementation `b86c2d5`. Default
`rollout: "interleaved"`; `"lockstep"` reproduces earlier runs. Known gap (harmless at defaults): an
episode truncated during the opponent's opening turn would hang the rollout (needs `max_agent_actions`
below an opening turn; same case M11 Q26 left untested).

**Milestone 12c closed (2026-10-28):** `decay10 >= decay20 >= const` on all 9 metrics; `elo_vs_init`
168 / 119 / 89, rush_e0 0.942 / 0.930 / 0.872; 9 of 9 paired seed comparisons ordered on
`elo_vs_init`. Conclusion: anneal the LR to zero within each run (the default); 10M-step runs suffice
for M14 comparisons. RL_REPORT "Milestone 12c".

Milestone 12 (closed 2026-10-26):
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
3. **M13 speed:** interleaved self-play rollout (brief `docs/MILESTONE_13.md`), then a re-profile.
4. **M14 architecture and scaling:** network width/depth ladder measured with the M12 harness.
5. **Game v2:** drafted after M14. The damage preview goes into v2, not into a separate v1
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

**Working rule while an experiment runs from this checkout:** do milestone work in a git worktree; in a
worktree never run `pixi run`, `maturin develop` or `check.sh` (they create a second pixi env or replace
the shared `polytopia_sim`); run tests with
`PYTHONPATH=$PWD/python /home/njweiss4/projects/polytopia_rl/.pixi/envs/default/bin/python -m pytest ...`.

## Waiting on the owner

1. Review `docs/MILESTONE_13b.md` (stacked snapshot forward; clarifications T1-T3), then start phase A.

## Open questions

- `QUESTIONS.md`: none open (next number Q38). Q1-Q37 are in `docs/DECISIONS.md`.
- Spec open questions: none. The Claude Doc master spec
  (https://claude.ai/artifact/3yVekWM5SAoUVxjZVmomTm) and `docs/SPEC.md` are in sync (rev 394).
- `.git/tests-lock` points at `b243472` (M13's test commit).
- Self-play default `p_bot` is 0.4 since `4c6740f` (owner, 2026-10-27). 11b and 12b `base` used 0.2.

## Next three actions

1. Gate 13a: profile fixture before/after, 2M smoke run, concurrency 1-4, re-profile, RL_REPORT section.
2. Orchestrator: from the re-profile, a second speed step or straight to the M14 brief (scaling ladder,
   10M runs annealed to zero, `p_bot` 0.4).
3. Small safeguard for the opening-turn truncation hang (raise instead of looping) in the next milestone.

## Change log

| Date | Change |
| --- | --- |
| 2026-10-28 | Owner chose option (b): M13b brief (stacked snapshot forward via torch.func) drafted. Option (a) and the hang safeguard deferred. |
| 2026-10-28 | Gate 13a: interleaved 1.66x rollout / 1.54x total at pool 10; smoke OK; --parallel 4. |
| 2026-10-28 | 12c done and reported (decay10 best); M13 merged (c8a0fcf), gate green, worktree removed; Q36-Q37 to DECISIONS.md. |
| 2026-10-27 | 12c-b launched by the owner. M13 S1-S3 confirmed; phase A moved to worktree `../polytopia_rl-m13` (first attempt stopped before any edit). |
| 2026-10-27 | Profiled self-play at pool 10 (92% of rollout in opponent calls, 58 calls per learner step, barrier per step); M13 brief (interleaved rollout) drafted; scaling becomes M14. |
| 2026-10-27 | M12c tests and implementation merged (7af81d4); Q35 answered A; 12b pbot40 runs linked as 12c decay10. |
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
