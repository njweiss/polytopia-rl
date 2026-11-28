# Project status

Maintained by the orchestrator session (`docs/ORCHESTRATOR.md`). A new orchestrator session continues
from this file. Update it at the end of every working session.

## Current state

All milestones **M1–M23** are merged on `main` (M23 merge commit `09f513a`, 2026-11-28).

- **Gate (`pixi run check`):** green — **742** cargo tests, **33** Python binding tests, **567** RL tests.
- **Test lock (`.git/tests-lock`):** `9ebcf4a` (M23 review guard tests).
- **Spec hashes (`size=11`):**
  - **v1:** `obs_hash d5f9b1033cfd55ca`, `rules_hash a8b52b91eae34503` (byte-identical to M11).
  - **v2:** `obs_hash c15247decc21ae4f`, `rules_hash ed271736a688e86d`.
- **Open questions:** none in `docs/QUESTIONS.md` (next number **Q60**). Q1–Q59 are logged in `docs/DECISIONS.md`.
- **Authoritative spec:** `docs/SPEC.md`. Empirical results live in `docs/RL_REPORT.md`.

## Done

| Milestone | Scope | Merge / Outcome |
| --- | --- | --- |
| M1–M6 | Data tables, `pcg32` RNG, combat, map gen, movement, economy, Explorer, Super Unit, capture, win, score | Gated, merged |
| M7 | Text renderer, state digest, replay log, HTML viewer, `polytopia` CLI | Gated, merged |
| M8 | `PlayerView` fog of war, `Random`/`Rush`/`Economy` bots, match runner | Gated, merged (Q13: Rush beats Economy 84%) |
| M9 | v1 observation encoding, factored action space and masks, `Env`, PyO3 (`polytopia_sim`) | Gated, merged |
| M10 | CleanRL-style PPO, 3-stage bot curriculum (20M steps) | `latest.pt`: 100% vs random/economy, 97.8% vs rush |
| M11 | `obs_hash`/`rules_hash`, checkpoints, 8-condition suite, loss review, Bradley-Terry Elo, snapshot-pool self-play | 11b passed (0.675 vs M10; see `RL_REPORT.md`) |
| M12, M12c | Batched suite, seed-varied maps, multi-seed experiment harness, LR annealing schedule | Merged (`RL_REPORT.md` 12a–12c; `p_bot` 0.4, anneal LR to 0) |
| M13, M13b | Interleaved self-play rollout, stacked snapshot forward (`torch.func`) | Merged (`c8a0fcf`, `074b0ef`; `RL_REPORT.md` 13a–13b) |
| M14 | Capacity ladder (`w32b4`–`w128b4`, `w64b8`) | Merged (`3501282`; `RL_REPORT.md` 14b: no capacity gap at fixed LR) |
| M15 | v2 foundation: `ruleset` flag, `data/v2/` tables, 7 v2 techs, v2 digest & `rules_hash` | Merged (`70c79e4`); v1 byte-identical |
| M16 | v2 roads, half-cost road movement, city connections (+1/-1 pop), Q48–Q49 | Merged (`9af6590`); v1 byte-identical |
| M16b | Rollout batching/sync fixes, `current_player()`, opening-turn safeguard, epoch-mean `target_kl`, batched eval, paired t-test, `osx-arm64` pixi, worktree-safe checks | Merged (`724e821`) |
| M17 | v2 economy buildings (`Sawmill`, `Windmill`, `Forge`, `Market`), `BurnForest`, `Disband`, `Destroy`, signed `i32` pop & score floor | Merged (`ba89c28`); v1 byte-identical |
| M18 | v2 corner lighthouses (+1 pop), v2 Explorer (+3.5 lighthouse bonus), monuments (`ParkOfFortune`, `GateOfPower`, `EyeOfGod`, 400 score) | Merged (`83c4003`); v1 byte-identical |
| M19 | v2 units (`Swordsman`, `Catapult`, `Knight`), `Stiff`/`Persist`, v2 bots, text/HTML renderer & CLI `--ruleset v2` | Merged (`4190c8a`); v1 byte-identical |
| M20 | v2 `PlayerView` & `attack_preview`, 64-ch/33-global obs (37/18 v1 prefix + 6 damage-preview channels), 15 factored `ActionType`s, `Env`/`PyEnv` v2 | Merged (`c966bea`); v1 byte-identical |
| M21 | `arch="v2"` `PolicyNet` (CoordConv, pre-norm `GroupNorm` + FiLM `ResBlockV2`, separate `move`/`attack` keys + QK `LayerNorm` + 2D `rel_bias`) | Merged (`5898c72`); `arch="v1"` & Rust untouched |
| M22 | v1→v2 policy transfer (`transfer.py`: prefix warm-start, 3-stage factored KL distillation), `v2_curriculum.json`, `v2_selfplay.json`, `transfer22.json` | Merged (`b575481`); Rust/hashes untouched |
| M23 | Revised self-play (`pfsp.py`, `compute_gae_segments`, `rollout="stream"`, live two-sided `p_self`, `opponent_forward="auto"`, `selfplay_bots`, cross-arch anchors, win-rate gating, shaping annealing, paired seeds), `v2_selfplay_revised.json`, `selfplay23.json` | Merged (`09f513a`); 0 blocking defects, all 4 review findings fixed |

## Next actions

1. Run the M22 4-arm v2 transfer experiment (`python/experiments/transfer22.json`: `scratch_v1`, `warmstart_v1`, `scratch_v2`, `distill_v2`) on the CUDA machine and record results in `docs/RL_REPORT.md`.
2. Run the M23 2-arm v2 self-play A/B experiment (`python/experiments/selfplay23.json`: `legacy` vs `revised`) on the CUDA machine and record results in `docs/RL_REPORT.md`.
3. Optional: run the M16b GPU profile check (`torch.profiler` sync counts on the RTX 4060) and add v2 road/building/monument/lighthouse badges to the HTML viewer (`crates/sim-tools/src/html.rs`).

## Change log

| Date | Change |
| --- | --- |
| 2026-11-28 | M23 merged (`09f513a`), `pixi run check` green (742 cargo, 33 + 567 pytest); `/review 23` 0 blocking defects; tests locked at `9ebcf4a`. |
| 2026-11-28 | M22 merged (`b575481`), `pixi run check` green (742 cargo, 33 + 483 pytest); Q59 to `DECISIONS.md`; tests locked at `f520118`. |
| 2026-11-27 | M21 merged (`5898c72`), `pixi run check` green (742 cargo, 33 + 455 pytest); Q58 to `DECISIONS.md`; tests locked at `9b2d0bb`. |
| 2026-11-27 | M20 merged (`c966bea`), `pixi run check` green (742 cargo, 33 + 425 pytest); Q56–Q57 to `DECISIONS.md`; tests locked at `13b27cd`. |
| 2026-11-27 | M19 merged (`4190c8a`), `pixi run check` green (725 cargo, 26 + 420 pytest); Q55 to `DECISIONS.md`. |
| 2026-11-27 | M18 merged (`83c4003`), `pixi run check` green (680 cargo, 26 + 420 pytest); Q52–Q54 to `DECISIONS.md`. |
| 2026-11-26 | M17 merged (`ba89c28`), `pixi run check` green (652 cargo, 26 + 420 pytest); Q50–Q51 to `DECISIONS.md`. |
| 2026-11-25 | M16b merged (`724e821`), `pixi run check` green on `osx-arm64` (625 cargo, 26 + 420 pytest); `docs/SPEC.md` sole spec, worktree-safe checks. |
| 2026-11-19 | M16 merged (`9af6590`), gate green (625 cargo, 26 + 368 pytest); Q47–Q49 to `DECISIONS.md`; `README.md` added. |
| 2026-11-09 | M15 merged (`70c79e4`), gate green (577 cargo, 26 + 368 pytest); Q42–Q46 to `DECISIONS.md`; v2 rules merged into `SPEC.md`. |
| 2026-10-31 | M14 closed (`3501282`, Q39–Q41); capacity ladder results logged in `RL_REPORT.md`. |
| 2026-10-28 | M13 (`c8a0fcf`, Q36–Q37) and M13b (`074b0ef`, Q38) merged; M12c LR schedule results logged in `RL_REPORT.md`. |
| 2026-10-26 | M12 closed (Q31–Q34); multi-seed harness and 12a/12b results logged in `RL_REPORT.md`; `p_bot` 0.4 adopted (`4c6740f`). |
| 2026-10-24 | M11 merged (`540a214`, 11b passed); Q1–Q30 moved to `DECISIONS.md`; turn cap closed at 30; `STATUS.md` created. |
