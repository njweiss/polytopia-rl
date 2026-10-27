# Decisions log

Answered agent questions, one line each, newest last. The full entries (options, the agent's reasoning,
test names) are in `docs/QUESTIONS.md` as of commit `540a214` for Q1-Q30 (`git show 540a214:docs/QUESTIONS.md`) `7542ffa` for Q31-Q34 and `7af81d4` for Q35.
Rule-level answers are also merged into `docs/SPEC.md` (the Claude Doc master spec); "Where" names the
spec section, or the brief when the answer is an implementation detail.

| Q | Date | Question | Answer | Where |
| --- | --- | --- | --- | --- |
| Q1 | 2026-10-21 | Starting value of the turn counter | `turn = 0`, incremented when player 1 ends | Spec: State definition, Turn flow |
| Q2 | 2026-10-21 | Tile actions after the M4 D4 overrule | Improvements keep the resource; one improvement per tile; harvests need only the resource; clear forest forfeits an animal | Spec: Data tables (tile actions) |
| Q3 | 2026-10-21 | `Tables::from_json` signature, reward-menu validation | Args `(units, techs, site_actions, rewards)`; menus cover every level from 2 exactly once; unknown or missing is `Err` | Spec: Data tables; M4 brief |
| Q4 | 2026-10-21 | The Explorer's 0.5 visited penalty never changes a choice | Keep the rule as written (a no-op; untested) | Spec: Explorer |
| Q5 | 2026-10-21 | The Explorer differential test cannot see the path | Compare the whole state after the walk; the revealed mask pins the path | M5 brief (tests) |
| Q6 | 2026-10-21 | Digest byte-stream details | No array length prefix; `TechSet` as its `u16` LE; `Outcome` as variant `u8` plus payload | M7 brief (digest) |
| Q7 | 2026-10-21 | Random policy returns an illegal EndTurn with a pending reward | If EndTurn is illegal, pick uniformly from `legal` with no `gen_f32` draw | M7 brief (random policy) |
| Q8 | 2026-10-21 | The 5 MB HTML bound needs compact frames | Raise the bound (option B) | M7 brief (HTML) |
| Q9 | 2026-10-21 | Formatting details the M7 tests assume | As the tests encode (option A) | M7 brief |
| Q10 | 2026-10-21 | Economy bot's first Warrior never moves in ~25% of games | `explore_step` falls back to a greedy step toward the nearest explored tile with vision gain | Spec: Tooling (bots); M8 brief |
| Q11 | 2026-10-21 | ZoC from units on unexplored tiles leaks through movement | Keep the rule; the no-cheating test uses deep fog only | Spec: Movement |
| Q12 | 2026-10-21 | Details the M8 tests assume | As the tests encode (option A) | M8 brief |
| Q13 | 2026-10-22 | Rush beats Economy 84%, outside the 25-75% target | Accept the imbalance | Spec: Tooling (bots) |
| Q14 | 2026-10-22 | `sim-py` needs `extension-module` to link | Keep the feature off; install the Python dev package where the gate runs (option B) | M9 brief |
| Q15 | 2026-10-22 | Details the M9 tests assume | As the tests encode (option A) | M9 brief |
| Q16 | 2026-10-22 | Which map sizes `Env::new` accepts | `MapParams::MIN_SIZE = 6`; smaller sizes are rejected | Spec: Map generation |
| Q17 | 2026-10-22 | pixi as the Python environment manager | Approved; run the gate as `pixi run check`; new Python deps only with approval | CLAUDE.md |
| Q18 | 2026-10-22 | The pixi env's numpy cannot load MKL | Pin BLAS to OpenBLAS | `pixi.toml` |
| Q19 | 2026-10-22 | Details the M10 tests assume | As the tests encode (option A) | M10 brief |
| Q20 | 2026-10-22 | How `evaluate` passes `greedy` to `policy_fn` | `policy_fn(env, greedy)` | M10 brief |
| Q21 | 2026-10-22 | The pixi env's torch has no CUDA | CUDA 12.9 torch build plus `cuda-cudart-dev` | `pixi.toml` |
| Q22 | 2026-10-23 | Exact bytes behind `obs_hash` and `rules_hash` | One flat `\n`-joined list; rules bytes per file `name\0contents`, then `RULES_VERSION` | M11 brief; Spec: Interface and training |
| Q23 | 2026-10-23 | A hook to play the Rust bots' moves from Python | `Env.bot_action(name)` from the mover's own view, one RNG stream per seat | M11 brief |
| Q24 | 2026-10-23 | Loss review definitions | Suite seeds, greedy; capital capture if the game ended before the cap; first city lost from observations | M11 brief |
| Q25 | 2026-10-23 | Pool thinning and the snapshot draw | Append, then thin keeping positions 0, 2, 4, ... and the newest; the uniform draw includes the newest | M11 brief; Spec: Interface and training |
| Q26 | 2026-10-23 | Truncation and the final observation in two-seat slots | A slot ends on any sub-step's terminal or truncation; final observation defined only after the learner's own action | M11 brief |
| Q27 | 2026-10-23 | Details the M11 tests assume | As the tests encode (option A) | M11 brief |
| Q28 | 2026-10-23 | Self-play rollout throughput (~200 sps) | Accept for now; speed work later after profiling | M11 brief; RL_REPORT |
| Q29 | 2026-10-23 | Speed after the 3M smoke run | R1 opponent `act(stats=False)`; R2 batched tournament; R3 later (done in Q30); R4 deferred; R6 lower LR if needed | RL_REPORT |
| Q30 | 2026-10-24 | Per-call cost of play-only policy calls | A no mask check and `inference_mode` on play paths; B opponent generator on device; config 128x32, `lr 1e-4`; C (CUDA graphs) deferred. Follow-up: `OMP_NUM_THREADS=1` default | RL_REPORT; `scripts/selfplay.sh` |
| Q31 | 2026-10-25 | Experiment file formats and report schema | `h2h_{step}.json` is `asdict(play_pair)` (run = a, init = b); players `{arm}_seed{seed}`; `ExperimentReport` as in the stub; `significant` is `None` for exploratory metrics; `n_flagged` counts flagged comparisons | M12 brief; `experiment.py` |
| Q32 | 2026-10-25 | `total_steps` of the smoke spec | `smoke.json` holds 200000 itself | M12 brief |
| Q33 | 2026-10-25 | Launcher and trainer details | Overlap check raises `ValueError`; `Popen` launcher; `run` exits non-zero after the others finish; `make_batch_fn` once per condition | M12 brief |
| Q34 | 2026-10-25 | Batched greedy GPU evaluation is not game-for-game identical (TF32) | Accept: W-L-D and outcomes match; a few game lengths differ through TF32 numerics; documented in RL_REPORT | RL_REPORT (12a) |
| Q35 | 2026-10-27 | Scope of the `lr_anneal_steps < total_steps` check | Raise regardless of `anneal_lr`; checked in `PPOConfig.from_json` (so also after `load_config` overrides) | M12c brief; `config.py` |
