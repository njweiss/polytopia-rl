# Milestone 10a: PPO smoke run report

Run by the implementer on 2026-10-22, on the owner's machine (Ryzen 5 5600X3D, WSL2). **CPU only**:
the pixi env's torch is the `cpu_generic` build (QUESTIONS.md Q21), so the RTX 4060 was not used.

Command (both runs in parallel, `OMP_NUM_THREADS=6` each):

```
python -m polytopia_rl.train --config python/configs/stage1_random.json --run-dir runs/smoke_shaped   total_steps=2000000 shaping_weight=1.0
python -m polytopia_rl.train --config python/configs/stage1_random.json --run-dir runs/smoke_unshaped total_steps=2000000 shaping_weight=0.0
python -m polytopia_rl.evaluate --checkpoint runs/smoke_<name>/latest.pt --opponents random --seeds 200
```

Default hyperparameters otherwise (32 envs x 128 steps, 488 updates, size 11, turn cap 30, seed 1).
Run dirs (`metrics.jsonl`, `eval.jsonl`, checkpoints, `train.log`, `final_eval.txt`) are under
`runs/` (gitignored). `python -m polytopia_rl.plot --run runs/smoke_shaped` draws the curves.

## Gate result

Final greedy evaluation, 200 eval seeds (>= 1e9) side-swapped = 400 games vs `random`:

| run | W-L-D | win rate | Wilson 95% | seat 0 / seat 1 | mean length | mean score diff |
| --- | --- | --- | --- | --- | --- | --- |
| shaped (`shaping_weight=1`) | 380-19-1 | **0.950** | [0.924, 0.967] | 0.955 / 0.945 | 243.4 | +1497 |
| unshaped (`shaping_weight=0`) | 400-0-0 | **1.000** | [0.990, 1.000] | 1.000 / 1.000 | 66.8 | +893 |

**The pipeline passes (>= 75% vs random) in both runs.**

## Metrics over time

Per update (`metrics.jsonl`); win rate, return and length are over the last 200 finished training
episodes. Return includes the shaping term in the shaped run.

### Shaped (`shaping_weight=1.0`)

| step | train win | return | entropy | KL | clipfrac | expl. var | end_turn_freq | ep. length |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4,096 | 0.483 | 0.021 | 2.104 | 0.007 | 0.098 | -0.169 | 0.281 | 105 |
| 126,976 | 0.975 | 0.702 | 2.536 | 0.068 | 0.266 | 0.917 | 0.097 | 228 |
| 253,952 | 0.995 | 0.285 | 2.580 | 0.078 | 0.285 | 0.978 | 0.068 | 385 |
| 503,808 | 0.980 | 0.381 | 1.864 | 0.072 | 0.257 | 0.986 | 0.100 | 322 |
| 1,003,520 | 0.945 | 0.647 | 1.160 | 0.044 | 0.216 | 0.954 | 0.149 | 207 |
| 1,503,232 | 0.980 | 0.654 | 1.214 | 0.022 | 0.159 | 0.989 | 0.112 | 250 |
| 1,998,848 | 0.980 | 0.633 | 1.120 | 0.000 | 0.000 | 0.982 | 0.097 | 283 |

Periodic greedy eval (50 seeds, 100 games): 0.92 at 0.5M, 0.92 at 1.0M, **0.55 at 1.5M**, final
0.95 (400 games).

### Unshaped (`shaping_weight=0.0`)

| step | train win | return | entropy | KL | clipfrac | expl. var | end_turn_freq | ep. length |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4,096 | 0.483 | -0.034 | 2.104 | 0.007 | 0.098 | -0.172 | 0.281 | 105 |
| 126,976 | 0.895 | 0.790 | 1.953 | 0.075 | 0.210 | 0.525 | 0.195 | 116 |
| 253,952 | 0.990 | 0.980 | 2.424 | 0.048 | 0.291 | 0.749 | 0.151 | 234 |
| 503,808 | 0.960 | 0.920 | 2.525 | 0.040 | 0.257 | 0.740 | 0.153 | 174 |
| 1,003,520 | 1.000 | 1.000 | 1.450 | 0.043 | 0.243 | 0.328 | 0.167 | 109 |
| 1,503,232 | 0.995 | 0.990 | 1.273 | 0.015 | 0.120 | 0.101 | 0.180 | 77 |
| 1,998,848 | 1.000 | 1.000 | 1.119 | 0.000 | 0.000 | 0.266 | 0.165 | 73 |

Periodic greedy eval (50 seeds): 0.97 at 0.5M, 1.00 at 1.0M, 1.00 at 1.5M, final 1.00 (400 games).

Action mix in the last rollout (`kind_freq`):

| run | move | attack | capture | recover | train | choose_reward | site | research | end_turn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| shaped | 0.551 | 0.013 | 0.009 | 0.097 | 0.052 | 0.029 | 0.126 | 0.025 | 0.097 |
| unshaped | 0.452 | 0.068 | 0.030 | 0.017 | 0.088 | 0.033 | 0.104 | 0.042 | 0.165 |

## Shaping ablation (K3)

Against `random`, the terminal-only reward was **not** too sparse: the unshaped run learned as fast
and ended stronger. The two runs learned different games:

- **Unshaped** learned to finish: episodes shrink from ~230 to ~73 agent actions, attacks and captures
  are 3-5x more frequent, and it wins every eval game. Its value function explains little of the
  return late in training (explained variance 0.1-0.3), which is expected: once nearly every game is a
  win, the discounted ±1 target is close to constant and its variance is small.
- **Shaped** learned to farm score: long games (~250-380 actions, near the turn cap), few attacks
  (1.3%), a lot of `recover` (10%), a +1497 score margin, but 19 losses and one draw in 400 games.
  Hypothesis (not tested): the score-difference potential pays for each point as it is gained. With `gamma = 0.999` and
  hundreds of steps per game, `F = gamma * phi' - phi` is policy-invariant in theory but not in
  practice: the value net fits the dense shaping stream very well (explained variance 0.98), and the
  policy follows that stream rather than the faint terminal signal at the end of the episode.
- The shaped run's greedy eval dropped to 0.55 at 1.5M and recovered to 0.95 by the end, while its
  training win rate stayed at 0.97-0.98 throughout. So its greedy policy was brittle at that point.
  The unshaped run's eval never dropped below 0.97.

Not a reason to change the rules or tests. Proposal for the owner, before 10b: run the curriculum
with `shaping_weight` reduced (0 or about 0.1-0.3) as well as at the default 1.0, and compare vs
`economy`. A scripted economy opponent is a harder test of "farm score vs win" than `random` is.

## Diagnostics checked

- **Entropy:** 2.10 at the start, a rise to ~2.5 (the policy spreads over more legal actions as games
  get longer), then a smooth fall to ~1.1. No collapse.
- **KL / clipfrac:** 0.04-0.08 and 0.2-0.29 early (on the high side for PPO with 4 epochs x 8
  minibatches, but stable, with no blow-up). Both fall as LR anneals; they reach 0 at the last update
  because LR is `lr / 488` there.
- **`end_turn_freq`:** falls from 0.28 (uniform-ish start) to 0.10-0.17. The agent did not learn to
  pass.
- **Value:** explained variance went from negative to above 0.9 within ~120k steps in the shaped run.

## Throughput (K9)

Averages over 488 updates, two runs sharing the CPU (6 threads each):

| quantity | value |
| --- | --- |
| overall sps (agent steps per wall second, rollout + update) | ~172 per run |
| env stepping only | ~28,800 steps/s |
| policy forward during rollout (incl. lazy mask fetch) | ~1,700 steps/s |
| rollout time per update (4,096 steps) | 2.9 s |
| PPO update time per update (4 epochs x 8 minibatches) | 21.9 s |
| env share of total time | **0.6%** |
| wall time per 2M-step run | 3.4 h |

The Rust env is far under 20% of the step time, so **subprocess workers are not worth it**. 88% of
the time is the PPO update's forward and backward pass on CPU. The GPU (Q21) is the obvious next lever
for 10b's 20M steps, which would take about 34 h at this CPU rate (about 22 h for one run alone on all 12
threads, which measured ~255 sps).

**Update (same day, after Q21):** with the CUDA build on the RTX 4060, the default config trains at
~1,550 sps (rollout 1.3 s + update 1.3 s per 4,096 steps), so 10b's 20M steps take about 3.6 h. The
rollout (single-env forward passes and mask fetches in Python) is now half the time, while the Rust
env is still only ~4% of it. Subprocess workers are still not worth it; batching the per-step mask
fetch is the next thing to profile.

# Milestone 10b: curriculum run and gate

Run by the owner on 2026-10-22/23 on the RTX 4060: `pixi run scripts/train.sh
python/configs/curriculum.json` (defaults: 20M steps, `shaping_weight 1.0`, seed 1), run dir
`runs/curriculum_seed1/`. 4,882 updates at a mean of ~1,300 sps, so 4.4 h of training (evals not
included).

## Curriculum

| stage | opponents | entered at | training win rate when it advanced |
| --- | --- | --- | --- |
| 0 | random | step 0 | 0.99 (at 1.0M, the `min_stage_steps` floor) |
| 1 | economy | 1.0M | ~0.95 (at 2.0M, the floor again) |
| 2 | economy + rush | 2.0M | final stage (0.99 at 20M) |

Each stage was passed as soon as `min_stage_steps` (1M) allowed. In-training greedy evals (50
seeds, 100 games) in stage 2 started at 0.89-0.97 vs economy and 0.76-0.89 vs rush (2.5-4.5M), and
were 0.99-1.00 vs economy and 0.96-1.00 vs rush over 17-19.5M. Over training, entropy fell from 2.10
to 0.52 and episodes shortened from ~210 to ~74 agent actions. KL stayed high (0.04-0.12) through
most of stage 2 without destabilizing, then fell with the LR.

## Gate (K2): greedy, 200 eval seeds, side-swapped (400 games per opponent)

`scripts/rl_gate.sh` on both final candidates (outputs in `runs/curriculum_seed1/gate_*.txt`):

| checkpoint | opponent | W-L-D | win rate | Wilson 95% | seat 0 / seat 1 | length | score diff | gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `best.pt` (13.0M) | random | 399-1-0 | 0.998 | [0.986, 1.000] | 1.000 / 0.995 | 118.7 | +1347 | PASS (>= 0.90) |
| | economy | 397-3-0 | 0.993 | [0.978, 0.997] | 0.990 / 0.995 | 114.7 | +959 | PASS (>= 0.60) |
| | rush | 389-11-0 | 0.973 | [0.951, 0.985] | 0.985 / 0.960 | 98.6 | +1454 | reported |
| `latest.pt` (20.0M) | random | 400-0-0 | 1.000 | [0.990, 1.000] | 1.000 / 1.000 | 74.4 | +1139 | PASS (>= 0.90) |
| | economy | 400-0-0 | 1.000 | [0.990, 1.000] | 1.000 / 1.000 | 70.0 | +903 | PASS (>= 0.60) |
| | rush | 391-9-0 | 0.978 | [0.958, 0.988] | 0.995 / 0.960 | 67.0 | +1223 | reported |

**10b passes with both checkpoints.** Notes for the owner:

- **No forgetting vs random.** The agent last trained against random at 1M steps, and nothing
  evaluated it against random after that, yet both checkpoints win 399-400 of 400.
- **`best.pt` is not the best checkpoint.** It was picked by a 50-seed eval (0.995 mean at 13.0M).
  Later checkpoints only tied that score, so it was never replaced. On 400 games, `latest.pt` is as
  good or better and finishes games ~40% faster. At these win rates, 50 seeds cannot separate
  checkpoints; use `latest.pt` (or rank by a larger eval) going forward.
- **Rush is the weak spot, mainly as seat 1** (0.960 for both checkpoints, vs 0.985-0.995 as seat 0).
  Every loss is a decisive game; there are no draws.
- **The bots no longer test the agent much.** With the gate this far exceeded, the next signal has to
  come from harder opposition (self-play, the spec's follow-up) or harder settings (turn cap, spec
  open question 1).

# Milestone 11: evaluation suite, versioning and snapshot-pool self-play

## What was built (11a)

- **Hashes.** `Env.spec()` has `obs_hash` (FNV-1a 64 of the observation and action layout: size,
  channel and global names, action types, arg lengths, unit/reward/tech/site-action names, one item
  per line) and `rules_hash` (FNV-1a 64 of every embedded `data/*.json`, by file name, then
  `RULES_VERSION`). Current build, size 11: obs `d5f9b1033cfd55ca`, rules `a8b52b91eae34503`.
  **Bumping `RULES_VERSION`:** a rules change made in Rust code rather than in `data/*.json` is not
  visible to `rules_hash`; increase `RULES_VERSION` in `crates/sim-env/src/hash.rs` by one (and the
  matching constant in `crates/sim-env/tests/hash.rs` and `python/tests/rl/sp_helpers.py`, a phase A
  change).
- **Checkpoints** (`polytopia_rl.checkpoint`) store `spec`; loading refuses another `obs_hash`
  (`SpecMismatch`), warns on another `rules_hash`, and loads M10 files (no `spec`) with a warning.
  Every CLI that takes a checkpoint prints its hashes and step on its first line.
- **Suite** (`scripts/suite.sh CKPT`, `--compare A.json B.json`): the 8 standing conditions,
  200 seeds side-swapped from `[1e9, 1.5e9)`.
- **Loss review** (`python -m polytopia_rl.losses --checkpoint CKPT --opponent rush --count 5 --out DIR`).
- **Ratings and tournaments** (`scripts/tournament.sh RUNDIR`): Bradley-Terry in Elo units, random
  bot = 0, seeds `>= 2e9`.
- **Self-play** (`scripts/selfplay.sh python/configs/selfplay.json`, warm start from
  `runs/curriculum_seed1/latest.pt` by default). The rating rows in `eval.jsonl` play the learner
  against each rated opponent only (a star graph), so opponents the learner beats equally often get
  equal ratings there (in the smoke run all three bots rated 0). The full round robin at the end
  (`tournament.md`) separates them.

## Sanity runs by the implementer (not the gate)

- Suite CLI on the M10 `latest.pt`, 3 seeds per condition: 6-0-0 in every condition. Not a result:
  far too few games.
- Self-play smoke, 2 updates (8,192 steps) from the M10 `latest.pt`: every output is written (pool,
  eval rows, ratings, `ckpt_final.pt`, `tournament.json`/`.md`, `train.log`). **Throughput is
  ~200-240 sps** against ~1,550 sps for M10 training. The opponent rounds dominate the rollout (see
  docs/QUESTIONS.md Q28). At this rate 20M steps take ~25 h.

## Gate 11a (owner's machine, 2026-10-23/24)

Build hashes for every run below: obs `d5f9b1033cfd55ca`, rules `a8b52b91eae34503`. The M10
checkpoint is an unstamped M10 file (it loads with a warning; the suite records the current build's
hashes).

### 1. Baseline suite: M10 `latest.pt` (step 20.0M)

`pixi run scripts/suite.sh runs/curriculum_seed1/latest.pt` -> `runs/curriculum_seed1/suite_latest/`
(200 seeds, 400 games per condition, 22 min).

| condition | games | W-L-D | win rate | 95% CI | seat 0 | seat 1 | mean len | score diff | losses capital / score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_e0 | 400 | 400-0-0 | 1.000 | [0.990, 1.000] | 1.000 | 1.000 | 74.4 | +1138.5 | 0.00 / 0.00 |
| economy_e0 | 400 | 400-0-0 | 1.000 | [0.990, 1.000] | 1.000 | 1.000 | 70.0 | +903.4 | 0.00 / 0.00 |
| rush_e0 | 400 | 391-9-0 | 0.978 | [0.958, 0.988] | 0.995 | 0.960 | 67.0 | +1223.0 | 1.00 / 0.00 |
| economy_e20 | 400 | 400-0-0 | 1.000 | [0.990, 1.000] | 1.000 | 1.000 | 69.4 | +913.2 | 0.00 / 0.00 |
| rush_e20 | 400 | 395-5-0 | 0.988 | [0.971, 0.995] | 0.990 | 0.985 | 68.3 | +1225.4 | 1.00 / 0.00 |
| economy_fullobs | 400 | 398-2-0 | 0.995 | [0.982, 0.999] | 1.000 | 0.990 | 67.9 | +361.0 | 0.50 / 0.50 |
| rush_fullobs | 400 | 386-14-0 | 0.965 | [0.942, 0.979] | 0.975 | 0.955 | 66.7 | +1043.5 | 0.93 / 0.07 |
| rush_e0_sampled | 400 | 388-12-0 | 0.970 | [0.948, 0.983] | 0.980 | 0.960 | 69.9 | +1210.2 | 0.92 / 0.08 |

This is the standing baseline for 11b. The `random_e0`, `economy_e0` and `rush_e0` rows match the
10b gate for `latest.pt` exactly (same checkpoint, greedy), though on suite seeds rather than eval
seeds.

### 2. Loss review: M10 `latest.pt` against rush

`PYTHONPATH=python pixi run python -m polytopia_rl.losses --checkpoint runs/curriculum_seed1/latest.pt --opponent rush --count 5 --out runs/curriculum_seed1/losses_rush`
(replays in `runs/curriculum_seed1/losses_rush/`):

| seed | seat | final turn | loss type | scores | first city lost |
| --- | --- | --- | --- | --- | --- |
| 1000000052 | 1 | 9 | capital_captured | 1380-1255 | 9 |
| 1000000059 | 1 | 5 | capital_captured | 1330-960 | 5 |
| 1000000069 | 0 | 11 | capital_captured | 1680-1690 | 11 |
| 1000000102 | 1 | 6 | capital_captured | 1270-935 | 6 |
| 1000000125 | 1 | 8 | capital_captured | 1305-705 | 8 |

5 losses; share by seat: seat 0 0.20, seat 1 0.80.

### 3. Self-play smoke run: 3M steps from M10 `latest.pt`

`pixi run scripts/selfplay.sh python/configs/selfplay.json total_steps=3000000 --run-dir runs/selfplay_smoke`:
732 updates, ~5 h including evals. The run stopped at step 2,998,272, just short of 3M, so there is
no 3M snapshot or 3M in-training eval. The owner stopped the run during the final round robin
(at the time ~1.6 h for this pool); the round robin was rerun after Q29 R2 (step 4).

In-training suite subset (50 seeds, 100 games per condition) and ratings (`eval.jsonl`):

| step | economy_e0 | rush_e0 | rush_e20 | regression flag |
| --- | --- | --- | --- | --- |
| 1.0M | 1.00 | 0.79 | 0.91 | true |
| 2.0M | 0.97 | 0.85 | 0.88 | true |

Rating at 2.0M (star graph): learner 921, snap_1003520 998, snap_0 852, rush 684, economy 193,
random 0. The 2M learner scored 0.39 against the 1M snapshot.

Training curves (`metrics.jsonl`, means per ~250k steps):

| steps | rollout sps | train win rate | entropy | approx_kl | clipfrac | explained var |
| --- | --- | --- | --- | --- | --- | --- |
| 0-0.25M | 279 | 0.66 | 0.70 | 0.235 | 0.29 | 0.81 |
| 0.75-1.0M | 334 | 0.77 | 0.80 | 0.155 | 0.28 | 0.97 |
| 1.25-1.5M | 162 | 0.42 | 0.84 | 0.103 | 0.25 | 0.90 |
| 2.0-2.25M | 180 | 0.55 | 0.77 | 0.059 | 0.20 | 0.93 |
| 2.75-3.0M | 187 | 0.63 | 0.71 | 0.005 | 0.04 | 0.94 |

Suite on the final weights, `pixi run scripts/suite.sh runs/selfplay_smoke/ckpt_final.pt` ->
`runs/selfplay_smoke/suite_ckpt_final/` (63 min, as games are about twice as long):

| condition | games | W-L-D | win rate | 95% CI | seat 0 | seat 1 | mean len | score diff | losses capital / score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_e0 | 400 | 400-0-0 | 1.000 | [0.990, 1.000] | 1.000 | 1.000 | 185.3 | +2345.6 | 0.00 / 0.00 |
| economy_e0 | 400 | 400-0-0 | 1.000 | [0.990, 1.000] | 1.000 | 1.000 | 161.3 | +1826.0 | 0.00 / 0.00 |
| rush_e0 | 400 | 347-53-0 | 0.868 | [0.831, 0.897] | 0.910 | 0.825 | 135.6 | +2017.2 | 1.00 / 0.00 |
| economy_e20 | 400 | 396-4-0 | 0.990 | [0.975, 0.996] | 0.995 | 0.985 | 170.1 | +1804.7 | 0.75 / 0.25 |
| rush_e20 | 400 | 371-29-0 | 0.927 | [0.898, 0.949] | 0.930 | 0.925 | 146.7 | +2181.6 | 1.00 / 0.00 |
| economy_fullobs | 400 | 373-27-0 | 0.932 | [0.904, 0.953] | 0.940 | 0.925 | 228.1 | +1006.9 | 0.19 / 0.81 |
| rush_fullobs | 400 | 267-133-0 | 0.667 | [0.620, 0.712] | 0.675 | 0.660 | 135.9 | +1499.0 | 1.00 / 0.00 |
| rush_e0_sampled | 400 | 350-50-0 | 0.875 | [0.839, 0.904] | 0.895 | 0.855 | 139.5 | +2055.9 | 1.00 / 0.00 |

`scripts/suite.sh --compare` (A = M10 baseline, B = smoke final):

| condition | A | B | diff | z |
| --- | --- | --- | --- | --- |
| random_e0 | 1.000 | 1.000 | +0.000 | 0.00 |
| economy_e0 | 1.000 | 1.000 | +0.000 | 0.00 |
| rush_e0 | 0.978 | 0.868 | -0.110 | -5.82 (significant) |
| economy_e20 | 1.000 | 0.990 | -0.010 | -2.01 (significant) |
| rush_e20 | 0.988 | 0.927 | -0.060 | -4.21 (significant) |
| economy_fullobs | 0.995 | 0.932 | -0.062 | -4.73 (significant) |
| rush_fullobs | 0.965 | 0.667 | -0.297 | -10.86 (significant) |
| rush_e0_sampled | 0.970 | 0.875 | -0.095 | -5.02 (significant) |

### 4. Round robin of the smoke pool and the bots

`pixi run scripts/tournament.sh runs/selfplay_smoke` (100 seeds, 200 games per pair, 1 min 37 s
after Q29 R2) -> `runs/selfplay_smoke/tournament.md`:

| player | Elo |
| --- | --- |
| snap_1003520 | 1296 |
| snap_0 | 1171 |
| snap_2002944 | 1167 |
| rush | 812 |
| economy | 531 |
| random | 0 |

Seat win rate: seat 0 0.524, seat 1 0.476.

Score of row against column (wins + draws / 2, over games):

| | random | economy | rush | snap_0 | snap_1003520 | snap_2002944 |
| --- | --- | --- | --- | --- | --- | --- |
| random |  | 0.00 | 0.04 | 0.00 | 0.00 | 0.00 |
| economy | 1.00 |  | 0.17 | 0.00 | 0.01 | 0.00 |
| rush | 0.95 | 0.83 |  | 0.01 | 0.12 | 0.20 |
| snap_0 | 1.00 | 1.00 | 0.99 |  | 0.23 | 0.47 |
| snap_1003520 | 1.00 | 0.99 | 0.88 | 0.77 |  | 0.65 |
| snap_2002944 | 1.00 | 1.00 | 0.81 | 0.53 | 0.35 |  |

The final weights (step 2,998,272) are not in the round robin: it covers pool snapshots only.

### Gate 11a verdict

**The pipeline passes, with one marginal ordering note.** Every step ran and wrote its outputs. The
numbers agree with each other:

- `random` is the lowest rated (0), as required.
- Elo order agrees with the head-to-head scores for every pair but one: snap_0 is rated 4 Elo above
  snap_2002944, while snap_2002944 won their match 0.53 (105-95 of 200, about 0.7 sd from 0.5, not
  significant). The joint fit puts snap_0 ahead because it scores 0.99 against rush to
  snap_2002944's 0.81.
- The in-training rating at 2M and the round robin agree on the order of the three snapshots
  (1M > snap_0 and 2M), and the suite, the in-training evals and the round robin all show the same
  decline against rush.

**No performance threshold applies at 11a, but self-play did not improve the agent in 3M steps.**
Per the gate, this is reported and no rule or test was changed.

### What we learned

- **How the M10 agent loses to rush.** Every loss in every rush condition is a capital capture
  (the few "score" losses in `rush_fullobs` and `rush_e0_sampled` are 1 game each). They come
  early: median turn 9-12.5, first city lost on turn 5-11 in the loss review. Seat 1 carries most of
  them (8 of 9 in `rush_e0`, 4 of 5 in the review): rush moves first as seat 0 and reaches the
  capital before the agent has defenders.
- **Noisy opponents and full observability (M10).** Noise helps the agent slightly
  (`rush_e20` 0.988 vs `rush_e0` 0.978; economy is 1.000 either way). Full observability costs
  ~1 point against rush (0.965) and almost nothing against economy (0.995), even though M10 never
  trained with it. Sampling instead of argmax costs ~1 point (0.970).
- **Self-play from the M10 checkpoint traded bot robustness for strength against nets.** The 1M
  snapshot beats snap_0 0.77 head to head and is the top-rated player, but every snapshot after the
  warm start is worse against rush (0.99 -> 0.88 -> 0.81 in the round robin; 0.978 -> 0.868 for the
  final weights in the suite). After 1M the learner also stopped improving against its own pool:
  the 2M snapshot loses to the 1M one 0.35.
- **The worst regression is under full observability against rush** (0.965 -> 0.667). M10 handled
  the unseen condition almost as well as fog; the self-play weights do not.
- **The self-play agent plays longer games and wins bigger.** Mean length roughly doubled
  (67-74 -> 135-228 agent actions) and score margins grew (e.g. +1223 -> +2017 against rush). But
  against economy with full observability 22 of its 27 losses are on score at the turn cap: it builds
  a lead and then fails to finish. Against rush it still loses by capital capture around turn 8-10,
  as M10 did, only five times as often.
- **The Elo curve is not monotone:** snap_0 1171, 1M 1296, 2M 1167. With 20% bot games and half the
  rest against the newest snapshot, the learner mostly chases its latest self.
- **The warm start loosened the policy (Q29 R6).** In the first 250k steps approx_kl averaged 0.235
  (M10's stage 2: ~0.05) with clipfrac 0.29, and entropy rose from 0.48 to ~0.8-0.9. Self-play starts
  a fresh Adam at the full `lr` (3e-4) on a converged policy; rush_e0 had already fallen to 0.79 by
  the 1M eval. The likely cause is that step size, though this is not tested. No change for now; if 11b repeats
  it, the first thing to try is a lower `lr` or an LR warm-up (config only).
- **Throughput.** Rollout ran at ~280-330 sps with snap_0 alone in the pool and ~160-190 sps
  after the 1M snapshot joined; the PPO update is ~6% of the time. A profile put ~90% of the rollout
  in opponent policy calls on batches of ~1.5 envs (Q29). Q29 R1 (skip opponent `log_prob` and
  `entropy`) raised profiled rollout throughput from 177 to 215 sps. Q29 R2 (batched tournament and
  rating games) cut a 100-seed pair from ~20 min to ~19 s for net vs net. The suite still plays one
  game at a time (~1 h per checkpoint at the smoke agent's game length).

**For 11b:** gate (ii) requires no suite condition more than 5 points below this baseline unless not
significant. The 3M smoke weights would fail it in five conditions. Running the 20M config
unchanged risks the same result; whether to change `lr`, `p_bot` or anything else first is the
owner's decision.

# Milestone 11b: 20M-step self-play run and gate

Run on the owner's machine (RTX 4060, WSL2) on 2026-10-24: `pixi run scripts/selfplay.sh
python/configs/selfplay.json`, run dir `runs/selfplay_seed1/`, warm start from
`runs/curriculum_seed1/latest.pt`. Build hashes: obs `d5f9b1033cfd55ca`, rules `a8b52b91eae34503`.

## Changes from the 11a smoke run

Every change is from the owner's answers to Q29 and Q30 (docs/QUESTIONS.md):

- **`lr` 1e-4** (was 3e-4). The smoke run's warm start had approx_kl ~0.2-0.38 and rush_e0 fell to
  0.79 by 1M (Q29 R6).
- **`num_envs` 128, `num_steps` 32** (was 32 x 128): the same 4,096-step batch, minibatches and
  update count, with larger opponent batches (Q29 R3).
- **Speed only, no change to results:** opponents skip `log_prob`/`entropy`/`value` (Q29 R1).
  Play-only calls skip the mask check and run under `torch.inference_mode` (Q30 A).
  `scripts/selfplay.sh` defaults `OMP_NUM_THREADS=1` (Q30 follow-up).
- **Changes the opponents' random stream:** opponents sample with a generator on the GPU (Q30 B).
- **Batched tournament and rating games** (Q29 R2): the same games, about 60x faster.

**Run history.** A first attempt (`runs/selfplay_lr1e-4`, the old 32 x 128 layout) was stopped at
180k steps to add the speed work. This run was then stopped at ~1.15M and resumed from its 1M
`latest.pt` with `OMP_NUM_THREADS=1`. `metrics.jsonl` and `train.log` were cut back to step
1,003,520 first; the originals are in `pre_resume_backup/`. On the GPU the resume is not
bit-identical (cuDNN), but the configuration and code are the same.

## Training

4,882 updates, 18.1 h of rollout and update time (evals, ratings and the final round robin not
included). Means per phase from `metrics.jsonl`:

| steps | sps | pool | approx_kl | entropy | train win rate | episode len | explained var |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0-1M | 656 | 1 | 0.073 | 0.64 | 0.70 | 199 | 0.88 |
| 1-2M | 528 | 2 | 0.068 | 0.64 | 0.61 | 227 | 0.91 |
| 2-5M | 391 | 5 | 0.061 | 0.67 | 0.59 | 224 | 0.90 |
| 5-10M | 305 | 10 | 0.041 | 0.83 | 0.62 | 201 | 0.89 |
| 10-15M | 291 | 15 | 0.018 | 0.93 | 0.65 | 195 | 0.90 |
| 15-20M | 259 | 20 | 0.004 | 1.04 | 0.68 | 199 | 0.91 |

Throughput falls as the pool grows, because each snapshot is a separate group of opponent calls
(Q30). Entropy rose steadily, from 0.5 at the warm start to ~1.05. KL fell with the LR.

### In-training monitoring (`eval.jsonl`, 50 seeds = 100 games per condition)

| step (M) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| economy_e0 | 1.00 | 0.98 | 0.94 | 1.00 | 0.99 | 0.99 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| rush_e0 | 0.96 | 0.93 | 0.77 | 0.87 | 0.83 | 0.75 | 0.88 | 0.85 | 0.82 | 0.89 | 0.89 | 0.87 | 0.86 | 0.80 | 0.91 | 0.83 | 0.92 | 0.86 | 0.94 |
| rush_e20 | 0.99 | 0.98 | 0.93 | 0.90 | 0.92 | 0.91 | 0.96 | 0.97 | 0.95 | 0.91 | 0.96 | 0.92 | 0.96 | 0.89 | 0.92 | 0.96 | 0.96 | 0.96 | 1.00 |

The regression flag (any rate below 0.90) was set from 3M to 14M and at 16M and 18M; it was clear
at 1M, 2M, 15M, 17M and 19M.

The learner's score in the rating rows (100 games per opponent):

| step (M) | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vs snap_0 | 0.68 | 0.51 | 0.46 | 0.56 | 0.61 | 0.60 | 0.60 | 0.62 | 0.70 |
| vs rush | 0.96 | 0.87 | 0.86 | 0.86 | 0.83 | 0.90 | 0.89 | 0.90 | 0.96 |

The learner's own rating in these rows is always 921. The star graph links it to `random` by a
1.0 score and nothing else, so only the opponents' ratings relative to it carry information.

## Final round robin (`tournament.md`, 100 seeds = 200 games per pair)

20 snapshots plus the three bots, 253 pairs. Bots: rush 801, economy 461, random 0. Seat 0 won
54.3% of games.

| snapshot | Elo | vs snap_0 | vs rush | vs economy |
| --- | --- | --- | --- | --- |
| 0.0M (M10) | 1138 | — | 0.99 | 1.00 |
| 1.0M | 1129 | 0.71 | 0.93 | 1.00 |
| 2.0M | 1112 | 0.69 | 0.95 | 0.98 |
| 3.0M | 981 | 0.42 | 0.83 | 0.91 |
| 4.0M | 1122 | 0.56 | 0.86 | 0.98 |
| 5.0M | 1101 | 0.49 | 0.88 | 0.99 |
| 6.0M | 1088 | 0.47 | 0.82 | 0.99 |
| 7.0M | 1153 | 0.50 | 0.85 | 0.99 |
| 8.0M | 1180 | 0.52 | 0.87 | 1.00 |
| 9.0M | 1177 | 0.52 | 0.81 | 0.99 |
| 10.0M | 1196 | 0.62 | 0.85 | 0.99 |
| 11.0M | 1219 | 0.58 | 0.90 | 1.00 |
| 12.0M | 1237 | 0.57 | 0.88 | 1.00 |
| 13.0M | 1219 | 0.56 | 0.90 | 1.00 |
| 14.0M | 1227 | 0.61 | 0.88 | 0.99 |
| 15.0M | 1294 | 0.67 | 0.92 | 1.00 |
| 16.0M | 1293 | 0.63 | 0.90 | 0.99 |
| 17.0M | 1346 | 0.70 | 0.95 | 1.00 |
| 18.0M | 1346 | 0.70 | 0.97 | 1.00 |
| 19.0M | 1383 | 0.68 | 0.94 | 1.00 |

**Consistency.** `random` is the lowest rated. In 13 of the 253 pairs the higher-rated player
scored below 0.5. Most of those scores are 0.43-0.475, within noise for 200 games. The two clear
ones are snap_0 against the 1M (0.285) and 2M (0.315) snapshots. snap_0 is rated above both
because it beats the bots and the later snapshots more often, but it loses to these two head to
head. The final weights (step 19,996,672) are not in the round robin, which covers pool snapshots
only.

## Gate 11b

**(i) Final weights against snapshot 0, 200 side-swapped games (greedy, seeds >= 2e9):** 135-65-0,
score **0.675**, Wilson 95% [0.607, 0.736]. As seat 0 the final weights win 0.74; as seat 1, 0.61.
**PASS** (>= 0.60).

**(ii) Suite on `ckpt_final.pt` against the 11a baseline** (`runs/selfplay_seed1/suite_ckpt_final/`,
200 seeds, 400 games per condition):

| condition | games | W-L-D | win rate | 95% CI | seat 0 | seat 1 | mean len | score diff | losses capital / score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_e0 | 400 | 400-0-0 | 1.000 | [0.990, 1.000] | 1.000 | 1.000 | 97.6 | +1592.8 | 0.00 / 0.00 |
| economy_e0 | 400 | 399-1-0 | 0.998 | [0.986, 1.000] | 1.000 | 0.995 | 84.6 | +1271.1 | 1.00 / 0.00 |
| rush_e0 | 400 | 379-21-0 | 0.948 | [0.921, 0.965] | 0.940 | 0.955 | 116.1 | +1774.0 | 1.00 / 0.00 |
| economy_e20 | 400 | 399-1-0 | 0.998 | [0.986, 1.000] | 1.000 | 0.995 | 84.8 | +1288.0 | 1.00 / 0.00 |
| rush_e20 | 400 | 393-7-0 | 0.983 | [0.964, 0.991] | 0.985 | 0.980 | 123.3 | +1834.2 | 1.00 / 0.00 |
| economy_fullobs | 400 | 400-0-0 | 1.000 | [0.990, 1.000] | 1.000 | 1.000 | 78.5 | +716.9 | 0.00 / 0.00 |
| rush_fullobs | 400 | 376-24-0 | 0.940 | [0.912, 0.959] | 0.960 | 0.920 | 111.5 | +1556.6 | 1.00 / 0.00 |
| rush_e0_sampled | 400 | 378-22-0 | 0.945 | [0.918, 0.963] | 0.960 | 0.930 | 115.8 | +1735.2 | 1.00 / 0.00 |

`scripts/suite.sh --compare` (A = M10 baseline, B = 11b final):

| condition | A | B | diff | z |
| --- | --- | --- | --- | --- |
| random_e0 | 1.000 | 1.000 | +0.000 | 0.00 |
| economy_e0 | 1.000 | 0.998 | -0.002 | -1.00 |
| rush_e0 | 0.978 | 0.948 | -0.030 | -2.23 (significant) |
| economy_e20 | 1.000 | 0.998 | -0.002 | -1.00 |
| rush_e20 | 0.988 | 0.983 | -0.005 | -0.58 |
| economy_fullobs | 0.995 | 1.000 | +0.005 | +1.42 |
| rush_fullobs | 0.965 | 0.940 | -0.025 | -1.66 |
| rush_e0_sampled | 0.970 | 0.945 | -0.025 | -1.75 |

No condition falls more than 5 points below the baseline. The largest drop, rush_e0 at -3.0 points,
is significant but under the limit. **PASS.**

**11b passes both criteria.**

### Loss review: final weights against rush

`python -m polytopia_rl.losses --checkpoint runs/selfplay_seed1/ckpt_final.pt --opponent rush --count 5 --out runs/selfplay_seed1/losses_rush`:

| seed | seat | final turn | loss type | scores | first city lost |
| --- | --- | --- | --- | --- | --- |
| 1000000008 | 1 | 11 | capital_captured | 1690-710 | 5 |
| 1000000011 | 1 | 8 | capital_captured | 1560-1005 | 8 |
| 1000000020 | 0 | 14 | capital_captured | 1535-1745 | 14 |
| 1000000028 | 0 | 17 | capital_captured | 870-2100 | 7 |
| 1000000043 | 1 | 16 | capital_captured | 1435-1945 | 5 |

5 losses; share by seat: seat 0 0.40, seat 1 0.60.

## What we learned

- **The lower LR fixed the warm-start collapse.** At 1M, rush_e0 was 0.96 (smoke run at lr 3e-4:
  0.79) and KL averaged 0.07 instead of ~0.2. Q29 R6 was right about the cause.
- **Self-play first drifted, then improved.** From 2M to ~9M the snapshots got worse against rush
  (round robin 0.81-0.88) and were no better than snap_0 head to head (0.42-0.56). The 3M snapshot
  is the weakest player in the pool (981). From ~10M on, Elo climbs steadily (1196 -> 1383) and
  results against rush recover (0.92-0.97 for 15-19M). The last five snapshots are the five
  highest rated. The rise coincides with the LR decaying and the pool growing more varied; this
  single run cannot say which mattered.
- **The final agent is stronger against the earlier networks, and almost as good against the
  bots.** It beats M10 0.675 head to head. Against rush it is 2.5-3 points below M10 in every rush
  condition (only rush_e0 significantly). It is at least as good as M10 everywhere else, including
  full observability against economy (1.000).
- **How it loses to rush:** still capital captures only, early to mid game (median turn 9-11,
  range 5-25). The seat split is more even than M10's (rush_e0: 9 of 21 losses as seat 1, against 8
  of 9 for M10). Rush can still take the capital in the opening; self-play did not teach a better
  defence.
- **Full observability no longer hurts much** (rush_fullobs 0.940, against 0.667 for the smoke
  run's final weights).
- **Games are longer and margins larger than M10's** (mean length 78-123 against 67-74, score
  margins about +300-550 higher). Unlike the smoke run, it still finishes games: no losses by score
  at the turn cap in any condition.
- **Entropy rose throughout** (0.5 -> 1.05). The policy became more stochastic, yet greedy play got
  stronger. Sampled play (`rush_e0_sampled` 0.945) is about the same as greedy (0.948).
- **Seat 0 is favoured:** 54.3% of the round robin's decisive games, and the final weights win 0.74
  as seat 0 against snap_0 but 0.61 as seat 1.
- **Throughput:** 656 sps with one snapshot, down to ~260 sps with 20 (18.1 h of training). The
  remaining cost is per-snapshot opponent calls on small batches. The deferred options are
  interleaved rollouts (Q29 R4) and CUDA graphs (Q30 C).

**For the owner's next-step decision** (the brief lists combat preview, a capacity experiment, the
turn cap, and game v2): the bots are close to solved, apart from early rush capital captures.
Self-play adds strength against networks, but only after a long drift. Any next self-play run
inherits ~18 h of training per 20M steps unless the R4/C speed work is done first.
