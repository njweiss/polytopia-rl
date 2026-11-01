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
  bots.** It beats M10 0.675 head to head. Against rush it is 2.5-3 points below M10 at eps 0
  (rush_e0, rush_fullobs, rush_e0_sampled; only rush_e0 significantly) and level at eps 0.2
  (-0.5). Against economy and random it matches M10 within 0.5 points, including
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

# Milestone 12a: multi-seed experiment tools and gate

Run on the owner's machine (RTX 4060 8 GB, WSL2 with 12 GB, 12 threads) on 2026-10-25, code at commit
`69412b6` (branch `milestone-12`). Build hashes: obs `d5f9b1033cfd55ca`, rules `a8b52b91eae34503`.
`pixi run check` passed first (Rust 534 tests, Python 26 + 264).

New: the batched `evaluate` path (used by the suite CLI, `scripts/suite.sh` and the in-training
suite subset), map seeds that vary with the run seed (`train_seed_base`), `gap_snapshot_0` in the
rating rows, `polytopia_rl.stats` (Welch's t-test) and `polytopia_rl.experiment` with
`scripts/experiment.sh run|report SPEC`.

## 1. Batched suite on the M10 `latest.pt`

`pixi run scripts/suite.sh runs/curriculum_seed1/latest.pt --out runs/curriculum_seed1/suite_latest_batched`
(200 seeds, 400 games per condition): **28 s, against 22 min** for the sequential M11 baseline
(`runs/curriculum_seed1/suite_latest/`), about 48x faster.

| condition | W-L-D baseline | W-L-D batched | games with a different outcome | games that differ at all | mean len (base / batched) |
| --- | --- | --- | --- | --- | --- |
| random_e0 | 400-0-0 | 400-0-0 | 0 | 8 | 74.36 / 74.12 |
| economy_e0 | 400-0-0 | 400-0-0 | 0 | 3 | 70.00 / 70.11 |
| rush_e0 | 391-9-0 | 391-9-0 | 0 | 4 | 67.04 / 67.07 |
| economy_e20 | 400-0-0 | 400-0-0 | 0 | 4 | 69.41 / 69.45 |
| rush_e20 | 395-5-0 | 395-5-0 | 0 | 6 | 68.26 / 68.18 |
| economy_fullobs | 398-2-0 | 398-2-0 | 0 | 7 | 67.88 / 67.83 |
| rush_fullobs | 386-14-0 | 386-14-0 | 0 | 9 | 66.68 / 66.83 |
| rush_e0_sampled | 388-12-0 | 391-9-0 | 21 | 383 | 69.92 / 70.86 |

**Greedy rows: the same W-L-D and the same outcome in all 2,800 games, but not game for game
identical.** 41 games (1.5%) differ in length and score diff, so `mean_length` and
`mean_score_diff` move slightly. The cause is TF32 convolutions on the GPU (PyTorch's default
`cudnn.allow_tf32 = True`). Their rounding depends on the batch size, so a near-tie argmax can flip.
Checked on 50 seeds of random_e0 and economy_e0 (`runs/gate12a/diag*.py`):

| device / setting | sequential vs M11 baseline | batched vs sequential |
| --- | --- | --- |
| GPU, TF32 on (default) | identical | 1 of 200 games differ |
| GPU, TF32 off | 3 of 200 differ | identical |
| CPU | 3 of 200 differ | identical |

So the batched path itself plays the same games (the CPU tests check this exactly). On the GPU,
exact identity needs TF32 off, which would also move the baseline. This is **docs/QUESTIONS.md Q34**
(owner decision; nothing changed in the code). `rush_e0_sampled` is equal in distribution only, as
the brief expects (M6): 0.978 against 0.970.

## 2. Concurrency measurement

1, 2 and 3 copies of the 11b config (`python/configs/selfplay.json`) started together from the M10
checkpoint, seeds 1 to N, `OMP_NUM_THREADS=1`, 36 updates each (`total_steps=147456`,
`tournament_seeds=1` so the end-of-run round robin does not overlap the others' measured updates);
script `runs/concurrency/measure.sh`. Total sps is the sum over trainers of each trainer's mean `sps`
over updates 5 to 30. Peak RSS is from `/usr/bin/time -v`. GPU memory is the peak whole-device use
from `nvidia-smi` (1,831 MiB idle; WSL does not report memory per process).

| trainers | total sps | sps per trainer | peak RSS per trainer (GiB) | peak GPU memory MiB |
| --- | --- | --- | --- | --- |
| 1 | 804 | 804 | 1.95 | 2344 (513 above idle) |
| 2 | 1264 | 648, 616 | 1.94, 1.94 | 2865 (1034 above idle) |
| 3 | **1570** | 511, 526, 532 | 1.96, 1.95, 1.94 | 3386 (1555 above idle) |

Three trainers give 1.95x the throughput of one. Memory allows it: each trainer is about 2 GiB RSS
and 0.5 GiB of GPU memory. During the smoke experiment below (3 trainers plus the launcher
evaluating) the whole machine peaked at 8.5 GiB used of 11.7 and 3.4 GiB GPU. **`scripts/experiment.sh`
defaults to `--parallel 3`.** The pool grows during a real run, and so does RSS (11b's RSS late in
the run was not measured), so watch memory in the first 12b hours.
These are warm-start rates with a pool of 1. 11b's rate fell to ~260 sps at 20 snapshots, so a
10M run with 3 at once will be slower per run than this table suggests.

## 3. Smoke experiment

`pixi run scripts/experiment.sh run python/experiments/smoke.json` (default `--parallel 3`), then
`report`. Spec: arms `base` and `pbot40` (`p_bot` 0.4), seeds 1 and 2, `total_steps` 200,000 (Q32
A; 48 updates = 196,608 steps), primary `rush_e0`. Both arms also set `eval_every` and
`snapshot_every` 65,536, `rating_every` 131,072 and `rating_seeds` 10, so that 200k steps give three
checkpoints, in-training suite rows, a pool and one rating row. Evaluation sizes, kept small for the
smoke: `suite_seeds` 50, `h2h_seeds` 50, `tournament_seeds` 20 (12b uses 200 / 100 / 100).

**Wall time 22 min** for the whole run: three trainers together finished at about 9 min (training
420-450 sps each, plus each run's own in-training evaluations and its 100-seed pool round robin). The
fourth started at about 11 min and finished at 18 min (556 sps). Each run's evaluation (3 checkpoints
x suite + head-to-head) took 2 to 5 min, and the 8-player round robin 1.5 min. `report` took 3 s.
Seed 2's runs trained on maps from 10,000,000 (`seed_base` in `ckpt_final.pt`), seed 1's from 0.
`command.txt` holds the command and the git commit. A repeated `run` launches nothing (tested).

Report (`runs/exp/smoke/report.md`; per-checkpoint tables omitted here):

| metric | base seed 1 | base seed 2 | base mean (sd) | pbot40 seed 1 | pbot40 seed 2 | pbot40 mean (sd) |
| --- | --- | --- | --- | --- | --- | --- |
| random_e0 | 1.000 | 1.000 | 1.000 (0.000) | 1.000 | 0.997 | 0.998 (0.002) |
| economy_e0 | 1.000 | 1.000 | 1.000 (0.000) | 0.990 | 0.997 | 0.993 (0.005) |
| rush_e0 | 0.970 | 0.990 | 0.980 (0.014) | 0.973 | 0.980 | 0.977 (0.005) |
| economy_e20 | 1.000 | 1.000 | 1.000 (0.000) | 1.000 | 0.993 | 0.997 (0.005) |
| rush_e20 | 0.990 | 0.993 | 0.992 (0.002) | 0.990 | 0.990 | 0.990 (0.000) |
| economy_fullobs | 0.993 | 0.997 | 0.995 (0.002) | 1.000 | 0.993 | 0.997 (0.005) |
| rush_fullobs | 0.963 | 0.947 | 0.955 (0.012) | 0.950 | 0.963 | 0.957 (0.009) |
| rush_e0_sampled | 0.973 | 0.973 | 0.973 (0.000) | 0.937 | 0.970 | 0.953 (0.024) |
| elo_vs_init | -26.7 | 7.0 | -9.9 (23.8) | 4.6 | 30.4 | 17.5 (18.2) |

Welch, pbot40 minus base (2 seeds per arm, so df is about 1-2 and only huge effects could show):

| metric | diff | t | df | p | flag |
| --- | --- | --- | --- | --- | --- |
| rush_e0 (primary) | -0.003 | -0.32 | 1.2 | 0.797 | not detected |
| rush_e0_sampled | -0.020 | -1.20 | 1.0 | 0.442 | exploratory |
| elo_vs_init | +27.4 | +1.29 | 1.9 | 0.333 | exploratory |
| economy_e0 | -0.007 | -2.00 | 1.0 | 0.295 | exploratory |

1 flagged comparison (the primary one). The other conditions differ by at most 0.003 (p >= 0.5).

Round robin of the final weights (20 seeds = 40 games per pair, exploratory): base_seed2 1129,
pbot40_seed2 1126, base_seed1 1119, init 1093, pbot40_seed1 1080, rush 566, economy 405, random 0.

The rating row in base/seed1's `eval.jsonl` at 131,072: learner 645.1, snapshot_0 611.9,
`gap_snapshot_0` +33.2.

**Smoke verdict:** the harness works end to end: launch with 3 at once and refill, per-checkpoint
suites and head-to-heads, the round robin, the report, the curves (`report_curves.png`) and the
hashes. 200k steps is far too short to separate the arms, as expected. Each run's own end-of-run pool
round robin (`tournament_seeds` 100 in the base config) costs a few minutes per run. It is unchanged
here, since the 11b config may not change.

**For 12b:** `pixi run scripts/experiment.sh run python/experiments/selfplay12b.json` (9 runs, 3 at
once), then `report`.

# Milestone 12b: self-play experiment (`selfplay12b`)

`pixi run scripts/experiment.sh run python/experiments/selfplay12b.json` (code `9e5fd5d`, build obs
`d5f9b1033cfd55ca`, rules `a8b52b91eae34503`), then `report`: `runs/exp/selfplay12b/report.md`,
`report.json`, `report_curves.png`, `tournament.md`. Three arms by three seeds, 10M steps each, warm
start from the M10 `latest.pt`, 3 trainers at once. Wall time about 36 h (2026-10-25 to 10-26): `base`
about 14.5 h including one interruption (the three `base` runs resumed from `latest.pt`), `pbot40`
about 9.5 h, `ent003` about 10.5 h. Per-trainer speed fell from ~550 sps (pool of 2) to ~180-250 sps
(pool of 10).

| arm | overrides | question |
| --- | --- | --- |
| `base` | none (the 11b config) | how far apart are seeds of one config? |
| `pbot40` | `p_bot: 0.4` | does more bot play prevent the drift against rush? |
| `ent003` | `ent_coef: 0.003` | does a smaller entropy bonus give a stronger policy? |

Each run's value is the mean over its checkpoints at 8.0M, 9.0M and 10.0M (400-game suite each;
`elo_vs_init` from 200 head-to-head games against the M10 checkpoint each). Seed `k` of every arm
trains on the same maps from the same starting weights, so arms can also be compared seed by seed.

## Primary metrics

| arm | rush_e0 (seeds 1, 2, 3) | mean | sd | elo_vs_init (seeds 1, 2, 3) | mean | sd |
| --- | --- | --- | --- | --- | --- | --- |
| base | 0.958, 0.935, 0.848 | 0.913 | 0.058 | 192, 142, 208 | 181 | 34 |
| pbot40 | 0.966, 0.936, 0.925 | 0.942 | 0.021 | 207, 113, 184 | 168 | 49 |
| ent003 | 0.893, 0.877, 0.882 | 0.884 | 0.008 | 182, 210, 177 | 190 | 18 |

Welch comparisons on the primary metrics (6 flagged):

| a | b | metric | diff | t | df | p | flag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base | pbot40 | rush_e0 | +0.029 | +0.81 | 2.5 | 0.488 | not detected |
| base | pbot40 | elo_vs_init | -12.8 | -0.37 | 3.6 | 0.731 | not detected |
| base | ent003 | rush_e0 | -0.029 | -0.87 | 2.1 | 0.473 | not detected |
| base | ent003 | elo_vs_init | +8.9 | +0.40 | 3.0 | 0.718 | not detected |
| pbot40 | ent003 | rush_e0 | -0.058 | -4.49 | 2.5 | 0.029 | significant |
| pbot40 | ent003 | elo_vs_init | +21.7 | +0.72 | 2.5 | 0.531 | not detected |

## Other rush conditions (exploratory, arm means)

| arm | rush_e20 | rush_fullobs | rush_e0_sampled |
| --- | --- | --- | --- |
| base | 0.962 | 0.870 | 0.909 |
| pbot40 | 0.976 | 0.910 | 0.956 |
| ent003 | 0.952 | 0.834 | 0.881 |

`random_e0`, `economy_e0`, `economy_e20` and `economy_fullobs` are 0.983-1.000 for every run. Round
robin of the final weights (exploratory): `pbot40` seeds 1st, 3rd and 4th (1372, 1315, 1307), `base`
2nd, 5th and 7th, `ent003` 6th, 8th and 9th (1241-1269); `init` 1112, `rush` 792.

## What we learned

- **Seed variance is as large as the effects we are after.** One config (`base`) scored 0.848 to
  0.958 against rush. Seed 3 is weak at all three of its checkpoints (0.82-0.87), so it is a weaker run,
  not a bad stopping point. Single-run results against rush (M10, 11b) are one draw from a spread of
  roughly +-5 points.
- **The one significant result is weak.** Six flagged tests at alpha 0.05 give about a 26% chance of at
  least one false positive if nothing is real; with a Bonferroni correction (0.05 / 6 = 0.0083) none
  passes. Neither variant is detected against `base`.
- **`pbot40` probably helps against bots, at no cost against networks** (consistent direction, not
  significant). Seed by seed it beats `base` on rush_e0 every time (+0.8, +0.1, +7.7 points), it is
  ahead on all four rush conditions (+1.4 to +4.6), its learning curve skips the 2-4M dip against rush
  that the other arms show, its `elo_vs_init` is within noise of `base`, and it ranks highest in the
  round robin. Bot games are also cheaper (its batch took ~9.5 h against ~10.5 h for `ent003`). A
  plausible mechanism, not tested: snapshots drift together away from defending fast rushes, and bot
  games keep that pressure on.
- **A smaller entropy bonus hurts against rush.** All three `ent003` seeds sit below the `base` mean on
  every rush condition with a small spread; strength against the M10 checkpoint is unchanged. Less
  exploration narrows play to what beats the current pool. Keep `ent_coef` 0.01.
- **The learning-rate schedule may matter more than either knob (hypothesis).** `base` seed 1 is the
  11b config on 11b's training maps, except that 10M steps decay the LR twice as fast. On the same
  rating seeds it scores 0.80 against the M10 checkpoint (`elo_vs_init` 241 at 10M) where the 11b final
  weights scored 0.675, and rush_e0 0.960 against 0.948. 11b also improved only after its LR had fallen
  below half. One run against one run; worth a direct test.
- **Report quirks:** when both arms have zero spread (e.g. economy_e20, 0.997 for every `base` seed and
  1.000 for every `pbot40` seed), Welch gives p = 0; those rows are exploratory and meaningless. The
  curves use a 0-1 axis, which hides the differences against rush.

# Milestone 12c: learning-rate schedule experiment (`selfplay12c`)

`pixi run scripts/experiment.sh run python/experiments/selfplay12c.json`, then `report`
(`runs/exp/selfplay12c/report.md`, `report.json`, `report_curves.png`, `tournament.md`). Code `7af81d4`
(lockstep rollout), build obs `d5f9b1033cfd55ca`, rules `a8b52b91eae34503`. Three schedules by three
seeds, 10M steps, `p_bot` 0.4, warm start from the M10 `latest.pt`. `decay10` reuses 12b's `pbot40` runs
(identical config, trained on code `9e5fd5d`, hard-linked; owner decision N1). The six new runs took
~20 h (2026-10-27 15:40 to 10-28 11:34), 3 at a time.

| arm | schedule | LR at 10M |
| --- | --- | --- |
| `decay10` | linear decay to 0 over 10M (default) | 0 |
| `decay20` | linear decay over a 20M horizon (11b's first half) | 5e-5 |
| `const` | no decay | 1e-4 |

## Primary metrics (mean over the 8/9/10M checkpoints)

| arm | rush_e0 (seeds 1, 2, 3) | mean | sd | elo_vs_init (seeds 1, 2, 3) | mean | sd |
| --- | --- | --- | --- | --- | --- | --- |
| decay10 | 0.966, 0.936, 0.925 | 0.942 | 0.021 | 207, 113, 184 | 168 | 49 |
| decay20 | 0.956, 0.901, 0.932 | 0.930 | 0.028 | 119, 105, 133 | 119 | 14 |
| const | 0.893, 0.861, 0.861 | 0.872 | 0.019 | 60, 80, 127 | 89 | 34 |

Welch (6 flagged): `const` below `decay10` on rush_e0 (-7.1 pp, p 0.013, significant) and below
`decay20` (-5.8 pp, p 0.046, significant); `elo_vs_init` differences not detected (p 0.09-0.27);
`decay20` vs `decay10` not detected on either. With a Bonferroni correction for 6 tests (0.0083), none
passes alone.

## All conditions (arm means)

| metric | decay10 | decay20 | const |
| --- | --- | --- | --- |
| random_e0 | 1.000 | 1.000 | 1.000 |
| economy_e0 | 0.998 | 0.997 | 0.994 |
| economy_e20 | 1.000 | 0.998 | 0.998 |
| economy_fullobs | 0.992 | 0.990 | 0.986 |
| rush_e0 | 0.942 | 0.930 | 0.872 |
| rush_e20 | 0.976 | 0.967 | 0.936 |
| rush_fullobs | 0.910 | 0.872 | 0.781 |
| rush_e0_sampled | 0.956 | 0.923 | 0.868 |
| elo_vs_init | 168 | 119 | 89 |

Round robin of the final weights (exploratory): the three `decay10` runs are the top three (1370,
1325, 1310), above every `decay20` run (1257-1291); two of three `const` runs are last (1200, 1228);
`init` 1153.

## What we learned

- **The hypothesis holds: how low the LR gets by the end of the run drives the result.** Every one of
  the nine metrics orders `decay10 >= decay20 >= const`, the dose series the brief predicted. Seed by seed
  (seed `k` of each arm shares maps and starting weights), `elo_vs_init` orders
  `decay10 > decay20 > const` in all three seeds (9 of 9 paired comparisons), and the round robin
  separates the arms almost completely. No single test survives a 6-test correction, but an ordered
  pattern this consistent across seeds and metrics is strong evidence; a test built for ordered
  alternatives (a trend test) would be the right tool and is not in the harness.
- **11b's slow start is explained.** 11b decayed over 20M, so at 10M it was where `decay20` is now:
  weaker against networks and against rush than a run that had annealed fully. Its late improvement
  came as its LR fell. A 10M run annealed to zero (`decay10`, `elo_vs_init` 168 = ~0.72 expected score)
  is at least as strong against the M10 checkpoint as 11b's 20M final (0.675 head to head; one run, at
  `p_bot` 0.2).
- **A constant LR hurts most against rush** (-7 to -13 pp), more than against networks. A large step size
  keeps chasing the latest pool and never settles into the robust play the bots punish the lack of.
- **For M14 (scaling):** anneal to zero within each run (the default), and 10M-step runs are enough to
  compare configurations; the final checkpoints of a fully annealed run are its best. Whether 20M steps
  annealed over 20M beats 10M annealed over 10M is untested.

# Milestone 13a: interleaved rollout gate

Run on the owner's machine (RTX 4060 8 GB, WSL2 with 11.7 GiB, 12 threads) on 2026-10-28, code at
commit `40f8711` (`main`, M13 merged at `c8a0fcf`). Build hashes: obs `d5f9b1033cfd55ca`, rules
`a8b52b91eae34503`. `pixi run check` was green before the gate. Every trainer ran with
`OMP_NUM_THREADS=1`; nothing else was running. Scripts and run dirs are in `runs/gate13a/` (not
tracked): `fixture.py`, `reprofile.py`, `smoke_compare.py`, `concurrency/measure.sh`, `concurrency/summarize.py`.

## 1. Profile fixture: lockstep vs interleaved

The M13 profile fixture: resume `runs/exp/selfplay12b/pbot40/seed1/latest.pt` (pool of 10, `p_bot`
0.4, 128 x 32) for 3 rollouts into a scratch dir, `tournament_seeds=1`, eval/rating/snapshot intervals
off. `runs/gate13a/fixture.py MODE` sets `rollout=MODE` and counts calls by wrapping
`SelfPlayVecEnv._policy` (opponent calls), `PolicyNet.trunk/act/value` outside opponent calls (learner
forward), `_advance` (rounds) and `tick`. Two runs per mode, alternated (order: interleaved 1, lockstep 1, interleaved 2, lockstep 2).
Per rollout = per 4,096 learner transitions. The lockstep counts reproduce the M13 profile exactly
(1,868 calls per rollout = 5,604 / 3, 2.2 groups per round).

| measure | lockstep run 1 / 2 | interleaved run 1 / 2 | ratio (means) |
| --- | --- | --- | --- |
| rollout sps | 360 / 340 | 585 / 580 | **1.66x** |
| update sps | 2,768 / 2,722 | 2,561 / 2,777 | 0.98x |
| total sps (rollout + update) | 319 / 303 | 476 / 480 | **1.54x** |
| rollout time, 3 rollouts (s) | 34.1 / 36.1 | 21.0 / 21.2 | |
| learner calls per rollout (mean batch) | 32 (128) | 109 (37.6) | 3.4x more |
| opponent calls per rollout | 1,868 | 659 / 650 | 2.86x fewer |
| opponent calls per learner transition | 0.456 | 0.161 / 0.159 | |
| opponent batch per call, mean (median) | 1.70 (1) | 4.39 / 4.45 (2) | |
| cost per opponent call (ms) | 5.5 / 5.9 | 5.9 / 6.2 | |
| rounds (lockstep) or ticks (interleaved) per rollout | 840 rounds | 159 / 147 ticks (5 without an opponent call) | 5.5x fewer |
| snapshot groups per round or tick: mean, max, share with 1 | 2.22, 7, 40% | 4.27 / 4.58, 10, 39% / 35% | 2.0x more |
| share of rollout time in opponent calls | 91% / 92% | 56% / 58% | |
| share of rollout time in learner forward | 6% / 5% | 18% / 16% | |
| share of rollout time in policy calls (both) | 97% / 97% | 74% / 74% | |

**The rollout is 1.66x faster and the whole loop 1.54x, not "several times".** The barrier did go:
840 opponent rounds per rollout became ~153 ticks. Three things eat most of that 5.5x:

- **Twice as many snapshot groups per tick (2.2 to 4.4).** A lockstep round only holds the few slots
  still in mid-turn; a tick holds the opponent move of every snapshot slot that is on its opponent's
  move (~77 slots spread over 10 snapshots), so more distinct snapshots meet in one tick. Calls fall
  only 2.86x, and a call costs ~6 ms whatever its batch (1.7 or 4.4 envs), so opponent time falls
  about as much as the call count.
- **The learner now makes 109 small calls (mean 38 rows) instead of 32 calls of 128**, at ~11 ms each
  (3.6 s of 20.7 s, see 4). Its share grows from 6% to 17%.
- **The end-of-rollout barrier (S2):** ~45 of the ~153 ticks per rollout have no learner call; they
  finish the opponent turns of slots that have used their 32 rows.

## 2. Smoke run: 2M steps interleaved from the M10 checkpoint

`pixi run scripts/selfplay.sh python/configs/selfplay.json total_steps=2000000 lr_anneal_steps=10000000
--run-dir runs/gate13a/smoke2m` (seed 1, 128 x 32, `p_bot` 0.4, LR on the 10M schedule of 12b). It
**completed** (exit 0; 488 updates, final step 1,998,848, pool of 2, `ckpt_final.pt`, `pool/`,
`tournament.json`/`.md`). **Wall time 41 min 13 s** for the whole command (training, the 1M suite
row, the end-of-run round robin of 100 seeds); training alone (rollout + update) 0.68 h at a mean of
**821 sps** (893 sps with a pool of 1, updates 1-244; ~780 sps with a pool of 2, updates 245-488).
Peak RSS 2.1 GiB. For scale only: the three 12b `pbot40` runs took 1.23-1.26 h (439-452 sps) for
their first 488 updates, but they ran lockstep and three at a time, so this is not a like-for-like
speed comparison (see 3 for that).

The run ends one update before 12b's 2M evaluation (2,002,944), so its 2M suite row is
`ckpt_final.pt` (1,998,848) evaluated afterwards with the in-training code (`SELFPLAY_SUBSET`, 50
seeds = 100 games, batched greedy; `runs/gate13a/smoke_compare.py`). Win rates:

| step (12b step) | condition | smoke | 12b `pbot40` seeds 1 / 2 / 3 | inside the 12b range |
| --- | --- | --- | --- | --- |
| 1,003,520 (1,003,520) | economy_e0 | 1.00 | 1.00 / 1.00 / 0.99 | yes |
| 1,003,520 (1,003,520) | rush_e0 | 0.87 | 0.86 / 0.93 / 0.97 | yes |
| 1,003,520 (1,003,520) | rush_e20 | 0.97 | 0.94 / 0.99 / 0.93 | yes |
| 1,998,848 (2,002,944) | economy_e0 | 0.98 | 1.00 / 0.99 / 1.00 | no (98 of 100 against 99-100) |
| 1,998,848 (2,002,944) | rush_e0 | 0.92 | 0.92 / 0.92 / 0.95 | yes |
| 1,998,848 (2,002,944) | rush_e20 | 0.95 | 0.97 / 0.96 / 0.94 | yes |

Training metrics, mean over the updates in the 100k steps up to each point:

| window | metric | smoke | 12b seeds 1 / 2 / 3 | inside |
| --- | --- | --- | --- | --- |
| to 1,003,520 | approx_kl | 0.0696 | 0.0636 / 0.0668 / 0.0808 | yes |
| to 1,003,520 | entropy | 0.634 | 0.562 / 0.642 / 0.603 | yes |
| to 1,003,520 | clipfrac | 0.165 | 0.168 / 0.158 / 0.164 | yes |
| to 1,003,520 | train_win_rate | 0.767 | 0.675 / 0.810 / 0.779 | yes |
| to 1,998,848 | approx_kl | 0.0569 | 0.0625 / 0.0565 / 0.0534 | yes |
| to 1,998,848 | entropy | 0.647 | 0.568 / 0.664 / 0.672 | yes |
| to 1,998,848 | clipfrac | 0.165 | 0.158 / 0.169 / 0.163 | yes |
| to 1,998,848 | train_win_rate | 0.650 | 0.734 / 0.651 / 0.805 | no (0.001 below) |

**Verdict: learning is not broken.** 5 of 6 suite rows and 7 of 8 training metrics fall inside the
three-seed range; the two outside are at its edge (one economy game, and a training win rate 0.001
below seed 2's, which depends on the opponent mix). Three reference seeds make a narrow range; this
is a sanity check, not a test.

## 3. Concurrency with the interleaved path

`runs/gate13a/concurrency/measure.sh N`: N copies of `python/configs/selfplay.json` (interleaved,
the default) from the M10 checkpoint, seeds 1 to N, started together, 36 updates each, as in 12a.
Total sps is the sum of each trainer's mean `sps` over updates 5 to 30; peak RSS from `/usr/bin/time
-v`; GPU memory is the peak whole-device use (1,555 MiB idle); system memory is sampled every 2 s
from `free` (2.5 GiB used and 9.2 GiB available when idle).

| trainers | total sps | sps per trainer | peak RSS per trainer (GiB) | peak GPU memory MiB | peak system RAM used (GiB) | min RAM available (GiB) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 909 | 909 | 1.98 | 2087 (532 above idle) | 4.1 | 7.6 |
| 2 | 1264 | 606, 658 | 1.99, 1.95 | 2615 (1060 above idle) | 5.6 | 6.1 |
| 3 | 1368 | 453, 480, 435 | 2.02, 1.96, 1.97 | 3144 (1589 above idle) | 7.1 | 4.6 |
| 4 | **1465** | 362, 363, 368, 373 | 1.94, 1.95, 1.96, 2.04 | 3641 (2086 above idle) | 8.6 | 3.1 |

Same-day lockstep reference (`measure.sh N lockstep`): **1 trainer 818 sps, 3 trainers 1,495 sps**
(515, 462, 518; 12a measured 804 and 1,570).

**`scripts/experiment.sh` now defaults to `--parallel 4`**: the best total, with 3.1 GiB still
available at the peak. In the 12a smoke experiment, 3 trainers plus the launcher's evaluation peaked
at 8.5 GiB used, ~1.4 GiB above the 3-trainer peak here; the same margin on top of 4 trainers would
leave ~1.7 GiB; watch memory early in the next experiment, and pass
`--parallel 3` if it gets tight. The gain over 3 is small (+7%), and each run is 20% slower.

Note on this table: it is measured at the warm start, **with a pool of 1**, where every snapshot slot
plays the same snapshot, so a lockstep round already makes a single call. There the interleaved path
gains little alone (+11%) and loses under contention (3 trainers: 1,368 against 1,495, -8%): its 3.4x
more, smaller learner calls cost CPU launch time, which is what concurrent trainers compete for.
Its gain grows with the pool (1.54x total at a pool of 10 in 1), and a 10M run spends most of its time
with a pool of 2 to 10, so the table understates the interleaved path over a real run.

## 4. Re-profile of the interleaved rollout

On the fixture of 1 (interleaved), three instruments, one run each (`runs/gate13a/reprofile.py`):
timing wrappers for the wall-clock split (`split`), `torch.profiler` with CPU and CUDA activities
over rollout 2 (`torch`), and cProfile (`cprofile`; used only for the time inside the Rust methods,
because under Python 3.12 its caller attribution came out incomplete).

Split of 3 rollouts (20.65 s, with `cuda.synchronize` at the rollout edges):

| part | seconds | share |
| --- | --- | --- |
| opponent forward (`trunk` + `act`, mask fetching and observations excluded) | 10.50 | 50.8% |
| learner forward (`trunk` + `act` + `value`, mask fetching excluded) | 3.55 | 17.2% |
| mask fetching, Python + Rust (`masks_fn`, learner 0.07 + opponents 0.14; the Rust mask calls alone 0.06) | 0.22 | 1.1% |
| env stepping (Rust `step_factored` 0.37, `info` 0.32, `observation` 0.14, `reset` 0.00) | 0.83 | 4.0% |
| Python loop and per-element buffer writes (the rest of `collect_rollout_interleaved` and `tick`) | 5.56 | 26.9% |

Per call (`torch.profiler`, rollout 2; CPU times are inflated by the profiler, ~7 ms against ~6 ms
without it):

| call | calls per rollout | kernel launches | host-device syncs (`cudaStreamSynchronize`) | memcpy | GPU kernel time (ms) | CPU time (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| opponent (`trunk` + `act`, stats off) | 655 | 171 | 10 | 12.9 | 1.11 | 7.1 |
| learner (`trunk` + `act` with stats) | 116 | 60 + 149 = 209 | 20 | 22.5 | 1.52 + 0.90 = 2.42 | 1.9 + 8.0 = 9.9 |
| everything outside these calls, per rollout | | 1,165 | 29,140 | 29,139 | 45 total | |

What this says:

- **Opponent forwards are still half the rollout, and they are host-bound.** A call has ~1.1 ms of GPU
  work but costs ~6 ms, made of 171 kernel launches (~14 µs each) and 10 syncs (the three sampling
  stages each go back to the CPU for the next mask). Its cost hardly depends on the batch (5.5 ms at
  1.7 envs, 6.2 ms at 4.4). The number of calls is what matters: 4.4 snapshot groups per tick.
- **The rest of the loop is a quarter of the rollout**, and the profiler shows why: ~29,000
  synchronous host-device copies per rollout outside the forward passes, about 7 per learner
  transition. `collect_rollout_interleaved` writes each transition's reward, terminated, truncated and
  next value into the GPU buffer one element at a time (4 x 4,096 copies), and reads each learner
  action back one element at a time with `int(out.actors[r])` (3 x 4,096).
- Mask fetching (1%) and the Rust env (4%) are not worth touching yet.

**Recommendation: stack the snapshot weights into one forward (`torch.func`), not CUDA graphs.**
- Stacking attacks the largest part directly: one opponent call per tick instead of 4.4. Since a
  call costs about the same at batch 2 or 10, the opponent part should fall from ~10.5 s to ~3-4 s
  per 3 rollouts (~150 ticks x ~7-8 ms for a somewhat larger vmapped call). The rollout would then
  take ~13-14 s instead of 20.7 s (~1.5x), more with a larger pool.
- CUDA graphs fit this `act` badly: the 10 syncs per call (masks fetched between stages, sampled
  actions read back) split it into at least 3 graph segments, and variable batch sizes need padded
  shape buckets. At best they remove the launch cost (~2.4 ms of the ~6 ms) from each of the 4.4
  calls, and they add more complexity than stacking.
- Outside the three options in the brief: the per-element buffer writes (27%) look like the cheapest
  win. Collecting a tick's transitions in NumPy and copying them to the buffer in one go, and reading
  the learner's actions back with one `.cpu()` per call, should remove most of the ~29,000 syncs.
  This changes `ppo.py`, so it needs an owner decision; it could go into the same speed step as
  stacking.

# Milestone 13b: stacked snapshot forward gate

Run on the owner's machine (RTX 4060 8 GB, WSL2, 12 threads) on 2026-10-28, code at commit `27458aa`
(`main`, M13b merged at `074b0ef`). Build hashes: obs `d5f9b1033cfd55ca`, rules `a8b52b91eae34503`.
`pixi run check` was green before the gate. Every trainer ran with `OMP_NUM_THREADS=1`; nothing else
was running; all runs were sequential (`runs/gate13b/run_all.sh`). Scripts and run dirs are in
`runs/gate13b/` (not tracked): `fixture.py`, `reprofile.py`, `torchprof.py`, `smoke_compare.py`.

`fixture.py` is the 13a fixture with `rollout=interleaved` and `opponent_forward=MODE`. An
"opponent call" is one per-snapshot batch-policy call (`per_snapshot`) or one stacked call
(`stacked`, one per tick with snapshot moves); the stack's `sync` is timed apart and is not in the
opponent time. Sampled stacked runs are not bit-identical to per-snapshot runs (draws in (group, slot)
order, MILESTONE_13b), so the two modes play different games from the first opponent move: counts
like ticks and learner calls differ between modes, not between runs of one mode at a pool of 10.

## 1. Profile fixture at a pool of 10

Resume `runs/exp/selfplay12b/pbot40/seed1/latest.pt` (pool of 10, `p_bot` 0.4, 128 x 32) for 3
rollouts, eval/rating/snapshot off. Order: per_snapshot 1, stacked 1, per_snapshot 2, stacked 2.
Per rollout = per 4,096 learner transitions.

| measure | per_snapshot run 1 / 2 | stacked run 1 / 2 | ratio (means) |
| --- | --- | --- | --- |
| rollout sps | 588 / 636 | 664 / 711 | **1.12x** |
| update sps | 2,818 / 2,811 | 2,897 / 3,114 | 1.07x |
| total sps (rollout + update) | 487 / 519 | 540 / 579 | **1.11x** |
| rollout time, 3 rollouts (s) | 20.9 / 19.3 | 18.5 / 17.3 | 0.89x |
| opponent calls per rollout | 650 | 165 | 3.9x fewer |
| opponent batch per call, mean (median) | 4.45 (2) | 17.9 (6) | |
| snapshot groups per tick with snapshot moves (mean, max) | 4.58, 10 | 4.26, 10 | |
| ticks per rollout | 147 | 172 | |
| time per opponent call (ms) | 6.1 / 5.7 | 16.9 / 16.1 | **2.8x** |
| share of rollout time in opponent calls | 57% / 58% | 45% / 46% | |
| share of rollout time in learner forward | 16% / 17% | 20% / 19% | |
| learner calls per rollout (mean batch) | 109 (37.6) | 119 (34.3) | |
| stack `sync`, 3 rollouts (s) | | 0.11 / 0.08 | |

**The rollout is 1.12x faster and the whole loop 1.11x, not the ~1.5x 13a estimated.** Calls fell
3.9x as planned (one per tick), but a stacked call costs ~16.5 ms, not the ~7-8 ms 13a assumed, so
opponent time per rollout only fell from ~3.8 s to ~2.8 s. Why (`torchprof.py`, `torch.profiler`
over rollout 2; CPU times inflated by the profiler):

| per call | kernel launches | host-device syncs | memcpy | GPU kernel time (ms) | CPU time (ms) |
| --- | --- | --- | --- | --- | --- |
| per-snapshot opponent call (13a) | 171 | 10 | 12.9 | 1.11 | 7.1 |
| stacked call, pool of 10 | 372 | 13 | 15.9 | 6.84 | 21.0 |
| stacked call, pool of 1 | 269 | 13 | 13.1 | 2.50 | 20.2 |

A stacked call makes twice the launches (`index_select` of every stacked parameter per call,
vmap's batching of each op, the pad and unpad gathers), runs ~6x the GPU work at a pool of 10 (every
group is padded to the largest group, mean 17.9 real envs in up to 10 x B padded rows), and its
CPU side (`functional_call` + `vmap` dispatch in Python) is ~3x a plain call.

## 2. Pool of 1 (start of a run)

A fresh self-play run from `runs/curriculum_seed1/latest.pt` with `python/configs/selfplay.json`,
10 updates (`total_steps=40960`, `tournament_seeds=1`, eval/rating/snapshot off), same order.

| measure | per_snapshot run 1 / 2 | stacked run 1 / 2 | ratio (means) |
| --- | --- | --- | --- |
| rollout sps | 1,166 / 1,309 | 932 / 1,002 | **0.78x** |
| total sps | 847 / 938 | 716 / 756 | **0.82x** |
| total sps, updates 2-10 | 848 / 938 | 707 / 742 | 0.81x |
| opponent calls per rollout (mean batch) | 117 / 130 (28.7 / 25.0) | 115 / 124 (28.2 / 25.7) | |
| time per opponent call (ms) | 6.7 / 5.8 | 13.3 / 12.2 | 2.0x |
| share of rollout time in opponent calls | 22% / 24% | 35% / 38% | |

**Stacked is slower at the start of a run: -18% total, -22% rollout.** With one snapshot there is
one group per tick in both modes, so the call count is the same, and the stacked call costs twice
as much (the overhead above with nothing to amortize). The two runs of one mode differ here (update
1's `approx_kl` already differs in the 4th digit: the GPU update is not bit-deterministic, and from a
fresh start the weights then move, so the games diverge), but both pairs show the same direction and
the per-call cost explains it. The 1M smoke (3) confirms it over 244 updates.

## 3. Smoke run: 1M steps with the default config

`pixi run scripts/selfplay.sh python/configs/selfplay.json total_steps=1000000
lr_anneal_steps=10000000 --run-dir runs/gate13b/smoke1m` (stacked, the default). It **completed**
(exit 0; 244 updates, final step 999,424, pool of 1, `ckpt_final.pt`, `pool/`, `tournament.json`/`.md`).
**Wall time 22 min 57 s**; training (rollout + update) 22.7 min at a mean of **735 sps** (rollout 962
sps); peak RSS 2.08 GiB. Over the same 244 updates (all at a pool of 1): 13a smoke (per-snapshot,
interleaved) 889 sps (rollout 1,234), 12b `pbot40` seeds (lockstep, 3 at a time) 529-549 sps. So
stacked is **17% slower than the 13a path over the first 1M steps**, as the pool-of-1 fixture says.

The run ends one update before the 1M evaluation (1,003,520), so it has no in-training suite row;
its 1M row is `ckpt_final.pt` (999,424) evaluated with the in-training code (`SELFPLAY_SUBSET`, 50
seeds = 100 games, batched greedy), as 13a did at 2M. Win rates:

| condition | smoke (999,424) | 13a smoke (1,003,520) | 12b `pbot40` seeds 1 / 2 / 3 (1,003,520) | inside the 12b range |
| --- | --- | --- | --- | --- |
| economy_e0 | 1.00 | 1.00 | 1.00 / 1.00 / 0.99 | yes |
| rush_e0 | 0.94 | 0.87 | 0.86 / 0.93 / 0.97 | yes |
| rush_e20 | 0.97 | 0.97 | 0.94 / 0.99 / 0.93 | yes |

Training metrics, mean over the updates in the 100k steps to 999,424:

| metric | smoke | 13a smoke | 12b seeds 1 / 2 / 3 | inside |
| --- | --- | --- | --- | --- |
| approx_kl | 0.0629 | 0.0700 | 0.0637 / 0.0682 / 0.0813 | no (0.0008 below) |
| entropy | 0.644 | 0.632 | 0.564 / 0.642 / 0.598 | no (0.002 above) |
| clipfrac | 0.160 | 0.164 | 0.168 / 0.158 / 0.163 | yes |
| train_win_rate | 0.824 | 0.769 | 0.673 / 0.808 / 0.778 | no (0.016 above) |

**Verdict: learning is not broken.** All three suite rows fall inside the three-seed range; the three
training metrics outside are at its edge and on the good side (a higher training win rate, depending
on the opponent mix). Three seeds make a narrow range; a sanity check, not a test.

## 4. Re-profile of the stacked interleaved rollout (pool of 10)

`runs/gate13b/reprofile.py`: the 13a timing wrappers (`cuda.synchronize` at the rollout edges), one
run, 3 rollouts (19.03 s, 495 stacked calls):

| part | seconds | share | 13a (per-snapshot) |
| --- | --- | --- | --- |
| opponent forward (stacked call; observations and mask Rust calls excluded) | 8.68 | 45.6% | 10.50 (50.8%) |
| stack `sync` | 0.09 | 0.5% | |
| learner forward (`trunk` + `act` + `value`, mask fetching excluded) | 4.20 | 22.1% | 3.55 (17.2%) |
| mask fetching (learner `masks_fn` 0.08 + opponent Rust mask calls 0.02) | 0.10 | 0.5% | 0.22 (1.1%) |
| env stepping (Rust `step_factored` 0.37, `info` 0.36, `observation` 0.15, `reset` 0.00) | 0.88 | 4.6% | 0.83 (4.0%) |
| Python loop and per-element buffer writes (the rest) | 5.07 | 26.6% | 5.56 (26.9%) |

The profiler (1) still shows ~29,200 synchronous host-device copies per rollout outside the forward
passes, as in 13a: the per-element buffer writes and action reads of `collect_rollout_interleaved`.

**Recommendation.** Stacking did what it was built to do (one opponent call per tick) but its per-call
overhead eats most of the gain at a pool of 10 (+11% total) and makes the start of a run slower (-18%
at a pool of 1), so over a 10M run, where the pool grows by one per 1M steps, the net is small and
may be negative in the first few million steps. Two things are worth doing before M14, both cheap:
(i) an owner decision on the default: either `per_snapshot` until the pool reaches some size, or a
stacked call that falls back to the plain per-snapshot call when a tick has a single group (one-line
dispatch in `tick`, recovers the pool-of-1 loss); (ii) deferred option (a), batching the
per-element buffer writes and action reads in `collect_rollout_interleaved`: 27% of the rollout at
every pool size, ~29,000 syncs per rollout, independent of the opponent path. Halving that part
would give ~1.15x on the rollout, more than stacking gave. Beyond that, the stacked call itself
could be trimmed (gather the parameters once per pool change instead of per call, pad less), but its
Python dispatch cost is inherent to `vmap`. Note also that M14 runs 4 trainers at once, where CPU
launch time is the contended resource; the stacked call's 2x launches may cost more there than
here (not measured), so the concurrency table of 13a (3) should be re-measured with the chosen default.

# Milestone 14b: network capacity ladder (`scale14_bots`, `scale14_sp`)

Driver `runs/gate14a/run14.sh` (14a smoke, passed 2026-10-29 00:09), then 14b: phase 1 `scale14_bots`
(M10 bot curriculum from scratch, 10M steps), phase 2 `scale14_sp` (self-play, each run warm-started
from its own phase-1 `ckpt_final.pt`, 10M steps). 4 arms x 3 seeds, 4 trainers at once, reference
opponent = the M10 `latest.pt` (20M curriculum steps). Code `98eba2d`, build obs `d5f9b1033cfd55ca`,
rules `a8b52b91eae34503`. Phase 1 took ~20 h plus a 1.6 h rerun of `w32b4` seed 3 (crashed 6 s in with
`CUDA_ERROR_UNKNOWN` while four trainers initialized the GPU; healthy afterwards); phase 2 took ~36 h
(2026-10-29 23:58 to 10-31 12:06). Reports: `runs/exp/scale14_bots/report.md`,
`runs/exp/scale14_sp/report.md`.

| arm | width x blocks | parameters | phase 1 sps | phase 2 sps |
| --- | --- | --- | --- | --- |
| w32b4 | 32 x 4 | 114,325 | 890 (seeds 1-2) | 335 |
| w64b4 | 64 x 4 (today's) | 422,869 | 708 | 277 |
| w128b4 | 128 x 4 | 1,623,637 | 384 | 217 |
| w64b8 | 64 x 8 | 718,293 | 667 | 231 |

(Phase 1 sps of `w32b4` seed 3 was 1,937 because it ran alone after the restart.)

## Results (mean over the 8/9/10M checkpoints, 3 seeds)

| arm | phase 1 elo_vs_ref | phase 1 rush_e0 | phase 2 elo_vs_ref (seeds) | phase 2 rush_e0 (seeds) |
| --- | --- | --- | --- | --- |
| w32b4 | -10 +- 60 | 0.959 | 152 +- 92 (195, 46, 215) | 0.957 (0.960, 0.941, 0.970) |
| w64b4 | -18 +- 12 | 0.968 | 147 +- 89 (184, 45, 212) | 0.944 (0.968, 0.925, 0.940) |
| w128b4 | -3 +- 59 | 0.946 | 75 +- 57 (71, 133, 20) | 0.856 (0.848, 0.868, 0.851) |
| w64b8 | -137 +- 33 | 0.846 | -118 +- 82 (-75, -212, -66) | 0.703 (0.754, 0.637, 0.718) |

Phase 2 Welch (12 flagged): `w64b8` is below every other arm on both metrics (p 0.012-0.044);
`w128b4` is below `w32b4` and `w64b4` on rush_e0 (p 0.001 and 0.009; the 0.001 survives a 12-test
Bonferroni correction, 0.0042); `w32b4` vs `w64b4` not detected on either (p 0.47, 0.95). Phase 1:
`w64b8` below the others on `elo_vs_ref` (p 0.015-0.047), nothing else detected.

Round robin of the phase-2 final weights (exploratory): `w64b4`, `w32b4` and `w128b4` interleave
(1,217-1,334); all three `w64b8` runs are below the reference (906-1,101); `ref` 1,112.

## What we learned

- **No capacity gap at these settings: bigger was not better, it was worse.** `w32b4` and `w64b4` are
  indistinguishable (Elo 152 vs 147, rush 0.957 vs 0.944, interleaved in the round robin). The 4x wider
  `w128b4` is about as strong against networks (round robin) but clearly weaker against rush (0.856).
  The 2x deeper `w64b8` is worse on everything, already after phase 1.
- **The likely cause is optimization, not capacity (hypothesis).** Hyperparameters were held fixed (P3):
  phase 1 at lr 3e-4, phase 2 at 1e-4, tuned for 64 x 4. Larger networks usually want a smaller step
  size, and the residual blocks have no normalization, so an 8-block stack starts with a larger
  activation scale than a 4-block one. A null or negative result for a larger size here means "not
  better at these settings and this budget (10M + 10M steps)", not "not better".
- **Today's 64 x 4 is a sound choice, and 32 x 4 is a cheaper equal** at ~20% higher self-play speed
  (335 vs 277 sps) and ~25% in phase 1. Per the roadmap's ladder ("a larger model only if a capacity
  experiment shows a gap"), there is no reason to grow the network for v1.
- **From scratch, 20M steps (10M curriculum + 10M self-play) beat the M10 checkpoint** (20M curriculum
  only): `w64b4` scores +147 Elo against it (~0.70 expected score).
- **Seed 2 is weak for every size in phase 2** (`elo_vs_ref` 46, 45, 133, -212; the lowest of the three
  seeds for three of four arms): the paired design shares maps and phase-1 seeds, so part of the spread
  is the seed, not the arm.
