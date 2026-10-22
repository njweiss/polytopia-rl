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
