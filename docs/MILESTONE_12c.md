# Milestone 12c: learning-rate schedule experiment

A small follow-up to milestone 12. Depends on milestones 1 to 12. In 12b, `base` seed 1 (the 11b config
on 11b's training maps, but 10M steps, so its learning rate decayed twice as fast) scored 0.80 against
the M10 checkpoint, where 11b's 20M final weights scored 0.675 on the same rating seeds. 11b also
improved only after its LR had fallen below half. Hypothesis: a low learning rate late in self-play is
what makes the gains, not the number of steps. This milestone adds the two config knobs needed to test
that and runs the test.

## Goal

1. Two config keys: switch LR annealing off, and anneal over a horizon longer than the run.
2. 12c: one experiment, three schedules by three seeds at 10M steps, with the M12 harness.
3. Decide the schedule for M13, and whether 10M-step runs are enough there.

## Decisions already made (owner)

- Self-play default `p_bot` is 0.4 (`python/configs/selfplay.json`, 2026-10-27). All arms use it.
- Same harness, metrics and statistics as 12b: 3 seeds, mean over the 8/9/10M checkpoints, Welch per
  pair, primary metrics `elo_vs_init` and `rush_e0`.

## In scope

1. `PPOConfig` keys `anneal_lr: bool = True` and `lr_anneal_steps: int | None = None`, used by the
   trainer (see Rules).
2. `python/experiments/selfplay12c.json` (see "Experiment (12c)").
3. An RL_REPORT section with the result.

## Out of scope

Other schedules (cosine, warm-up, step decay), per-group learning rates, any change to the default
behavior, to `selfplay.json`, or to the harness.

## Public API

```python
# polytopia_rl.config.PPOConfig (new keys)
anneal_lr: bool = True          # False: the LR stays at `lr` for the whole run
lr_anneal_steps: int | None = None   # None: anneal over `total_steps` (today's behavior)
```

## Rules

- With `anneal_lr` true, update `u` (1-based) uses `lr * (1 - (u - 1) / H)`, where
  `H = (lr_anneal_steps or total_steps) // (num_envs * num_steps)`. With `lr_anneal_steps` unset this is
  exactly today's formula. With `anneal_lr` false, every update uses `lr`.
- `lr_anneal_steps` below `total_steps` raises `ValueError` when the config is loaded (the LR would reach
  zero and go negative before the run ends).
- Resume needs nothing new: the LR is a function of the update index and the config.
- The `lr` metric in `metrics.jsonl` already records the value used.

## Experiment (12c)

`python/experiments/selfplay12c.json`: base config `python/configs/selfplay.json`, init checkpoint
`runs/curriculum_seed1/latest.pt`, seeds 1 to 3, `total_steps` 10M, primary `elo_vs_init` and
`rush_e0`, `suite_seeds` 200, `h2h_seeds` 100, `tournament_seeds` 100, `eval_checkpoints` 3.

| arm | overrides | LR at 10M | question |
| --- | --- | --- | --- |
| `decay10` | none | 0 | the control (today's schedule) |
| `decay20` | `lr_anneal_steps: 20000000` | 5e-5 | 11b's schedule for its first 10M: does reaching a low LR matter? |
| `const` | `anneal_lr: false` | 1e-4 | no annealing at all |

The three arms are a dose series in how low the LR gets (0, 5e-5, 1e-4). If the hypothesis holds,
`elo_vs_init` and `rush_e0` should order `decay10 > decay20 > const`.

**Reuse of the 12b runs:** `decay10` has exactly the config of 12b's `pbot40` arm (same base config
values, seeds, maps, init and code path; the new keys keep their defaults). So the owner may reuse
those runs instead of retraining:
`mkdir -p runs/exp/selfplay12c/decay10 && cp -al runs/exp/selfplay12b/pbot40/seed1 runs/exp/selfplay12b/pbot40/seed2 runs/exp/selfplay12b/pbot40/seed3 runs/exp/selfplay12c/decay10/`
(hard links, no extra disk). The launcher then sees them as done, with their evaluations, and trains
only the six new runs. The report notes that `decay10` ran on code `9e5fd5d`.

Cost: six runs, 3 at a time, about 20 h at 12b's speed.

## Clarifications (owner: confirm or overrule)

- **N1:** reuse 12b's `pbot40` runs as `decay10` (saves ~10 h). Retraining them is the alternative;
  the result would differ only by GPU nondeterminism.
- **N2:** constant LR is tested at 1e-4 (11b's starting LR), not at a lower value; a lower constant LR
  would be a separate question (LR size rather than schedule).
- **N3:** 3 seeds as in 12b. A dose series with three ordered arms is easier to read than one pair:
  a consistent ordering across seeds is informative even where Welch flags nothing.

## Changes to earlier tests

None. The defaults reproduce today's schedule exactly.

## Tests (phase A guidance)

1. Defaults: `anneal_lr` true, `lr_anneal_steps` None; the LR of updates 1, 2 and the last on a tiny
   config equals today's formula exactly.
2. `lr_anneal_steps = 2 * total_steps`: the LR halves linearly; the last update's LR is just above
   `lr / 2` (hand computation).
3. `anneal_lr` false: every update's LR equals `lr` (read from the optimizer and from `metrics.jsonl`).
4. `lr_anneal_steps < total_steps` raises `ValueError` from `load_config` and `PPOConfig.from_json`.
5. Resume mid-run keeps the schedule (the LR sequence equals an uninterrupted run's).
6. `selfplay12c.json` loads and has exactly the arms above.

Mutants to aim for: horizon ignored; horizon in updates vs steps confused; `anneal_lr` false still
annealing; the check missing; resume restarting the schedule.

## Gate

**12c-a:** `./scripts/check.sh` passes.
**12c-b (owner):** `pixi run scripts/experiment.sh run python/experiments/selfplay12c.json` (after the
optional reuse step), then `report`, pasted into RL_REPORT with a short "what we learned" and a
recommendation for M13's schedule and run length.

## Notes for the implementer

- Small and mechanical: phase A and B may run back to back, tests first.
- Do not change any default, `selfplay.json`, or the 12b spec.
