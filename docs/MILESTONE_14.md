# Milestone 14: network capacity (scaling ladder)

Depends on milestones 1 to 13b. The roadmap's ladder is bots, then the self-play pool, then a larger
model "only if a capacity experiment shows a gap". This milestone is that experiment: does a wider or
deeper network play better under the same training pipeline and budget, and what does it cost in speed?

## Goal

1. Train several network sizes from scratch through the same two-phase pipeline (bot curriculum, then
   self-play), 3 seeds each, and compare them on a fixed reference and against each other.
2. Report the cost of each size (steps per second, parameters) next to its strength.
3. Decide the network size for game v2 (or keep 64 x 4).

## Decisions already made (owner)

- Self-play settings from M12c/M13: `p_bot` 0.4, LR annealed to zero within each run, interleaved rollout,
  `opponent_forward: "per_snapshot"` (`python/configs/selfplay.json`), 10M-step self-play runs.
- Statistics as in M12: per-seed tables, mean over the last 3 checkpoints, Welch per pair with only the
  primary metrics flagged.
- Experiments run 4 trainers at once (`--parallel 4`, gate 13a).

## Why from scratch

A network of another size cannot load the M10 weights, so warm-starting only the current size would
compare a pre-trained network against untrained ones. Every arm, including today's 64 x 4, therefore
starts from random weights and goes through the same pipeline. The M10 checkpoint becomes a fixed
reference opponent instead of a starting point.

## In scope

1. Two optional spec fields in `polytopia_rl.experiment` (Public API).
2. The report gains cost columns per arm: parameter count and mean training sps per phase.
3. Specs `python/experiments/scale14_bots.json` (phase 1) and `python/experiments/scale14_sp.json`
   (phase 2).
4. RL_REPORT section.

## Out of scope

New architectures (attention, other heads), per-arm hyperparameter tuning, distillation, changes to the
model code, the game, the bots, PPO, the rollout or the observation.

## Public API (fixed so tests can import it)

```python
# polytopia_rl.experiment.Spec (new optional fields; existing specs behave exactly as before)
reference: str | None = None    # a checkpoint every evaluated checkpoint plays head to head
                                #   (h2h_{step}.json, h2h_seeds); metric name "elo_vs_ref"; the round
                                #   robin includes it as player "ref"
init_from: str | None = None    # the name of an earlier experiment: run (arm, seed) warm-starts from
                                #   runs/exp/{init_from}/{arm}/seed{seed}/ckpt_final.pt
```

## Rules

- `init_from` and `init_checkpoint` are mutually exclusive (`ValueError` on load). With `init_from`, a
  run whose source `ckpt_final.pt` does not exist fails with a clear message (the launcher's normal
  failure path: the other runs continue, `run` exits non-zero); `command.txt` records the actual path.
- With `reference` set, the head-to-head is played against it instead of the init checkpoint, and the
  metric is `elo_vs_ref` (computed exactly like `elo_vs_init`, including the clipping). Without
  `reference`, behavior is unchanged: the head-to-head is against `init_checkpoint` when there is one.
  `primary` may name `elo_vs_ref`.
- The reference may be a different architecture from the runs (it is loaded with its own config, as
  `load_net` already does).
- Cost columns: per arm, the parameter count of its network (from its final checkpoint) and the mean
  `sps` of its runs over all updates (from `metrics.jsonl`), mean and sd over seeds. Reported, not
  tested across arms.

## Experiment

Arms (the same names in both phases):

| arm | width | blocks | parameters vs base (trunk convs) | question |
| --- | --- | --- | --- | --- |
| `w32b4` | 32 | 4 | ~0.25x | is the current size above what is needed? |
| `w64b4` | 64 | 4 | 1x (today's network) | the control |
| `w128b4` | 128 | 4 | ~4x | does width help? |
| `w64b8` | 64 | 8 | ~2x | does depth help? |

**Phase 1, `scale14_bots`:** base config `python/configs/curriculum.json` (the M10 curriculum: random,
then economy, then economy and rush), no init, seeds 1 to 3, `total_steps` 10M, `reference`
`runs/curriculum_seed1/latest.pt`, primary `elo_vs_ref` and `rush_e0`. Arms override `width` and
`blocks`.

**Phase 2, `scale14_sp`:** base config `python/configs/selfplay.json`, `init_from` `scale14_bots`, seeds
1 to 3, `total_steps` 10M, the same `reference` and primary metrics.

The phase 2 report is the result. The phase 1 report shows where each size starts self-play (expected to
be near the bots' ceiling for every size, as M10 was).

Cost estimate: 12 runs per phase, 4 at a time. Phase 1 (bot training, ~1,300 sps alone in M10) about
3-5 h per batch of four; phase 2 (self-play, ~400-600 sps per trainer) about 6-8 h per batch. Three
batches per phase: roughly 10-15 h and 20-25 h, so 1.5-2 days in total. Wider and deeper networks run
somewhat slower; the cost columns will show how much.

## Clarifications (owner: confirm or overrule)

- **P1:** four arms including a smaller one (`w32b4`). Dropping it saves a quarter of the compute but
  loses the check of whether today's size is already more than enough.
- **P2:** phase 1 is 10M steps, not M10's 20M: M12c showed that a run annealed to zero within its
  length does well, and M10's best checkpoint was at 13M. Every arm gets the same 10M.
- **P3:** hyperparameters are held fixed across sizes (M10's for phase 1, the self-play config for phase
  2). Larger networks often want a lower learning rate, so a null result for a larger size means "not
  better at these settings", not "not better".
- **P4:** the M10 checkpoint is the reference opponent (`elo_vs_ref`). Because it was trained for 20M
  curriculum steps and every arm here gets 10M, the from-scratch `w64b4` arm may score below 0 against
  it after phase 1; the comparison across arms is what matters.
- **P5:** the cross-arm round robin of the final weights is reported (exploratory), not tested; it is
  the most direct ranking of the sizes, but its ratings are fitted jointly, so they are not independent
  across seeds.

## Changes to earlier tests

None: both new fields are optional, and specs without them must behave exactly as before (12b and 12c
reports must be reproducible).

## Tests (phase A guidance)

1. **Spec:** the new fields load, default to None, and `init_from` together with `init_checkpoint`
   raises; the shipped `scale14_bots.json` and `scale14_sp.json` have the arms, seeds, steps, reference
   and primary metrics above.
2. **Launcher with `init_from`:** the command of run (arm, seed) has `--init-checkpoint
   runs/exp/{init_from}/{arm}/seed{seed}/ckpt_final.pt`; a missing source fails that run only, with a
   message naming the path; resume of a partly trained run still uses `--resume`.
3. **Reference:** with `reference`, head-to-heads are played against it (mock `play_pair`, check the
   opponent's checkpoint path), the files are `h2h_{step}.json`, the metric is `elo_vs_ref` with the same
   clipping as `elo_vs_init`, the round robin includes "ref" and not "init". Without `reference`,
   everything is as before (an existing 12b-style spec gives the same commands, files and report keys).
4. **Mixed architectures:** a round robin and a head-to-head between tiny checkpoints of different
   `width`/`blocks` run (each loaded with its own config).
5. **Cost columns:** parameter counts match `sum(p.numel())` of each arm's network; mean sps matches the
   metrics files (hand-written fixtures).

Mutants to aim for: `init_from` path built with the wrong arm or seed; reference ignored (h2h against
init); metric still named `elo_vs_init` with a reference; "init" player kept in the round robin; old specs
changed by the new fields; parameter count of the wrong network.

## Gate

**14a:** `./scripts/check.sh` passes; a smoke of both phases (2 arms x 1 seed, `w32b4` and `w64b4`,
200k steps each, small seed counts) runs end to end, including `init_from` and the reference.
**14b (owner):** `scripts/experiment.sh run python/experiments/scale14_bots.json`, then `report`; then
the same for `scale14_sp.json`. Results in RL_REPORT with a recommendation for the network size.

## Notes for the implementer

- `width` and `blocks` are existing `PPOConfig` keys; arms override them like any other key.
- Keep the 12b and 12c behavior byte-for-byte when the new fields are absent.
- If a test contradicts this brief or the spec, stop and add a QUESTIONS.md entry (next number Q39).
