# Open questions from agents

Only open questions live here, plus the answered ones of the milestone in progress (the implementer
needs their full text). At the end of each milestone, answered entries move to `docs/DECISIONS.md` as one
line (Q1-Q38 are there with pointers to their full text). Number new entries from **Q39**.

Append entries below. Format:

## Q<n>: <short title>

- Needed for: <milestone / spec section>
- Ambiguity: <what the spec does not say>
- Options: <A / B>
- Agent's pick and why: <...>
- Owner answer: (blank until answered)

## Q39: Where the cost columns live in the report, and what an unreadable run gives

- Needed for: milestone 14 "In scope" 2, "Rules" (cost columns), test 5.
- Ambiguity: the brief fixes the cost columns' content (parameter count from the final checkpoint,
  mean `sps` over all updates of `metrics.jsonl`, mean and sd over seeds) but not their place in
  `ExperimentReport` / `report.json`, nor what happens when a run's `ckpt_final.pt` cannot be loaded or
  `metrics.jsonl` is missing. The milestone 12 report tests (which may not change) use empty placeholder
  checkpoints and have no `metrics.jsonl`, so `report` must not raise there.
- Options: A. a new field `ExperimentReport.cost`: arm -> `{"params": int, "sps": {"per_seed", "mean",
  "sd", "min", "max"}}` (the `_summary` shape), one new top-level key `cost` in `report.json` for every
  spec, a cost table in `report.md`; an unloadable checkpoint or missing metrics file gives None there
  (not tested). B. cost columns inside `tables` as extra metrics (they would then get Welch comparisons,
  which the brief says are not wanted: "reported, not tested across arms").
- Agent's pick and why: A (the stub declares `cost: dict | None = None`). The tests check
  `rep.cost[arm]["params"]` against `sum(p.numel())` of a `PolicyNet` of the arm's size,
  `rep.cost[arm]["sps"]["per_seed" / "mean" / "sd"]` against the hand-written `metrics.jsonl` (mean over a
  run's rows, then `np.mean` / `np.std(ddof=1)` over seeds), that `report.md` shows each parameter count
  (plain or comma-grouped) and the word "sps", and that `cost` is the only new top-level key of
  `report.json` for a milestone 12 spec.
- Owner answer: A on 2026-10-28

## Q40: Report and file names with a reference

- Needed for: milestone 14 "Rules" (reference), tests 3.
- Ambiguity: the brief names the metric (`elo_vs_ref`), the files (`h2h_{step}.json`) and the round robin
  player ("ref"), but not the per-run keys, the head-to-head's `b` name, the round robin order, or what
  happens to the init checkpoint when both `reference` and `init_checkpoint` are set.
- Options / agent's pick (tested): per-run keys `elo_vs_ref` and `elo_vs_ref_mean` (the `elo_vs_init`
  pattern); `h2h_{step}.json` has `b` = "ref"; `tournament_players` is the bots, then "ref" where "init"
  was, then the runs; with both fields the init checkpoint still starts training (`--init-checkpoint`)
  but is neither loaded as an opponent nor a round robin player, and the only Elo metric is
  `elo_vs_ref`; with a reference `report.md` does not mention `elo_vs_init` (its explanatory line names
  the metric in use). Without a reference the per-run keys, tables, players, files and `report.md`
  metric names are exactly the milestone 12 ones.
- Owner answer: A on 2026-10-28

## Q41: A missing `init_from` source: who reports it

- Needed for: milestone 14 "Rules" (`init_from`), test 2.
- Ambiguity: "fails with a clear message (the launcher's normal failure path)". The trainer launched with
  a missing `--init-checkpoint` would fail on its own, but its error would only be in that run's
  `train.out`, and the launcher's echo of the command (`start ...`) names the path without saying
  anything is wrong.
- Options: A. the launcher checks the source before launching (after writing `command.txt`), prints a
  failure line naming the path, counts the run as failed, and goes on. B. launch anyway and rely on the
  trainer's error in `train.out`.
- Agent's pick and why: A. The test requires a launcher output line that names the missing path and is
  not the echoed train command, `command.txt` with the path, no `ckpt_final.pt` for that run, the other
  runs trained and evaluated, and a non-zero exit. Whether the trainer is launched for the failing run
  is not tested. Not tested either: `init_from` without `reference` (the agent reads the brief as "no
  head-to-head", since only `init_checkpoint` gives one), and a missing source for a run that resumes
  from its own `latest.pt` (the source is not needed then).
- Owner answer: A on 2026-10-28
