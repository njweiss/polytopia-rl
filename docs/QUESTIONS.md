# Open questions from agents

Only open questions live here, plus the answered ones of the milestone in progress (the implementer
needs their full text). At the end of each milestone, answered entries move to `docs/DECISIONS.md` as one
line (Q1-Q30 are there; their full text is in `git show 540a214:docs/QUESTIONS.md`). Number new
entries from **Q31**.

Append entries below. Format:

## Q<n>: <short title>

- Needed for: <milestone / spec section>
- Ambiguity: <what the spec does not say>
- Options: <A / B>
- Agent's pick and why: <...>
- Owner answer: (blank until answered)

## Q31: Experiment file formats and the report schema the M12 tests assume

- Needed for: milestone 12, "Experiment runs" and "Report" (tests `test_experiment_run.py`, `test_experiment_report.py`, `test_train_seeds.py`)
- Ambiguity: the brief names the files and the `ExperimentReport` return type but not their contents.
- Options: A (as the tests encode, documented in the `polytopia_rl/experiment.py` stub):
  1. `h2h_{step}.json` is `dataclasses.asdict(play_pair(run, init, h2h_seeds, env_kwargs))`, the run's checkpoint as `a`, the init checkpoint as `b` named `"init"`; `s = (a_wins + draws / 2) / games`.
  2. Round robin players: the bots `random`, `economy`, `rush`, then `init` (if any), then each run's `ckpt_final.pt` named `{arm}_seed{seed}`.
  3. `ExperimentReport` fields `name, arms, seeds, runs, tables, comparisons, n_flagged, elo, obs_hash`, with `runs` (per run: `steps`, per-checkpoint `suite` win rates, `suite_mean`, `elo_vs_init` list, `elo_vs_init_mean`, hashes), `tables` (metric -> arm -> `per_seed`, `mean`, `sd`, `min`, `max`; suite conditions then `elo_vs_init`), `comparisons` (one dict per pair and metric, Welch fields, `primary`, `significant` a bool for primary metrics and `None` for exploratory ones). `report.json` is `dataclasses.asdict(report)`.
  4. A suite condition's value is its `win_rate`. "The number of flagged comparisons" (`n_flagged`) counts the comparisons that carry a flag (pairs x primary metrics), not the significant ones.
  5. `report` reads every `suite_{step}/suite.json` present, takes the run hashes from them, and opens no checkpoint; mixed `obs_hash` raises `ValueError`.
  6. The rating row's `gap_snapshot_0` is a top-level key of the row, next to `rating`.
  B: another layout (the tests change in a new phase A commit).
- Agent's pick and why: A. It reuses the existing JSON forms (`PairResult`, `write_tournament`, `suite_to_json`) and keeps the report machine-readable; `None` makes "no flag" explicit rather than a `False` that reads like a negative result.
- Owner answer: A on 2026-10-25

## Q32: `total_steps` of the smoke spec

- Needed for: milestone 12, gate 12a (3) and `python/experiments/smoke.json`
- Ambiguity: the gate runs `scripts/experiment.sh run python/experiments/smoke.json` "with `total_steps=200000`", but the CLI (`run SPEC [--parallel K] [--only ARM]`) has no override, and the brief's example `smoke.json` shows `10000000`.
- Options: A: `smoke.json` itself has `"total_steps": 200000` / B: the CLI gains a `total_steps=N` override / C: the example's 10M, edited by hand for the gate.
- Agent's pick and why: A (no new CLI surface; a smoke spec is for smoke runs). The tests pin only the smoke spec's name, base config, init checkpoint, seeds [1, 2], two arms and valid primary metrics, so any option passes them.
- Owner answer: A on 2026-10-25

## Q33: Launcher and trainer details the M12 tests assume

- Needed for: milestone 12, "Seed-varied training maps" and "Experiment runs"
- Ambiguity: mechanics the brief leaves open.
- Options: A (as the tests encode):
  1. The overlap check raises `ValueError` (like the existing seed-range checks) and reads `train.TRAIN_SEEDS_PER_RUN` at call time: the test shrinks it to 6 to trigger the check in a tiny run.
  2. The launcher starts trainers with `subprocess.Popen` (the test replaces it, and `time.sleep`, with fakes) and evaluates in its own process; `argv[0]` is `sys.executable` (or `python`/`python3`); the first line of `command.txt` is the command, `shlex.split`-able back to the argv; overrides are `key=json.dumps(value)` (`p_bot=0.4`); `OMP_NUM_THREADS` is set in the env passed to `Popen` (or `os.environ`); a resume command adds `--resume DIR/latest.pt` (with or without `--init-checkpoint`, both accepted).
  3. "`run` exits non-zero": `main(["run", SPEC])` raises (`SystemExit` with a non-zero code, or an exception) after every other run has been trained and evaluated; the printed message contains the exit code and the run directory.
  4. `make_batch_fn` is called once per condition, in condition order.
  5. `t_sf` of a negative `t` is `1 - t_sf(-t)` (the definition `P(T > t)`).
  6. Not tested: `--only ARM`, the CLI's default `--parallel`, whether a run that was already done is re-evaluated, the round robin after a failed run, `df` in the zero-variance case, the git commit in `command.txt`, the curves beyond the PNG existing.
  B: other choices (the tests change in a new phase A commit).
- Agent's pick and why: A: each is the plain reading of the brief or the existing code's convention. No earlier test needed a change: none pins absolute training seeds for `seed != 1`.
- Owner answer: A on 2026-10-25
