# Milestone 12: multi-seed experiments

Depends on milestones 1 to 11 (M11 merged into `main` first). Every RL result so far comes from one
training seed. The differences we care about now (a few points against rush, a self-play variant against
another) are about as large as the run-to-run noise, so one run cannot tell them apart. This milestone
builds the tools to run one config over several seeds and compare arms honestly. It also takes the cheap
speed wins that the M11 profiles already found, since experiments multiply the compute cost. The
experiments themselves are chosen after the 11b result (12b).

## Goal

1. Run an experiment (arms x seeds) from one spec file, resumable, with a fixed evaluation at the end.
2. Report each arm with every seed's value, the mean and the spread, and compare arms with a test across
   seeds, not across pooled games.
3. Cut the evaluation time per checkpoint (batched suite) and measure whether running several trainers at
   once raises total throughput.
4. 12b: run the owner's experiments (see "Experiments (12b)").

## Decisions already made (owner)

- Each experiment run is **10M steps** unless its spec says otherwise.
- **Training maps vary with the run seed** (seed 1 keeps today's maps exactly).
- **Statistics:** a per-seed table always, plus Welch's t-test across seeds. Pooling games across seeds
  is not used for arm comparisons (it ignores seed-to-seed variance).
- **Experiments (12b):** one experiment, three arms by three seeds, listed in "Experiments (12b)"
  (owner, 2026-10-24, after 11b). The harness must not assume this particular experiment.
- **Each run is scored on its last 3 checkpoints, not only the final weights.** In 11b the late
  checkpoints swung between 0.80 and 0.94 against rush on the same 50 seeds, so one stop point is noisy.
- **Learning rate:** the LR anneals linearly over `total_steps`, so a 10M run decays twice as fast as
  11b's 20M run. Every arm of an experiment uses the same `total_steps`, so arms compare fairly, but a
  10M base arm is not a replica of 11b.
- Cheap, profiled speed wins that save experiment time are in scope. Larger structural speed work (Q29 R4
  interleaved rollout, Q30 C CUDA graphs) is not.
- No new dependencies: numpy, torch and matplotlib (already in the env) only. No scipy.

## In scope

1. **Speed (12a, first):** a batched `evaluate` path used by the suite and the in-training evaluations; a
   concurrency measurement of 1, 2 and 3 trainers running at once.
2. **Seed-varied training maps** in `train.py`.
3. **In-training rating fix:** each `eval.jsonl` rating row gains `gap_snapshot_0` (learner minus
   snapshot 0). In M11's star graph the learner's own rating is fixed by its record against `random`
   (921.28 at every step of 11b), so only the gap carries information.
4. **`polytopia_rl/experiment.py`:** the spec format, the `run` launcher (subprocesses, resume, the
   evaluation of the last checkpoints, the final round robin) and the `report` aggregation.
5. **`polytopia_rl/stats.py`:** Welch's t-test with the Student t distribution, written with numpy/math.
6. `scripts/experiment.sh`, the smoke spec `python/experiments/smoke.json`, the 12b spec
   `python/experiments/selfplay12b.json`, and an RL_REPORT section.

## Out of scope (do not build)

Hyperparameter search or sweeps beyond a listed set of arms, early stopping of runs, schedulers or job
queues beyond the `--parallel K` launcher, multiple-comparison corrections beyond the primary-condition
rule below, rollout restructuring (Q29 R4), CUDA graphs or `torch.compile` (Q30 C), any game rule,
observation, network or bot change, new dependencies.

## Public API (fixed so tests can import it)

```python
# polytopia_rl.evaluate (extended; the sequential path and its results are unchanged)
def evaluate(policy_fn, opponent, seeds, env_kwargs, greedy, *, seed_base=EVAL_SEED_BASE, epsilon=0.0,
             batch_fn=None) -> EvalResult
#   batch_fn(envs) -> one (actor, kind, arg) per env, greediness bound in. When given, policy_fn is
#   ignored and all 2 * seeds games run together in rounds (see Rules).

# polytopia_rl.suite (extended)
def run_suite(policy_fn, conditions, seeds, env_kwargs, *, step=None, make_batch_fn=None) -> SuiteResult
#   make_batch_fn(greedy: bool) -> batch_fn; when given, each condition plays batched.

# polytopia_rl.train
TRAIN_SEEDS_PER_RUN = 10_000_000
def train_seed_base(seed: int) -> int     # (seed - 1) * TRAIN_SEEDS_PER_RUN; ValueError unless 1 <= seed <= 100

# polytopia_rl.stats
@dataclass class Welch: mean_a, mean_b, diff, t, df, p, significant   # diff = mean_b - mean_a; p two-sided
def welch(a: Sequence[float], b: Sequence[float], alpha: float = 0.05) -> Welch
def t_sf(t: float, df: float) -> float    # P(T > t) for Student t with df degrees of freedom (df may be non-integer)

# polytopia_rl.experiment
@dataclass class Spec: name, base_config, seeds, arms, total_steps, init_checkpoint, primary, suite_seeds, tournament_seeds, eval_checkpoints, h2h_seeds
def eval_checkpoints(run_dir, k) -> list[str]   # the last k distinct steps among ckpt_{step}.pt, oldest first
def load_spec(path) -> Spec
def run_dir(spec, arm, seed) -> str       # runs/exp/{name}/{arm}/seed{seed}
def run(spec, parallel=1) -> None
def report(spec) -> ExperimentReport      # also writes runs/exp/{name}/report.md and report.json
```

CLI: `python -m polytopia_rl.experiment run SPEC [--parallel K] [--only ARM]`,
`python -m polytopia_rl.experiment report SPEC`, wrapped by `scripts/experiment.sh run|report SPEC [...]`.

Spec file (`python/experiments/{name}.json`):

```json
{
  "name": "smoke",
  "base_config": "python/configs/selfplay.json",
  "init_checkpoint": "runs/curriculum_seed1/latest.pt",
  "seeds": [1, 2],
  "total_steps": 10000000,
  "arms": {"base": {}, "pbot40": {"p_bot": 0.4}},
  "primary": ["rush_e0"],
  "suite_seeds": 200,
  "tournament_seeds": 100,
  "eval_checkpoints": 3,
  "h2h_seeds": 100
}
```

`init_checkpoint` is optional (null: train from scratch; then there is no head-to-head and no
`elo_vs_init`). `eval_checkpoints` defaults to 3 and `h2h_seeds` to 100. `arms` maps an arm name to config overrides
(the `key=value` overrides of `train.main`, as JSON values). `seed` and `total_steps` are set by the
launcher and may not appear in an arm. Unknown keys raise `ValueError`.

## Rules

### Batched evaluation

- With `batch_fn`, `evaluate` builds every game first (seed `seed_base + j`, seat 0 then seat 1, the
  same envs as the sequential path) and plays them in rounds. Each round makes one `batch_fn` call over
  the live games, in game order, and steps each of them once. A game leaves when it terminates or
  truncates. `per_game` keeps the sequential order (seed order, seat 0 first).
- Greedy batched results equal the sequential results game for game (the same per-game dicts). Sampled
  batched results use the generator in a different order, so they are equal in distribution, not
  identical.
- The suite CLI, `scripts/suite.sh` and the in-training suite subset use the batched path with
  `net_batch_policy`. The tournament (`play_pair`) is unchanged; it already plays in rounds.

### Seed-varied training maps

- The trainer starts its vec env at `train_seed_base(cfg.seed)` instead of 0. Seed 1 gives 0, so every
  earlier run is reproduced exactly. A resume keeps the stored `seed_base` as today.
- A run whose next training seed would reach `train_seed_base(cfg.seed + 1)` raises (no overlap between
  runs). Ten million seeds cover far more episodes than a 20M-step run plays.

### Experiment runs

- `run` visits the (arm, seed) pairs in spec order (arms outer, seeds inner) and runs each as
  `python -m polytopia_rl.train --config BASE --run-dir DIR [--init-checkpoint CKPT] seed=S total_steps=N k=v ...`
  in a subprocess, at most `parallel` at a time, each with `OMP_NUM_THREADS=1` unless set. It writes
  `DIR/command.txt` with the exact command first.
- A run with `ckpt_final.pt` is done and skipped. A run with `latest.pt` but no `ckpt_final.pt` is
  resumed with `--resume DIR/latest.pt`. A run with neither starts fresh. So `run` can be repeated after a
  crash and only redoes the unfinished work.
- After a run finishes, each of its `eval_checkpoints(DIR, k)` checkpoints gets the suite (`suite_seeds`,
  all `DEFAULT_SUITE` conditions, batched) into `DIR/suite_{step}/`, and, when there is an init
  checkpoint, a head-to-head against it (`play_pair`, `h2h_seeds` seeds, so `2 * h2h_seeds` games) into
  `DIR/h2h_{step}.json`. A run's value for a metric is the mean over its evaluated checkpoints.
- When every run is done, one round robin is played among all final weights, the bots, and the init
  checkpoint (if any) as `init`, with `tournament_seeds` seeds per pair, to
  `runs/exp/{name}/tournament.json` and `.md`. It is reported but not tested across arms (exploratory).
- A failed subprocess is reported with its exit code and log path; the other runs continue. `run` exits
  non-zero if any run failed.

### Report

`report` reads every run's `suite_{step}/suite.json` and `h2h_{step}.json`, the experiment tournament and
each run's `eval.jsonl`, and writes `report.md` and `report.json`. A run's value is the mean over its
evaluated checkpoints; the per-checkpoint values are shown too.

1. A table per suite condition: one row per arm with each seed's value, the mean, the sample standard
   deviation (`ddof=1`) and min to max, and a second table with every checkpoint's value.
2. `elo_vs_init`: per run, `400 * log10(s / (1 - s))` of the head-to-head score `s` against the init
   checkpoint (wins plus half the draws, over games), with `s` clipped to `[0.5 / n, 1 - 0.5 / n]` for
   `n` games, averaged over the evaluated checkpoints; per arm the mean and the sd over seeds.
3. Elo per final policy from the experiment round robin (exploratory).
4. For each pair of arms (spec order, the first arm as `a`), a Welch comparison per suite condition and
   for `elo_vs_init`. Only the conditions named in `primary` carry the `significant` flag (`primary` may
   name `elo_vs_init`); the rest are reported as exploratory (t and p shown, no flag). The report states
   the number of flagged comparisons.
5. Learning curves: for each in-training suite-subset condition, the win rate against the step for each
   arm, as the mean over seeds with a min to max band (`report_curves.png`, matplotlib).
6. Each run's obs and rules hashes. `report` refuses to mix runs with different `obs_hash` values.

### Welch's t-test

`t = (mean_b - mean_a) / sqrt(s_a^2 / n_a + s_b^2 / n_b)`, with Welch-Satterthwaite
`df = (s_a^2/n_a + s_b^2/n_b)^2 / ((s_a^2/n_a)^2/(n_a-1) + (s_b^2/n_b)^2/(n_b-1))`, `p = 2 * t_sf(|t|, df)`,
`significant = p < alpha`. Both variances 0: `t = 0, p = 1` if the means are equal, else `t = +-inf,
p = 0`. Fewer than 2 values in an arm raises `ValueError`. `t_sf` uses the regularized incomplete beta
function, `t_sf(t, df) = 0.5 * I_{df/(df+t^2)}(df/2, 1/2)` for `t >= 0`, evaluated with a continued
fraction (Lentz's method, `math.lgamma` for the prefactor) to an absolute error below `1e-10`.

### Concurrency measurement (12a)

Measure total training throughput (sum of `sps` over the trainers, mean over updates 5 to 30) for 1, 2
and 3 copies of the 11b config started together from the M10 checkpoint, with seeds 1 to 3, on the
owner's machine after the WSL memory increase. Also record peak resident memory per trainer and GPU
memory. The default `--parallel` in `scripts/experiment.sh` is the count with the best total, unless
memory forbids it.

## Clarifications (owner: confirm or overrule)

- **M1:** each run seed owns 10M training map seeds; run seeds 1 to 100.
- **M2:** arms are compared on the mean over each run's last 3 checkpoints (the suite and the
  head-to-head against the init checkpoint). The round robin and the learning curves are shown but not
  tested.
- **M3:** only the spec's `primary` metrics get a significance flag, named before the run, so
  eight conditions times several arm pairs do not produce false positives by chance. The other numbers
  are exploratory.
- **M4:** self-play arms warm-start from the same M10 checkpoint, so the spread measures self-play
  variance only, not curriculum variance.
- **M5:** with 3 seeds per arm, Welch's test detects only large effects (df about 4, so `|t|` must be
  about 2.8 or more). The report says so in one line, and a null result is reported as "not detected",
  not "no effect".
- **M6:** batched sampled evaluation changes which random draws each game gets, so `rush_e0_sampled`
  numbers from before and after this milestone are equal in distribution, not identical. The greedy
  conditions are unchanged.

## Changes to earlier tests

- Trainer tests that use `seed` other than 1 train on different maps after the seed offset. Tests that
  compare two runs or a run with its resume are unaffected. Any test that pins absolute training seeds for
  `seed != 1` changes in phase A.
- `evaluate`, `run_suite` and the suite CLI keep their outputs for greedy conditions; no expected value
  changes.

## Tests (phase A guidance)

1. **Batched evaluate:** greedy batched equals sequential game for game on a few seeds against each bot
   (a scripted policy that has both a `policy_fn` and a `batch_fn` form); `per_game` order; one `batch_fn`
   call per round over exactly the live games (mock); games that end early leave the round set; sampled
   batched runs are repeatable for a fixed generator.
2. **Suite:** `make_batch_fn` is called with the condition's greedy flag; batched and sequential suites
   match for greedy conditions.
3. **Seed base:** `train_seed_base(1) == 0`, `(2) == 1e7`, out-of-range seeds raise; the trainer's first
   episode seeds start at the base; a run that would cross into the next base raises; resume keeps the
   base.
4. **Welch:** `t_sf` against known values (`t_sf(0, df) = 0.5`; df 1 is Cauchy: `t_sf(1, 1) = 0.25`;
   `2 * t_sf(2.776445, 4) = 0.05` to 1e-5; `2 * t_sf(12.7062, 1) = 0.05`; large df approaches the normal,
   `2 * t_sf(1.959964, 1e7) = 0.05` to 1e-5); `welch` on a hand-computed example with unequal variances
   (t, df, p); the zero-variance cases; fewer than 2 values raises; swapping `a` and `b` negates `t` and
   `diff` and keeps `p`.
5. **Spec:** loading, defaults, unknown keys, `seed`/`total_steps` in an arm rejected, `run_dir` layout.
6. **Launcher (tiny, CPU, the M10/M11 fake-env style or a tiny real config):** the commands built per
   (arm, seed) (exact argv, `command.txt`); skip/resume/fresh by the files present; `--parallel` never
   exceeds K (mock subprocess); a failing run does not stop the others and `run` exits non-zero;
   `eval_checkpoints` picks the last k distinct steps (and fewer when a run has fewer); the per-checkpoint
   suites, head-to-heads and the round robin are written.
7. **Report:** on synthetic run directories (hand-written `suite.json`, `h2h_{step}.json`, `eval.jsonl`,
   tournament): the per-checkpoint means, the per-seed tables, mean and sd, `elo_vs_init` with its
   clipping, the flags only on `primary` metrics, the comparison count, refusing mixed `obs_hash`, and
   the curves PNG exists.
8. **Rating gap:** a rating row's `gap_snapshot_0` equals `learner - snapshot_0` on a mocked rating.

Mutants to aim for: batched path stepping a finished game, or reordering `per_game`; the seed base
ignoring the seed or off by one run; overlap check missing; Welch with pooled variance (Student's t) or
`ddof=0`; `t_sf` one-sided where two-sided is required; flags on non-primary metrics; only the final
checkpoint evaluated; the per-run mean taken over games pooled across checkpoints of different runs;
`elo_vs_init` without clipping (infinite at 100%); resume restarting a finished run; `--parallel` ignored.

## Experiments (12b)

`python/experiments/selfplay12b.json`: base config `python/configs/selfplay.json` (the 11b config), init
checkpoint `runs/curriculum_seed1/latest.pt`, seeds 1 to 3, `total_steps` 10M, primary metrics
`rush_e0` and `elo_vs_init`.

| arm | overrides | question |
| --- | --- | --- |
| `base` | none | How much do runs of the 11b config differ by seed alone? (the spread of this arm) |
| `pbot40` | `p_bot: 0.4` | Does more bot play prevent the drift against rush that 11b showed from 2M to 9M? |
| `ent003` | `ent_coef: 0.003` | Entropy rose from 0.5 to 1.15 over 11b while KL fell to 0. Does a smaller entropy bonus give a stronger, sharper policy? |

Nine runs. At 11b's ~300-650 sps each run takes about 6-8 h alone; the concurrency measurement sets
how many run at once.

## Gate

**12a (implementation, Claude Code):** `./scripts/check.sh` passes. Then on the owner's machine:
(1) the suite on the M10 `latest.pt` with the batched path: greedy rows identical to the M11 baseline
(`runs/curriculum_seed1/suite_latest/`), and the wall time before and after; (2) the concurrency
measurement table; (3) `scripts/experiment.sh run python/experiments/smoke.json` with
`total_steps=200000` (2 arms x 2 seeds), then `report`, end to end, with the report pasted in RL_REPORT.
No performance threshold.

**12b (owner):** `scripts/experiment.sh run python/experiments/selfplay12b.json`, then `report`, pasted
into RL_REPORT with a short "what we learned". The owner reads the per-seed tables before the significance flags.

## Notes for the implementer

- `evaluate` stays the one place that plays an evaluation game; the batched path lives inside it.
- The launcher is plain `subprocess.Popen` with polling; no threads that touch torch.
- Do not change `python/configs/selfplay.json` or any default that would change the 11b run.
- Record every number with the hashes and the git commit of the code that produced it
  (`git rev-parse HEAD` into `command.txt`).
- If a test contradicts this brief or the spec, stop and add a QUESTIONS.md entry.
