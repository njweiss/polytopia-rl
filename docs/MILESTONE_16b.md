# Milestone 16b: rollout speedups, harness correctness, paired stats and repo hygiene

Depends on milestones 1 to 16. Before continuing game v2 (M17–M21) and launching a new round of multi-seed
training runs, this milestone clears the remaining low-effort speed wins from the M13a/M13b profiles, fixes
the deferred correctness gaps in the RL harness, adds paired statistical testing for our paired-seed
experiment design, and cleans up repo/workflow friction.

**The hard constraint:** v1 and v2 game rules, `obs_hash`, `rules_hash`, digests, replays and `Config` JSON
are completely unchanged. Every existing Rust and Python test passes with no expected value changed.

## Goal

1. Eliminate the remaining host-bound rollout overhead identified in M13a/M13b: per-element GPU buffer
   writes/reads (~27% of rollout time, ~29,000 syncs/rollout), redundant mask-check GPU syncs inside
   `PolicyNet.act`/`evaluate`, and full `env.info()` dict + score allocations on every self-play tick check.
2. Fix four deferred RL harness gaps: the opening-turn truncation hang in `SelfPlayVecEnv.tick`, epoch-mean
   `target_kl` early stopping, batched evaluation in the `evaluate.py` CLI and curriculum trainer, and hash
   validation in `train.load_policy`.
3. Add paired t-test statistics to `polytopia_rl.stats` and `experiment.report` (matching the paired-seed
   experiment design) alongside the existing Welch comparisons, and run the deferred `experiment.py`
   simplify pass.
4. Clean up repo and workflow friction (stale docs, orphan `src/polytopia_rl`, worktree-safe `py_check.sh` /
   `rl_check.sh`, and optional `osx-arm64` CPU support in `pixi.toml`).

## Decisions already made (owner)

- Deferred option (a) from M13a/M13b (batching per-element buffer writes and action reads in
  `collect_rollout_interleaved`) and the opening-turn truncation safeguard are open follow-ups in
  `docs/STATUS.md`.
- `experiment.py` simplify pass is on `docs/STATUS.md` ("Next three actions").
- Play-only calls already skip the mask check (`check=False`, Q30 A) because env masks always have at least
  one legal option.
- No new Python or Rust dependencies.
- `PPOConfig.opponent_forward` keeps its existing default (`"stacked"`) so M13b config tests remain
  unchanged; `python/configs/selfplay.json` keeps `"per_snapshot"`.

## In scope

1. `crates/sim-py`: lightweight `Env.current_player() -> int` method (no dict allocation, no score
   computation).
2. `python/polytopia_rl/categorical.py` and `model.py`: optional `check: bool = True` on `log_prob` and
   `entropy`; `PolicyNet.act` and `PolicyNet.evaluate` skip redundant mask-check syncs on the inner path.
3. `python/polytopia_rl/ppo.py`: batched action readback and batched transition scatter in
   `collect_rollout_interleaved`; epoch-mean `target_kl` check in `train_update`.
4. `python/polytopia_rl/selfplay.py`: use `env.current_player()` when available (falling back to
   `env.info()["current_player"]` for fake/mock envs); handle opening-turn episode ends in `tick` so
   `collect_rollout_interleaved` never hangs.
5. `python/polytopia_rl/evaluate.py` and `train.py`: batched evaluation in `evaluate.main` and curriculum
   periodic eval; `train.load_policy` delegates to `checkpoint.load_net`.
6. `python/polytopia_rl/stats.py` and `experiment.py`: `PairedT` + `paired_t(a, b, alpha=0.05)`; paired
   results reported alongside Welch in `experiment.report` / `report.md`; readability/simplify pass on
   `experiment.py`.
7. Repo and workflow hygiene: remove empty `src/polytopia_rl/__init__.py`; fix stale lines in `CLAUDE.md`
   and `docs/ORCHESTRATOR.md`; worktree-safe Python extension loading in `scripts/py_check.sh` and
   `scripts/rl_check.sh`; `osx-arm64` CPU target in `pixi.toml` (if confirmed in B6).
8. Before/after measurement on the M13a profile fixture recorded in `docs/RL_REPORT.md` ("Milestone 16b").

## Out of scope

Any game rule change (v1 or v2), observation or action-space change, `obs_hash` or `rules_hash` change,
`PolicyNet` architecture changes (M20), reward shaping or discounting changes (M21), PFSP / block-batched
opponents / retiring `stacked.py` (M22), `sim-core` `update_connections` or `city(id)` changes (M17).

## Public API (fixed so tests can import it)

```rust
// crates/sim-py (new method on PyEnv; all existing methods unchanged)
fn current_player(&self) -> PyResult<u8>   // RuntimeError(NotReset) before first reset
```

```python
# polytopia_rl.categorical (new keyword defaults preserve all existing callers)
def log_prob(logits: Tensor, mask: Tensor, index: Tensor, check: bool = True) -> Tensor
def entropy(logits: Tensor, mask: Tensor, check: bool = True) -> Tensor

# polytopia_rl.stats (new dataclass and function; Welch and welch unchanged)
@dataclass
class PairedT:
    mean_a: float
    mean_b: float
    diff: float        # mean(b_i - a_i) == mean_b - mean_a
    sd_diff: float     # sample sd (ddof=1) of d_i = b_i - a_i
    t: float           # diff / (sd_diff / sqrt(n))
    df: float          # float(n - 1)
    p: float           # two-sided: 2 * t_sf(|t|, df)
    significant: bool  # p < alpha

def paired_t(a: Sequence[float], b: Sequence[float], alpha: float = 0.05) -> PairedT

# polytopia_rl.experiment (ExperimentReport top-level keys unchanged: OLD_REPORT_KEYS | {"cost"})
# Each comparison dict in `ExperimentReport.comparisons` gains one key:
#   "paired": {"t": float, "df": float, "p": float, "significant": bool | None}
# where `significant` is `bool(p < 0.05)` for primary metrics and `None` for exploratory ones.
```

All other public signatures (`SelfPlayVecEnv`, `collect_rollout_interleaved`, `train_update`, `evaluate`,
`train.load_policy`, `ExperimentReport`) are unchanged.

## Rules

### 1. `PyEnv.current_player()` and `SelfPlayVecEnv` polling

- `PyEnv.current_player()` returns `self.env()?.view().current()` (a `u8`, `0` or `1`), raising
  `RuntimeError("NotReset")` before the first `reset`, identical to `info()`.
- `SelfPlayVecEnv` reads the current player via a helper that calls `env.current_player()` when the env
  exposes it and falls back to `env.info()["current_player"]` otherwise (so fake/mock envs in tests keep
  working unchanged). Both `_opponent_to_move` and `learner_slots` use this helper.

### 2. Batched buffer writes and action readback (`collect_rollout_interleaved`)

- After `net.act(...)`, convert `out.actors`, `out.kinds`, `out.args` to host Python lists in a single
  batched transfer per tensor (`out.actors.cpu().tolist()`, etc.) instead of `int(out.actors[r])` inside a
  per-slot loop.
- After `vec_env.tick(actions)`, collect the completed transitions `done` into host lists/arrays
  (`slots`, `t_idx`, `rewards`, `terminated`, `truncated`), convert once to tensors on `device`, and scatter
  them into `buffer.rewards`, `buffer.terminated`, `buffer.truncated`, and `buffer.next_values` (zeroed for
  those `(t_idx, slots)` first) in one indexed write per field per tick.
- For truncated transitions (`cut`), compute `values = net.value(net.trunk(fb, fg))` in one batch as today
  and scatter `values` into `buffer.next_values[cut_t, cut_slots]` in one indexed write before advancing
  `t[slot] += 1`.
- Exact numerical equivalence: on CPU, for a fixed seed, `collect_rollout_interleaved` produces bit-identical
  buffer tensors, episode stats, and final vec-env state to today's implementation.

### 3. Mask-check sync removal (`categorical.py`, `model.py`)

- `categorical.log_prob` and `categorical.entropy` accept `check: bool = True` and forward it to
  `masked_logits(logits, mask, check=check)`. Calling either with default arguments still raises `ValueError`
  on an all-false mask row.
- `PolicyNet.act`: when `stats=True`, validate masks at most once or skip (`check=False`, see Clarification
  B1); `log_prob` and `entropy` always receive `check=False`.
- `PolicyNet.evaluate`: passes `check=False` to `log_prob` and `entropy` (masks in the rollout buffer were
  produced by `Env` and already validated during collection).
- On every valid mask, outputs and gradients are bit-identical to today.

### 4. Opening-turn truncation/termination safeguard (`SelfPlayVecEnv.tick`)

- In `SelfPlayVecEnv.tick`, after applying opponent moves, if a snapshot slot has `self._before[i] is None`
  and `self._ended[i] is not None` (meaning the episode terminated or truncated during the opponent's opening
  turn before the learner ever acted):
  - Append `self._episode_stat(i)` to `self._stats` (length 0, return 0.0, outcome from `info()["outcome"]`).
  - Call `self._reset(i)` so the slot starts a fresh episode (not advanced until the next tick, preserving
    M13's reset rule).
  - Do **not** emit a `Transition` for slot `i` (there is no learner action to assign a transition to).
- Thus `collect_rollout_interleaved` never hangs when an episode ends during an opponent's opening turn.

### 5. Epoch-mean `target_kl` early stopping (`ppo.py`)

- In `train_update`, accumulate `approx_kl` across all minibatches of an epoch and compare the **epoch mean**
  (`epoch_kl / num_minibatches`) against `cfg.target_kl` at the end of the epoch.
- When `cfg.target_kl is None` (the default), behavior is unchanged.

### 6. Batched evaluation and checked `load_policy` (`evaluate.py`, `train.py`)

- `evaluate.main` passes `batch_fn=net_batch_policy(net, torch.Generator().manual_seed(0), greedy=True,
  device=device)` to `evaluate(...)` (matching `suite.py`).
- Curriculum periodic evaluation in `train.run` (`elif evaluated:` branch) passes `batch_fn=net_batch_policy(
  net, torch.Generator().manual_seed(0), greedy=True, device=device)` to `evaluate(...)`.
- `train.load_policy(path, device="cpu")` delegates to `checkpoint.load_net(path, device=device)` and returns
  `(net, ck.cfg)`, so it enforces `obs_hash` (`SpecMismatch`) and warns on `rules_hash` mismatches and
  unstamped M10 checkpoints.

### 7. Paired t-test (`stats.py`, `experiment.py`)

- `paired_t(a, b, alpha=0.05)`:
  - Requires `len(a) == len(b)` and `len(a) >= 2`; otherwise raises `ValueError`.
  - Computes paired differences `d_i = float(b_i) - float(a_i)` in exact/float64 arithmetic (`statistics`).
  - `mean_a = mean(a)`, `mean_b = mean(b)`, `diff = mean(d)`, `sd_diff = stdev(d)` (`ddof=1`), `df = float(n - 1)`.
  - Zero-variance edge case (`sd_diff == 0.0`): `t = 0.0, p = 1.0` if `diff == 0.0`, else `t = copysign(inf, diff),
    p = 0.0` (matching `welch`'s convention).
  - Otherwise `t = diff / (sd_diff / sqrt(n))`, `p = 2 * t_sf(abs(t), df)`, `significant = bool(p < alpha)`.
- `experiment.report`:
  - Because `per_seed[a]` and `per_seed[b]` are aligned in `spec.seeds` order, every comparison in
    `rep.comparisons` computes `paired_t(per_seed[a], per_seed[b])` in addition to `welch(...)` and attaches
    `"paired": {"t": pt.t, "df": pt.df, "p": pt.p, "significant": pt.significant if primary else None}`.
  - `report_markdown` includes the paired `t`, `df`, `p`, and flag (either as columns in the Comparisons table
    or a short paired table directly beneath it — see Clarification B2).
  - Top-level keys of `ExperimentReport` / `report.json` remain unchanged (`OLD_REPORT_KEYS | {"cost"}`).
- `experiment.py` simplify pass: reduce boilerplate and separate runner / aggregation / formatting helpers
  internally while keeping every public symbol, file layout, and CLI flag identical.

### 8. Repo & workflow hygiene

- Delete empty `src/polytopia_rl/__init__.py` and its empty parent directory `src/polytopia_rl/`.
- Update `CLAUDE.md` layout section (remove stale "Future crates (not yet): `sim-py`, `bots`, `render`") and
  `docs/ORCHESTRATOR.md` (replace the stale M11 "Where things stand" block with a pointer to `docs/STATUS.md`).
- Worktree-safe checks (`scripts/py_check.sh`, `scripts/rl_check.sh`): when `git rev-parse --git-dir` differs
  from `git rev-parse --git-common-dir` (inside a git worktree), `py_check.sh` builds a wheel into
  `target/wheels`, unpacks it to `target/pyext`, and runs pytest with `PYTHONPATH="$PWD/target/pyext:..."`
  instead of running `maturin develop` into the shared `.pixi` environment; `rl_check.sh` likewise prepends
  `target/pyext` to `PYTHONPATH` when in a worktree. In the main checkout, behavior is unchanged.
- `pixi.toml` (pending B6): add `osx-arm64` platform with CPU `pytorch` under `[target.osx-arm64.dependencies]`,
  moving the CUDA-specific `pytorch` and `cuda-cudart-dev` entries to `[target.linux-64.dependencies]` with
  identical version/build pins.

## Clarifications (owner: confirm or overrule)

- **B1 (mask checks in `PolicyNet.act` / `evaluate`):** Skip the GPU→CPU mask check (`check=False`) in
  `PolicyNet.act` (even when `stats=True`) and `PolicyNet.evaluate`, keeping `check=True` as the default on
  `categorical.masked_logits`, `sample`, `log_prob`, and `entropy`. — Confirmed (owner, 2026-11-25)
- **B2 (paired stats placement in `ExperimentReport` and `report.md`):** Attach `"paired": {...}` inside each
  dict in `rep.comparisons` (keeping top-level `report.json` keys identical so M14's
  `test_milestone_12_report_keys_are_unchanged` passes with zero edits), and render a separate
  `"## Comparisons (paired t, b minus a)"` table right below the Welch table in `report.md`. — Confirmed (owner, 2026-11-25)
- **B3 (opening-turn truncation with `EpisodeStat`):** When an episode ends on the opponent's opening turn
  before the learner acts, record an `EpisodeStat` with `length=0, ret=0.0` and reset the slot (no
  `Transition` emitted). — Confirmed (owner, 2026-11-25)
- **B4 (`target_kl` semantics):** Change `train_update` to check epoch-mean `approx_kl > cfg.target_kl`
  instead of last-minibatch `approx_kl > cfg.target_kl`. — Confirmed (owner, 2026-11-25)
- **B5 (`SPEC.md` as single source of truth):** Retire the external Claude Doc as master and treat
  `docs/SPEC.md` in git as the sole authoritative spec going forward (updating `docs/ORCHESTRATOR.md` and
  `docs/STATUS.md` accordingly). — Confirmed (owner, 2026-11-25)
- **B6 (`osx-arm64` in `pixi.toml`):** Add an `osx-arm64` CPU-only target to `pixi.toml` alongside the
  existing `linux-64` CUDA 12.9 target so `pixi run check` works on Apple Silicon Macs. — Confirmed (owner, 2026-11-25)

## Changes to earlier tests

None expected. Every new parameter has a backwards-compatible default, `ExperimentReport`'s top-level keys
are unchanged, and `SelfPlayVecEnv` falls back to `info()["current_player"]` for test fakes. If any locked
test breaks, stop and record it in `docs/QUESTIONS.md` (next number **Q50**).

## Tests (phase A guidance)

1. **`PyEnv.current_player()` (`python/tests/test_env.py` or a new `python/tests/rl/` test):** matches
   `env.info()["current_player"]` at reset and after every step in both bot mode and two-seat mode; raises
   `RuntimeError` before first `reset`. Spy test on `SelfPlayVecEnv`: when an env implements
   `current_player()`, `learner_slots()` and `_opponent_to_move()` do not call `env.info()`.
2. **Batched `collect_rollout_interleaved` equivalence:** on CPU (fixed seed, mixed bots + snapshots, shaping
   on/off, truncations), the batched collector still passes every hand-traced and lockstep-equivalence check
   and matches a sequential reference tensor-for-tensor.
3. **`categorical` `check=False`:** `log_prob(..., check=False)` and `entropy(..., check=False)` match
   `check=True` bit-for-bit on valid masks; `check=True` (default) still raises `ValueError` on an all-false
   row. Verify `PolicyNet.act` and `PolicyNet.evaluate` outputs and gradients match the reference.
4. **Opening-turn truncation safeguard:** scripted `FakeTwoSeatEnv` with `k=3, truncate_at=2` on seat 1
   (opponent moves first, truncates before learner ever acts), followed by a seat-0 seed (or a wrapper whose
   first episode truncates on the opening turn and second does not): `tick` emits no `Transition`, records
   the `EpisodeStat` (`length=0`), resets the slot, and `collect_rollout_interleaved` completes all `T` rows
   without hanging.
5. **Epoch-mean `target_kl`:** construct a tiny update (2 epochs × 2 minibatches) where minibatch 1 has high
   KL and minibatch 2 has low KL (so epoch mean > `target_kl` while last minibatch < `target_kl`), and the
   reverse (epoch mean < `target_kl` while last minibatch > `target_kl`); assert early stopping follows the
   epoch mean in both cases.
6. **`train.load_policy` hash enforcement & batched eval:** `load_policy` raises `SpecMismatch` on a wrong
   `obs_hash`, warns on a wrong `rules_hash`, and loads valid checkpoints identically. `evaluate.main` and
   curriculum periodic eval produce the same greedy W-L-D on CPU as sequential eval (and spy confirms
   `batch_fn` is passed).
7. **`paired_t` and `experiment.report`:**
   - Hand-worked golden example: `a = [1.0, 2.0, 3.0, 4.0]`, `b = [2.0, 4.0, 4.0, 6.0]` → `d = [1, 2, 1, 2]`,
     `diff = 1.5`, `sd_diff = sqrt(1/3)`, `t = 1.5 / sqrt(1/12) = 3 * sqrt(3) ≈ 5.1961524`, `df = 3.0`,
     `p = 2 * t_sf(3*sqrt(3), 3)`.
   - Seed-shift invariance: adding any per-seed offset `c_i` to both `a_i` and `b_i` leaves `diff`, `sd_diff`,
     `t`, `df`, `p` unchanged (while Welch's `t` shrinks).
   - Edge cases: swapped arms negate `diff` and `t`, keep `p`; zero-variance `d` (`diff == 0` → `t=0, p=1`;
     `diff != 0` → `t=±inf, p=0`); mismatched lengths or `n < 2` raise `ValueError`.
   - `experiment.report`: every entry of `rep.comparisons` carries `"paired"` matching `paired_t`,
     `paired["significant"]` is `None` for non-primary metrics, `report.md` renders the paired results, and
     `set(report.json)` still equals `OLD_REPORT_KEYS | {"cost"}`.
8. **Hygiene checks:** `src/polytopia_rl` does not exist; `check.sh` passes in both the main checkout and a
   fresh git worktree without modifying the main checkout's installed extension.

Mutants to aim for: forgetting to reset `buffer.next_values` to `0.0` on non-truncated rows when batching;
mixing `t[i]` before/after increment on truncated rows; `tick` dropping `EpisodeStat` or failing to reset on
an opening-turn end; `tick` emitting a spurious `Transition` on an opening-turn end; `SelfPlayVecEnv` still
calling `env.info()` in `learner_slots`; `target_kl` still checking only the last minibatch; `paired_t`
using `ddof=0` or `df = 2n - 2` instead of `n - 1`; `paired_t` flagging non-primary metrics; `load_policy`
swallowing `SpecMismatch`.

## Gate

1. `pixi run check` exits 0 (all Rust, Python, and RL tests green; zero edits to locked tests).
2. `rules_hash()` == `"a8b52b91eae34503"` and `obs_hash(11)` == `"d5f9b1033cfd55ca"` (v1 byte-identity).
3. On the owner's GPU machine (idle): re-run the M13a profile fixture (`pool` = 10, 3 rollouts,
   `rollout=interleaved`, `opponent_forward=per_snapshot`) before and after M16b, plus `torch.profiler`
   sync counts: report rollout sps, total sps, syncs/rollout outside forward passes, and learner/opponent
   syncs per call in `docs/RL_REPORT.md` ("Milestone 16b"). Expect ~1.2–1.35× rollout speedup and a sharp
   drop in host-device syncs (from ~29,000/rollout to a few hundred).

## Notes for the implementer

- Keep the `SelfPlayVecEnv` lockstep path (`step`, `_advance`) and `collect_rollout` unchanged as references.
- In `collect_rollout_interleaved`, be careful with the two-step truncation order: write the non-truncated
  fields and zero `next_values` for all `done`, compute `net.value` for `cut`, scatter `next_values` for
  `cut`, and increment `t[i]` for every `tr.slot` in `done`.
- In `experiment.py`'s simplify pass, do not change `report.json` keys, directory layout, CLI flags, or
  `RunsFailed` / `Spec` / `ExperimentReport` names.
- If a test contradicts this brief or the spec, stop and add a `docs/QUESTIONS.md` entry (next number
  **Q50**).
