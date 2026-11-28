# Milestone 23: Revised Self-Play Environment (PFSP, Live Two-Sided Self-Play, Barrier-Free Stream Rollouts, Anchors, Gating & Variance Reduction)

Pure-Python RL milestone upgrading the self-play training environment ahead of v2 self-play experiments,
while keeping 100% backward compatibility with every existing M11–M22 config and test:

1. **PFSP matchmaking (`pfsp.py` + `SnapshotPool`):** per-opponent EMA win-rate tracking (`WinRateTracker`)
   and AlphaStar-style prioritized opponent sampling (`"off"`, `"hard"`, `"variance"`) across both
   historical/anchor pool opponents and individual bots.
2. **Barrier-free stream rollout collector & live two-sided self-play (`rollout="stream"`, `p_self`):**
   - `collect_rollout_stream` collects exactly `batch = num_envs * num_steps` completed transitions into
     per-`(slot, seat)` GAE lanes without draining mid-turn slots at rollout boundaries.
   - Live `"self"` episodes (probability `p_self`) execute **both** seat 0 and seat 1 in the same single
     batched learner forward pass (zero separate opponent forward calls) and emit valid on-policy
     transitions for **both seats** into separate GAE lanes (`compute_gae_segments`).
   - `opponent_forward="auto"` dispatches single-group ticks via plain per-snapshot forward and multi-group
     stackable ticks via `StackedSnapshots` (never stacking cross-architecture pinned anchors).
3. **Flexible bot pool & pinned external anchors (`selfplay_bots`, `anchor_checkpoints`):**
   - Configurable bot list with per-bot `name`, `bot`, `epsilon`, and `weight` (so e.g. `rush_e0`,
     `rush_e20`, `economy_e0` are tracked and PFSP-weighted individually).
   - Pinned external anchor checkpoints (`pinned=True`) loaded once via `load_net(..., strict_obs=True)`,
     supporting mixed architectures (`arch="v1"` / `arch="v2"` on the same `obs_hash`), never evicted by
     `SnapshotPool.thin()`, and included in the end-of-run tournament.
4. **Win-rate-gated snapshots, per-opponent metrics & self-play `best.pt`:**
   - Optional win-rate gating (`snapshot_win_rate`, `snapshot_min_games`, `snapshot_max_steps`).
   - Structured per-opponent win rates (`sp_win_rate`, `win_rate_bots`, `win_rate_pool`, `win_rate_latest`)
     and current `shaping_weight` in `metrics.jsonl`.
   - Non-regressed `best.pt` saving during self-play based on `rating["learner"]` when the latest suite
     eval passed `regression_floor`.
5. **Shaping annealing & paired seat-0 / seat-1 seeds (`shaping_anneal_steps`, `shaping_end`, `paired_seeds`):**
   - Linear `shaping_weight` annealing across all slots (bots, snapshots, anchors, live `"self"`), using
     Python-side M9 `float32` shaping on bot slots when annealing is enabled (zero Rust edits).
   - Zero-stall FIFO mirror queue (`paired_seeds=True`) so every non-`"self"` map seed and opponent is
     played from both seat 0 and seat 1.
6. **Shipped v2 revised self-play config & A/B experiment spec:**
   - `python/configs/v2_selfplay_revised.json` and `python/experiments/selfplay23.json`.

**The hard constraint:** zero changes to `crates/`, `data/`, `obs_hash`, `rules_hash`, `experiment.py`,
`python/configs/selfplay.json`, `python/configs/v2_selfplay.json`, or any pre-existing tests. When all new
`PPOConfig` fields stay at their defaults, every existing M10–M22 training/eval path behaves identically.

## Clarifications confirmed (owner, 2026-11-28)

- **SP1:** Pure-Python, 100% backward-compatible extension of `SelfPlayVecEnv` and `PPOConfig`; zero Rust or
  hash changes; Python-side M9 `float32` shaping on bot slots when shaping annealing is enabled.
- **SP2:** Barrier-free `rollout="stream"` collector (`collect_rollout_stream`) + two-sided live `"self"`
  (`p_self`) collecting both seats into separate `(slot, seat)` GAE lanes via `compute_gae_segments`, with
  pending/carryover state serialized for exact resume; `p_self > 0` requires `rollout == "stream"`;
  `opponent_forward="auto"` uses per-snapshot for 1 group and `stacked` for `>= 2` stackable groups.
- **SP3:** `WinRateTracker` (EMA with warmup `max(alpha, 1/n)`, prior `0.5`, win=1 / draw=0.5 / loss=0) and
  `pfsp_mode` in `("off", "hard", "variance")` with `pfsp_power=2.0`, `eps=1e-3`, applied to both bots and
  pool+anchors; sampling hierarchy `p_bot`, `p_self`, then `1 - p_bot - p_self` (`p_latest` newest unpinned
  snapshot else PFSP over pool+anchors).
- **SP4:** `selfplay_bots` list of `{name, bot, epsilon, weight}` dicts; `anchor_checkpoints` loaded once
  via `load_net(..., strict_obs=True)` preserving CPU torch RNG, `pinned=True` (immune to `thin()`),
  cross-arch safe via per-anchor `PolicyNet`, included in end-of-run tournament.
- **SP5:** Win-rate-gated snapshots (`snapshot_win_rate`, `snapshot_min_games`, `snapshot_max_steps`);
  per-opponent metrics in `metrics.jsonl`; self-play `best.pt` saved at `rating_every` when latest suite
  passed `regression_floor` and `rating["learner"]` beats `best_score`.
- **SP6:** Linear `shaping_weight` annealing (`shaping_anneal_steps`, `shaping_end`) and zero-stall FIFO
  mirror queue (`paired_seeds`) serialized in `state_dict()`.
- **SP7:** Leave `selfplay.json` and `v2_selfplay.json` untouched; ship `python/configs/v2_selfplay_revised.json`
  and `python/experiments/selfplay23.json`.

## In scope

1. `python/polytopia_rl/pfsp.py` (new module):
   - `BotSpec`, `parse_bot_specs`, `pfsp_weights`, `WinRateTracker`, `shaping_schedule`.
2. `python/polytopia_rl/gae.py`:
   - `compute_gae_segments(rewards, values, terminated, truncated, next_values, segment_end, gamma, lam)`.
3. `python/polytopia_rl/config.py`:
   - Add the 12 new `PPOConfig` fields with legacy defaults; extend `rollout` (`"stream"`) and
     `opponent_forward` (`"auto"`); validate all new fields in `PPOConfig.from_json`.
4. `python/polytopia_rl/selfplay.py`:
   - `LIVE_SELF = "self"`, `Transition.seat: int = 0`, `Entry.pinned: bool = False`,
     `Entry.net_template: Any | None = None`.
   - `SnapshotPool`: pinned entries immune to `thin()`, `latest_snapshot()`, PFSP-aware `sample(...)`.
   - `draw_opponent(...)`: backward-compatible default path + revised `p_self` / `bots` / `tracker` /
     `pfsp_mode` / `pfsp_power` path.
   - `SelfPlayVecEnv`: live `"self"` two-sided turn & reward tracking, flexible `BotSpec` envs, pinned
     cross-arch anchors, `"auto"` single-group fallback & non-stackable anchor dispatch,
     `set_shaping_weight` (with Python-side M9 bot shaping when enabled), `paired_seeds` FIFO mirror queue,
     and full `state_dict()` / `load_state_dict()` persistence.
   - `make_selfplay_vec_env`: wire all M23 `PPOConfig` fields cleanly.
5. `python/polytopia_rl/ppo.py`:
   - `RolloutBuffer.segment_end: Tensor | None = None`.
   - `StreamState` + `collect_rollout_stream(net, vec_env, buffer, generator, stream_state)` (exact `batch`
     transitions per call, lane-grouped packing, exact bootstrapping, serializable `stream_state`).
   - `train_update`: use `compute_gae_segments` when `buffer.segment_end is not None`, else exact legacy
     `compute_gae`.
6. `python/polytopia_rl/train.py` & `python/polytopia_rl/tournament.py`:
   - Load pinned `anchor_checkpoints` preserving CPU torch RNG; maintain & checkpoint `WinRateTracker`,
     `StreamState`, `last_snapshot_step`, `last_regression`; support win-rate-gated snapshots, shaping
     annealing, `rollout="stream"`, `opponent_forward="auto"`, per-opponent metrics in `metrics.jsonl`,
     non-regressed self-play `best.pt`, and `pool/anchor_*.pt` in `finish_selfplay` and `tournament.main`.
7. Shipped configs:
   - `python/configs/v2_selfplay_revised.json`
   - `python/experiments/selfplay23.json`

## Out of scope

Running multi-million-step GPU training runs (owner runs on CUDA); modifying Rust (`crates/`), `data/`,
`obs_hash`, `rules_hash`, `experiment.py`, `python/configs/selfplay.json`, `python/configs/v2_selfplay.json`,
or any pre-existing test file.

## Public API (fixed so tests can import it)

```python
# python/polytopia_rl/pfsp.py
@dataclass(frozen=True)
class BotSpec:
    name: str
    bot: str
    epsilon: float = 0.0
    weight: float = 1.0

def parse_bot_specs(raw: list[dict] | None, default_epsilon: float = 0.0) -> list[BotSpec]: ...

def pfsp_weights(
    win_rates: Sequence[float],
    mode: str,
    power: float = 2.0,
    eps: float = 1e-3,
    base_weights: Sequence[float] | None = None,
) -> np.ndarray: ...

class WinRateTracker:
    def __init__(self, alpha: float = 0.05, prior: float = 0.5): ...
    def record(self, opponent: str, outcome: str) -> float: ...
    def win_rate(self, opponent: str) -> float: ...
    def games(self, opponent: str) -> int: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...

def shaping_schedule(
    update: int,
    start: float,
    end: float,
    anneal_updates: int | None,
) -> float: ...
```

```python
# python/polytopia_rl/gae.py
def compute_gae_segments(
    rewards: Tensor,
    values: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    next_values: Tensor,
    segment_end: Tensor,
    gamma: float,
    lam: float,
) -> tuple[Tensor, Tensor]: ...
```

```python
# python/polytopia_rl/selfplay.py
LIVE_SELF = "self"

@dataclass
class Transition:
    slot: int
    reward: float
    terminated: bool
    truncated: bool
    final_board: np.ndarray | None
    final_globals: np.ndarray | None
    seat: int = 0

@dataclass
class Entry:
    step: int
    state: dict
    name: str
    pinned: bool = False
    net_template: Any | None = None

class SnapshotPool:
    def add(
        self,
        model_state: dict,
        step: int,
        *,
        pinned: bool = False,
        name: str | None = None,
        net_template=None,
    ) -> Entry: ...
    def latest_snapshot(self) -> Entry | None: ...
    def sample(
        self,
        rng: np.random.Generator,
        *,
        tracker: WinRateTracker | None = None,
        pfsp_mode: str = "off",
        pfsp_power: float = 2.0,
    ) -> Entry: ...

def draw_opponent(
    pool: SnapshotPool,
    rng: np.random.Generator,
    p_bot: float,
    *,
    p_self: float = 0.0,
    bots: Sequence[BotSpec] | None = None,
    tracker: WinRateTracker | None = None,
    pfsp_mode: str = "off",
    pfsp_power: float = 2.0,
): ...
```

```python
# python/polytopia_rl/ppo.py
class StreamState:
    def __init__(self): ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...

def collect_rollout_stream(
    net,
    vec_env,
    buffer: RolloutBuffer,
    generator: torch.Generator,
    stream_state: StreamState | None = None,
) -> list: ...
```

## Rules

### 1. PFSP, Bot Specs, Win-Rate Tracker & Shaping Schedule (`pfsp.py`)

- **`parse_bot_specs(raw, default_epsilon=0.0)`**:
  - When `raw is None`: returns `[BotSpec("economy", "economy", default_epsilon, 1.0), BotSpec("rush", "rush", default_epsilon, 1.0)]`.
  - When `raw` is a list: must be non-empty; each item must be a dict with keys subset of `{"name", "bot", "epsilon", "weight"}` containing `"name"` (non-empty `str`, unique, not `"self"`, not starting with `"snap_"` or `"anchor_"`), `"bot"` in `("random", "economy", "rush")`, `"epsilon"` in `[0.0, 1.0]` (default `0.0`), `"weight"` `> 0.0` (default `1.0`). Raises `ValueError` on any violation.
- **`pfsp_weights(win_rates, mode, power=2.0, eps=1e-3, base_weights=None)`**:
  - Raises `ValueError` if `len(win_rates) == 0`, `mode not in ("off", "hard", "variance")`, `power <= 0.0`, `eps < 0.0`, or `base_weights` has wrong length or any value `<= 0.0`.
  - Clamp each `w` in `win_rates` to `[0.0, 1.0]`. Base weight `b_i = 1.0` if `base_weights is None` else `float(base_weights[i])`.
  - Raw weights (`float64`):
    - `"off"`: `b_i`
    - `"hard"`: `b_i * ((1.0 - w_i) ** power + eps)`
    - `"variance"`: `b_i * ((w_i * (1.0 - w_i)) ** power + eps)`
  - Return `raw / raw.sum()` (1D `np.ndarray` of `float64`, strictly finite, sums to `1.0`).
- **`WinRateTracker(alpha=0.05, prior=0.5)`**:
  - Validates `0.0 < alpha <= 1.0` and `0.0 <= prior <= 1.0` (`ValueError` otherwise).
  - `record(opponent, outcome)`: `outcome` must be `"win"` (`1.0`), `"draw"` (`0.5`), or `"loss"` (`0.0`) (else `ValueError`).
    - Increment `n = count + 1`.
    - Effective step size `eta = max(self.alpha, 1.0 / n)`.
    - `prev = self.prior` if `n == 1` else previous EMA (note: for `n == 1`, `eta = 1.0`, so `new = score` exactly; for `n == 2`, `eta = max(alpha, 0.5)`, giving the exact running mean until `1 / n <= alpha`, then smooth EMA with rate `alpha`!).
    - Store `new = prev + eta * (score - prev)` and `count = n`; return `new`.
  - `win_rate(opponent)`: returns stored EMA if seen, else `self.prior`.
  - `games(opponent)`: returns stored count if seen, else `0`.
  - `state_dict()` / `load_state_dict(state)`: round-trips `alpha`, `prior`, and sorted `{name: [ema, count]}`.
- **`shaping_schedule(update, start, end, anneal_updates)`**:
  - If `anneal_updates is None`: returns `float(start)`.
  - Raises `ValueError` if `anneal_updates <= 0`.
  - Otherwise `frac = min(1.0, max(0.0, (update - 1) / anneal_updates))`, returns `float(start + frac * (end - start))`.

### 2. Segmented GAE (`compute_gae_segments` in `gae.py`)

- All inputs are 1D tensors of shape `(N,)` on the same device (`terminated`, `truncated`, `segment_end` boolean).
- For `N == 0`: returns two empty tensors of shape `(0,)` matching `rewards`'s dtype and device.
- `segment_end[N - 1]` must be treated as `True` (last element always terminates its segment).
- Semantics for `i = N - 1, ..., 0`:
  - `is_end = bool(segment_end[i]) or (i == N - 1)`
  - `v_next = next_values[i]` if `(truncated[i] or is_end)` else `values[i + 1]`
  - `not_term = 0.0 if terminated[i] else 1.0`
  - `delta = rewards[i] + gamma * v_next * not_term - values[i]`
  - `cont = 0.0 if (terminated[i] or truncated[i] or is_end) else 1.0`
  - `a_next = delta + gamma * lam * cont * a_next`
  - `advantages[i] = a_next`
- Returns `(advantages, advantages + values)`, both shape `(N,)`.
- Must match `compute_gae` bit-for-bit (within `atol=1e-6`) when a `(T, E)` buffer is flattened in lane-major `(E, T)` order with `segment_end` True at each lane's final step and `next_values` populated with `last_value` at non-truncated segment ends.

### 3. `PPOConfig` Fields & Validation (`config.py`)

New fields and defaults:
- `p_self: float = 0.0`
- `pfsp_mode: str = "off"`
- `pfsp_power: float = 2.0`
- `pfsp_ema_alpha: float = 0.05`
- `selfplay_bots: list[dict] | None = None`
- `anchor_checkpoints: list[str] = field(default_factory=list)`
- `snapshot_win_rate: float | None = None`
- `snapshot_min_games: int = 20`
- `snapshot_max_steps: int | None = None`
- `shaping_anneal_steps: int | None = None`
- `shaping_end: float = 0.0`
- `paired_seeds: bool = False`

Extended validation in `PPOConfig.from_json`:
- `rollout in ("interleaved", "lockstep", "stream")`
- `opponent_forward in ("stacked", "per_snapshot", "auto")`
- `0.0 <= p_bot <= 1.0`, `0.0 <= p_self <= 1.0`, `p_bot + p_self <= 1.0 + 1e-12` (clamped / exact `<= 1.0`), `0.0 <= p_latest <= 1.0`
- `p_self > 0.0` requires `rollout == "stream"`
- `pfsp_mode in ("off", "hard", "variance")`
- `pfsp_power > 0.0`
- `0.0 < pfsp_ema_alpha <= 1.0`
- `selfplay_bots` validated via `parse_bot_specs(cfg.selfplay_bots)` when not `None`
- `anchor_checkpoints`: must be a `list` of non-empty `str`
- `snapshot_win_rate is None or 0.0 <= snapshot_win_rate <= 1.0`
- `snapshot_min_games >= 1`
- `snapshot_max_steps is None or snapshot_max_steps >= snapshot_every`
- `shaping_anneal_steps is None or shaping_anneal_steps > 0`
- `shaping_end >= 0.0`

### 4. `SnapshotPool`, `draw_opponent` & `SelfPlayVecEnv` (`selfplay.py`)

- **Pinned anchors in `SnapshotPool`:**
  - `add(..., pinned=False, name=None, net_template=None)` appends an `Entry` with cloned CPU tensors.
  - `thin()` preserves every `pinned=True` entry in order and applies the M11 even-index + newest thinning rule strictly to the unpinned entries. `add` calls `thin()` while the count of **unpinned** entries exceeds `pool_max`.
  - `latest_snapshot()` returns the newest unpinned entry (or `entries[-1]` if all entries are pinned, or `None` if empty).
- **`draw_opponent` & `SnapshotPool.sample`:**
  - Legacy default path (`p_self == 0.0`, `bots is None`, `pfsp_mode == "off"`, and no pinned entries): exact M11 RNG draw sequence.
  - Revised path: single `u = rng.random()` selects bot (`u < p_bot`), live `"self"` (`p_bot <= u < p_bot + p_self`), or pool (`u >= p_bot + p_self`).
    - Bot selection uses `pfsp_weights` over `bots` (respecting `base_weights=[b.weight for b in bots]`). Returns `BotSpec` when `bots is not None`, else legacy `str` bot name.
    - Pool selection: if `rng.random() < pool.p_latest`, returns `pool.latest_snapshot()`; else samples from `pool.entries` (unpinned + pinned anchors) using `pfsp_weights`.
- **Live `"self"` in `SelfPlayVecEnv`:**
  - Slot `i` with `self._opp[i] == LIVE_SELF` runs a two-seat env (`opponent="none"`).
  - `_opponent_to_move(i)` is `False`; `learner_slots()` includes `i` whenever `self._ended[i] is None`.
  - `learner_seat(i)` helper returns `_current_player(self.envs[i])` for `LIVE_SELF` slots, else `self._seat[i]`.
  - In `tick(learner_actions)`:
    - Stepping a `LIVE_SELF` slot `i` when seat `s = _current_player(env)` is acting:
      - Records `before_scores = env.info()["scores"]`, steps `env.step_factored(*action)`, logs `action` in `_log[i]` and increments seat `s` length.
      - If `terminated or truncated`:
        - Emits a `Transition(slot=i, seat=s, ...)` for acting seat `s` with reward `_seat_reward(i, s, before_scores)`, `terminated`, `truncated`, `final_board`, `final_globals`.
        - If `other = 1 - s` had a pending `_self_before[i][other]`, also emits `Transition(slot=i, seat=other, ...)` with reward `_seat_reward(i, other, _self_before[i][other])`, `terminated`, `truncated`, `final_board`, `final_globals`.
        - Emits one `EpisodeStat(env_index=i, seed=self._seed[i], opponent=LIVE_SELF, agent_seat=0, outcome=agent_outcome(out, 0), length=sum(_self_len[i]), ret=_self_ret[i][0])` and resets slot `i`.
      - Else if `_current_player(env) == s` (same seat continues turn):
        - Emits `Transition(slot=i, seat=s, reward=_seat_reward(i, s, before_scores), terminated=False, truncated=False, final_board=None, final_globals=None)`.
      - Else (turn passed to `other = 1 - s`):
        - Stores `_self_before[i][s] = before_scores` (seat `s` is now pending).
        - If `_self_before[i][other]` is not None, completes `other`'s pending transition and emits `Transition(slot=i, seat=other, reward=_seat_reward(i, other, _self_before[i][other]), terminated=False, truncated=False, final_board=None, final_globals=None)` and clears `_self_before[i][other] = None`.
    - Returned transitions from `tick` are sorted by `(tr.slot, tr.seat)`.
- **Snapshot & anchor dispatch (`opponent_forward`):**
  - Pinned anchors with `entry.net_template is not None` build their policy from `copy.deepcopy(entry.net_template)` (with `entry.state` loaded) rather than `copy.deepcopy(learner_net)`.
  - When `stacked_policy` is provided:
    - Any group whose entry has `entry.net_template is not None` (non-stackable) is always dispatched via `self._policy(entry)`.
    - Remaining stackable groups: under `"stacked"`, always run through `stacked_policy`; under `"auto"` (`auto_stack=True`), run through `stacked_policy` only when `len(stackable_groups) >= 2`, else via `self._policy(entry)`.
    - `_stack_entries()` only includes stackable entries (`e.net_template is None`).
- **Python-side bot shaping (`set_shaping_weight`):**
  - `vec.set_shaping_weight(w)` sets `self.reward["shaping_weight"] = np.float32(w)`.
  - When `python_bot_shaping=True` (set by `make_selfplay_vec_env` iff `cfg.shaping_anneal_steps is not None`), bot-mode envs are constructed with `shaping_weight=0.0` and `SelfPlayVecEnv` adds `self.reward["shaping_weight"] * (gamma * after - before)` in `np.float32` (matching M9 exactly for any annealed `shaping_weight`).
- **Paired seeds (`paired_seeds=True`):**
  - FIFO `_mirror_queue: list[tuple[int, object, int]]` + monotonic `_next_seed_idx: int`.
  - Fresh episode on non-`"self"` opponent plays `seat = 0` and appends `(seed, opp, 1)` to `_mirror_queue`; next resetting slot pops the front of `_mirror_queue` and plays `(seed, opp, 1)`.
  - Fresh `"self"` episode plays both seats on `seed` and enqueues nothing.
  - Bot envs are cached per `(bot_name, epsilon, seat_key)` where `seat_key = "seed_parity"` when `paired_seeds=False` and `seat` (`0` or `1`) when `paired_seeds=True`.
- **State serialization (`state_dict` / `load_state_dict`):**
  - Backward-compatible with existing keys; additionally stores `_seat` (when `paired_seeds`), `_mirror_queue`, `_next_seed_idx`, `_self_before` / `_self_ret` / `_self_len`, and `before` (for mid-turn snapshot slots in `rollout="stream"`!) so resuming mid-turn or mid-mirror-queue reproduces the uninterrupted run bit-for-bit.

### 5. Barrier-Free Stream Collector (`collect_rollout_stream` in `ppo.py`)

- `collect_rollout_stream(net, vec_env, buffer, generator, stream_state=None) -> list[EpisodeStat]`:
  - `batch = buffer.num_steps * buffer.num_envs`.
  - If `stream_state is None`, creates a fresh `StreamState()`.
  - Drains `stream_state.carryover` into per-`(slot, seat)` lane lists first (up to `batch`).
  - Loops while `collected < batch`:
    - `acting = vec_env.learner_slots()`
    - If `acting`: one batched `net.trunk` + `net.act` pass over `[vec_env.envs[i] for i in acting]`; stores each `(slot, vec_env.learner_seat(slot))` step into `stream_state.pending[(slot, seat)]`, passes `actions` to `vec_env.tick(actions)`.
    - For each `tr` in `vec_env.tick(actions)`: pops `step = stream_state.pending.pop((tr.slot, tr.seat))`, combines into a completed record, appends to `lanes[(tr.slot, tr.seat)]` if `collected < batch` (incrementing `collected`), else appends to `stream_state.carryover`.
    - Collects `vec_env.episode_stats()`.
  - Computes bootstrap `next_values`:
    - For every truncated completed transition: batch-evaluate `net.value(net.trunk(final_board, final_globals))`.
    - For every lane's final transition that is neither `terminated` nor `truncated`:
      1. If `lane_id` appears in `stream_state.carryover`, use the first carryover record's `.value`.
      2. Else if `lane_id` is in `stream_state.pending`, use `stream_state.pending[lane_id].value`.
      3. Else (that seat is currently to move in `vec_env.envs[slot]`), batch-evaluate `net.value(net.trunk(obs))` across all such lanes in one forward pass.
  - Packs all non-empty lanes in ascending `(slot, seat)` order into `buffer` (reshaping length-`batch` arrays to `(T, E)`), and sets `buffer.segment_end` (bool tensor `(batch,)` True on each lane's final flat index).
  - `train_update`: if `buffer.segment_end is not None`, calls `compute_gae_segments` on the flattened `(batch,)` arrays; otherwise runs the exact existing `compute_gae` path.

### 6. `train.py` & `tournament.py` Wiring

- **Anchors:** Load `cfg.anchor_checkpoints` once at startup via `load_net(..., device=device, strict_obs=True)` (saving/restoring CPU torch RNG around loading). Validate `anchor_spec["obs_hash"] == spec["obs_hash"]` (`SpecMismatch` on mismatch). Add each as `pool.add(anchor_net.state_dict(), 0, pinned=True, name=f"anchor_{i}", net_template=anchor_net)`.
- **Win-rate tracker:** Created for `cfg.selfplay` with `alpha=cfg.pfsp_ema_alpha`. Updated after every rollout for each finished `EpisodeStat` with `s.opponent != LIVE_SELF`. Saved/restored in checkpoints (`ck.extra["win_rate_tracker"]`).
- **Snapshots:**
  - When `cfg.snapshot_win_rate is None`: exact legacy `crossed(cfg.snapshot_every)`.
  - When `cfg.snapshot_win_rate is not None`: add snapshot when `global_step - last_snapshot_step >= cfg.snapshot_every` and either `(latest is not None and tracker.games(latest.name) >= cfg.snapshot_min_games and tracker.win_rate(latest.name) >= cfg.snapshot_win_rate)` or `(cfg.snapshot_max_steps is not None and global_step - last_snapshot_step >= cfg.snapshot_max_steps)`. Update `last_snapshot_step = global_step`.
- **Shaping annealing:** If `cfg.shaping_anneal_steps is not None`, compute `cur_shaping = shaping_schedule(update, cfg.shaping_weight, cfg.shaping_end, max(1, cfg.shaping_anneal_steps // batch))` before each rollout and call `vec.set_shaping_weight(cur_shaping)` (self-play; raises `ValueError` in curriculum mode only? No: if curriculum doesn't support `set_shaping_weight`, only call it in self-play or document it as self-play-only — wait: keep curriculum unchanged and call `vec.set_shaping_weight(cur_shaping)` when `cfg.selfplay`).
- **Collector selection:**
  - `cfg.selfplay and cfg.rollout == "stream"` -> `collect_rollout_stream(net, vec, buffer, gen, stream_state)`
  - `cfg.selfplay and cfg.rollout == "interleaved"` -> `collect_rollout_interleaved(net, vec, buffer, gen)`
  - Else -> `collect_rollout(net, vec, buffer, gen)` (setting `buffer.segment_end = None`).
- **Metrics in `metrics.jsonl` (when `cfg.selfplay`):**
  - Add `"shaping_weight"` (current float), `"sp_win_rate"` (dict `{name: ema}` for bots, anchors, and latest snapshot that have `games > 0`), `"win_rate_bots"` (game-count-weighted mean EMA or simple mean across seen bots, `None` if 0 bot games — specify: game-count-weighted mean over seen bots, `None` if none), `"win_rate_pool"` (game-count-weighted mean over seen pool+anchor opponents, `None` if none), `"win_rate_latest"` (EMA vs `latest.name` if `games > 0` else `None`).
- **Self-play `best.pt`:**
  - Track `last_regression: bool | None` (updated at `eval_every`).
  - At `rating_every`: if `last_regression is False` and `(best_score is None or ratings["rating"]["learner"] > best_score)`, set `best_score = ratings["rating"]["learner"]` and save `best.pt`.
- **Tournament (`finish_selfplay` & `tournament.py`):**
  - `finish_selfplay` saves unpinned entries as `pool/{e.name}.pt` (`snap_*.pt`) and pinned anchors as `pool/{e.name}.pt` (`anchor_*.pt`, using each anchor's `net_template` + `e.state`) and includes all of them in the round-robin tournament.
  - `tournament.main` loads both `pool/snap_*.pt` (sorted by step) and `pool/anchor_*.pt` (sorted by index).
- **Exact resume:** Checkpoint `extra` persists `win_rate_tracker`, `last_snapshot_step`, `last_regression`, and `stream_state` (when `rollout == "stream"`), so resuming mid-run produces identical metrics, evals, pool, and final weights.

### 7. Shipped Configs & Experiment Spec

- `python/configs/v2_selfplay_revised.json`:
  ```json
  {
    "ruleset": "v2",
    "arch": "v2",
    "selfplay": true,
    "rollout": "stream",
    "opponent_forward": "auto",
    "total_steps": 20000000,
    "num_envs": 128,
    "num_steps": 32,
    "lr": 0.0001,
    "opponents": ["economy", "rush"],
    "stages": [{"opponents": ["economy", "rush"]}],
    "p_bot": 0.25,
    "p_self": 0.35,
    "p_latest": 0.25,
    "pfsp_mode": "hard",
    "pfsp_power": 2.0,
    "pfsp_ema_alpha": 0.05,
    "selfplay_bots": [
      {"name": "rush_e0", "bot": "rush", "epsilon": 0.0},
      {"name": "rush_e20", "bot": "rush", "epsilon": 0.2},
      {"name": "economy_e0", "bot": "economy", "epsilon": 0.0}
    ],
    "snapshot_every": 500000,
    "snapshot_win_rate": 0.55,
    "snapshot_min_games": 20,
    "snapshot_max_steps": 2000000,
    "pool_max": 40,
    "opponent_greedy": false,
    "shaping_anneal_steps": 5000000,
    "shaping_end": 0.0,
    "paired_seeds": true,
    "eval_every": 1000000,
    "eval_seeds": 50,
    "regression_floor": 0.9,
    "rating_every": 2000000,
    "rating_seeds": 50,
    "tournament_seeds": 100
  }
  ```
- `python/experiments/selfplay23.json`: 2-arm A/B experiment comparing `"legacy"` (exact `v2_selfplay.json` settings) vs `"revised"` (M23 stream + PFSP + live self-play + paired seeds + shaping annealing + win-rate-gated snapshots) over `seeds: [1, 2, 3]`, `total_steps: 10_000_000`, `primary: ["rush_e0", "elo_vs_init"]`, `init_checkpoint` pointing to the post-transfer v2 curriculum checkpoint (`runs/exp/transfer22/distill_v2/seed1/ckpt_final.pt`).

## Tests (Phase A)

Write `python/tests/rl/test_selfplay_v2.py` covering:
1. **`pfsp.py` unit tests:** `parse_bot_specs` (defaults, custom specs, validation errors), `pfsp_weights` (hand-computed `"off"`, `"hard"`, `"variance"`, `base_weights`, edge cases & errors), `WinRateTracker` (warmup exact mean -> EMA transition, prior, win/draw/loss, round-trip `state_dict`), `shaping_schedule` (linear decay, clamping, `None` constant, error).
2. **`compute_gae_segments` unit tests:** exact equivalence to `compute_gae` on lane-major flattened `(T, E)` tensors; hand-computed variable-length segments with `terminated`, `truncated` (`next_values`), and ongoing segment-end bootstrapping; empty `(0,)` input.
3. **`PPOConfig` validation & shipped files:** all 12 new field defaults, round-trip JSON, every validation error (`p_bot + p_self > 1`, `p_self > 0` without `rollout="stream"`, invalid `pfsp_mode`, bad `selfplay_bots`, `snapshot_max_steps < snapshot_every`, etc.), and schema/load verification of `v2_selfplay_revised.json` and `selfplay23.json` (while confirming `selfplay.json` and `v2_selfplay.json` are untouched).
4. **Pool, PFSP sampling & pinned anchors:** pinned entries immune to `thin()`; `latest_snapshot()` ignores pinned anchors; statistical frequency checks (5-sigma bounds) for `draw_opponent` with `p_bot`, `p_self`, `p_latest`, `pfsp_mode="hard"` / `"variance"`, and weighted `selfplay_bots`.
5. **Live two-sided `"self"` & `SelfPlayVecEnv` mechanics:**
   - Hand-verified two-seat trace on `FakeTwoSeatEnv`: both seat 0 and seat 1 emit exact M9 rewards, zero opponent calls, simultaneous terminal completion of acting seat + pending other seat.
   - `paired_seeds=True`: FIFO mirror queue plays identical `(seed, opponent)` on seat 0 and seat 1; `"self"` skips mirror; exact `state_dict()` / `load_state_dict()` round-trip mid-queue.
   - Python-side bot shaping (`set_shaping_weight`): exact `float32` match against Rust `Env(shaping_weight=w)` for multiple `w` values (`1.0`, `0.5`, `0.0`).
   - `opponent_forward="auto"` uses per-snapshot for 1 group and `stacked` for `>= 2` stackable groups, and never passes cross-arch pinned anchors (`arch="v1"` anchor with `arch="v2"` learner) to `StackedSnapshots`.
6. **Barrier-free `collect_rollout_stream`:**
   - Produces exact `batch = T * E` transitions per rollout across consecutive rollouts with live `"self"` + snapshots + bots, with zero drain ticks at rollout boundary (slots remain mid-turn across rollouts).
   - Carryover & pending bootstrapping verified against an independent hand-computed reference; `train_update` runs cleanly with `buffer.segment_end` (both plain PPO and with M22 distillation!).
7. **End-to-end CPU `train.run` (revised self-play):**
   - Runs end-to-end with `ruleset="v2"`, `arch="v2"`, `rollout="stream"`, `opponent_forward="auto"`, `p_self=0.4`, `p_bot=0.25`, `pfsp_mode="hard"`, `selfplay_bots`, cross-arch `anchor_checkpoints` (a v2 `arch="v1"` checkpoint!), `snapshot_win_rate` gating + `snapshot_max_steps`, `shaping_anneal_steps`, `paired_seeds=True`, and verifies:
     - `metrics.jsonl` contains `shaping_weight` (decaying), `sp_win_rate`, `win_rate_bots`, `win_rate_pool`, `win_rate_latest`.
     - `best.pt` saved when non-regressed and NOT saved when regressed.
     - `pool/anchor_0.pt` saved and included in `tournament.json` + rebuilt via `tournament.main`.
     - **Exact bit-for-bit mid-run resume** (mid-stream, mid-anneal, mid-mirror-queue, with tracker + carryover + pending state).
     - Unchanged v1 `rules_hash` (`a8b52b91eae34503`) and `obs_hash(11)` (`d5f9b1033cfd55ca`).

## Definition of done

`pixi run check` exits 0 (`.git/tests-lock`, `cargo fmt`, `cargo clippy -D warnings`, all Rust tests, all
Python tests, all RL tests including `test_selfplay_v2.py`), zero Rust/hash/existing-test changes, and
`/review 23` passes with 0 blocking defects.
