# Milestone 9: Python interface (observation, masks, environment)

The spec's milestone 9. Depends on milestones 1 to 8. Spec gate: "a random agent runs thousands of
games from Python." After this the PPO work in milestone 10 needs nothing from Rust beyond this API.

## Goal

A single-environment Gym-style API for one learner seat: fixed-size float32 observations, factored
action masks, shaped-or-terminal reward, a Rust-side scripted opponent (or two-seat mode for later
self-play), and a thin PyO3 wrapper. All logic lives in a pure-Rust crate so it is tested with
`cargo test`; the Python layer only converts types.

## Decisions already made (owner)

- Opponent: **both** a Rust-side opponent (agent plays one seat, the opponent's turns run inside `step`)
  **and** a two-seat mode (caller supplies every action).
- Reward shaping lives in the Rust env, configurable; the default shaping weight is 0.
- Observation: fixed-size `float32` tensors, channels-first board plus a global vector, written for a
  map size fixed per env at construction.
- Packaging: maturin, crate `sim-py`, pytest tests; vectorization stays in Python and is milestone 10.

## In scope

1. New crate `crates/sim-env` (pure Rust, depends on `sim-core`, `bots`): observation encoding, the
   factored action space (encode, decode, masks), reward, and the `Env` type.
2. New crate `crates/sim-py` (PyO3, depends on `sim-env`): Python module `polytopia_sim` exposing `Env`.
3. `python/` directory: `pyproject.toml` (maturin), `tests/` (pytest), `bench/random_agent.py`.
4. `scripts/py_check.sh` and `scripts/py_gate.sh`; `scripts/check.sh` calls `py_check.sh` when `maturin`
   and `python3` are on the PATH and prints `SKIPPED python tests (maturin not found)` otherwise.

## Out of scope (do not build)

Vectorized envs, PPO or any training code, library choice (milestone 10), multi-threading,
self-play bookkeeping, a flat-index action encoding, observation of anything beyond `PlayerView`,
zero-copy buffer tricks (allowed later if the benchmark says they matter), GPU anything.
New dependencies approved by this brief and no others: Rust `pyo3` and `numpy` (the rust-numpy crate)
for `sim-py` only; Python `maturin`, `numpy`, `pytest` (dev). Pin versions and record them.

## Public API (fixed so tests can import it)

```rust
// sim_env::obs  -- reads ONLY a PlayerView, never a State
pub struct ObsSpec { pub height: usize, pub width: usize, pub channels: usize, pub globals: usize,
                     pub channel_names: Vec<String>, pub global_names: Vec<String> }
pub fn obs_spec(size: u8) -> ObsSpec;
pub fn encode_board(view: &PlayerView, out: &mut [f32]);     // len channels*height*width, channel-major (C,H,W); overwrites all of out
pub fn encode_globals(view: &PlayerView, out: &mut [f32]);   // len globals; overwrites all of out

// sim_env::actions
#[repr(u8)] pub enum ActionType { Move = 0, Attack, Capture, Recover, Promote, Train, ChooseReward, Site, Research, EndTurn }
pub const NUM_ACTION_TYPES: usize = 10;                      // Copy, Eq, Debug; ActionType::ALL: [ActionType; 10]
pub struct Factored { pub actor: usize, pub kind: ActionType, pub arg: usize }   // Copy, Eq, Debug
pub fn num_actors(size: u8) -> usize;                        // 3 * size*size + 1
pub fn arg_len(size: u8, kind: ActionType) -> usize;
pub fn encode_action(state: &State, action: &Action) -> Factored;           // state is the state BEFORE the action
pub fn decode_action(state: &State, f: &Factored) -> Result<Action, DecodeError>;   // identification only; legality is checked by Env::step
pub struct ActionMasks { /* built once from a legal-action list */ }
impl ActionMasks {
    pub fn new(state: &State, legal: &[Action]) -> ActionMasks;
    pub fn actor(&self) -> &[bool];                          // len num_actors
    pub fn kind(&self, actor: usize) -> [bool; NUM_ACTION_TYPES];
    pub fn arg(&self, actor: usize, kind: ActionType) -> &[bool];   // len arg_len(kind); empty for kinds without an argument
}

// sim_env::env
pub enum Opponent { TwoSeat, Bot { kind: BotKind, epsilon: f32 } }       // Bot kinds: Random, Rush, Economy
pub enum AgentSeat { Fixed(PlayerId), SeedParity }                       // SeedParity: seat = seed % 2
pub struct RewardConfig { pub terminal_weight: f32 /*1.0*/, pub shaping_weight: f32 /*0.0*/,
                          pub gamma: f32 /*1.0*/, pub potential_scale: f32 /*1000.0*/ }   // Default = the values shown
pub struct EnvConfig { pub map: MapParams, pub config: Config, pub opponent: Opponent, pub agent_seat: AgentSeat,
                       pub reward: RewardConfig, pub max_agent_actions: u32 /*5000*/ }
pub struct StepOut { pub reward: f32, pub rewards: [f32; 2], pub terminated: bool, pub truncated: bool }
pub enum EnvError { NotReset, Finished, Illegal, BadIndex, Decode(DecodeError) }
pub struct Env { /* private */ }
impl Env {
    pub fn new(cfg: EnvConfig) -> Env;
    pub fn reset(&mut self, seed: u64);
    pub fn state(&self) -> &State;                           // for tests and tools only; Python never sees it
    pub fn seed(&self) -> u64;           pub fn agent_seat(&self) -> PlayerId;
    pub fn observer(&self) -> PlayerId;                      // Bot mode: the agent seat. TwoSeat: state.current (the seat to act), or the last actor once terminal
    pub fn view(&self) -> PlayerView<'_>;                    // PlayerView::new(state, observer())
    pub fn legal_actions(&self) -> &[Action];                // empty when terminal
    pub fn masks(&self) -> &ActionMasks;
    pub fn step(&mut self, action: &Action) -> Result<StepOut, EnvError>;
    pub fn step_legal(&mut self, index: usize) -> Result<StepOut, EnvError>;   // index into legal_actions()
    pub fn step_factored(&mut self, f: &Factored) -> Result<StepOut, EnvError>;
    pub fn action_log(&self) -> &[(PlayerId, Action)];       // every applied action incl. the opponent's, in order
    pub fn is_terminal(&self) -> bool;
}
```

```python
# polytopia_sim (sim-py).  Arrays are numpy, freshly allocated per call.
Env(size=11, turn_cap=30, full_observability=False, opponent="rush"|"random"|"economy"|"none",
    opponent_epsilon=0.0, agent_seat=0|1|"seed_parity", terminal_weight=1.0, shaping_weight=0.0,
    gamma=1.0, potential_scale=1000.0, max_agent_actions=5000)    # other MapParams/Config fields keep their defaults
env.reset(seed) -> None
env.observation() -> (board: float32[C,H,W], globals: float32[G])
env.actor_mask() -> bool[3N+1];  env.kind_mask(actor) -> bool[10];  env.arg_mask(actor, kind) -> bool[arg_len]
env.legal_count() -> int;        env.step_legal(i) -> StepResult;   env.step_factored(actor, kind, arg) -> StepResult
StepResult = (reward: float, rewards: float32[2], terminated: bool, truncated: bool)
env.info() -> dict(seed, turn, current_player, observer, agent_seat, scores=(s0, s1), outcome=None|"win_0"|"win_1"|"draw", agent_actions)
env.replay_json() -> str          # milestone 7 replay log of the episode so far (incl. opponent actions)
env.spec() -> dict(height, width, channels, globals, channel_names, global_names, action_types, num_actors, arg_lens, unit_types, rewards, techs, site_actions)
```

`Env::new` asserts `size` is accepted by `MapParams`. `N = size * size`, tile index `= row * size + col`.
Errors in Python: `ValueError` for `Illegal`/`BadIndex`/`Decode`, `RuntimeError` for `NotReset`/`Finished`.
An illegal call changes nothing (the env state, rng, log and counters are untouched).

## Rules

### Observation (everything from `PlayerView`; an unexplored tile is all zeros, including the `explored` channel)

Board channels, in this order. A "group" is one channel per enum variant in declaration order (the data
enum's variants, named by their `snake_case` data names); `channel_names` lists them as
`"group.variant"` or `"name"`. Add an `ALL`/count helper to an enum if it lacks one.

| Group | Channels | Value |
| --- | --- | --- |
| `explored` | 1 | 1 if the viewer explored the tile |
| `terrain` | one per `Terrain` | one-hot; unexplored all zero |
| `resource` | one per `Resource` | one-hot, only if `VisibleTile.resource` is `Some` (tech-hidden crops/metal stay zero) |
| `improvement` | one per `Improvement` | one-hot |
| `village` | 1 | 1 if a village stands there |
| `territory` | 2 | `mine`, `enemy` (both zero if unowned) |
| `city` | 4 | `mine`, `enemy`, `level` (`min(level,10)/10`), `capital` |
| `my_unit` | one per `UnitType` | one-hot of own unit on the tile |
| `enemy_unit` | one per `UnitType` | one-hot of visible enemy unit |
| `health` | 1 | `health / max_health` of any unit on the tile |
| `phase` | 4 | one-hot `Fresh, Moved, Attacked, Done`, own units only |
| `veteran` | 1 | 1 for a veteran unit of either side |
| `can_promote` | 1 | own unit with a legal `Promote` |
| `can_harvest` | 1 | a legal `Site` harvest action (animal or fruit) on the tile |
| `can_build` | 1 | a legal `Site` action on the tile that is not a harvest |

Global vector, in this order (`global_names` lists them): `stars` (`min(stars,50)/50`), `income`
(`min(income,20)/20`), `turn` (`turn/turn_cap`), `pending_reward` (0/1), `my_score` and `opp_score`
(each `min(score,2000)/2000`), `score_diff` (`clamp((my - opp)/1000, -1, 1)`), `my_cities`
(`min(n,8)/8`), `my_units` (`min(n,16)/16`), then one 0/1 entry per `Tech` (researched). Opponent stars,
techs and income never appear. The observation is always from the observer's own perspective ("mine" =
the observer), so there is no seat leakage and nothing is mirrored. Under `full_observability` the same
encoding is used (explored all 1).

### Factored action space

- **Actor index** (`3N + 1` slots): unit actor `tile` (the tile of an own unit), city actor `N + tile`,
  site actor `2N + tile`, global actor `3N`.
- **Kinds and their actor:** `Move, Attack, Capture, Recover, Promote` use a unit actor; `Train,
  ChooseReward` a city actor; `Site` a site actor; `Research, EndTurn` the global actor.
- **Arguments:** `Move` destination tile index, `Attack` target tile index (`arg_len = N`); `Train` the
  `UnitType` index; `ChooseReward` the `Reward` index; `Site` the `SiteAction` index; `Research` the
  `Tech` index (enum indices are the enum's declaration order); `Capture, Recover, Promote, EndTurn`
  have `arg_len = 0` and take `arg = 0`.
- **Masks come from the legal list:** `actor[a]` is true iff some legal action has actor `a`;
  `kind(a)[k]` iff some legal action has actor `a` and kind `k`; `arg(a, k)[x]` iff the legal action
  with that actor, kind and argument exists. With a pending reward only `ChooseReward` actions of that
  city exist, so only that city actor is true. Terminal: all masks false.
- `encode_action` and `decode_action` are inverse on legal actions; `decode_action` returns
  `DecodeError` when the actor slot holds no own unit/city (for unit and city actors), when `kind` does
  not belong to the actor type, or when `arg` is out of range or nonzero for a no-argument kind.
- `step_legal(i)` indexes `legal_actions()` in canonical order (M3 to M6); `step_factored` decodes and
  then requires membership in the legal list.

### Env

- **reset(seed):** `new_game_from_seed(seed, &map, &config)`; choose the agent seat (`SeedParity`: `seed
  % 2`); clear counters and the log. In `Bot` mode, if the opponent is to move it plays now, so after
  `reset` it is always the agent's turn. In `TwoSeat` mode nothing is played.
- **step (Bot mode):** apply the agent's action; if the game is not terminal and it is now the
  opponent's turn, loop: the opponent bot chooses from `PlayerView::new(state, opponent)` with its own
  `bot_rng(seed, opponent)` (a single rng per seat for the whole episode, never re-seeded) and
  `make_bot(kind, epsilon)`, and applies it, until the agent is to move or the game is terminal. One
  agent action is one step regardless of how many opponent actions followed.
- **step (TwoSeat mode):** apply the action for `state.current`. `observer()` after the step is the new
  `state.current`.
- **Counters:** `agent_actions` counts the caller's actions. If it reaches `max_agent_actions` without a
  terminal state, `truncated = true` (and the episode is finished). Reaching the turn cap is `terminated`,
  not truncated.
- **After the end:** `step*` returns `Err(Finished)`; before any `reset` it returns `Err(NotReset)`.
- **Opponent diversity:** the opponent is a bot with the given epsilon; its draws never touch the game rng.

### Reward

For seat `s` with `me = score(s)`, `opp = score(1 - s)`: potential `phi_s(state) = (me - opp) /
potential_scale` for a non-terminal state and `0` for a terminal one. On each step, for each seat:
`rewards[s] = terminal_weight * t_s + shaping_weight * (gamma * phi_s(after) - phi_s(before))` where
`t_s` is `+1` if the game ended this step with `Win(s)`, `-1` if it ended with the other player
winning, `0` otherwise (draws included), and "before"/"after" are the states immediately before the
agent's action and immediately after the whole step (opponent turn included in Bot mode). `reward` is
`rewards[agent seat]` in Bot mode and `rewards[acting player]` in TwoSeat mode. With the default
weights only terminal steps are nonzero. Rewards are zero-sum at gamma 1.

### Replay

`action_log` records `(player, action)` for every applied action. `replay_json` returns the milestone 7
replay log for `(seed, map, config, actions)` that `sim-tools verify` accepts and that reproduces the
final state. Use `sim_tools::replay`; if `sim-tools` is not usable as a library dependency, stop and
add a QUESTIONS.md entry.

### Python layer

Pure conversion: no game logic, no caching of state beyond the `Env` handle. Arrays are C-contiguous
with the dtypes above. `spec()` must be consistent with `obs_spec` and the enum orders. No GIL
release is required.

## Clarifications to the spec (owner: confirm or overrule, then update the doc)

- **J1: crate split.** `sim-env` holds everything testable; `sim-py` has no Rust tests of its own. - Confirmed
- **J2: observation layout and normalizers** above are my choices (caps 10, 50, 20, 2000, 1000, 8, 16).
  Level and stars beyond the caps are clipped. Enemy health and phase: health shown, phase hidden
  (own units only). - Confirmed
- **J3: terminal observations** are still valid, from the observer's perspective, with all masks false. - Confirmed
- **J4: reward in Bot mode spans the opponent's turn,** terminal potential is 0, and the draw reward is
  0. Shaping with `gamma = 1` leaves the optimal policy unchanged. - Confirmed
- **J5: truncation** is a safety net (5000 agent actions) separate from the turn cap. - Confirmed
- **J6: actor slots** are three per tile plus one global (spec: "four actor slots: unit, city, site,
  global"). A unit and a city never share an actor slot because they use different slot families. - Confirmed
- **J7: Capture, Recover, Promote, EndTurn have no argument head.** `Site` has an enum head over
  `SiteAction`. - Confirmed
- **J8: no vectorization and no zero-copy yet;** milestone 10 decides after profiling. - Confirmed
- **J9: map size and observation shape are fixed per Env** (the default `size=11` is the spec's map). - Confirmed
- **J10: `SeedParity` seats** make evaluation over a seed range side-swapped automatically. - Confirmed

## Changes to earlier tests

None expected. `scripts/check.sh` gains the conditional Python step; the workspace `Cargo.toml` lists
the two new crates. `sim-py` builds without the `extension-module` feature under `cargo clippy/test
--workspace` (maturin enables it through `pyproject.toml`) so the Rust gate never needs a Python
interpreter to link. If `pyo3` macro output conflicts with the `unsafe` ban, stop and record it:
the owner decides whether `sim-py` alone gets an exception.

## Tests (phase A guidance)

Golden values are derived by hand from the rules above or from independent counting.

1. **Spec and layout:** `obs_spec(11)` channel/global counts equal the sums implied by the enum sizes;
   names are unique and in the documented order; `encode_*` overwrites the whole buffer (pre-filled
   garbage is gone).
2. **Board goldens:** a scripted small board with every row of the table exercised (own/enemy unit
   types, health fractions, phases, veteran, promotion-ready, city levels above and below the clip,
   capital, territory, village, improvements, harvest/build flags from real `legal_actions`, hidden
   crops/metal with and without Organization/Climbing, unexplored tiles all zero). Tile positions are
   asserted at exact `(channel, row, col)` indexes.
3. **Global goldens:** each entry on scripted states, incl. clipping at and beyond the caps, negative
   and clamped `score_diff`, opponent stars/techs changed with no effect.
4. **No-cheating (metamorphic):** reuse the milestone 8 idea: twins that differ only in hidden
   information (unexplored tile contents, opponent stars/techs/unit internals, rng state) give
   bit-identical `encode_board`, `encode_globals`, masks and `legal_actions`-independent outputs.
5. **Action encoding:** golden `Factored` values for each of the 10 kinds (including the global slot and
   the last tile); `encode_action`/`decode_action` round trip on every legal action in thousands of
   random playout states; decode errors for each failure case.
6. **Mask correctness (spec property tests):** across random playouts (all opponent kinds, including
   states with a pending reward and terminal states): every `true` path `(actor, kind, arg)` decodes to a
   legal action; every legal action is reached by exactly one true path; the set of decoded paths equals
   `legal_actions` as a set with no duplicates; with a pending reward only that city's actor is true;
   terminal states have all masks false; masks for no-argument kinds have length 0.
7. **Env behavior:** reset places the agent so it is to move (both seats, `SeedParity`); a Bot-mode
   step runs the opponent exactly until the agent is to move; an agent that mirrors the same bot as the
   opponent (same kind, epsilon 0, the agent's choices made through `step`) reproduces `bots::play_game`
   for the same seed and seat order, same outcome, scores, turn and action log; two-seat mode applies
   exactly one action per step and flips `observer`; `step_legal`, `step` and `step_factored` give the
   same result for the same action; illegal calls (`Illegal`, `BadIndex`, `Decode`, `Finished`,
   `NotReset`) leave everything unchanged (compare the full state, log and counters); truncation at a
   tiny `max_agent_actions`; terminal-by-cap is `terminated`.
8. **Reward:** hand-computed values for: win, loss, draw, shaping on a scripted score change (including
   `gamma < 1` and the zero terminal potential), the opponent's turn included in the delta, zero-sum
   at `gamma = 1`, the default config giving zeros until the last step, `terminal_weight` scaling.
9. **Determinism and replay:** same seed and same agent policy give identical observations, rewards and
   logs; different opponent epsilon streams do not change the game rng; `replay_json` verifies with
   the milestone 7 verifier and reproduces the final state, including the opponent's actions.
10. **Python (pytest, requires maturin):** shapes/dtypes/contiguity of every array; `spec()` matches the
    Rust spec; a random agent (`step_legal(rng.integers(legal_count))`) finishes 200 games for each
    opponent; the error types; seed determinism across two `Env` objects; `replay_json` round trips
    through `sim-tools verify` in a subprocess; factored random sampling (sample actor, kind, arg from
    masks) always steps without error for 50 games.

Mutants to aim for: board indexed (W,H) or channel-last; a channel off by one group; hidden crop/metal
shown; opponent score/stars leaked; unexplored tile nonzero; health fraction not divided by max;
`can_build`/`can_harvest` swapped; actor offsets for city/site swapped; global slot wrong; encode and
decode disagreeing for one kind; a mask built from the wrong actor; pending-reward mask too wide;
opponent's turn skipped or run past the agent's turn; opponent rng re-seeded each turn; opponent
epsilon drawn from the game rng; reward sign flipped for seat 1; shaping using the post-opponent
state wrongly or without the terminal zero; truncation counted on opponent actions; illegal action
mutating state; `SeedParity` inverted; `replay_json` missing the opponent's actions.

## Gate

`./scripts/check.sh` passes (Rust tests, plus the Python tests when `maturin` exists), and the owner has
run `./scripts/py_gate.sh`, which builds a release wheel into a throwaway venv, runs pytest, then
`python/bench/random_agent.py`: 2000 complete games (500 per opponent kind `random`, `rush`,
`economy`, and `none` driven by a random second seat) with no exception, every episode ending
`terminated`, and a printed table of agent steps/second and games/second per mode. The implementer
pastes that table; there is no speed threshold (milestone 10 decides whether it matters).

## Notes for the implementer

- Keep all logic in `sim-env`; `sim-py` should be a few hundred lines of glue.
- Build the legal list and masks once per step, not per query.
- `encode_*` take slices so a later zero-copy path needs no change in `sim-env`.
- No `HashMap`/`HashSet` where order can affect a result.
- If the tests contradict this brief or the spec, stop and add a QUESTIONS.md entry. If a layout detail
  here is impossible with the real enums, record it and choose the closest reading.
