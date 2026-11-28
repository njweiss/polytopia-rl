# Milestone 20: v2 observation, factored action space, Env/Python support, and damage preview (v2)

Sixth milestone of game v2 — wiring the completed v2 simulator (M15–M19) into the RL observation,
factored action space, `sim_env::Env`, PyO3 `polytopia_sim.Env`, and the Python RL training/evaluation
stack, including the spec's damage preview observation feature (`docs/SPEC.md` "Game v2 rules" ->
"Population loss, score and the damage preview").

**The hard constraint (as in M15–M19):** under `ruleset: v1` everything is byte-identical: `obs_hash(11)`
== `"d5f9b1033cfd55ca"`, `rules_hash()` == `"a8b52b91eae34503"`, `CHANNELS = 37`, `GLOBALS = 18`,
`NUM_ACTION_TYPES = 10`, `PyEnv.spec()` keys/values, board/global encoding, action masks, `Env` stepping,
and every v1 Rust, Python, and RL test & checkpoint pass unchanged.

## Goal

1. Extend `PlayerView` (`sim_core::view`) with explored v2 tile state (`VisibleTile`), the viewer's own
   v2 counters/monument sets, and `attack_preview(unit, target)` (using the defender's true defense
   bonus and `Stiff`/range retaliation rules).
2. Define the `V2` observation layout in `sim_env::obs` (**64** board channels = 37 v1 prefix + 27 v2
   channels including **6** damage-preview channels; **33** globals = 18 v1 prefix + 15 v2 globals) and
   `obs_hash_for(size, ruleset)` in `sim_env::hash`.
3. Extend the factored action space in `sim_env::actions` to all **15** v2 `ActionType`s across the
   existing 4 actor families (`3 * size * size + 1` actors — including neutral roads in family 2!),
   supporting lossless `encode_action`, `decode_action`, and `ActionMasks` under both `V1` and `V2`.
4. Enable `Ruleset::V2` in `sim_env::env::Env`, `sim_py::PyEnv`, and `python/polytopia_rl/` (`PPOConfig`,
   `vec_env`, `train`, `checkpoint`, `suite`) so `PolicyNet`, `SyncVecEnv`, `InterleavedVecEnv`, PPO
   updates, evaluation, and checkpointing run end-to-end on `v2` games.

## Decisions already made (owner)

- Damage preview is an observation feature (not an action): for each attack the observer could make,
  attack damage and retaliation from the combat formula using the defender's true defense bonus — a
  deliberate, documented exception to hidden techs that bumps `obs_hash` (`docs/SPEC.md` line 686).
- Road building is a site action on the target tile, including neutral explored unoccupied tiles
  (`docs/SPEC.md` line 640).
- `v1` stays runnable and byte-identical behind `ruleset: v1` (`obs_hash` and `rules_hash` unchanged);
  `Env` lifts its M15 `V2` refusal in M20.

## In scope

1. `sim_core::view`:
   - `VisibleTile`: add `pub road: bool`, `pub building: Option<Building>`, `pub building_level: u8`,
     `pub lighthouse: bool`, `pub monument: Option<Monument>` (populated for explored tiles only; always
     default under `V1`).
   - `PlayerView`: add viewer-only getters `kills(&self) -> u32`, `lighthouses_discovered(&self) -> u8`,
     `monuments_unlocked(&self) -> MonumentSet`, `monuments_placed(&self) -> MonumentSet`.
   - `AttackPreview` struct and `PlayerView::attack_preview(&self, unit: UnitId, target: Pos) -> Option<AttackPreview>`.
2. `sim_env::obs`:
   - Keep `CHANNELS = 37`, `GLOBALS = 18`, `obs_spec(size)` unchanged for `V1`.
   - Add `V2_CHANNELS = 64`, `V2_GLOBALS = 33`, `obs_spec_for(size, ruleset)`.
   - `encode_board`, `encode_board_with_legal`, `encode_globals`: dispatch on `view.ruleset()` (`V1`:
     exact 37/18 encoding; `V2`: 37 v1 prefix + 27 v2 channels including 6 damage-preview channels, and
     18 v1 prefix + 15 v2 globals).
3. `sim_env::actions`:
   - Append 5 `ActionType` variants: `Disband = 10`, `BuildRoad = 11`, `BuildBuilding = 12`,
     `BuildMonument = 13`, `Destroy = 14`.
   - Keep `NUM_ACTION_TYPES = 10`, `ActionType::ALL` (10), `arg_len(size, kind)`, and
     `ActionMasks::kind(actor) -> [bool; 10]` unchanged for `V1`.
   - Add `V2_NUM_ACTION_TYPES = 15`, `action_types(ruleset) -> &'static [ActionType]`,
     `arg_len_for(size, kind, ruleset) -> usize`, and `ActionMasks::kind_slice(&self, actor) -> &[bool]`.
   - Update `encode_action`, `decode_action`, and `ActionMasks::new` to support all 15 v2 action types
     and v2 arg widths while rejecting v2 types/args under `V1`.
4. `sim_env::hash`:
   - Keep `obs_canonical(size)` and `obs_hash(size)` unchanged (`"d5f9b1033cfd55ca"` at size 11).
   - Add `obs_canonical_for(size, ruleset)` and `obs_hash_for(size, ruleset)`.
5. `sim_env::env`:
   - Remove the M15 `V2` refusal in `Env::new`; add `pub fn ruleset(&self) -> Ruleset`.
6. `sim_py`:
   - Remove the M15 `V2` refusal in `PyEnv::new`; wire `ruleset` into `observation`, `kind_mask`
     (`kind_slice`), `arg_mask`, `step_factored`, and `spec` (keeping `V1` `spec()` 100% unchanged).
7. `python/polytopia_rl/`:
   - `config.py`: add `ruleset: str = "v1"` (validated `"v1"` | `"v2"`).
   - `vec_env.py`, `train.py`, `checkpoint.py`, `suite.py`: pass `ruleset` through `env_kwargs_of` and
     `Env(size=..., ruleset=...).spec()` calls.

## Out of scope

v1 → v2 weight transfer / distillation experiments and long GPU v2 training runs (M21); `PolicyNet`
architecture changes (the existing dynamic `PolicyNet` already handles arbitrary `channels`, `globals`,
`action_types`, and `arg_lens` from `Env.spec()`); water/naval units and ruins (v3).

## Public API (fixed so tests can import it)

```rust
// sim_core::view
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttackPreview {
    /// Actual HP lost by the defender (capped at defender's current health).
    pub dealt: u32,
    /// Actual HP lost by the attacker from retaliation (0 if defender dies, out of range, or Stiff).
    pub retaliation: u32,
    pub defender_dies: bool,
    pub attacker_dies: bool,
    /// Defender's true defense bonus (1.0, 1.5, or 4.0).
    pub defense_bonus: f32,
}

pub struct VisibleTile {
    pub terrain: Terrain,
    pub resource: Option<Resource>,
    pub improvement: Option<Improvement>,
    pub village: bool,
    pub city: Option<CityView>,
    pub territory: Option<PlayerId>,
    pub unit: Option<UnitView>,
    pub road: bool,
    pub building: Option<Building>,
    pub building_level: u8,
    pub lighthouse: bool,
    pub monument: Option<Monument>,
}

impl<'a> PlayerView<'a> {
    pub fn kills(&self) -> u32;
    pub fn lighthouses_discovered(&self) -> u8;
    pub fn monuments_unlocked(&self) -> MonumentSet;
    pub fn monuments_placed(&self) -> MonumentSet;
    pub fn attack_preview(&self, unit: UnitId, target: Pos) -> Option<AttackPreview>;
}

// sim_env::obs
pub const CHANNELS: usize = 37;      // V1 (unchanged)
pub const GLOBALS: usize = 18;       // V1 (unchanged)
pub const V2_CHANNELS: usize = 64;   // V2
pub const V2_GLOBALS: usize = 33;    // V2
pub fn obs_spec(size: u8) -> ObsSpec;                              // V1 (unchanged)
pub fn obs_spec_for(size: u8, ruleset: Ruleset) -> ObsSpec;        // V1 == obs_spec(size)

// sim_env::actions
#[repr(u8)]
pub enum ActionType {
    Move = 0, Attack, Capture, Recover, Promote,
    Train, ChooseReward, Site, Research, EndTurn,
    Disband, BuildRoad, BuildBuilding, BuildMonument, Destroy,
}
pub const NUM_ACTION_TYPES: usize = 10;     // V1 (unchanged)
pub const V2_NUM_ACTION_TYPES: usize = 15;  // V2
pub fn action_types(ruleset: Ruleset) -> &'static [ActionType];
pub fn arg_len(size: u8, kind: ActionType) -> usize;                           // V1 (unchanged for v1 kinds)
pub fn arg_len_for(size: u8, kind: ActionType, ruleset: Ruleset) -> usize;     // V1/V2
impl ActionMasks {
    pub fn kind(&self, actor: usize) -> [bool; NUM_ACTION_TYPES];   // V1 compatibility (first 10)
    pub fn kind_slice(&self, actor: usize) -> &[bool];              // len 10 (V1) or 15 (V2)
}

// sim_env::hash
pub fn obs_canonical(size: u8) -> String;                             // V1 (unchanged)
pub fn obs_hash(size: u8) -> String;                                  // V1 (unchanged)
pub fn obs_canonical_for(size: u8, ruleset: Ruleset) -> String;       // V1 == obs_canonical(size)
pub fn obs_hash_for(size: u8, ruleset: Ruleset) -> String;            // V1 == obs_hash(size)

// sim_env::env
impl Env {
    pub fn ruleset(&self) -> Ruleset;
}
```

## Rules

### `PlayerView::attack_preview` (O4)

- Returns `Some(AttackPreview)` iff:
  - `self.state.outcome.is_none()` and `self.state.current == self.player`,
  - `unit` is alive and `u.owner == self.player`,
  - `target` is in-bounds and `self.is_explored(target)`,
  - `target` holds a living enemy unit `d` (`d.owner != self.player`),
  - `chebyshev(u.pos, target) <= state.tables.unit(u.kind).range` (and `target != u.pos`).
  Otherwise returns `None`.
- Computes combat identically to `sim_core::actions::apply_attack`:
  - `bonus = defense_bonus(state, defender_id)` (defender's **true** defense bonus: 1.0, 1.5, or 4.0),
  - `can_retaliate = chebyshev(u.pos, target) <= sd.range && !sd.skills.contains(&Skill::Stiff)`,
  - `outcome = resolve_attack(&ca, &cd, bonus, can_retaliate)`,
  - `dealt = d.health - outcome.defender_health`,
  - `retaliation = u.health - outcome.attacker_health`,
  - `defender_dies = outcome.defender_health == 0`, `attacker_dies = outcome.attacker_health == 0`.

### V2 Observation Layout (O1, O2)

Under `Ruleset::V1`, `encode_board_with_legal` and `encode_globals` are 100% unchanged (`37` channels,
`18` globals).

Under `Ruleset::V2`, channels `0..37` and globals `0..18` are the **exact** v1 layout (with v1 unit
types `0..5` encoded on channels `18..23` / `23..28`, v2 unit types `5..8` encoded on the appended
channels `37..43`, `health`/`phase`/`veteran` encoded for all 8 unit types, and `can_build` remaining
non-harvest `Action::Site` actions including `BurnForest`), followed by:

#### Appended V2 Board Channels (`37..64`, 27 channels -> `V2_CHANNELS = 64`)

| Offset | Count | Channel name(s) | Value |
| --- | --- | --- | --- |
| 37 | 3 | `my_unit.swordsman`, `my_unit.catapult`, `my_unit.knight` | binary (`1.0` if own unit of kind on tile) |
| 40 | 3 | `enemy_unit.swordsman`, `enemy_unit.catapult`, `enemy_unit.knight` | binary (`1.0` if visible enemy unit of kind on tile) |
| 43 | 1 | `road` | binary (`1.0` if explored tile has `road == true`) |
| 44 | 1 | `city.connected` | binary (`1.0` if tile holds an **own** city with `connected_to.is_some()`; never set on enemy cities so fogged enemy roads never leak) |
| 45 | 4 | `building.sawmill`, `building.windmill`, `building.forge`, `building.market` | binary one-hot on explored tiles |
| 49 | 1 | `building.level` | `t.building_level.min(8) as f32 / 8.0` on explored tiles (`0.0` if no building) |
| 50 | 1 | `lighthouse` | binary (`1.0` if explored tile has `lighthouse == true`; `0.0` while unexplored so unexplored tiles stay all-zero) |
| 51 | 3 | `monument.park_of_fortune`, `monument.gate_of_power`, `monument.eye_of_god` | binary one-hot on explored tiles |
| 54 | 1 | `can_build_road` | binary (`1.0` if `Action::BuildRoad { at }` in `legal`) |
| 55 | 1 | `can_build_building` | binary (`1.0` if any `Action::BuildBuilding { at, .. }` in `legal`) |
| 56 | 1 | `can_build_monument` | binary (`1.0` if any `Action::BuildMonument { at, .. }` in `legal`) |
| 57 | 1 | `can_destroy` | binary (`1.0` if `Action::Destroy { at }` in `legal`) |
| 58 | 1 | `preview.target_can_kill` | on `target`, `1.0` if any legal `Action::Attack { unit, target }` has `defender_dies`, else `0.0` |
| 59 | 1 | `preview.target_max_dmg` | on `target`, `max(p.dealt as f32 / d.health as f32)` across legal attacks targeting `target` (`0.0` if none) |
| 60 | 1 | `preview.target_min_retal` | on `target`, `min(p.retaliation as f32 / a.health as f32)` across legal attacks targeting `target` (`0.0` if none) |
| 61 | 1 | `preview.target_def_bonus` | on `target`, `p.defense_bonus / 4.0` if at least one legal attack targets `target`, else `0.0` |
| 62 | 1 | `preview.attacker_can_kill` | on `a.pos`, `1.0` if `a` has any legal attack with `defender_dies`, else `0.0` |
| 63 | 1 | `preview.attacker_max_dmg` | on `a.pos`, `max(p.dealt as f32 / d.health as f32)` across `a`'s legal attacks (`0.0` if none) |

All 64 channels are bounded in `[0.0, 1.0]` and `0.0` on unexplored tiles. When `legal` is empty
(terminal or opponent's turn), all action-hint and damage-preview channels are `0.0`.

#### Appended V2 Global Features (`18..33`, 15 globals -> `V2_GLOBALS = 33`)

| Offset | Count | Global name(s) | Value |
| --- | --- | --- | --- |
| 18 | 7 | `tech.roads`, `tech.free_spirit`, `tech.trade`, `tech.chivalry`, `tech.mathematics`, `tech.construction`, `tech.smithery` | binary |
| 25 | 1 | `my_kills` | `view.kills().min(10) as f32 / 10.0` |
| 26 | 1 | `my_lighthouses` | `view.lighthouses_discovered().count_ones() as f32 / 4.0` |
| 27 | 3 | `monument_unlocked.park_of_fortune`, `monument_unlocked.gate_of_power`, `monument_unlocked.eye_of_god` | binary |
| 30 | 3 | `monument_placed.park_of_fortune`, `monument_placed.gate_of_power`, `monument_placed.eye_of_god` | binary |

### Factored Action Space (`V1` vs `V2`, O3)

Actor slots remain `num_actors(size) = 3 * N + 1` (`N = size * size`):
- Family 0 (`0..N`, unit at tile): `Move` (arg `N`), `Attack` (arg `N`), `Capture` (0), `Recover` (0),
  `Promote` (0), and under `V2`: `Disband` (0).
- Family 1 (`N..2N`, city at tile): `Train` (arg `5` in `V1`, `8` in `V2`), `ChooseReward` (arg `8`).
- Family 2 (`2N..3N`, site at tile — owned or neutral): `Site` (arg `6` in `V1`, `7` in `V2`), and under
  `V2`: `BuildRoad` (0), `BuildBuilding` (arg `4`: `ALL_BUILDINGS`), `BuildMonument` (arg `3`:
  `ALL_MONUMENTS`), `Destroy` (0).
- Family 3 (`3N`, global): `Research` (arg `9` in `V1`, `16` in `V2`), `EndTurn` (0).

`decode_action(state, f)`:
- Rejects any `ActionType` not in `action_types(state.config.ruleset)` with `DecodeError::WrongKind`.
- Rejects any `f.arg >= arg_len_for(size, f.kind, state.config.ruleset).max(1)` with `DecodeError::BadArg`.
- Decodes `Train` via `units(ruleset)[f.arg]`, `Site` via `site_actions(ruleset)[f.arg]`, `Research` via
  `techs(ruleset)[f.arg]`, `BuildBuilding` via `ALL_BUILDINGS[f.arg]`, `BuildMonument` via
  `ALL_MONUMENTS[f.arg]`.

### Hashes & `PyEnv.spec()` (O5)

- `obs_canonical_for(size, Ruleset::V1)` == `obs_canonical(size)` -> `obs_hash_for(11, V1)` ==
  `"d5f9b1033cfd55ca"`.
- `obs_canonical_for(size, Ruleset::V2)`: flat `\n`-joined list of: `size`, `spec.channel_names` (64),
  `spec.global_names` (33), `action_types(V2)` names (15), `arg_len_for(size, k, V2)` (15),
  `units(V2)` names (8), `ALL_REWARDS` names (8), `techs(V2)` names (16), `site_actions(V2)` names (7),
  `ALL_BUILDINGS` names (4), `ALL_MONUMENTS` names (3).
- `PyEnv.spec()` under `V1`: exact existing keys and values unchanged.
- `PyEnv.spec()` under `V2`: same keys updated to v2 (`channels: 64`, `globals: 33`, 15 `action_types`,
  v2 `arg_lens`, 8 `unit_types`, 8 `rewards`, 16 `techs`, 7 `site_actions`, `obs_hash: obs_hash_for(size, V2)`,
  `rules_hash: rules_hash_for(V2)`) plus `"buildings"`, `"monuments"`, and `"ruleset": "v2"`.

### Python RL Wiring (O6)

- `PPOConfig`: add `ruleset: str = "v1"`; `PPOConfig.from_json` raises `ValueError` if `ruleset` is not
  `"v1"` or `"v2"`.
- `env_kwargs_of(cfg)`: include `ruleset=getattr(cfg, "ruleset", "v1")`.
- `train.py` (`_spec_of`), `checkpoint.py` (`load_net`), and `suite.py`: pass `ruleset` when fetching
  `polytopia_sim.Env(...).spec()`.

## Clarifications (owner: confirm or overrule)

- **O1 (V2 observation prefix layout):** Keep all 37 v1 board channels (`0..37`) and all 18 v1 globals
  (`0..18`) in exact v1 order, appending 27 v2 board channels (`V2_CHANNELS = 64`) and 15 v2 globals
  (`V2_GLOBALS = 33`) so M21 v1 -> v2 warm-start is a clean prefix copy. — Confirmed (owner, 2026-11-27)
- **O2 (Damage preview encoding):** Encode damage preview as 6 bounded `[0, 1]` spatial board channels
  (`preview.target_{can_kill,max_dmg,min_retal,def_bonus}` on target tiles and
  `preview.attacker_{can_kill,max_dmg}` on attacker tiles) via `PlayerView::attack_preview`, keeping the
  `(board, globals)` tuple and `PolicyNet` architecture unchanged. — Confirmed (owner, 2026-11-27)
- **O3 (V2 factored action space):** Append 5 `ActionType` variants (`Disband = 10`, `BuildRoad = 11`,
  `BuildBuilding = 12`, `BuildMonument = 13`, `Destroy = 14`) into the existing 4 actor families
  (`num_actors = 3N + 1`, with neutral `BuildRoad` in family 2); keep `ActionType::ALL` (10),
  `NUM_ACTION_TYPES = 10`, and `ActionMasks::kind` unchanged for v1 and add `action_types(ruleset)`,
  `arg_len_for(size, kind, ruleset)`, and `ActionMasks::kind_slice(actor)`. — Confirmed (owner, 2026-11-27)
- **O4 (`PlayerView` v2 visibility):** Add `road`, `building`, `building_level`, `lighthouse`, `monument`
  to `VisibleTile` (explored tiles only), add viewer-only `kills`, `lighthouses_discovered`,
  `monuments_unlocked`, `monuments_placed`, and `attack_preview` on `PlayerView`; encode `city.connected`
  for own cities only (never leaking enemy fogged roads). — Confirmed (owner, 2026-11-27)
- **O5 (`obs_hash_for` & `PyEnv.spec()`):** Keep `V1` `obs_hash(11)` (`d5f9b1033cfd55ca`) and `V1`
  `PyEnv.spec()` 100% byte-identical; add `obs_canonical_for`/`obs_hash_for` and return v2 spec +
  `buildings`/`monuments`/`ruleset` under `V2`. — Confirmed (owner, 2026-11-27)
- **O6 (Python RL `ruleset` plumbing):** Add `PPOConfig.ruleset: str = "v1"` and wire it through
  `vec_env.py`, `train.py`, `checkpoint.py`, and `suite.py` so PPO training, self-play, evaluation, and
  checkpoints work end-to-end on `v2` right in M20. — Confirmed (owner, 2026-11-27)
- **O7 (Allowed earlier test edits):** Update `VisibleTile` literals in `crates/sim-core/tests/view.rs`,
  add 5 `ActionType` exhaustiveness arms in `crates/sim-env/tests/common/mod.rs`, and replace the two
  temporary M15 "refuses v2 until milestone 20" tests in `crates/sim-env/tests/v2_hash.rs` and
  `python/tests/rl/test_ruleset.py` with assertions that `v2` is now accepted. Zero v1 expectations change. — Confirmed (owner, 2026-11-27)

## Changes to earlier tests

Allowed only for the following (no v1 expected value may change; no test may be removed or weakened; list
every edit in the phase A commit message):

1. `crates/sim-core/tests/view.rs`: add `road: false, building: None, building_level: 0, lighthouse: false,
   monument: None` to `VisibleTile` struct literals (O4/O7).
2. `crates/sim-env/tests/common/mod.rs`: add exhaustiveness arms for `Disband`, `BuildRoad`,
   `BuildBuilding`, `BuildMonument`, `Destroy` in the v1 helper matches (O3/O7).
3. `crates/sim-env/tests/v2_hash.rs:166`: update `env_refuses_v2_naming_milestone_20` to assert `Env::new`
   now accepts `Ruleset::V2` and steps cleanly (all `rules_hash_for` tests untouched, O7).
4. `python/tests/rl/test_ruleset.py:13`: update `test_v2_is_refused_with_a_value_error_naming_milestone_20`
   to assert `ps.Env(ruleset="v2")` now constructs and exposes the v2 spec & hashes (all v1 tests untouched, O7).

## Tests (phase A guidance)

1. **v1 identity:** all existing v1 tests across `sim-core`, `bots`, `sim-tools`, `sim-env`, `sim-py`, and
   `python/tests/rl/` pass unchanged; `obs_hash(11) == "d5f9b1033cfd55ca"`,
   `rules_hash() == "a8b52b91eae34503"`, v1 replay fixture verifies unchanged.
2. **`PlayerView` v2 & `attack_preview` (`crates/sim-core/tests/v2_view.rs`):**
   - `VisibleTile` exposes `road`, `building`, `building_level`, `lighthouse`, `monument` on explored
     tiles and returns `None` on unexplored tiles.
   - `kills`, `lighthouses_discovered`, `monuments_unlocked`, `monuments_placed` return the viewer's own
     state (never the opponent's).
   - `attack_preview` matches `apply_attack` HP deltas exactly across: plain combat, hidden enemy Archery
     (forest 1.5x) & Climbing (mountain 1.5x), city (1.5x) & walled city (4.0x) with/without Fortify,
     Knight 3.5 attack, Catapult `Stiff` & `defense == 0`, out-of-range defender (0 retaliation), lethal
     attack (`defender_dies == true`, `retaliation == 0`); returns `None` on unexplored/out-of-range/wrong
     turn/terminal/friendly target.
3. **V2 observation, damage preview, factored actions, masks, hashes, and no-cheating (`crates/sim-env/tests/v2_env.rs`):**
   - `obs_spec_for(size, V2)`: 64 channels, 33 globals, exact names, exact v1 prefix (`0..37`, `0..18`).
   - `encode_board` / `encode_globals` under `V2`: verify every single one of the 27 new board channels
     and 15 new global features on scripted boards (including `city.connected` own-only, neutral
     `can_build_road`, multi-attacker damage preview aggregation `max_dmg`/`min_retal`/`can_kill`, hidden
     Archery/Climbing reflected in `preview.target_def_bonus` ONLY when targetable, and all-zero preview
     when opponent's turn or terminal).
   - Factored actions & masks under `V2`: round-trip `encode_action` <-> `decode_action` for every one of
     the 15 `ActionType`s (including neutral `BuildRoad`, `BurnForest`, `BuildBuilding` x4, `BuildMonument`
     x3, `Disband`, `Destroy`, v2 `Train` x3, v2 `Research` x7); mask bijection (`ActionMasks` matches
     `legal_actions` 1-to-1 across multi-seed v2 playouts); `V1` rejects all v2 action types & arg indices.
   - `obs_hash_for(size, V2)`: deterministic, matches independent canonical FNV-1a, distinct from `V1`,
     changes when any ingredient changes; `V1` unchanged.
   - `Env` under `V2`: multi-seed games in `TwoSeat` and `Bot` modes (`random`, `rush`, `economy`),
     `step_factored`, `bot_action`, `replay_json` verified by `sim_tools::replay::replay`, determinism.
   - No-cheating under `V2`: twin states differing in hidden v2 state (opponent `kills`,
     `lighthouses_discovered`, `monuments_unlocked`, `monuments_placed`, hidden v2 techs that do NOT
     change defense bonus of any currently attackable enemy, unexplored roads/buildings/monuments/lighthouses,
     enemy city `connected_to`) yield identical v2 board, globals, and masks; AND a dedicated test
     confirming that hidden enemy `Archery` on a **currently attackable** forest defender *does* change the
     damage preview channels (and does *not* when that enemy is out of attack range!).
4. **Python `polytopia_sim` & RL end-to-end (`python/tests/test_env_v2.py` and `python/tests/rl/test_v2_rl.py`):**
   - `ps.Env(ruleset="v2")` shapes (`(64, H, W)`, `(33,)`, `kind_mask` `(15,)`), `spec()` keys & hashes,
     factored sampling finishing 50 v2 games across all opponents, `bot_action`, verifiable v2 `replay_json`,
     and error checks (`kind == 15` raises `ValueError`; `kind == 10` still raises `ValueError` in `v1`).
   - `PPOConfig(ruleset="v2")` validation (`"v3"` raises `ValueError`), `PolicyNet(v2_spec)` forward/act/
     evaluate, `SyncVecEnv` & `InterleavedVecEnv` on `v2`, short end-to-end `train()` + `evaluate()` +
     `save_checkpoint`/`load_net` round trip on `ruleset="v2"`, and `SpecMismatch` when loading a `v1`
     checkpoint against a `v2` spec (and vice versa).

Mutants to aim for: `V1` `obs_hash` or `spec()` changed; `encode_board` leaking unexplored lighthouse/road/
building/monument or enemy `city.connected`; `attack_preview` using `1.0` instead of true `defense_bonus`,
forgetting `Stiff` or defender death (`retaliation != 0` on kill), or leaking defense bonus on non-attackable
enemies; `target_min_retal` defaulting to `1.0` or `+inf` on non-targeted tiles instead of `0.0`; `V2`
`ActionMasks` dropping neutral `BuildRoad`, `BuildMonument`, `BuildBuilding`, `Destroy`, `Disband`, or
`BurnForest`; `decode_action` accepting v2 action types/args under `V1`; `PolicyNet` or `SyncVecEnv`
hardcoding 10 kinds or v1 `obs_hash`; `checkpoint.load_net` fetching the v1 spec for a v2 checkpoint.

## Gate

`pixi run check` exits 0 (fmt, clippy `-D warnings`, all Rust, Python, and RL tests green). `rules_hash()`
== `"a8b52b91eae34503"`, `obs_hash(11)` == `"d5f9b1033cfd55ca"`, and `polytopia verify` accepts the v1
replay fixture unchanged. Architecture milestone: a `/review 20` phase is recommended.

## Notes for the implementer

- Keep `ALL_UNITS` (5), `ALL_TECHS` (9), `ALL_SITE_ACTIONS` (6), `ActionType::ALL` (10), `CHANNELS` (37),
  and `GLOBALS` (18) untouched; all v2 extensions sit alongside them via `*_for(..., ruleset)` helpers.
- `sim_env::obs` must continue reading only `&PlayerView` (never `&State`).
- Keep `PolicyNet` (`python/polytopia_rl/model.py`) unchanged unless a genuine bug appears — it is already
  driven dynamically by `Env.spec()`.
- If a test contradicts this brief or the spec, stop and add a `docs/QUESTIONS.md` entry (next number
  **Q56**).
