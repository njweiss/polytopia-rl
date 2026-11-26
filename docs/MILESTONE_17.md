# Milestone 17: economy buildings, v2 tile actions and population loss (v2)

Third milestone of game v2. The rules are in `docs/SPEC.md` "Game v2 rules" ("Buildings and tile actions",
"Population loss, score and the damage preview"). Under `ruleset: v2` this milestone turns on the four
economy buildings (`Sawmill`, `Windmill`, `Forge`, `Market`), `BurnForest`, `Disband`, `Destroy`, and
signed city population (`i32`) with population loss and non-negative score surplus. It also folds in the
carry-over items from M16 (Q49 scripted test, `road_step_cost` load validation, unique-capital debug assert)
and two `sim-core` hot-path cleanups ($O(1)$ `State::city`, single-BFS gated `update_connections`).

**The hard constraint (as in M15–M16):** under `ruleset: v1` everything is byte-identical: legal actions,
`apply`, digests, replay files, `Config` JSON, `rules_hash`, `obs_hash`, observations, bots. Every v1
identity test and replay fixture passes unchanged.

## Goal

1. Under `V2`, a player with the matching tech can build `Sawmill`, `Windmill`, `Forge`, and `Market` (5
   stars each, one per city) on eligible field tiles next to matching friendly improvements/buildings;
   levels, city population, and Market income stay automatically synced as neighbors are added, destroyed,
   or change owner on city capture.
2. Under `V2`, `BurnForest` (Construction, 3 stars), `Disband` (Free Spirit, half-cost refund, any phase),
   and `Destroy` (Chivalry, removes any improvement or building except roads, reversing its population)
   are legal and work per spec.
3. `City.population` becomes `i32`: population loss can push population below the current level's threshold
   or below 0 without lowering city level; score surplus floors at 0.
4. M16 carry-overs and `sim-core` hot-path cleanups (`State::city` $O(1)$, capital-BFS gated
   `update_connections`) land cleanly without changing any v1 or M16 behavior.

## Decisions already made (owner)

- All building, tile-action, Disband, Destroy, and population-loss rules in `docs/SPEC.md` "Game v2 rules"
  (owner, 2026-11-01/02) and the M15 data tables (`data/v2/buildings.json`, `site_actions.json`,
  `rules.json`).
- `City.population` changes to a signed type in M17 (M15 V3, M16 R4).
- Roads cannot be destroyed; economy buildings give no score of their own (only via population); surplus
  population in score floors at 0.
- Super Unit disband value is 10 (refund 5); Disband is legal in any phase and is not a kill.
- Burn forest turns a forest with no improvement into a field with crop and forfeits any animal.

## In scope

1. `sim_core::state`: `City.population: i32`; `TileState.building: Option<Building>` and
   `TileState.building_level: u8` (always `None` / `0` under `V1`); $O(1)$ `State::city` / `city_mut`.
2. `sim_core::buildings` (new module): `building_level`, `market_income`, and internal `sync_buildings`.
3. `sim_core::actions`: `Action::BuildBuilding`, `Action::Disband`, `Action::Destroy`; `SiteAction::BurnForest`
   enabled under `V2`; legality, canonical ordering, `apply`, signed `level_up_check`, and gated post-apply
   `sync_buildings` + `update_connections`.
4. `sim_core::economy` and `sim_core::score`: `income` adds Market stars under `V2`; `score_breakdown`
   floors city surplus population at 0.
5. `sim_core::data` and `sim_core::roads`: load-time check in `Tables::from_json_v2` that
   `2.0 * road_step_cost` is a positive integer; single-BFS `update_connections` gated to
   road/city/territory-changing actions; `roads::lose` uses signed subtraction; `debug_assert!` for unique
   capital per player outside terminal states.
6. `sim_core::testing`: `StateBuilder::building(p, b)` helper (optional convenience, sets `building` and
   `building_level`).
7. `sim_tools::digest`, `sim_tools::render`, `sim_env::actions`, `bots`: v2 digest covers `building` and
   `building_level` under `V2` only (v1 byte-identical); match arms for the 3 new `Action` variants.

## Out of scope

Monuments and lighthouses (M18); new units (`Swordsman`, `Catapult`, `Knight`), Stiff/Persist, and bot
strategies for v2 (M19); v2 observation, factored action slots, `Env` support, and damage preview (M20);
rendering buildings in the text/HTML viewer and exposing buildings on `PlayerView` (M19/M20).

## Public API (fixed so tests can import it)

```rust
// sim_core::data
pub const ALL_BUILDINGS: [Building; 4] = [
    Building::Sawmill,
    Building::Windmill,
    Building::Forge,
    Building::Market,
];  // promote existing private const to pub

// sim_core::state
pub struct City {
    ...,
    pub population: i32,   // was u32; always >= 0 under V1
    ...
}
pub struct TileState {
    ...,
    pub building: Option<Building>,  // always None under V1
    pub building_level: u8,          // synced level (0 when None or under V1)
}

// sim_core::actions (appended after BuildRoad, preserving existing discriminants)
pub enum Action {
    ...,
    BuildRoad { at: Pos },
    BuildBuilding { at: Pos, building: Building },
    Disband { unit: UnitId },
    Destroy { at: Pos },
}

// sim_core::buildings (new module)
pub fn building_level(state: &State, pos: Pos) -> u32
pub fn market_income(state: &State, pos: Pos) -> u32

// sim_core::testing
impl StateBuilder {
    pub fn building(self, p: Pos, building: Building) -> StateBuilder
    //   sets tile.building = Some(building); caller sets building_level via tile/build or lets tests
    //   call sync through apply (or builder computes level on build — see Rules below)
}
```

## Rules

### Signed population and population loss

- `City.population` is `i32`. Threshold for level `L` remains `cumulative_population_for_level(L)` (`u32`,
  compared as `i32`).
- `level_up_check(state, id)`: if `!c.pending_reward && c.population >= cumulative_population_for_level(c.level + 1) as i32`,
  increment `c.level` and set `c.pending_reward = true`. A city's level never decreases.
- `score_breakdown`: city surplus is `(c.population - cumulative_population_for_level(c.level) as i32).max(0) as u32`.
- `roads::lose` subtracts `amount as i32` directly (no longer panics if building losses pushed population
  low before a connection broke).

### Building levels and Market income (`sim_core::buildings`)

All constants (`cost`, `tech`, `adjacent`, `level_from`, `population_per_level`, `stars_per_level`,
`max_output`, `one_per_city`) are read from `state.tables.building(b)`, never hard-coded.

- **Adjacency:** 8 surrounding tiles (`grid::neighbors`). Only neighbors whose `owner_city` belongs to the
  **same player** as the building tile's `owner_city` count (cross-city friendly tiles count; enemy and
  neutral tiles do not; P2).
- **`building_level(state, pos)`:**
  - Returns `0` if `tile(pos).building` is `None` or `tile(pos).owner_city` is `None`.
  - For `LevelFrom::Count` (`Sawmill`, `Windmill`, `Forge`): number of matching friendly neighbors whose
    `improvement` matches a name in `info.adjacent`.
  - For `LevelFrom::Levels` (`Market`): sum over matching friendly neighbors whose `building` matches a
    name in `info.adjacent` of `building_level(state, q)`.
- **`market_income(state, pos)`:**
  - If `tile(pos).building` is `Some(b)` with `info.stars_per_level > 0`, returns
    `building_level(state, pos) * info.stars_per_level`, capped at `info.max_output` if `Some(cap)`.
    Otherwise `0`.
- **`income(state, player)`:** under `V2`, adds `market_income(state, pos)` for every tile owned by a city
  of `player` (under `V1`, unchanged).

### `BurnForest` (under `V2` only; never legal under `V1`)

- Uses existing `Action::Site { at, action: SiteAction::BurnForest }`, reading `cost` (3) and `tech`
  (`Construction`) from `tables.site_action(SiteAction::BurnForest)`.
- Legal when: ruleset is `V2`, no reward pending, `at` is owned by current player's city, no enemy unit on
  `at`, player has `Construction` and `stars >= 3`, and `t.terrain == Terrain::Forest && t.improvement.is_none() && t.building.is_none()`.
- Apply: deduct `cost` stars, set `t.terrain = Terrain::Field`, `t.resource = Some(Resource::Crop)` (any
  animal on the forest is overwritten/forfeited).
- `legal_actions` iterates `site_actions(state.config.ruleset)` per tile in row-major order, so under `V1`
  `BurnForest` is never checked or emitted.

### `BuildBuilding` (under `V2` only; never legal under `V1`)

- Legal when: ruleset is `V2`, no reward pending, player has `info.tech` and `stars >= info.cost`, `at` is
  in-bounds and owned by a city `c` of `state.current`, no enemy unit on `at` (own unit is fine),
  `t.terrain == Terrain::Field`, `t.city.is_none()`, `t.improvement.is_none()`, `t.building.is_none()`,
  `t.resource.is_none()` (P1), if `info.one_per_city` then no tile with `owner_city == Some(c)` already has
  `building == Some(building)`, and `at` has at least one 8-adjacent friendly tile matching `info.adjacent`
  (P2).
- Apply: deduct `info.cost` stars, set `t.building = Some(building)` and `t.building_level = 0`; the
  post-apply `sync_buildings` sets `t.building_level` to the computed level and credits
  `level * info.population_per_level` to `c.population`.

### `Disband` (under `V2` only; never legal under `V1`)

- Legal when: ruleset is `V2`, no reward pending, player has `v2_rules().disband_tech` (`FreeSpirit`), and
  `unit` is a living unit owned by `state.current` (any `Phase`: `Fresh`, `Moved`, `Attacked`, `Done`).
- Apply:
  - `value = stats.cost.unwrap_or(rules.super_unit_value)` (Warrior 2, Rider/Archer/Defender 3, Swordsman 5,
    Catapult/Knight 8, SuperUnit 10).
  - `refund = value / rules.disband_refund_divisor` (integer floor: 1, 1, 1, 1, 2, 4, 4, 5).
  - Add `refund` to owner's `stars`; clear `tiles[index(u.pos)].unit = None`; remove unit from `state.units`.
  - Not a kill (does not affect any kill counts).

### `Destroy` (under `V2` only; never legal under `V1`)

- Legal when: ruleset is `V2`, no reward pending, player has `v2_rules().destroy_tech` (`Chivalry`), `at`
  is in-bounds and owned by a city of `state.current`, no enemy unit on `at` (own unit is fine), and
  `t.improvement.is_some() || t.building.is_some()`. (Roads are never destroyed; cities/villages/terrain
  are never destroyed.)
- Apply (no cost, no star refund):
  - If `t.improvement` is `Some(imp)`: set `t.improvement = None` (terrain and resource stay, P3); subtract
    `tables.site_action(site_action_of(imp)).population as i32` from `owner_city.population` (LumberHut 1,
    Farm 2, Mine 2).
  - Else if `t.building` is `Some(b)`: subtract
    `(t.building_level as i32) * (tables.building(b).population_per_level as i32)` from
    `owner_city.population`; set `t.building = None`, `t.building_level = 0`.
  - Post-apply `sync_buildings` updates any neighboring buildings whose level changed.

### Post-apply sync (`sync_buildings` + `update_connections`) and `legal_actions` order

- **Canonical `legal_actions` order** (outside pending rewards and terminal states):
  1. Units by ascending id: Move (row-major), Attack (row-major), Capture, Recover, Promote, **Disband** (v2).
  2. Train per city by ascending id (`ALL_UNITS` order; v2 units wait for M19).
  3. Site actions row-major in `site_actions(ruleset)` order (`BurnForest` last on each tile under v2).
  4. `BuildRoad` row-major (v2).
  5. `BuildBuilding` row-major, then `ALL_BUILDINGS` order (`Sawmill`, `Windmill`, `Forge`, `Market`) (v2).
  6. `Destroy` row-major (v2).
  7. Research in `techs(ruleset)` order.
  8. `EndTurn`.
- **Post-apply sync under `V2` (when `state.outcome.is_none()`, Q48):**
  - Record `before: Vec<i32>` of city populations before post-apply syncs (note: `apply_site` already ran
    `level_up_check` on the site's own city, or all level-up checks run cleanly; `level_up_check` is
    idempotent when `pending_reward` is already set).
  - Run `sync_buildings` on actions that can affect building presence or friendly adjacency (`Site`,
    `BuildBuilding`, `Destroy`, `Capture`):
    1. Pass 1 (`LevelFrom::Count` buildings, row-major): compute `new_level = building_level(state, p) as u8`,
       apply `(new_level as i32 - old_level as i32) * pop_per_level as i32` to `owner_city.population`, set
       `tile.building_level = new_level`.
    2. Pass 2 (`LevelFrom::Levels` buildings, row-major): set `tile.building_level = building_level(state, p) as u8`.
  - Run `update_connections` on actions that can affect roads, cities, or territory (`BuildRoad`, `Capture`,
    `ChooseReward { reward: Reward::BorderGrowth, .. }`), implemented via one BFS from each player's capital
    instead of per-city DFS (exact same connection semantics as M16).
  - Run `level_up_check(state, cid)` in ascending `CityId` order for every city whose population increased
    during `sync_buildings` / `update_connections` (P4).
- **`StateBuilder::building(p, b)`:** sets `tile.building = Some(b)` without computing `building_level`
  immediately, OR sets `tile.building_level` on `build()`? Setting `tile.building = Some(b)` and computing
  `tile.building_level = building_level(&state, p) as u8` for every tile with a building inside
  `StateBuilder::build()` ensures scripted test boards start with consistent `building_level` without
  mutating hand-set `city.population`! (Tests can also override `tile.building_level` after `build()` if
  needed.)

### M16 carry-overs & hot-path cleanups

- **Load-time `road_step_cost` check (`sim_core::data`):** `Tables::from_json_v2` returns `Err(...)` if
  `2.0 * rules.road_step_cost` is not finite, not `> 0.0`, or has a non-zero fractional part.
- **Unique-capital debug assert (`sim_core::roads`):** `capital_of` checks
  `debug_assert!(state.outcome.is_some() || count <= 1)` per player.
- **`State::city(id)` / `city_mut(id)`:** index `self.cities[id.0 as usize]` directly (with
  `debug_assert_eq!(c.id, id)`).
- **Q49 scripted test:** verify a neutral road built by player 0 connects player 1's city, raising its
  level and setting `pending_reward = true` without blocking player 0's remaining turn; player 1 resolves
  it at the start of their turn; and if player 0 captures that city while `pending_reward` is true, player
  0 must choose the reward immediately.

### Digest (under `V2` only)

Each tile's record under `V2` appends, after the M16 `road` byte:
- `building`: `opt_u8(t.building.map(|b| b as u8))` (0 if `None`, else 1 then discriminant `0..=3`),
- `building_level`: `u8`.

`City.population` is encoded as 4 little-endian bytes (`to_le_bytes()`), which is bit-identical to `u32`
for every non-negative value, so **v1 digests are completely unchanged**.

### Hashes

`rules_hash_for(V1)` (`a8b52b91eae34503`), `rules_hash_for(V2)`, and `obs_hash(11)` (`d5f9b1033cfd55ca`)
are all unchanged (`data/v2/*.json` files do not change; `RULES_VERSION` is not bumped, per M16 R5).

## Clarifications (owner: confirm or overrule)

- **P1 (building placement on resource tiles):** Require `tile.resource.is_none()` to place a `Sawmill`,
  `Windmill`, `Forge`, or `Market` (plain empty field only — harvest fruit first, crops are for farms,
  matching real Polytopia). — Confirmed (owner, 2026-11-25)
- **P2 (friendly cross-city adjacency & Market placement):** Adjacent matching improvements/buildings in
  **any city owned by the same player** count (never enemy/neutral tiles), and placing a `Market` requires
  at least one adjacent friendly `Sawmill`, `Windmill`, or `Forge` (presence of the building, even if its
  level is currently 0). — Confirmed (owner, 2026-11-25)
- **P3 (Destroy keeps terrain & resource on improvements):** Destroying a `LumberHut`, `Farm`, or `Mine`
  removes `tile.improvement` and its population while leaving terrain (`Forest`/`Field`/`Mountain`) and
  underlying resource (`Animal`/`Crop`/`Metal`) intact (matching v1's "improvements do not consume
  resources"). — Confirmed (owner, 2026-11-25)
- **P4 (post-apply sync order):** On non-terminal v2 actions, run `sync_buildings` (on `Site` /
  `BuildBuilding` / `Destroy` / `Capture`), then `update_connections` (on `BuildRoad` / `Capture` /
  `BorderGrowth`), then `level_up_check` in ascending `CityId` order for every city whose population rose.
  — Confirmed (owner, 2026-11-25)
- **P5 (`Action` variants & `legal_actions` order):** Add `BuildBuilding`, `Disband`, `Destroy` to `Action`
  (with `BurnForest` using `Action::Site`), ordered as specified in Rules above. — Confirmed (owner, 2026-11-25)
- **P6 (allowed edits to earlier tests):** Approve the narrow edits listed under "Changes to earlier tests"
  below (M15 `ruleset.rs` v2 "nothing else unlocked" assertions for M17 actions, M15/M16 `v2_digest.rs` /
  `road_digest.rs` v2-only tile bytes, and type-only `i32` / `TileState` / `Action` exhaustiveness edits —
  zero v1 expectations changed). — Confirmed (owner, 2026-11-25)
- **P7 (M16 carry-overs & hot-path cleanups):** Fold in the 3 M16 carry-overs (Q49 scripted test,
  `road_step_cost` load check, unique-capital debug assert) and 2 `sim-core` cleanups ($O(1)$ `State::city`,
  single-BFS gated `update_connections`). — Confirmed (owner, 2026-11-25)

## Changes to earlier tests

Allowed only for the following (no v1 expected value may change; no test may be removed or weakened; list
every edit in the phase A commit message):

1. **Type & exhaustiveness edits:** `City.population: u32 -> i32` (`as i32` / signed literals in
   `sim-core/tests/*`, `bots/tests/no_cheating.rs`, `sim-tools/tests/*`); `TileState` struct literals /
   destructurings (`building: None, building_level: 0` or `..`); exhaustive `Action` match arms
   (`BuildBuilding`, `Disband`, `Destroy`) in `sim-core/tests/common/mod.rs` and `sim-env/tests/common/mod.rs`.
2. **M15 "not yet unlocked" v2 assertions (`crates/sim-core/tests/ruleset.rs`):** update
   `under_v2_with_every_tech_no_new_unit_or_burn_forest_is_legal` and
   `random_v2_playouts_research_v2_techs_and_nothing_else` so `BurnForest` (and `BuildBuilding`, `Disband`,
   `Destroy`) are allowed under `V2`, while the 3 new units (`Swordsman`, `Catapult`, `Knight`) stay illegal
   until M19. Under `V1`, every v2 action remains strictly illegal.
3. **v2-only digest reference encoders (`crates/sim-tools/tests/v2_digest.rs`, `road_digest.rs`):** append
   `building` (`Option<u8>`) and `building_level` (`u8`) to each tile record in `expected_v2_digest` /
   `expected_v2_bytes` (same precedent as M16 Q47 A; all v1 digest expectations untouched).

## Tests (phase A guidance)

1. **v1 identity:** all existing v1 identity tests pass unchanged; under `V1`, `BurnForest`,
   `BuildBuilding`, `Disband`, and `Destroy` are never legal (random playouts + rich board) and `apply`
   refuses them.
2. **`BurnForest`:** legality (needs `V2`, Construction, 3 stars, own forest without improvement/building,
   no enemy unit, blocked by pending reward); apply deducts 3 stars, changes terrain to `Field` with
   `Resource::Crop`, forfeits any `Animal`, leaves road intact if present.
3. **Economy buildings (`Sawmill`, `Windmill`, `Forge`, `Market`):**
   - Legality: tech, 5 stars, field only (not forest/mountain/city), empty (no improvement/building/resource),
     own territory, no enemy unit, `one_per_city`, needs >= 1 matching friendly neighbor (enemy neighbor
     does not qualify; cross-city friendly neighbor does).
   - Population & dynamic levels: placing next to $k$ huts/farms/mines gives $+k \times \text{pop\_per\_level}$
     to the building's owning city (Forge gives $+2$/mine); adding a new adjacent improvement raises level
     and population (including level-up cascade when Mine +2 + Forge +2 crosses two thresholds!); destroying
     an adjacent improvement drops level and population; capturing a neighboring city transfers adjacency
     and adjusts both players' building levels (skipped on capital capture ending the game, Q48).
   - Market income: sums levels of adjacent friendly `Sawmill`/`Windmill`/`Forge` (each times
     `stars_per_level`), capped at `max_output` (8), updates automatically when a Sawmill/Windmill/Forge
     gains/loses levels or is destroyed, and is reflected in `income()` and turn-start stars.
4. **`Disband`:** legality (Free Spirit, own unit, any phase `Fresh`/`Moved`/`Attacked`/`Done`, blocked by
   pending reward); refund matches `value / 2` for every unit type (including SuperUnit → 5); removes unit
   and frees tile; does not increment kills; does not count toward capacity afterward.
5. **`Destroy`:** legality (Chivalry, own territory, no enemy unit, has improvement or building; illegal on
   road-only tile, empty tile, city-only tile, enemy/neutral tile); removes improvement (keeping terrain
   and resource, subtracting `SiteAction.population`) or building (subtracting `level * pop_per_level`);
   leaves any road on the tile intact; cascades adjacency loss to neighboring buildings.
6. **Signed population & score floor:** destroying buildings/improvements or breaking connections can drop
   `City.population` below `cumulative_population_for_level(level)` and below `0`; `city.level` never drops;
   growing again does not level up until `population >= cumulative_population_for_level(level + 1)`;
   `score_breakdown` counts 0 surplus (never negative, never underflows) when population is below threshold
   or negative.
7. **M16 carry-overs & hot-path tests:**
   - Q49 scripted test (opponent's pending reward during current turn; captured city keeps it and capturer
     resolves immediately).
   - `Tables::from_json_v2` rejects `road_step_cost` = `0.0`, `-0.5`, `0.3` (non-positive or non-half-integer).
   - Gated single-BFS `update_connections` matches the brute-force path rule (`is_connected`) after every
     action across random v2 playouts (including `BorderGrowth` claiming a neutral road tile!).
8. **Digest & determinism:** v1 digests unchanged; under `V2`, `building`, `building_level`, and negative
   `City.population` each change the digest; random v2 playouts exercising all M17 actions replay
   identically and maintain the invariant `tile.building_level as u32 == building_level(&s, p)` after every
   non-terminal step.

Mutants to aim for: counting enemy-territory neighbors in `building_level`; restricting adjacency to the
same city instead of same owner; ignoring `one_per_city`; omitting `max_output` cap on Market; Market
summing neighbor count instead of neighbor levels; `Forge` giving +1 instead of +2; `BurnForest` leaving
`Terrain::Forest` or keeping `Animal`; `Disband` rounding up instead of down or refunding 0 for SuperUnit;
`Disband` illegal in `Done` phase; `Destroy` removing a road or clearing `Crop`/`Metal`; `Destroy`
forgetting to sync neighboring buildings; `score_breakdown` panicking/wrapping on negative population;
`City.level` dropping on population loss; `sync_buildings` running after a game-ending capital capture;
`update_connections` skipping `BorderGrowth`; `road_step_cost` validator accepting `0.3` or `0.0`; digest
writing v2 building fields under `V1`.

## Gate

`pixi run check` exits 0 (fmt, clippy `-D warnings`, all Rust, Python, and RL tests green). `rules_hash()`
== `"a8b52b91eae34503"`, `obs_hash(11)` == `"d5f9b1033cfd55ca"`, and `polytopia verify` accepts the v1
replay fixture unchanged. Rules milestone: a `/review 17` phase is recommended.

## Notes for the implementer

- Read all building stats (`cost`, `tech`, `adjacent`, `level_from`, `population_per_level`,
  `stars_per_level`, `max_output`, `one_per_city`), `BurnForest` stats, and `v2_rules()` (`disband_tech`,
  `destroy_tech`, `disband_refund_divisor`, `super_unit_value`) from `state.tables`; do not hard-code
  constants.
- For `State::city(id)` / `city_mut(id)` $O(1)$ indexing, keep a `debug_assert_eq!(c.id, id)` so any
  broken test fixture that constructs non-dense city ids fails loudly in debug builds.
- In `sync_buildings`, always update `LevelFrom::Count` buildings before `LevelFrom::Levels` (`Market`),
  so Market reads up-to-date neighbor levels in one clean pass.
- If a test contradicts this brief or the spec, stop and add a `docs/QUESTIONS.md` entry (next number
  **Q50**).
