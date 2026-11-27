# Milestone 18: monuments, lighthouses and the v2 Explorer (v2)

Fourth milestone of game v2. The rules are in `docs/SPEC.md` "Game v2 rules" ("Monuments", "Lighthouses and
the Explorer", "Population loss, score and the damage preview"). Under `ruleset: v2` this milestone turns on
the four corner lighthouses, their discovery reward, the v2 Explorer's lighthouse bonus (+3.5), and the three
monuments (`ParkOfFortune`, `GateOfPower`, `EyeOfGod`) with task tracking, placement, destruction, and 400
score each. It also folds in the two small `/review 17` defensive nits (`disband_refund_divisor == 0` load
validation and $O(1)$ `train_legal` city lookup).

**The hard constraint (as in M15–M17):** under `ruleset: v1` everything is byte-identical: legal actions,
`apply`, map generation, Explorer walks, digests, replay files, `Config` JSON, `rules_hash`, `obs_hash`,
observations, bots. Every v1 identity test and replay fixture passes unchanged.

## Goal

1. Under `V2`, each of the map's four corner tiles holds a lighthouse (`Field`, no resource/village/city,
   starts fogged when `!full_observability` even if within capital starting vision). The first time each
   player removes fog over a lighthouse, that player's capital gains +1 population.
2. Under `V2`, the Explorer walk applies step 3b: a candidate whose uncovered fog tiles include a lighthouse
   gets `+explorer_lighthouse_bonus` (`+3.5`) after the cap of 4 on fog count.
3. Under `V2`, the three monuments (`ParkOfFortune` at city level 5, `GateOfPower` at 10 kills, `EyeOfGod`
   at 4 lighthouses) unlock permanently once their task is completed, can be placed for free once per player
   per game (+3 population to the owning city, 400 score), and can be destroyed by `Destroy` (removing
   population and score without allowing rebuild).
4. Carry-over `/review 17` nits (`disband_refund_divisor == 0` validation, $O(1)$ `train_legal` lookup,
   and relaxing the latent `pending_reward` invariant in `v2_m17_playouts.rs`) land cleanly.

## Decisions already made (owner)

- All monument, lighthouse, and Explorer rules in `docs/SPEC.md` "Game v2 rules" (owner, 2026-11-01/02)
  and the M15 data tables (`data/v2/monuments.json`, `data/v2/rules.json`: `lighthouse_population: 1`,
  `explorer_lighthouse_bonus: 3.5`, `monument_population: 3`, `monument_score: 400`).
- Monument placement is a site action; kills count as in v1 (attack or retaliation, Super Unit included;
  Disband and forced-spawn removal are not kills); a destroyed monument cannot be placed again.
- Ruins and the remaining real-game monuments stay out of v2.

## In scope

1. `sim_core::data`: promote `ALL_MONUMENTS` to `pub const`; reject `disband_refund_divisor == 0` in
   `Tables::from_json_v2`.
2. `sim_core::state`: `MonumentSet` bitset (like `TechSet`); `TileState.lighthouse: bool` and
   `TileState.monument: Option<Monument>` (always `false` / `None` under `V1`); `PlayerState.kills: u32`,
   `lighthouses_discovered: u8`, `monuments_unlocked: MonumentSet`, `monuments_placed: MonumentSet` (all
   zero/empty under `V1`).
3. `sim_core::setup`: under `V2`, configure the four corner tiles as field, no resource, `lighthouse: true`,
   and `explored = false` when `!config.full_observability` (even when a capital is at Chebyshev 2).
4. `sim_core::monuments` (new module): `corners(size)`, `corner_index(size, pos)`, lighthouse discovery on
   fog removal, and table-driven `update_monument_unlocks`.
5. `sim_core::explorer`: step 3b lighthouse bonus (`tables.v2_rules().explorer_lighthouse_bonus`) under `V2`.
6. `sim_core::actions`: `Action::BuildMonument { at: Pos, monument: Monument }`; kill tracking in
   `apply_attack`; lighthouse discovery through unified fog reveal (`reveal_within`, `apply_capture`,
   `border_growth`); `Destroy` support for monuments; `BuildRoad` / `BuildBuilding` exclusivity with
   lighthouses/monuments; post-apply non-terminal `update_monument_unlocks` and `level_up_check`; $O(1)$
   `train_legal` city lookup.
7. `sim_core::score`: add `pub monuments: u32` to `ScoreBreakdown` (`0` under `V1`, `count * monument_score`
   under `V2`).
8. `sim_core::testing`: `StateBuilder` helpers (`lighthouse`, `monument`, `kills`, `monuments_unlocked`,
   `monuments_placed`, `lighthouses_discovered`, or `player_with`).
9. `sim_tools::digest`, `sim_tools::render`, `sim_env::actions`: v2-only digest bytes for the new tile and
   player fields (v1 byte-identical); `action_label` and `encode_action` arms for `BuildMonument`.

## Out of scope

New v2 units (`Swordsman`, `Catapult`, `Knight`), Stiff/Persist, and bot strategies for v2 (M19); v2
observation, factored action slots, `Env` support, and damage preview (M20); rendering lighthouses/monuments
in the text/HTML viewer and exposing them on `PlayerView` (M19/M20).

## Public API (fixed so tests can import it)

```rust
// sim_core::data
pub const ALL_MONUMENTS: [Monument; 3] = [
    Monument::ParkOfFortune,
    Monument::GateOfPower,
    Monument::EyeOfGod,
];  // promote existing private const to pub

// sim_core::state
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct MonumentSet(u8);
impl MonumentSet {
    pub fn new() -> MonumentSet
    pub fn from_slice(monuments: &[Monument]) -> MonumentSet
    pub fn contains(&self, m: Monument) -> bool
    pub fn insert(&mut self, m: Monument)
    pub fn bits(&self) -> u8
}

pub struct TileState {
    ...,
    pub lighthouse: bool,            // always false under V1
    pub monument: Option<Monument>,  // always None under V1
}

pub struct PlayerState {
    pub stars: u32,
    pub techs: TechSet,
    pub explored: Vec<bool>,
    pub kills: u32,                          // always 0 under V1
    pub lighthouses_discovered: u8,          // 4-bit mask indexed by corner_index (0..4); 0 under V1
    pub monuments_unlocked: MonumentSet,     // empty under V1
    pub monuments_placed: MonumentSet,       // empty under V1
}

// sim_core::actions (appended after Destroy, preserving existing discriminants)
pub enum Action {
    ...,
    Destroy { at: Pos },
    BuildMonument { at: Pos, monument: Monument },
}

// sim_core::monuments (new module)
pub fn corners(size: u8) -> [Pos; 4]              // row-major: (0,0), (0,n-1), (n-1,0), (n-1,n-1)
pub fn corner_index(size: u8, pos: Pos) -> Option<usize>  // 0..=3 for the 4 corners, else None

// sim_core::score
pub struct ScoreBreakdown {
    pub units: u32,
    pub territory: u32,
    pub exploration: u32,
    pub cities: u32,
    pub parks: u32,
    pub techs: u32,
    pub monuments: u32,   // always 0 under V1
}

// sim_core::testing
impl StateBuilder {
    pub fn lighthouse(self, p: Pos) -> StateBuilder
    pub fn monument(self, p: Pos, monument: Monument) -> StateBuilder
    pub fn player_with(self, player: PlayerId, edit: impl FnOnce(&mut PlayerState)) -> StateBuilder
}
```

## Rules

### Lighthouses (under `V2` only; never present under `V1`)

- **Placement at game start (`new_game`):** for each `p` in `corners(map.size)`:
  - `tile.lighthouse = true`, `tile.terrain = Terrain::Field`, `tile.resource = None`.
  - If `!config.full_observability`, set `players[0].explored[idx(p)] = false` and
    `players[1].explored[idx(p)] = false` (overriding capital starting radius-2 vision when a capital sits
    at distance 2 from a corner).
  - If `config.full_observability`, `explored` stays all-true and `lighthouses_discovered` starts at `0`
    (M1).
  - `generate_map` itself is unchanged, so v1 maps and RNG streams stay byte-identical.
- **Discovery:** whenever an action under `V2` (while `state.outcome.is_none()`, Q48) reveals a tile `p` for
  `player` (`explored` transitions `false -> true`) and `tile(p).lighthouse` is `true` (M2):
  - Determine bit `b = 1u8 << corner_index(size, p)` (or if a scripted test places a lighthouse on a
    non-corner tile via `StateBuilder`, wait: `lighthouse` is only meaningful on corners, or bit indexed by
    `corner_index`; tests place lighthouses on corners).
  - If `players[player].lighthouses_discovered & b == 0`:
    - Set `players[player].lighthouses_discovered |= b`.
    - If `player` owns a capital (`capital_of(state, player)`), add
      `state.tables.v2_rules().lighthouse_population as i32` (`+1`) to that capital's `population`.
    - Run `level_up_check` on that capital (either immediately or during post-apply ascending-id checks;
      note: for the Explorer walk, multiple lighthouses discovered during one walk each add +1 before
      `level_up_check` runs, and any multi-level threshold cascade resolves via `apply_choose_reward` as
      usual).
- **Tile exclusivity (M3):** a tile with `lighthouse == true` rejects `BuildRoad`, `BuildBuilding`, and
  `BuildMonument`. Units may enter and stand on it as a normal `Field` tile.

### Explorer lighthouse bonus (under `V2` only; `V1` unchanged)

- In `explorer_walk`, step 3 scores each candidate `q` at minimum BFS distance `dmin` by:
  - `base = f.min(FOG_CAP) as f32 - if visited[q] { VISITED_PENALTY } else { 0.0 }` (v1 rule).
  - **Step 3b (under `V2`):** if any tile `t` in `neighbors(n, q).chain(once(q))` is unexplored by `owner`
    (`!state.is_explored(owner, t)`) **and** `state.tile(t).lighthouse`, add
    `state.tables.v2_rules().explorer_lighthouse_bonus` (`3.5`) to the candidate's score.
- Once a lighthouse tile is explored by `owner`, it is no longer an uncovered fog tile and contributes no
  bonus.
- Under `V1`, `tile.lighthouse` is always `false`, so step 3b never fires and v1 Explorer walks are
  byte-identical.

### Monument task tracking and unlocks (under `V2` only)

- **Kills (`PlayerState.kills`):** under `V2`, in `apply_attack`, whenever a unit dies (`defender_died` ->
  attacker's owner; `attacker_died` -> defender's owner), increment `players[killer_owner].kills += 1`.
  Counts attack kills and retaliation kills, including Super Unit kills. Disband and forced-spawn removal
  never increment `kills`. Under `V1`, `PlayerState.kills` stays `0`.
- **Unlock check (`update_monument_unlocks`):** after every non-terminal v2 action (`state.outcome.is_none()`),
  for each player `p in 0..2` and each `m` in `ALL_MONUMENTS`:
  - Skip if `players[p].monuments_unlocked.contains(m)`.
  - Read `info = state.tables.monument(m)` (`task`, `threshold`):
    - `MonumentTask::CityLevel`: unlocked if any city owned by `p` has `c.level >= info.threshold`.
    - `MonumentTask::Kills`: unlocked if `players[p].kills >= info.threshold`.
    - `MonumentTask::Lighthouses`: unlocked if `players[p].lighthouses_discovered.count_ones() >= info.threshold`.
  - Once set in `monuments_unlocked`, a bit is never cleared (M5).

### `BuildMonument` (under `V2` only; never legal under `V1`)

- Legal when: ruleset is `V2`, no reward pending, `me.monuments_unlocked.contains(monument)`,
  `!me.monuments_placed.contains(monument)`, `at` is in-bounds and owned by a city `c` of `state.current`,
  no enemy unit on `at` (own unit is fine), `t.terrain == Terrain::Field`, `t.city.is_none()`,
  `t.improvement.is_none()`, `t.building.is_none()`, `t.monument.is_none()`, `!t.lighthouse`, and
  `t.resource.is_none()` (M3). Cost is 0 stars.
- Apply:
  - Set `t.monument = Some(monument)`.
  - Insert `monument` into `players[state.current].monuments_placed`.
  - Add `state.tables.v2_rules().monument_population as i32` (`+3`) to `c.population`.
  - `level_up_check` runs on `c` (and post-apply sync checks `update_monument_unlocks`, so a monument
    leveling a city to 5 can immediately unlock `ParkOfFortune`!).
- `BuildBuilding` legality also requires `t.monument.is_none()` and `!t.lighthouse` (M3).

### `Destroy` on a monument (under `V2` only)

- `destroy_legal`: `t.improvement.is_some() || t.building.is_some() || t.monument.is_some()`.
- `apply_destroy`: if `t.monument.is_some()`, set `t.monument = None` and subtract
  `state.tables.v2_rules().monument_population as i32` (`3`) from `owner_city.population`.
  `monuments_placed` is **not** cleared, so the destroyed monument can never be placed again. Any road on
  the tile stays intact.

### Score (`ScoreBreakdown.monuments`)

- Under `V1`: `monuments = 0`.
- Under `V2`: `monuments = count_of_tiles_with(t.monument.is_some() && owner_of(t.owner_city) == player) *
  state.tables.v2_rules().monument_score` (`400`).
- Included in `ScoreBreakdown::total()`. Capturing a city transfers its tiles (and any monuments on them)
  to the new owner, transferring their 400 score (M5).

### Canonical `legal_actions` order and post-apply sync

- **Order in `legal_actions`:** `BuildMonument` appears row-major, then `ALL_MONUMENTS` order
  (`ParkOfFortune`, `GateOfPower`, `EyeOfGod`), right after `BuildBuilding` and before `Destroy` (M4).
- **Post-apply sync under `V2` (when `state.outcome.is_none()`, Q48):**
  1. `sync_buildings` (on `Site | BuildBuilding | Destroy | Capture`),
  2. `update_connections` (every non-terminal v2 action, Q50 C),
  3. `level_up_check` in ascending `CityId` order for every city whose population rose during the action or
     sync (including capital population from lighthouse discoveries and `BuildMonument`),
  4. `update_monument_unlocks` (runs *after* `level_up_check` so a level-up to level 5 on this action
     unlocks `ParkOfFortune` immediately!).

### M17 review nits folded in (M7)

- `Tables::from_json_v2` returns `Err(...)` if `rules.disband_refund_divisor == 0`.
- `train_legal` in `actions.rs` looks up the city via `state.cities.get(city.0 as usize).filter(|c| c.id == city)`.

### Digest (under `V2` only)

- Per tile, after `building_level` (under `V2` only):
  - `lighthouse`: `bool` (`0` or `1`),
  - `monument`: `opt_u8(t.monument.map(|m| m as u8))` (`0` if `None`, else `1` then discriminant `0..=2`).
- Per player, after `explored` (under `V2` only):
  - `kills`: `u32`,
  - `lighthouses_discovered`: `u8`,
  - `monuments_unlocked`: `u8` (`bits()`),
  - `monuments_placed`: `u8` (`bits()`).
- Under `V1`: none of these fields are written, so **v1 digests are byte-identical**.

### Hashes

`rules_hash_for(V1)` (`a8b52b91eae34503`), `rules_hash_for(V2)`, and `obs_hash(11)` (`d5f9b1033cfd55ca`)
are all unchanged (`data/v2/*.json` unchanged; `RULES_VERSION` not bumped).

## Clarifications (owner: confirm or overrule)

- **M1 (lighthouses under `full_observability: true`):** Under `V2` with `full_observability: false`, the 4
  corner tiles start `explored = false` even if within a capital's starting radius-2 vision. Under
  `full_observability: true`, `explored` stays all-true (preserving the `full_observability` invariant) and
  `lighthouses_discovered` starts at `0` (no turn-0 bonus). — Confirmed (owner, 2026-11-26)
- **M2 (what triggers lighthouse discovery):** Any fog transition (`false -> true`) on a lighthouse tile
  under `V2` (unit move, melee kill-move, forced Super Unit push, Explorer walk, city/village capture
  claiming the corner, or `BorderGrowth` claiming the corner) counts as discovering that lighthouse (once
  per corner per player, skipped once `outcome.is_some()` per Q48). — Confirmed (owner, 2026-11-26)
- **M3 (lighthouse & monument tile exclusivity):** A lighthouse tile blocks `BuildRoad`, `BuildBuilding`,
  and `BuildMonument` (units may still stand on it). Placing a monument requires an owned plain `Field`
  with `resource.is_none()`, `city.is_none()`, `improvement.is_none()`, `building.is_none()`,
  `monument.is_none()`, `!lighthouse` (matching M17 P1), and a road may share a tile with a monument.
  `BuildBuilding` also rejects `monument.is_some()` and `lighthouse`. — Confirmed (owner, 2026-11-26)
- **M4 (`Action::BuildMonument` & ordering):** Add `Action::BuildMonument { at: Pos, monument: Monument }`
  after `Destroy`; order in `legal_actions` row-major then `ALL_MONUMENTS` right after `BuildBuilding` and
  before `Destroy`. — Confirmed (owner, 2026-11-26)
- **M5 (monument unlock & capture transfer):** Unlocks are permanent once earned; `monuments_placed` is
  permanent (cannot rebuild after `Destroy`); `ScoreBreakdown.monuments` derives 400 per monument on tiles
  owned by `player`'s cities so capturing a city transfers its monument's 400 score to the new owner.
  — Confirmed (owner, 2026-11-26)
- **M6 (allowed edits to earlier tests):** Approve the narrow v2-only and type/exhaustiveness edits listed
  under "Changes to earlier tests" below (zero v1 expectations changed). — Confirmed (owner, 2026-11-26)
- **M7 (`/review 17` nits folded in):** Add `disband_refund_divisor == 0` rejection in
  `Tables::from_json_v2` and $O(1)$ `train_legal` lookup in `actions.rs`. — Confirmed (owner, 2026-11-26)

## Changes to earlier tests

Allowed only for the following (no v1 expected value may change; no test may be removed or weakened; list
every edit in the phase A commit message):

1. **Type & exhaustiveness edits:** `TileState` (`lighthouse: false, monument: None` or `..`), `PlayerState`
   (`kills: 0, lighthouses_discovered: 0, monuments_unlocked: MonumentSet::new(), monuments_placed:
   MonumentSet::new()` or `..`), `ScoreBreakdown` (`monuments: 0` in `sim-core/tests/score.rs`), and
   `Action::BuildMonument` arms (`sim-core/tests/common/mod.rs`, `sim-env/tests/common/mod.rs`).
2. **M15 `crates/sim-core/tests/ruleset.rs`:** allow `BuildMonument` under `V2` while keeping v2 units
   illegal until M19 (and everything v2 illegal under `V1`).
3. **v2-only digest reference encoders (`crates/sim-tools/tests/{v2_digest,road_digest,buildings_digest}.rs`):**
   append v2-only tile bytes (`lighthouse`, `monument`) and v2-only player bytes (`kills`,
   `lighthouses_discovered`, `monuments_unlocked`, `monuments_placed`); all v1 expectations untouched.
4. **M16 `crates/sim-core/tests/connections.rs:486`:** extend `check_update`'s population-delta skip guard
   to also skip `Action::BuildMonument { .. }`, `Action::Capture { .. }`, and actions where either player's
   `lighthouses_discovered` changed (still checking `connected_to` on every non-terminal step).
5. **M17 `crates/sim-core/tests/v2_m17_playouts.rs:331`:** relax the overly-strong `if c.pending_reward {
   assert!(c.population >= exp_cum(c.level) as i32); }` check noted in `/review 17` Nit #1 (keeping the
   `else` branch `c.population < exp_cum(c.level + 1) as i32`).
6. **M15 `crates/sim-core/tests/ruleset.rs:174` (Q52 A):** update `new_game_reads_the_tables_of_the_configs_ruleset`
   to verify the four corner tiles are fogged `Field` lighthouses with no resource under `V2` while keeping
   exact equality with `V1` on RNG, seats, cities, units, and all non-corner tiles/explored entries.
7. **M16 `crates/sim-core/tests/connections.rs:611` (Q53 A, superseded by Q54 B):** keep seed range `0..30`
   and relax the playout vacuity guard from `losses > 2` to `losses >= 2` under M18's shifted action draws.

## Tests (phase A guidance)

1. **v1 identity:** all existing v1 tests pass unchanged; under `V1`, no tile has `lighthouse == true`,
   `BuildMonument` is never legal/applicable, `ScoreBreakdown.monuments == 0`, `PlayerState` v2 fields stay
   zero, and v1 Explorer walks and digests are unchanged.
2. **Lighthouse setup & discovery:**
   - Across many v2 seeds (including seeds where a capital sits at `(2, 2)`, `(2, n-3)`, `(n-3, 2)`, or
     `(n-3, n-3)`!), all 4 corners are `Field`, no resource/village/city, `lighthouse == true`, and
     `explored == false` for both players when `!full_observability`.
   - Under `full_observability: true`, all tiles including corners start `explored == true` and
     `lighthouses_discovered == 0`.
   - Discovering a lighthouse by unit move, melee kill-move, Explorer walk, forced Super Unit push, and
     `BorderGrowth` each grant `+1` population to the discoverer's capital (once per corner per player;
     both players can each claim the same corner once; discovering does not fire again on revisiting).
   - Discovering a lighthouse that crosses the capital's threshold sets `pending_reward = true` (including
     opponent's capital via forced push, Q49).
   - `BuildRoad`, `BuildBuilding`, and `BuildMonument` are illegal on a lighthouse tile; units may stand on
     it.
3. **Explorer lighthouse bonus (step 3b):**
   - Scripted board where candidate A (at distance `dmin`) uncovers 4 plain fog tiles (score 4.0) and
     equally near candidate B uncovers 1 fog tile that hides a lighthouse (score `1.0 + 3.5 = 4.5`): under
     `V2`, Explorer deterministically moves toward B (no tie draw); once that lighthouse is already explored,
     Explorer picks A.
   - Under `V1`, the same geometry has no lighthouse bonus and picks A.
   - Nearest-distance rule (step 2) still takes precedence over step 3b when a closer fog candidate exists.
4. **Monument unlocks:**
   - `ParkOfFortune`: unlocks at city level 5 via growth (including cascade or monument placement) AND via
     capturing an enemy level-5 city; stays unlocked if that city is later lost.
   - `GateOfPower`: unlocks on the 10th kill (counting attack kills, retaliation kills, and Super Unit
     kills; Disband and forced-spawn removal do not increment `kills`).
   - `EyeOfGod`: unlocks when the 4th distinct corner lighthouse is discovered (not at 3; not by one player
     re-visiting or the other player discovering).
5. **`BuildMonument`, `Destroy`, score & capture:**
   - Legality: unlocked & not yet placed, free (0 stars legal!), owned plain `Field`, no resource, no city,
     no improvement/building/monument, not lighthouse, no enemy unit (own unit ok), road on tile ok, blocked
     while reward pending; canonical order in `legal_actions`.
   - Apply: sets `tile.monument`, sets `monuments_placed`, adds `+3` pop (and levels up / cascades if
     threshold crossed), adds `+400` to `ScoreBreakdown.monuments` and `score()`.
   - `BuildBuilding` illegal on a tile with a monument.
   - `Destroy` removes monument, subtracts `3` pop (can go negative without lowering level), removes 400
     score, leaves road intact, and `monuments_placed` prevents rebuilding it.
   - Capturing a city with a monument transfers its 400 score to the new owner.
6. **M17 review nit:** `Tables::from_json_v2` rejects `"disband_refund_divisor": 0`.
7. **Digest & v2 playouts:** v1 digests unchanged; under `V2`, `lighthouse`, `monument`, `kills`,
   `lighthouses_discovered`, `monuments_unlocked`, and `monuments_placed` each change the digest; multi-seed
   random v2 playouts exercise lighthouses, Explorer bonus, monument unlocks/placement/destroy, maintain
   invariants, and replay identically.

Mutants to aim for: corners left explored at start when capital is at distance 2; corner resource/forest not
cleared under `V2`; lighthouse giving +1 pop twice to the same player; second player blocked from claiming a
lighthouse the first player already found; Explorer adding `+3.5` before the cap of 4 (capped away to 4.0!)
or adding it for an already-explored lighthouse; `Disband` or push-removal incrementing `PlayerState.kills`;
retaliation or Super Unit kills not incrementing `kills`; `ParkOfFortune` not unlocking on capture of a
level-5 city; `BuildMonument` costing stars or allowing rebuild after `Destroy`; `Destroy` not subtracting 3
pop on a monument; `ScoreBreakdown.monuments` not included in `total()` or not transferring on capture;
`BuildRoad`/`BuildBuilding` allowed on a lighthouse tile; `disband_refund_divisor == 0` accepted; digest
writing v2 lighthouse/monument/player fields under `V1`.

## Gate

`pixi run check` exits 0 (fmt, clippy `-D warnings`, all Rust, Python, and RL tests green). `rules_hash()`
== `"a8b52b91eae34503"`, `obs_hash(11)` == `"d5f9b1033cfd55ca"`, and `polytopia verify` accepts the v1
replay fixture unchanged. Rules milestone: a `/review 18` phase is recommended.

## Notes for the implementer

- Read all monument thresholds/tasks from `state.tables.monument(m)` and all scalars (`lighthouse_population`,
  `explorer_lighthouse_bonus`, `monument_population`, `monument_score`) from `state.tables.v2_rules()`; do
  not hard-code constants.
- Keep `generate_map` untouched so v1 mapgen tests and RNG streams remain byte-identical; configure corner
  lighthouses in `setup::new_game` under `Ruleset::V2`.
- Route all fog reveals (`reveal_within`, `apply_capture`, `border_growth`) through one helper so
  lighthouse discovery cannot be missed on any code path.
- If a test contradicts this brief or the spec, stop and add a `docs/QUESTIONS.md` entry (next number
  **Q52**).
