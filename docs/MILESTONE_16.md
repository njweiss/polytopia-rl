# Milestone 16: roads and city connections (v2)

Second milestone of game v2. The rules are in `docs/SPEC.md` "Game v2 rules" ("Roads and city
connections"; Claude Doc tab "v2 rules"). Under `ruleset: v2` this milestone turns on building roads, road
movement and city connections. Everything else from v2 except research (M15) stays off.

**The hard constraint (as in M15):** under `ruleset: v1` everything is byte-identical: legal actions,
`apply`, digests, replay files, `Config` JSON, `rules_hash`, `obs_hash`, observations, bots. The M15
v1-identity tests (playout fold, env episode fold, replay fixture, hashes) must pass unchanged.

## Goal

1. Under `V2`, a player with Roads can build a road (3 stars) on the tiles the spec allows, including
   neutral tiles.
2. Under `V2`, movement uses roads: a step between two usable road tiles costs 0.5 and ignores the forest
   stop.
3. Under `V2`, a city linked to its owner's capital by roads is connected: the city and the capital each
   gain +1 population, and lose it again when the connection breaks.

## Decisions already made (owner)

- All road rules in the spec section "Roads and city connections" (owner, 2026-11-01/02), and the M15
  data: `rules.json` `road_cost` 3, `road_tech` roads, `road_step_cost` 0.5, `connection_population` 1.
- Building a road is a site action in the rules sense: it acts on a tile, not through a unit. Its actor
  slot in the RL action space is an M20 design point.
- Roads give no score of their own; they score only through the population they give.

## In scope

1. `sim_core`: the road flag on tiles, the `BuildRoad` action (legality, apply), road movement in
   `reachable_tiles`, city connections and their population, and the road under a neutral village
   disappearing on capture.
2. The digest covers the new state under `V2` only.
3. The M15 review nit: `StateBuilder::config` also sets the tables of the config's ruleset.
4. The match arms the new `Action` variant forces in `sim-tools` (render, describe), `bots` and
   `sim-env`.

## Out of scope

Every other v2 rule (buildings, burn forest, disband, destroy, monuments, lighthouses, the new units and
skills, population loss from buildings). Observation, action space and `Env` support for v2 (M20). Bot
road use (M19). Showing roads in the text renderer, the HTML viewer and `PlayerView` (M19/M20).

## Public API (fixed so tests can import it)

```rust
// sim_core::state
pub struct TileState { ..., pub road: bool }               // always false under V1
pub struct City { ..., pub connected_to: Option<CityId> }  // the capital whose connection this
                                                           // city holds; None for capitals, unconnected
                                                           // cities, and always under V1
// sim_core::actions
pub enum Action { ..., BuildRoad { at: Pos } }             // appended last

// sim_core::roads (new module)
pub fn road_usable(state: &State, player: PlayerId, p: Pos) -> bool
pub fn is_connected(state: &State, city: CityId) -> bool  // the path rule below, ignores connected_to

// sim_core::testing
impl StateBuilder { pub fn config(self, config: Config) -> StateBuilder }  // now also swaps the tables
```

## Rules

### Road tiles

- `TileState.road` is a bool; at most one road per tile. It never changes under `V1`.
- **Usable:** `road_usable(state, player, p)` is true when the tile has a road or a city, and is not in
  the other player's territory (its `owner_city` is `None` or a city `player` owns). So neutral roads work
  for both players, a player's own roads and cities work for that player, and nothing in enemy territory
  works for either. A city tile counts as a road (R2).

### BuildRoad legality (under V2 only; never legal under V1)

All of:

- the current player has `road_tech` (Roads) and at least `road_cost` (3) stars;
- the tile is on the map, has no road, has no city, and is not a mountain;
- **own territory** (`owner_city` is a city the current player owns): no enemy unit on the tile (the v1
  site-action rule; an own unit is fine);
- **neutral territory** (`owner_city` is `None`): the tile is explored by the current player and has no
  unit at all. A neutral village tile is allowed;
- **enemy territory:** never.

Order in `legal_actions`: after the site actions, `BuildRoad` row-major, then Research, then EndTurn.
Like a site action it is blocked while a reward is pending.

### BuildRoad apply

Deduct `road_cost` stars, set `road = true`, then update connections (below). It changes nothing else:
terrain, resource, improvement and village stay.

### Road movement (under V2; `reachable_tiles`, so Move and Escape both use it)

Movement is counted in half-points: the budget is `2 × movement`.

- A **road step** is a step from `a` to `b` where `road_usable(owner, a)` and `road_usable(owner, b)`. It
  costs `2 × road_step_cost` = 1 half-point. Every other step costs 2.
- A step is allowed when the remaining budget is at least 1 half-point. The budget then drops by the step's
  cost, floored at 0 ("costs round up": with 0.5 left a unit can still take a normal step).
- The target tile must be enterable as in v1 (explored, empty, terrain allowed).
- **Stops:** a unit that arrives on a tile stops there, with no further steps, when the tile is next to an
  enemy unit (Zone of Control, any step), or when it is a forest or mountain and the step was not a road
  step. The start tile is exempt, as in v1.
- `reachable_tiles` returns every tile reachable by some sequence of allowed steps, row-major, excluding
  the start tile. The search must keep, per tile, enough state (remaining budget and whether the unit
  stopped) to find every such tile; a breadth-first search by step count is no longer enough.

Under `V1` there are no roads, so every step is a normal step and the result equals today's exactly.

### City connections (under V2)

- **Path rule:** a non-capital city `c` is connected when there is a sequence of 8-adjacent tiles from
  `c`'s tile to its owner's capital's tile in which every tile in between is `road_usable` for `c`'s owner
  (own cities on the way are fine, being road-usable). Adjacent city and capital are connected directly.
- **State:** `City.connected_to` records the capital (by id) whose connection the city currently holds.
- **Update:** after every applied action under `V2`, for every city in ascending id order:
  - if `connected_to` is `Some(k)` and either `c` no longer satisfies the path rule or `k` is no longer its
    owner's capital: `c` and city `k` each lose `connection_population` (1); `connected_to = None`;
  - then if `connected_to` is `None`, `c` is not a capital and the path rule holds: `c` and its owner's
    capital each gain 1; `connected_to = Some(capital)`.
  - After all cities: the level-up check for every city whose population rose, in ascending id order (the
    existing one-level-at-a-time rule with pending rewards).
- The update is skipped when the action ended the game (`outcome` is set), so after a capital capture
  no population changes from connections (Q48, owner answer C).
- Population loss: a city's level never goes down. A loss can never take population below 0 here, because
  every loss reverses an earlier +1 to the same city (R4).
- Ways a connection breaks in v2 so far: a city on the path, or the city itself, is captured by the other
  player; a tile on the path becomes enemy territory (a captured city's territory); a road under a village
  disappears when the village is captured.

### Capture of a village with a road

When a village becomes a city (Capture), its tile's `road` becomes `false`; the new city tile counts as a
road for its owner anyway. Then connections update as above.

### Digest (under V2 only)

Each tile's record gets one trailing byte, `road` (0 or 1), and each city's record a trailing
`connected_to` (`Option<CityId>` encoded as the digest already encodes options), only when the ruleset is
`V2`. v1 digests are unchanged.

### Hashes

`rules_hash_for(V1)` and `obs_hash` unchanged. `RULES_VERSION` is not bumped (R5).

## Clarifications (owner: confirm or overrule)

- **R1:** the action is a new `Action::BuildRoad { at }`, not a `SiteAction` variant. A `SiteAction` would
  need a row in `data/v2/site_actions.json` that duplicates `road_cost` and `road_tech` from `rules.json`,
  and its legality (neutral tiles allowed) differs from every site action. It is still "a site action" in
  the spec's sense and goes to the site slot in M20. - Confirmed (owner, 2026-11-11)
- **R2:** a city tile counts as a road for its owner: for movement ("a road and a city") and as a link in
  a connection path. A step from city to city also costs 0.5. A road cannot be built on a city tile (it
  would change nothing). - Confirmed (owner, 2026-11-11)
- **R3:** connections are recomputed after every action under v2, in ascending city id order (gains are
  given after losses within each city). Recomputing everything is simplest and costs little on an 11x11
  map; it also catches every way a connection breaks. - Confirmed (owner, 2026-11-11)
- **R4:** no signed population type yet. In M16 every population loss reverses an earlier +1 to the same
  city, so it cannot go below 0; the subtraction is checked and panics if that ever fails. The signed type
  stays with M17, where building loss first allows otherwise (M15 V3). - Confirmed (owner, 2026-11-11)
- **R5:** no rules-version bump: v1 rules are unchanged, so the v1 `rules_hash` (and every v1 checkpoint)
  stays valid. No v2 checkpoint can exist until M20, which introduces a v2 rules version. - Confirmed (owner, 2026-11-11)
- **R6:** a connected city that the other player captures: its old connection is lost first (the city and
  the old owner's capital each lose 1), then it may connect to its new owner's capital (each gain 1). - Confirmed (owner, 2026-11-11)

## Changes to earlier tests

Allowed only where a type changes: struct literals of `TileState` and `City` gain `road: false` and
`connected_to: None` (or destructuring patterns gain `..`/the field), and exhaustive matches on `Action`
gain a `BuildRoad` arm. No expected value may change; no test may be removed or weakened. Each such edit
is listed in the phase A commit message.

One owner-approved exception (Q47, answer A): in `crates/sim-tools/tests/v2_digest.rs`, `expected_v2_digest`
changes to the M16 v2 encoding (trailing `road` byte per tile, trailing `connected_to` per city). Its
v1 expectations and every other value stay unchanged.

## Tests (phase A guidance)

1. **v1 identity:** the M15 identity tests pass unchanged; under `V1`, `BuildRoad` is never legal (random
   playouts) and `apply` refuses it.
2. **BuildRoad legality:** without Roads; with fewer than 3 stars; on a mountain; on a city; on a tile
   that already has a road; own territory with an own unit (legal) and with an enemy unit (illegal);
   neutral explored empty (legal), neutral unexplored (illegal), neutral with any unit (illegal), neutral
   village (legal); enemy territory (illegal); blocked while a reward is pending; order in
   `legal_actions`.
3. **BuildRoad apply:** stars −3, `road` set, terrain/resource/improvement/village unchanged.
4. **Road movement:** scripted boards with an oracle (an independent brute-force search over step
   sequences in the test): a Warrior (movement 1) goes 2 road steps; a Rider (2) goes 4; road then normal
   step with 0.5 left; road step onto a forest does not stop, a normal step onto a forest does; ZoC stops a
   road step; enemy-territory roads give no discount for either player; neutral roads work for both; a
   city counts as a road; the final tile needs no road. Property test: on random v2 boards with random
   roads, `reachable_tiles` equals the oracle. Escape after an attack uses roads.
5. **Connections:** a city linked by roads gains +1 and so does the capital (once, not per action); a path
   through an own city; adjacent city and capital; neutral roads count; a path through enemy territory
   does not; capture of a city on the path breaks it (−1 each); capture of the connected city itself (R6);
   capture of a village with a road removes the road; a +1 that crosses a threshold levels the city up
   with a pending reward; losing it never lowers the level.
6. **Digest:** v1 digests unchanged (goldens); under `V2` a road and a `connected_to` change the digest.
7. **StateBuilder:** `config(v2 config)` gives the v2 tables (a v2 tech's cost can be read).
8. **Determinism:** random v2 playouts with roads replay identically from their action log.

Mutants to aim for: road steps that ignore ZoC; forest stop applied to road steps; enemy-territory roads
usable; neutral roads usable only by one player; the budget check `> 0` replaced by `>= 2` (no
round-up); the connection +1 given on every action; the capital not getting its +1; a lost connection not
reversed; the village road kept after capture; `BuildRoad` legal on a neutral tile with a unit; digest
fields serialized under v1.

## Gate

`./scripts/check.sh` passes with all existing tests unchanged in value. `rules_hash` and `obs_hash` of the
build equal the recorded v1 values, and `polytopia verify` accepts the v1 replay fixture unchanged.
Rules milestone: a `/review 16` phase is recommended.

## Notes for the implementer

- Branch on `state.config.ruleset` at the few entry points (legal actions, movement, the connection
  update after `apply`, digest), not throughout the rules.
- Read `road_cost`, `road_tech`, `road_step_cost` and `connection_population` from `tables.v2_rules()`;
  do not hard-code them.
- Work in the worktree recipe in `docs/STATUS.md` (Rust changes: build the extension into the worktree).
- If a test contradicts this brief or the spec, stop and add a QUESTIONS.md entry (next number Q47).
