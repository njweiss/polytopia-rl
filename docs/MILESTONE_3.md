# Milestone 3: game state and unit actions

Spec milestone 4, sliced: state, fog/explored mask, movement, combat integration, and the unit
actions. Capture is milestone 6; train, site actions, research and rewards are milestone 5.
Depends on milestone 1 (tables, combat) and milestone 2 (Rng, map generator).

## Goal
A `State` that holds ground truth for a game, `new_game` to create one from a generated map, and
`legal_actions` / `apply` for the actions below, so that two scripted or random players can move,
fight, recover, promote and end turns with every spec invariant holding.

## In scope
1. State types (plain public data) and `new_game`.
2. Vision and the per-player explored mask.
3. Movement: `reachable_tiles`.
4. Attacks: legality, `defense_bonus`, resolution (damage, retaliation, kill credit, kill-move).
5. `Recover`, `Promote`, `EndTurn`.
6. `legal_actions(state)` and `apply(state, action)` for exactly these actions.
7. Test support: `StateBuilder` (scripted boards) and a random-playout harness (see Tests).

## Out of scope (do not build)
Capture, villages becoming cities, Train, site actions (harvest, farm, ...), Research, Choose
reward, income, level-ups, border growth, Explorer, Super Unit and forced spawn, score, winning,
turn-cap termination, observation encoding, bots, rendering, PyO3.
Fields these later milestones need exist in the state types now (so layout does not change), but
nothing reads or writes them yet except where stated.

## Module layout and public API (fixed so tests can import it)

```rust
// sim_core::state
pub type PlayerId = u8;                       // 0 or 1
pub struct UnitId(pub u32);                   // Copy, Eq, Ord, Hash, Debug
pub struct CityId(pub u32);                   // same derives
pub enum Phase { Fresh, Moved, Attacked, Done }
pub struct Dir { pub dr: i8, pub dc: i8 }     // each in -1..=1, not both 0
pub struct TechSet(/* bitset */);             // new(), from_slice(&[Tech]), contains(Tech), insert(Tech)

pub struct Config { pub full_observability: bool, pub turn_cap: u32 } // Default: false, 50

pub struct Unit {
    pub id: UnitId, pub kind: UnitType, pub owner: PlayerId, pub pos: Pos,
    pub health: u32, pub max_health: u32,
    pub origin: Option<CityId>,
    pub kills: u32, pub veteran: bool, pub promotion_available: bool,
    pub phase: Phase,
    pub last_dir: Option<Dir>,                // ONE direction per unit, see C6
}
pub struct City {
    pub id: CityId, pub owner: PlayerId, pub pos: Pos, pub level: u32, pub population: u32,
    pub is_capital: bool, pub workshop: bool, pub parks: u32, pub wall: bool,
    pub border_growth_used: bool,
    pub pending_reward: bool,                 // placeholder; milestone 5 refines it
}
pub enum Improvement { LumberHut, Farm, Mine }
pub struct TileState {
    pub terrain: Terrain, pub resource: Option<Resource>, pub improvement: Option<Improvement>,
    pub village: bool,                        // unclaimed village (capture is milestone 6)
    pub city: Option<CityId>,                 // a city stands here
    pub owner_city: Option<CityId>,           // territory; written at most once
    pub unit: Option<UnitId>,                 // occupant
}
pub struct PlayerState { pub stars: u32, pub techs: TechSet, pub explored: Vec<bool> } // row-major
pub enum Outcome { Win(PlayerId), Draw }      // unused until milestone 6
pub struct State {                            // Clone, Debug, PartialEq, Eq
    pub size: u8, pub turn: u32, pub current: PlayerId, pub rng: Rng, pub config: Config,
    pub tables: Arc<Tables>,                  // Tables must derive PartialEq, Eq (tiny M1 follow-up)
    pub tiles: Vec<TileState>,                // row-major
    pub units: Vec<Unit>,                     // ascending id, dead units removed
    pub cities: Vec<City>,                    // ascending id, never removed
    pub players: [PlayerState; 2],
    pub next_unit_id: u32, pub outcome: Option<Outcome>,
}
impl State { pub fn tile(&self, p: Pos) -> &TileState; pub fn unit(&self, id: UnitId) -> Option<&Unit>;
             pub fn city(&self, id: CityId) -> &City; pub fn is_explored(&self, player: PlayerId, p: Pos) -> bool; }

// sim_core::setup
pub fn new_game(map: &GeneratedMap, config: &Config, rng: Rng) -> State;
pub fn new_game_from_seed(seed: u64, params: &MapParams, config: &Config) -> State;
//   = { let mut rng = Rng::new(seed); let map = generate_map(&mut rng, params); new_game(&map, config, rng) }

// sim_core::movement
pub fn reachable_tiles(state: &State, unit: UnitId) -> Vec<Pos>;   // row-major, excludes own tile
pub fn defense_bonus(state: &State, defender: UnitId) -> f32;

// sim_core::actions
pub enum Action { Move { unit: UnitId, to: Pos }, Attack { unit: UnitId, target: Pos },
                  Recover { unit: UnitId }, Promote { unit: UnitId }, EndTurn }   // Copy, Eq, Hash, Debug
pub enum ActionError { Illegal }
pub fn legal_actions(state: &State) -> Vec<Action>;
pub fn apply(state: &mut State, action: &Action) -> Result<(), ActionError>;

// sim_core::testing   (compiled in normal builds; plain data assembly)
pub struct StateBuilder;  // scripted boards, see Tests
```
`apply` is a pure function of (state, action) in behavior: no globals, no clock, and the only
randomness is `state.rng` (nothing in this milestone draws from it after `new_game`).

## Rules

### new_game
1. One draw: `let flip = rng.gen_range(2)` after map generation. Player 0 gets `map.capitals[flip]`,
   player 1 the other. Player 0 always moves first.
2. Tiles copy terrain and resource from the map. Village sites become `village: true`.
3. City 0 is player 0's capital, city 1 player 1's: level 1, population 0, `is_capital`, no
   buildings. Each owns the 3x3 block around it (`owner_city` set; border radius 1).
4. Each player starts with 5 stars, techs `{Organization}`, and one Warrior on the capital tile:
   `origin = capital`, full health, phase Fresh, no kills, no `last_dir`. Unit ids 0 and 1 follow
   player order; `next_unit_id = 2`.
5. Explored mask: every tile within Chebyshev 2 of the player's capital (25 tiles). With
   `config.full_observability`, all tiles for both players, always.
6. `turn = 1`, `current = 0`, `outcome = None`. `state.rng` is the rng after the flip.

### Vision and the explored mask
- A unit sees every tile within Chebyshev radius 1 of its tile; radius 2 if it stands on a
  mountain. A city sees every tile with `owner_city == that city`.
- After any change of a unit's position (Move, kill-move, Escape move) the owner's explored mask
  gains the tiles that unit sees from its new position. The mask never shrinks.
- Explored tiles always show live state (decided). There is no separate "currently visible" set.

### reachable_tiles(state, unit)
Pure geometry for the unit's owner; it ignores the unit's phase (phase is checked by legality).
Budget = the unit's movement stat (the full stat also for an Escape move). Search over steps to the
8 neighbors, each step costs 1:
- A tile may be entered only if: it is in bounds; it is explored by the unit's owner (mask as it
  is BEFORE the move, so tiles revealed mid-move do not extend reach); it holds no unit, friendly or
  enemy; and it is not a mountain unless the owner has Climbing.
- Cities and villages are ordinary tiles for movement: enterable when unoccupied, whoever owns them.
- Because occupied tiles cannot be entered, units cannot pass through friendly units.
- Rough terrain: after entering a forest or mountain tile the unit cannot take further steps from it
  (the tile is still a valid destination).
- Zone of Control: after entering a tile adjacent (Chebyshev 1) to any enemy unit, the unit cannot
  take further steps from it. The tile is still a valid destination. Enemy cities/villages exert no
  ZoC. A unit's own starting tile is exempt: it may leave forest, mountain or ZoC with its full
  budget, and may re-enter ZoC.
- Result: all tiles reachable by some valid path of at most `movement` steps, row-major, without
  the starting tile.

### Move
Legal when the mover is a unit of the current player, `phase == Fresh` or (`phase == Attacked` and
the unit type has Escape), and `to` is in `reachable_tiles`. Effect: `pos = to` (tile occupancy
updated); `last_dir` = sign of the net displacement per axis; phase Fresh -> Moved, Attacked -> Done;
the owner's explored mask grows from the new position.

### Attack
Legal when the attacker belongs to the current player, `phase == Fresh` or (`phase == Moved` and
the type has Dash), `target` has 1 <= Chebyshev distance <= `range` from the attacker, the target tile is
explored by the attacker's owner, and an enemy unit stands on it. No path or line-of-sight
requirement, and attacks need not be preceded by a move.

Effect, in this order:
1. `bonus = defense_bonus(defender)`.
2. `combat::resolve_attack(attacker, defender, bonus, in_range)` where each `Combatant` uses the
   unit's table attack/defense and its current `health` / `max_health` (a veteran's max health is
   already +5), and `in_range = chebyshev(attacker, defender) <= defender.range stat`.
3. Apply the new healths. Dead units are removed (their tile is freed, ids are never reused).
4. Kill credit: if the defender died, attacker `kills += 1`; if the attacker died from
   retaliation, the defender `kills += 1`. Then for any unit: if `kills >= 3`, not `veteran`, and
   the type does not have `Static`, set `promotion_available = true`.
5. Kill-move: if the defender died, the attacker survived, and the attacker is melee (stat
   `range == 1`), it moves onto the defender's tile, unless that tile is a mountain and the owner
   lacks Climbing. The kill-move updates position, occupancy, `last_dir` and the explored mask. It
   does not consume the Escape move.
6. A surviving attacker's phase becomes Attacked. A surviving ranged attacker (stat `range > 1`)
   sets `last_dir` toward the target (sign of target minus attacker per axis).

### defense_bonus(state, defender)
Evaluated for the defender's owner and tile; the largest applicable value wins (no stacking):
- 1.0 default.
- 1.5 if the tile is a forest and the owner has Archery; 1.5 if it is a mountain and the owner has
  Climbing.
- If the tile holds a city owned by the defender's owner and the unit type has Fortify: 1.5, or 4.0
  if that city has `wall`.

### Recover
Legal for a current-player unit with `phase == Fresh`; legal at full health. Heals 4 if the unit's
tile has `owner_city` whose city belongs to the unit's owner, else 2; health capped at `max_health`.
Phase becomes Done.

### Promote
Legal for a current-player unit with `promotion_available && !veteran`, in any phase. Effect:
`max_health += 5`, `health = max_health`, `veteran = true`, `promotion_available = false`. Phase and
`last_dir` unchanged.

### EndTurn
Legal when the game has no outcome and no city of the current player has `pending_reward`.
Effect: `current = 1 - current`; if the new current player is 0, `turn += 1`; every unit of the new
current player resets to `Fresh` (`last_dir` is never reset). Income is added here in milestone 5
(first-turn income is decided then).

### legal_actions and apply
- `legal_actions` returns only the current player's actions, in this canonical order: units by
  ascending id; per unit, Move (destinations row-major), Attack (targets row-major), Recover,
  Promote; then `EndTurn`. If `outcome` is `Some`, it returns an empty list.
- `apply` returns `Err(Illegal)` and leaves the state completely unchanged (including `rng`) for
  any action not in `legal_actions`. It returns `Ok` exactly for actions that are in it. It does not
  have to call `legal_actions` internally.

## Clarifications to the spec (owner: confirm or overrule, then update the doc)
- **C1: cities and villages are enterable** when unoccupied (needed so capture can happen the next
  turn). The spec's "cannot enter occupied tiles, including cities and villages" means occupied ones. - Confirmed
- **C2: reach uses the pre-move explored mask**, for every step of the path. - Confirmed
- **C3: attack targets must be on explored tiles** (owner decision: explored tiles are always visible).
- **C4: ZoC comes from enemy units only** (owner decision), applies on arrival, never to the start tile.
- **C5: melee means stat range 1.** Ranged units (Archer) never kill-move. An Archer may attack at
  distance 1 or 2. - Confirmed
- **C6: one `last_dir`** per unit: net displacement sign for moves (including kill-move and Escape),
  toward the target for ranged attacks; unchanged by Recover, Promote and (later) pushes. - Confirmed
- **C7: Recover is legal at full health** (it is a pass for that unit). - Confirmed
- **C8: city defense bonus applies only in the defender's own city**, only to units with Fortify;
  wall (4.0) replaces, not multiplies, the city bonus; terrain bonuses use the defender's techs. - Confirmed
- **C9: both sides earn kills** (retaliation kills count). Kills keep counting after promotion. - Confirmed
- **C10: turn counter** is a round counter starting at 0, incremented when player 1 ends. Player 0
  always moves first; capitals are assigned by one coin flip.
- **C11: after a kill-move the phase is Attacked**, so a Rider (Escape) can still move afterwards. - Confirmed
- **C12: `apply` is atomic**: an illegal action changes nothing. - Confirmed

## Tests (phase A guidance)
Golden values must come from outside the code under test: the wiki formula by hand (reuse the
milestone 1 golden rows for damage numbers), counting tiles on a drawn board, and the rules above.
Reachability expectations are checked against an independent brute-force path enumerator written
in the test (budgets are at most 2 steps, so exhaustive search is trivial), not against the output
of `reachable_tiles`.

1. **StateBuilder + scripted boards.** Support at least: board size, per-tile terrain/resource,
   placing cities and villages, units (owner, type, position, optional health/max health/kills/
   veteran/phase), techs, `explore_all()` / explicit explored tiles, current player. Plain data
   assembly; it is part of the test author's stubs and is allowed to be real code.
2. **Movement goldens** (on drawn boards): open-field Warrior and Rider; friendly unit blocking a
   corridor; detour needed; forest and mountain stop (arrive vs start inside); mountain with and
   without Climbing; ZoC stop, leaving ZoC, re-entering ZoC, diagonal adjacency; unexplored tile as
   destination and as a path tile; empty village and empty enemy city enterable, occupied not.
3. **Attack and combat integration:** the five milestone 1 rows played through `apply`; retaliation
   kill; no retaliation at range; melee kill-move and its mountain exception; ranged no-move;
   Dash and Escape sequences (Rider: move, attack, move; Warrior: attack then no move; Moved + no
   Dash cannot attack); `defense_bonus` table (forest/mountain with and without tech, own city with
   and without Fortify, wall, enemy city, Defender vs Warrior in city); kill credit for both sides;
   promotion becomes available at exactly 3 kills; Super Unit never; Promote effects.
4. **Recover / EndTurn / new_game:** territory 4 vs 2 healing and the cap; turn counter and phase
   reset across several end turns; `new_game` field-by-field (stars, techs, explored = 25 tiles,
   units, cities, coin flip effect, full observability, determinism from the seed).
5. **Legality vs apply (property):** over many random games, at every step compare `legal_actions`
   with a brute-force enumeration built in the test from the rules above (fine to build from
   `reachable_tiles` once that is validated by item 2), and check that `apply` is Ok exactly for
   members of the list, over the list plus a sample of random non-members, with unchanged state on Err.
6. **Random playouts with invariants.** At least 300 games, at least 150 actions each (or until no
   progress), choosing uniformly among legal actions but weighting `EndTurn` so games advance.
   After every action check, with checkers written in the test: one unit per tile and occupancy
   matches `unit.pos`; units ascending by id; `owner_city` never changes once set; explored masks
   only grow; `0 < health <= max_health` for living units; stars never negative; phase never
   regresses within a turn; all of the current player's units are `Fresh` right after EndTurn; no
   action is legal for a non-current player's units.
7. **Determinism and replay:** same seed + same action list -> identical `State` (derive
   `PartialEq`); a cloned state evolves identically to the original.

Mutants the test author should aim to kill: ZoC off or applied to the start tile; forest/mountain
stop removed; passing through friendly units; explored check removed (destination only, path
only); Dash/Escape gating swapped; kill-move onto an unclimbable mountain; kill-move for ranged
units; retaliation ignoring range or surviving check; kill credit lost on retaliation; wrong
recover amounts; phase reset on the wrong player; `turn` incremented at the wrong time; city
bonus applied to an enemy city or without Fortify; `apply` accepting an action not in the list.

## Gate
`./scripts/check.sh` passes, including the property and playout tests. Runtime of the whole suite
should stay under about 60 seconds in a debug build; the test author tunes game counts to fit.

## Notes for the implementer
- Keep the hot paths allocation-light but do not optimize yet: correctness and clarity first.
- No `HashMap`/`HashSet` in anything that affects state or action order.
- If the tests contradict this brief or the spec, stop and add a QUESTIONS.md entry.