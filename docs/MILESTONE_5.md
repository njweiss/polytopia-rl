# Milestone 5: Explorer, Super Unit and forced spawn

Second half of the spec's "economy and progression". Depends on milestones 1 to 4. Capture, winning
and score are milestone 6.

## Goal
Add the two reward options milestone 4 left out: Explorer (level 2) and Super Unit (level 5 and up),
including the Explorer's random fog-seeking walk and the push rules for a Super Unit spawning onto an
occupied city tile.

## In scope
1. Reward enum and menus gain `Explorer` and `SuperUnit` (data file change).
2. `ChooseReward { Explorer }`: the 15-move walk, consuming `state.rng`, updating the explored mask.
3. `ChooseReward { SuperUnit }`: spawn on the city tile, pushing any occupant.
4. `supported_units` ignores Super Units.
5. Updating the milestone 4 tests that this changes (list below).

## Out of scope (do not build)
Capture, villages becoming cities, winning, turn-cap termination, score, observation encoding,
bots, rendering, PyO3, roads, lighthouses.

## Public API additions (fixed so tests can import them)

```rust
// sim_core::data
pub enum Reward { Workshop, Explorer, CityWall, Resources, PopulationGrowth, BorderGrowth, Park, SuperUnit }
//   (Explorer and SuperUnit are new; keep the existing variants and snake_case names)

// sim_core::explorer
pub const EXPLORER_MOVES: u32 = 15;
pub fn explorer_walk(state: &mut State, city: CityId);   // the whole walk; used by ChooseReward

// sim_core::spawn
pub fn spawn_super_unit(state: &mut State, city: CityId) -> UnitId;  // push + place; used by ChooseReward
pub fn push_direction(state: &State, occupant: UnitId, city_owner: PlayerId) -> Dir; // intended direction
```
No State field changes. `Action` is unchanged (`ChooseReward { city, reward }` just gains two legal
`reward` values). Both new functions are called by `apply` and must also be callable directly, so
tests can set up a state and exercise them without a whole turn.

## Rules

### Menus (replaces D6 of milestone 4)
| Level reached | Menu |
| --- | --- |
| 2 | Workshop, Explorer |
| 3 | CityWall, Resources |
| 4 | PopulationGrowth, BorderGrowth (BorderGrowth omitted if used) |
| 5 and up | Park, SuperUnit |

Order inside each menu is the order written (it fixes `legal_actions` order).

### Explorer walk
Triggered by `ChooseReward { city, Explorer }` after `pending_reward = false` is set and before the
level-up check runs. The explorer is not a unit and never appears in the state. Let `owner` be the
city's owner. The explorer starts on the city's tile and makes exactly `EXPLORER_MOVES` moves, in
this order each move (the walk updates `owner`'s explored mask as it goes):

0. Before move 1 (only): add the starting tile to the visited set and reveal radius 1 around it.
1. **Passable tile:** in bounds, and not a mountain unless `owner` has Climbing. The explorer ignores
   units, Zone of Control, fog, forests and cities; it may pass through anything passable.
2. Compute BFS distance `d(t)` from the current tile over passable tiles (8-neighbors, each step 1).
3. `fog(t)` = number of in-bounds tiles within Chebyshev 1 of `t` that `owner` has not explored (using
   the mask as it is right now). **Candidates:** passable tiles with `1 <= d(t) <= 4` and `fog(t) >= 1`.
4. If there are candidates: keep those with the smallest `d`. Score each as `min(fog(t), 4) - 0.5` if
   `t` is in the visited set, else `min(fog(t), 4)` (use `f32`). Keep the highest score. If more than
   one remains, order them row-major and pick one with `rng.gen_range(k)`; if exactly one, **no draw**.
   That is the target.
5. Step toward the target: among the current tile's passable neighbors `n` with
   `dist(n, target) == d(target) - 1` (BFS from the target over passable tiles), order row-major and,
   if more than one, pick with `rng.gen_range(k)` (no draw if exactly one).
6. If there are no candidates (no reachable fog within 4 steps): wander. Take the passable neighbors
   of the current tile that are not in the visited set; if there are none, take all passable
   neighbors; order row-major and pick with `rng.gen_range(k)` (no draw if exactly one; if the set
   is empty the explorer stays put).
7. After the step: add the new tile to the visited set and reveal radius 1 (no mountain bonus) around
   it in `owner`'s explored mask.

Draw order within one move: target tie-break first, then first-step tie-break. With
`full_observability` there is never any fog, so the explorer always wanders.

### Super Unit spawn
`ChooseReward { city, SuperUnit }` (after `pending_reward = false`, before the level-up check):
1. If a unit (the occupant) stands on the city's tile, push it (below). If it cannot be placed it is
   removed from the game; this is not a kill and credits nobody.
2. Create the Super Unit: `kind = SuperUnit`, `owner = city.owner`, health and max health from the
   table, `pos = city.pos`, `origin = Some(city)`, no kills, not veteran, phase **Done**, no
   `last_dir`, id from `next_unit_id`.
3. `supported_units` (and so Train capacity) never counts `SuperUnit`s.

**Push direction** (`push_direction`): with `occupant.last_dir`:
- `Some(d)`, occupant owned by the city's owner: `d`.
- `Some(d)`, occupant owned by the other player: the opposite of `d` (negate both components).
- `None`: toward the map center: per axis the sign of `center - pos`, where `center = (size - 1) / 2`
  (integer division); if both axes give 0 (the exact center) the direction is south (`dr = +1, dc = 0`).
The spec's "ranged" rule has no independent effect with one `last_dir` per unit (D11 of milestone 4),
so there is no separate ranged branch.

**Placement.** Directions form a clockwise ring as seen on the board (rows grow downward):
`N(-1,0), NE(-1,1), E(0,1), SE(1,1), S(1,0), SW(1,-1), W(0,-1), NW(-1,-1)`. Starting from the intended
direction, probe in the order: intended, 1 step counterclockwise, 1 step clockwise, 2 ccw, 2 cw, 3 ccw,
3 cw, the opposite. The first neighbor tile of the city that is in bounds, holds no unit, and is not a
mountain when the occupant's owner lacks Climbing receives the occupant (cities, villages and fog are
fine). Moving the occupant changes only `pos` and tile occupancy, plus its owner's explored mask (as
for any position change). Its `phase`, `last_dir`, `kills`, health and origin are unchanged.

## Clarifications to the spec (owner: confirm or overrule, then update the doc)
- **E1: visited set** includes the starting tile; the explorer reveals radius 1 at its start. - Confirmed
- **E2: random draws happen only when there is a tie** (more than one candidate), in the order given,
  so a seed always means the same walk. The spec's "random" says nothing about draws when there is no tie. - Confirmed
- **E3: wander prefers unvisited neighbors**; it falls back to any passable neighbor. That is how I read
  "backtracks only when nothing else is open". - Confirmed
- **E4: candidates exclude the current tile and use BFS distance over passable tiles.** The explorer
  moves through forests, fog and occupied tiles freely. - Confirmed
- **E5: all 15 moves are always taken,** even once nothing is left to explore (wandering costs rng draws). - Confirmed
- **E6: the Super Unit spawns Done,** like trained units, with `origin = city` but excluded from capacity. - Confirmed
- **E7: clockwise table above** defines "counterclockwise then clockwise" probing. Enemy occupants are
  pushed opposite to their `last_dir` (D11). - Confirmed
- **E8: a pushed unit reveals fog** for its owner at its new tile; it may be pushed onto an empty city
  or village tile. - Confirmed
- **E9: center** for the push is `(size-1)/2` per axis (integer division); only odd sizes are tested. - Confirmed

## Changes to milestone 4 tests
The test author updates these in phase A, in the same `tests(m5)` commit, and lists every change:
- Reward menus: level 2 now offers `[Workshop, Explorer]`, level 5+ `[Park, SuperUnit]`; any test
  asserting the reduced menus or the exact `ChooseReward` action lists for those levels changes.
- Data tests for `rewards.json` (menus, `explorer` and `super_unit` names).
- `supported_units` tests gain a Super Unit case.
- Playouts and the legal/apply property test include the new rewards (see below).

## Tests (phase A guidance)
Golden expectations come from hand-drawn boards and the rules above, never from the code under test.
1. **Explorer without ties** (no rng draw, so the rng must be unchanged, which is itself asserted):
   a drawn board where each move has a unique best candidate and unique first step; the full 15-step
   path and the final explored mask are listed by hand. Include: nearest candidate beats a richer far
   one; the cap at 4 (a tile that would reveal 5 scores 4); the 0.5 visited penalty flips a choice; a
   mountain blocks without Climbing and is used with it; forests and units do not block.
2. **Explorer with ties:** differential test. The test file contains its own independent
   implementation of the walk using the same `Rng` and the draw rules, run on many random boards and
   seeds, comparing path, mask and final rng state with `explorer_walk`. (The reference must be written
   from this brief, not from the implementation.)
3. **Explorer properties:** at most `EXPLORER_MOVES` steps; each step is to a neighbor; never out of
   bounds; explored mask only grows and is a superset of radius 1 around every visited tile; wander
   when nothing is in reach; `full_observability` consumes draws but changes no mask; same seed -> same
   result.
4. **Super Unit spawn goldens:** empty city tile; friendly occupant with `last_dir` east is pushed east;
   enemy occupant with `last_dir` east is pushed west; occupant with no `last_dir` pushed toward the
   center from each of the 8 octants of a drawn 11x11 board, and south from the exact center; each
   probe step (intended blocked by a unit, by a mountain without Climbing, by the map edge) lands on the
   correct counterclockwise-first neighbor; all 8 blocked removes the occupant without kill credit; the
   pushed unit's phase, `last_dir`, kills, health preserved; fog revealed for its owner; Super Unit
   fields and `supported_units`/Train capacity unaffected.
5. **Menus and ChooseReward:** level 2 and 5+ menus; Explorer and SuperUnit effects reachable through
   `legal_actions`/`apply`; atomicity on illegal choices.
6. **Property / playout:** extend the milestone 4 playouts so Explorer and SuperUnit are chosen often
   (pre-grown cities and extra population); invariants still hold: one unit per tile, occupancy matches
   positions, ownership immutable, explored masks only grow, capacity counts, level/pending invariants,
   no dead units retained.

Mutants to aim for: tie-break draws made when there is no tie (rng changes); draws in the wrong order;
visited penalty dropped or applied to the wrong tile; fog cap missing; candidate distance limit off by
one; explorer blocked by forests or units; mountain rule inverted; wander ignores the unvisited
preference; fewer or more than 15 moves; start reveal missing; enemy pushed the same way as friendly;
probe order clockwise-first; toward-center using Euclidean/abs instead of sign; exact center pushed
wrong way; pushed unit's phase reset; occupant removal counted as a kill; Super Unit counted in
capacity or spawned Fresh.

## Gate
`./scripts/check.sh` passes, including the differential test and extended playouts, under about
60 seconds in a debug build.

## Notes for the implementer
- Keep the Explorer and the push logic in the modules named above so tests can call them directly.
- No `HashMap`/`HashSet` where order affects state; use sorted `Vec`s or grid arrays for BFS.
- If the tests contradict this brief or the spec, stop and add a QUESTIONS.md entry.

## Appendix: `data/rewards.json` (replace the file)
```json
{
  "menus": [
    { "min_level": 2, "max_level": 2,    "options": ["workshop", "explorer"] },
    { "min_level": 3, "max_level": 3,    "options": ["city_wall", "resources"] },
    { "min_level": 4, "max_level": 4,    "options": ["population_growth", "border_growth"] },
    { "min_level": 5, "max_level": null, "options": ["park", "super_unit"] }
  ],
  "resources_stars": 5,
  "population_growth_amount": 3
}
```