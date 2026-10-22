# Milestone 6: capture, win and score

The spec's milestone 6. After this the simulator plays complete games: every playout ends with a
winner or a draw. Depends on milestones 1 to 5.

## Goal

Add the `Capture` action (villages become cities, enemy cities change hands), the win condition
(capital capture), turn-cap termination decided by score, and the derived `score` function, so
random playouts always finish with a valid `Outcome`.

## In scope

1. `Action::Capture` for villages and enemy cities, including orphaning and origin migration.
2. Ending the game by capital capture.
3. Turn-cap termination in `EndTurn` (higher score wins, equal scores draw).
4. `score` and `score_breakdown` (derived on demand, never stored) and `ScoreWeights` in `Config`.
5. Updating earlier tests that this changes (list below).

## Out of scope (do not build)

Observation encoding and the public opponent-score feature (milestone 9), reward shaping, bots,
rendering and replay tooling (milestone 7), PyO3, roads, monuments, temples. No other elimination or
victory condition exists.

## Public API additions (fixed so tests can import them)

```rust
// sim_core::state
pub struct ScoreWeights {                       // Clone, Copy, Debug, PartialEq, Eq; Default = the values below
    pub unit_per_star: u32,        // 5   (x unit cost in stars)
    pub super_unit: u32,           // 50  (flat; Super Units have no cost)
    pub territory_per_tile: u32,   // 20
    pub exploration_per_tile: u32, // 5
    pub city_base: u32,            // 100
    pub city_per_level: u32,       // 50  (x (level - 1))
    pub population_surplus: u32,   // 5   (x surplus population)
    pub park: u32,                 // 250
    pub tech_per_tier: u32,        // 100 (x tier, per researched tech)
}
pub struct Config {                             // gains one field, Default keeps its other values
    pub full_observability: bool, pub turn_cap: u32, pub score_weights: ScoreWeights,
}
impl State { pub fn is_terminal(&self) -> bool; }   // outcome.is_some()

// sim_core::actions
pub enum Action { /* all existing variants, plus: */ Capture { unit: UnitId } }

// sim_core::score
pub struct ScoreBreakdown { pub units: u32, pub territory: u32, pub exploration: u32,
                            pub cities: u32, pub parks: u32, pub techs: u32 }   // Debug, PartialEq, Eq
impl ScoreBreakdown { pub fn total(&self) -> u32; }
pub fn score_breakdown(state: &State, player: PlayerId) -> ScoreBreakdown;
pub fn score(state: &State, player: PlayerId) -> u32;   // == score_breakdown(..).total()
```

`new_game` asserts `config.turn_cap >= 1`.

## Rules

### Capture

Legal when no reward is pending, the game has no outcome, the unit belongs to the current player,
`phase == Fresh`, and the unit's tile is either a village (`village == true`) or holds a city owned
by the other player. Any unit type may capture (including the Super Unit).

**Village.** Create a city: `id = CityId(cities.len())`, owner = the unit's owner, `pos` = the tile,
`level 1`, `population 0`, not capital, no workshop/wall/parks, `border_growth_used = false`,
`pending_reward = false`. The tile gets `village = false`, `city = Some(id)`. Every tile within
Chebyshev 1 of the village (including the village tile) that is in bounds and has no `owner_city` gets
`owner_city = Some(id)`.

**Enemy city.** `city.owner` becomes the capturer. Everything else on the city is kept (level,
population, buildings, `is_capital`, `border_growth_used`). Tiles follow automatically because tile
ownership is by city id; no tile's `owner_city` changes. Every living unit whose `origin` is this city
(they belong to the previous owner) gets `origin = None` and never counts toward any capacity again.

**Both cases.** The capturing unit's `origin` becomes the captured/new city (so it counts toward that
city's capacity unless it is a Super Unit); its `phase` becomes Done; its `last_dir` is unchanged. The
capturer's explored mask gains every tile the city owns (cities see all their tiles).

**Win.** If the captured city `is_capital`, set `outcome = Some(Outcome::Win(capturer))`
immediately. The state is final: `legal_actions` is empty and `apply` returns `Err(Illegal)`.

### Turn cap (changes EndTurn)

`EndTurn` behaves as before, except: if the ending player is player 1 and `turn + 1 == config.turn_cap`,
then `turn += 1`, `current = 0`, and `outcome` is set from the scores of both players
(`Win(p)` for the higher score, `Draw` if equal). In that case income and the unit reset are NOT
applied. So each player plays exactly `turn_cap` turns (turns `0..turn_cap`). A capital capture on the
final turn wins before the cap is evaluated.

### Score

`score_breakdown(state, player)`, using `state.config.score_weights` `w` and the tables:

- **units:** over the player's living units: `w.super_unit` for a Super Unit, else
  `w.unit_per_star * cost` (the unit table's cost).
- **territory:** `w.territory_per_tile` times the number of tiles whose `owner_city` belongs to a city
  owned by the player.
- **exploration:** `w.exploration_per_tile` times the number of `true` entries in the player's mask.
- **cities:** over the player's cities: `w.city_base + w.city_per_level * (level - 1) +
  w.population_surplus * (population - cumulative_population_for_level(level))` (surplus saturates at 0).
- **parks:** `w.park` times the sum of `parks` over the player's cities.
- **techs:** `w.tech_per_tier * tier` summed over the player's researched techs (the starting
  Organization counts).

Score is public to both players in the observation (milestone 9) but nothing here hides it.

### legal_actions order (extends the milestone 5 order)

Per unit: Move, Attack, **Capture**, Recover, Promote. Everything else unchanged.

## Clarifications to the spec (owner: confirm or overrule, then update the doc)

- **F1: new city id** is `cities.len()` (ids are dense and never reused). The village tile is claimed
  together with its unclaimed 3x3 neighbors. - Confirmed
- **F2: capturing an enemy city keeps its population and buildings**, and orphans the previous
  owner's units of that city (origin `None`). Units of other cities are unaffected. - Confirmed
- **F3: the game ends immediately on capital capture.** No later state changes happen. - Confirmed
- **F4: any unit type can capture,** only from `Fresh`, with a pending reward blocking it like every
  other action. - Confirmed
- **F5: turn cap** evaluated when player 1 ends the final turn; no income or reset after that.
  `turn_cap >= 1`. A draw only on exactly equal scores. - Confirmed
- **F6: score is `u32`,** derived on demand. Super Units score a flat 50 (spec: "Super Unit 50"). - Confirmed
- **F7: surplus population** is measured against `cumulative_population_for_level(level)` (the
  threshold of the city's current level), as in the spec's formula. - Confirmed
- **F8: exploration counts the mask,** so under `full_observability` both players score 605 for it. - Confirmed
- **F9: no other elimination rule.** A player with no units and no money can still end turns. - Confirmed

## Changes to earlier tests

The test author updates these in phase A, in the same `tests(m6)` commit, and lists every change:

- A unit standing on a village or an enemy city now has a `Capture` action. Tests comparing a unit's
  full action list in such positions (milestone 3 enterable-city/village tests, playout enumerators)
  change; the brute-force legality enumerator gains the Capture rule.
- `Config` gains `score_weights`; builders and struct-literal uses update.
- `EndTurn` at the turn cap ends the game; multi-turn tests that run past the default cap of 50 set a
  larger `turn_cap` or stay below it.
- Playouts may now reach a terminal state; they must stop (not fail) when `is_terminal()`.

## Tests (phase A guidance)

Golden values come from the rules above and hand arithmetic, never from the code under test.

1. **Score goldens:** the starting score is 515 per player with breakdown
   `{units: 10, territory: 180, exploration: 125, cities: 100, parks: 0, techs: 100}`. Then scripted
   states, each changing exactly one term by exactly its weight: add a Rider (+15), a Defender (+15), a
   Super Unit (+50); own one more territory tile (+20); explore one more tile (+5); raise a city to
   level 3 with population 7 (cities: 100 + 100 + 5*(7-5) = 210); add a park (+250); research Hunting
   (+100) and Archery (+200). Custom `ScoreWeights` change totals proportionally. `score ==
   breakdown.total()`.
2. **Capture, village:** unit on a village captures; new city fields; tile flags; the exact claimed set
   on a drawn board (skips already owned tiles and off-map); origin migration (the old city's
   `supported_units` drops, the new city's rises; Super Unit not counted); phase Done; explored mask
   grows; research cost for the capturer rises with the city count; second capture gets the next id.
3. **Capture, enemy city:** owner flips, level/population/buildings kept, tiles unchanged by id but
   now count as the capturer's territory (score and `Recover` healing amounts follow), previous owner's
   units of that city orphaned while other units keep their origin, capturer's origin migrates,
   explored mask gains the city's tiles, defense bonus in the captured city now applies for the
   capturer (Fortify) and no longer for the loser.
4. **Capture legality:** requires Fresh (a unit that moved onto the tile cannot capture until next
   turn), a unit of the current player, not on an own city, not on a plain tile, blocked by a pending
   reward and after the game ends; enemy-occupied tiles cannot be entered, so the defender must die
   first (scripted kill-move then capture next turn).
5. **Win:** capturing the enemy capital sets `Win(capturer)`, no other state change afterwards,
   `legal_actions` empty, `apply` returns `Err` for every action kind; capturing a non-capital city
   does not end the game.
6. **Turn cap:** with `turn_cap = 2` scripted EndTurns: the game ends exactly after player 1's second
   EndTurn; outcomes for higher score each way and for equal scores; no income added at the end;
   capital capture on the last turn wins regardless of score; EndTurn before the cap behaves as in
   milestone 4.
7. **Property / playouts:** a large set of random games (small `turn_cap`, e.g. 10 to 30, and
   scenario variants where units start next to villages and the enemy capital) must all reach a
   terminal state within a bounded number of actions with a valid `Outcome`. After every action:
   all earlier invariants plus: cities are dense by id; `tile.city` and `cities[].pos` agree; a
   village tile has `city == None`; at most one `is_capital` city per original owner and the capital
   only changes owner by capture; orphaned units have `origin == None`; no unit's origin names a
   city that does not exist; score is `>= 0` and equals the breakdown total; the outcome, once
   set, never changes; `apply` after terminal always errs. The legal/apply equivalence property
   covers Capture.
8. **Determinism and replay:** same seed + actions -> identical states through captures and the
   cap; cloned states evolve identically.

Mutants to aim for: capture allowed from Moved; enemy city's units not orphaned (or all units
orphaned); origin not migrated; level/buildings reset on capture; village city not claiming
neighbors or claiming owned tiles; new city id wrong; capital capture not ending the game (or any
capture ending it); win awarded to the wrong player; cap evaluated one turn early/late or for the
wrong player; income applied after the cap; draw/win inverted on ties; each score term's weight, a
dropped term, surplus measured against the next threshold, starting tech omitted, Super Unit scored
by cost; `apply` accepting actions after the game ended.

## Gate

`./scripts/check.sh` passes, including the terminating playouts, under about 60 seconds in a debug
build.

## Notes for the implementer

- Keep capture and scoring in clearly named modules (`capture` logic may live in `actions`), and keep
  `score` free of side effects; it will be called often by shaping and observations later.
- No `HashMap`/`HashSet` where order affects state.
- If the tests contradict this brief or the spec, stop and add a QUESTIONS.md entry.
