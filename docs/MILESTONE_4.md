# Milestone 4: economy and progression

First half of the spec's "economy and progression" (the spec's milestone 5). The second half
(Explorer, Super Unit, forced spawn) is milestone 5; capture, winning and score are milestone 6.
Depends on milestones 1 to 3 (tables, combat, Rng, map, state, unit actions).

## Goal
Add income, research, training, tile (site) actions, population and level-ups, and the forced
reward choices to `legal_actions` / `apply`, so a player can build an economy and an army and the
spec invariants still hold in random playouts.

## In scope
1. Income at the start of a player's turn (changes `EndTurn`).
2. `Research`.
3. `Train` (with unit capacity).
4. Site actions: harvest animal, harvest fruit, lumber hut, clear forest, farm, mine.
5. Population, level thresholds, and level-up cascades.
6. `ChooseReward` and the forced-reward rule, with the reward menus below.
7. New rules data tables (Appendix) and the `economy` helper functions.
8. Updating the milestone 3 tests where this milestone's rules change their expectations (list below).

## Out of scope (do not build)
Explorer and Super Unit rewards (not offered in menus yet, see D6), forced unit spawn, capture,
villages becoming cities, winning, turn-cap termination, score, observation encoding, bots,
rendering, PyO3, `max_city_level` (no cap exists, so no config key yet).

## Public API additions (fixed so tests can import them)

```rust
// sim_core::data  (new enums used as table keys; derive Copy, Eq, Hash, Debug, Deserialize snake_case)
pub enum SiteAction { HarvestAnimal, HarvestFruit, LumberHut, ClearForest, Farm, Mine }
pub enum Reward { Workshop, CityWall, Resources, PopulationGrowth, BorderGrowth, Park }
pub const ALL_SITE_ACTIONS: [SiteAction; 6];
pub struct SiteActionInfo { pub cost: u32, pub tech: Tech, pub population: u32, pub stars: u32 }
impl Tables {                                             // additions; load()/from_json keep working
    pub fn site_action(&self, a: SiteAction) -> &SiteActionInfo;
    pub fn reward_menu(&self, level: u32) -> &[Reward];   // menu offered when a city reaches `level`
    pub fn resources_reward_stars(&self) -> u32;           // 5
    pub fn population_growth_amount(&self) -> u32;         // 3
}
// from_json/load must now also take the two new JSON files (see Appendix); keep a clear Err for
// missing, duplicated or malformed entries as in milestone 1.

// sim_core::actions
pub enum Action {                                          // M3 variants unchanged, plus:
    Train { city: CityId, kind: UnitType },
    Site { at: Pos, action: SiteAction },
    Research { tech: Tech },
    ChooseReward { city: CityId, reward: Reward },
    /* Move, Attack, Recover, Promote, EndTurn as before */
}

// sim_core::economy
pub fn income(state: &State, player: PlayerId) -> u32;
pub fn cumulative_population_for_level(level: u32) -> u32;   // L(L+1)/2 - 1  (level 1 -> 0)
pub fn supported_units(state: &State, city: CityId) -> u32;  // living units whose origin == city
pub fn research_cost(state: &State, player: PlayerId, tech: Tech) -> u32;
pub fn reward_menu(state: &State, city: CityId) -> Vec<Reward>; // menu for the city's current level,
                                                                // BorderGrowth removed if already used
```
`City.pending_reward` stays a `bool`. The menu is determined by the city's current `level` (see D5).
No other state field changes. All of `State`'s derives stay.

## Rules

### Start of turn: income (changes EndTurn)
`EndTurn` behaves as in milestone 3, then, for the player whose turn begins: if `state.turn >= 1`
(after the possible increment), `stars += income(player)`. Unit reset to Fresh is as before.
So neither player gets income on their first turn (turn 0), and both do from the second round on.

`income(player)` = sum over that player's cities of `level + (is_capital ? 1 : 0) + (workshop ? 1 : 0)
+ parks`.

### Research
Legal when no reward is pending, the tech is not yet researched, its prerequisite (if any) is
researched, and `stars >= research_cost`. `research_cost = tables.tech_cost(tech, cities owned by the
player)`. Effect: `stars -= cost`; tech added.

### Train
Legal when no reward is pending and: the city belongs to the current player; the city tile holds no
unit; `kind` is trainable (`cost` is `Some`) and unlocked (`unlocked_by` is `None` or researched by
the owner); `stars >= cost`; `supported_units(city) < level + 1`.
Effect: `stars -= cost`; a new unit (`id = next_unit_id`, then increment) of `kind` at the city tile,
owner = city owner, full health from the table, `origin = Some(city)`, no kills, not veteran,
phase **Done**, no `last_dir`. The explored mask is unchanged.

### Site actions
Legal when no reward is pending and all hold: the tile is owned by a city (`owner_city`) belonging to
the current player; no enemy unit stands on it (a friendly unit is fine); the owner has the action's
tech (`site_action(a).tech`); `stars >= cost`; and the per-action condition:

| Action | Tile condition | Effect besides `stars -= cost` |
| --- | --- | --- |
| HarvestAnimal | `resource == Animal` | `resource = None`; population + `info.population` |
| HarvestFruit | `resource == Fruit` | `resource = None`; population + `info.population` |
| LumberHut | `terrain == Forest`, `resource == None`, `improvement == None` | `improvement = LumberHut` (terrain stays Forest, D3); population + `info.population` |
| ClearForest | `terrain == Forest`, `resource == None`, `improvement == None` | `terrain = Field`; `stars += info.stars` |
| Farm | `resource == Crop`, `improvement == None` | `resource = None`; `improvement = Farm`; population + `info.population` |
| Mine | `resource == Metal`, `improvement == None` | `resource = None`; `improvement = Mine`; population + `info.population` |

Population goes to the city that owns the tile (`owner_city`), then the level-up check runs.

### Population and levels
- `cumulative_population_for_level(L) = L(L+1)/2 - 1`: levels 2..5 at 2, 5, 9, 14.
- Level-up check (run after every population gain and after every reward resolves), for a city:
  if `pending_reward == false` and `population >= cumulative_population_for_level(level + 1)`, then
  `level += 1` and `pending_reward = true`. Only one reward is pending per city at a time; if the
  population still exceeds the next threshold after the reward resolves, the check fires again
  (cascade). There is no level cap.

### Reward menus (data: `rewards.json`) and ChooseReward
`reward_menu(city)` is the table menu for the city's current `level`:

| Level reached | Offered in this milestone |
| --- | --- |
| 2 | Workshop |
| 3 | CityWall, Resources |
| 4 | PopulationGrowth, BorderGrowth (BorderGrowth omitted if `border_growth_used`) |
| 5 and up | Park |

(Explorer at level 2 and Super Unit at level 5+ are added in milestone 5.)

`ChooseReward { city, reward }` is legal when the city belongs to the current player, has
`pending_reward`, and `reward` is in its menu. It is the only kind of action legal while any of the
current player's cities has a pending reward, and only for the lowest-id pending city (forced
rewards). Effects, then `pending_reward = false`, then the level-up check:
- Workshop: `workshop = true`. CityWall: `wall = true`. Park: `parks += 1`.
- Resources: `stars += resources_reward_stars()`.
- PopulationGrowth: `population += population_growth_amount()`.
- BorderGrowth: `border_growth_used = true`; every in-bounds tile within Chebyshev distance 2 of the
  city that has no `owner_city`, is not a village site and holds no city gets `owner_city = this
  city`; the owner's explored mask gains those tiles. Legal even if nothing is claimed (D7).

### legal_actions order (extends the milestone 3 order)
If the current player has a pending reward: only that city's `ChooseReward` actions, in
`reward_menu` order. Otherwise, in this order: units by ascending id (Move, Attack, Recover, Promote
as before); then cities by ascending id (Train in `ALL_UNITS` order); then site actions by tile
row-major, per tile in `ALL_SITE_ACTIONS` order; then Research in `ALL_TECHS` order; then `EndTurn`.
`apply` stays atomic: an illegal action returns `Err(Illegal)` and changes nothing.

## Clarifications to the spec (owner: confirm or overrule, then update the doc)
- **D1: first-turn income.** Neither player gets income on their first turn (owner decision):
  income is added to the incoming player only when `turn >= 1`. - Confirmed
- **D2: tech cost uses the number of cities the player owns** (always 1 until capture exists). - Confirmed
- **D3: Lumber hut keeps the forest** (owner decision): terrain stays Forest (still stops movement and
  still gives the Archery bonus); only ClearForest turns forest into field. - Confirmed
- **D4: tile requirements.** LumberHut and ClearForest need a forest tile with no resource and no
  improvement (an animal must be hunted first). Harvest, Farm and Mine consume the resource;
  Farm/Mine leave an improvement; improvements never change or disappear in v1. - Overruled, see spec for details
- **D5: levels rise immediately.** `level` increments the moment the threshold is crossed and
  `pending_reward` marks the unresolved choice, so the menu is a function of `level`. Invariants:
  `pending ⇒ population >= cum(level)`; `!pending ⇒ population < cum(level + 1)`. - Confirmed
- **D6: menus are reduced until milestone 5.** Level 2 offers only Workshop, level 5+ only Park. - Confirmed
- **D7: Border Growth.** Radius 2, unowned tiles only, never village or city tiles, legal as a no-op,
  once per city. (A village tile claimed by someone else's border would break "owning city never
  changes once set" when the village becomes a city.) - Confirmed
- **D8: Train capacity** counts living units whose `origin == city`. Orphans (origin `None`, later
  milestones) never count. - Confirmed
- **D9: forced rewards block everything** else, including unit actions, Promote and EndTurn. - Confirmed
- **D10: site actions with a friendly unit on the tile are legal;** an enemy unit blocks them. - Confirmed
- **D11: Super Unit push precedence (for milestone 5, owner decision).** The spec's order stands
  literally: rule 1 friendly with a last direction, rule 2 enemy with a last direction (pushed
  opposite), rule 3 ranged, rule 4 toward center. With one `last_dir` per unit, a ranged unit with
  a `last_dir` is caught by rule 1 or 2, so enemy Archers that only ever attacked are pushed opposite
  to their attack direction. Logged here so milestone 5 does not rediscover it. - Confirmed

## Tests (phase A guidance)
Golden values come from the rules above and hand arithmetic, never from running the code.
1. **economy helpers:** `cumulative_population_for_level` for levels 1..8 (0, 2, 5, 9, 14, 20, 27,
   35); `income` table: level 1 capital = 2, + workshop = 3, level 3 non-capital = 3, + 2 parks;
   sum over several cities; `research_cost` for tier 1 and 2 with 1 and 3 cities.
2. **income timing:** stars across a scripted sequence of EndTurns: no income on turn 0 for either
   player, income for player 0 after the second EndTurn of player 1, and so on.
3. **Research / Train:** prerequisite, cost, already-researched, insufficient stars; Train blocked by
   occupied city tile, locked unit (Rider without Riding), Super Unit never trainable, capacity
   (level 1 city supports 2 units: Warrior from new_game + 1 trained, third illegal), stars exact;
   new unit fields (phase Done, origin, id sequence).
4. **Site actions:** one golden per action (cost, pop, resource/terrain/improvement result, stars for
   ClearForest); blocked by enemy unit, allowed with friendly unit, not owned tile, wrong tech, animal
   on forest blocks LumberHut/ClearForest until hunted, improvement blocks a second action.
5. **Levels and rewards:** population 1 -> 2 triggers level 2 with a pending Workshop; each menu
   entry's effect; Resources +5 stars; PopulationGrowth +3 pop; BorderGrowth claims exactly the
   expected set on a drawn board (skips owned, village and city tiles, off-map) and extends the
   explored mask; BorderGrowth absent from the menu once used; forced-reward blocking (every other
   action illegal, lowest-id pending city first on a synthetic two-pending state).
   Cascade golden on a synthetic state (city level 4, pending, population 12; choose
   PopulationGrowth; expect level 5 pending, then Park). It is unreachable in normal play, so build it
   with the StateBuilder.
6. **Property / playout:** extend the milestone 3 random playouts and the apply-vs-legal property
   test to all new actions (weight Train, Site, Research, ChooseReward so they actually occur; start a
   fraction of games with extra stars, techs and a pre-grown city so levels 3 to 5 are visited).
   Extra invariants checked after every action: supported units per city <= level + 1; stars never
   negative; the D5 level/pending invariants for every city; `improvement` never set on a tile that
   already had one; ownership still immutable; a pending reward exists only for the current player's
   cities.
7. **Data tests:** `Tables::load` parses both new files; malformed/duplicate/missing entries are
   `Err`; menus cover levels 2..4 and an open-ended 5+.

Mutants to aim for: income granted on turn 0 or to the wrong player; workshop/park/capital bonus
dropped; tech cost ignoring tier or city count; Train ignoring capacity, tech, or the occupied tile;
new unit phase Fresh; site action allowed on enemy-occupied or unowned tile; wrong tile condition
(animal on forest allowed hut); harvest not consuming the resource; population to the wrong city;
level threshold off by one; pending reward not blocking other actions; reward effect wrong (e.g.
Resources +3); cascade skipped; BorderGrowth claiming villages or already-owned tiles; BorderGrowth
reusable.

## Gate
`./scripts/check.sh` passes, including extended property tests and playouts, under about 60 seconds
in a debug build.

## Notes for the implementer
- Rules data (costs, techs, menus, gains) comes from the tables; structural tile conditions and
  effects live in code as written above.
- No `HashMap`/`HashSet` in anything that affects state or action order.
- If the tests contradict this brief or the spec, stop and add a QUESTIONS.md entry.

## Appendix: data files to add (copy exactly)

`data/site_actions.json`
```json
[
  { "action": "harvest_animal", "cost": 2, "tech": "hunting",      "population": 1, "stars": 0 },
  { "action": "harvest_fruit",  "cost": 2, "tech": "organization", "population": 1, "stars": 0 },
  { "action": "lumber_hut",     "cost": 3, "tech": "forestry",     "population": 1, "stars": 0 },
  { "action": "clear_forest",   "cost": 0, "tech": "forestry",     "population": 0, "stars": 1 },
  { "action": "farm",           "cost": 5, "tech": "farming",      "population": 2, "stars": 0 },
  { "action": "mine",           "cost": 5, "tech": "mining",       "population": 2, "stars": 0 }
]
```

`data/rewards.json`
```json
{
  "menus": [
    { "min_level": 2, "max_level": 2,    "options": ["workshop"] },
    { "min_level": 3, "max_level": 3,    "options": ["city_wall", "resources"] },
    { "min_level": 4, "max_level": 4,    "options": ["population_growth", "border_growth"] },
    { "min_level": 5, "max_level": null, "options": ["park"] }
  ],
  "resources_stars": 5,
  "population_growth_amount": 3
}
```