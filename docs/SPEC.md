# Polytopia Headless Sim: Spec v0.1

Oct 24, 2026 · @Nicholas Weiss

## Overview and goals

This spec defines a deterministic, headless Rust simulator of a reduced 1v1 Battle of Polytopia, built to train a PPO agent. The owner makes every rules decision. Agents implement against this document and its tests.

**Success criteria.** A working agent that makes sensible decisions given the rules and state, and a solid understanding of why it succeeds or fails. The sim never has to interoperate with the real game.

**Priorities, in order:**

1. RL understanding
2. Agentic coding workflow skill
3. Optimization (a side product, not a goal)

**Sources of truth:** the Polytopia wiki first, then the owner's play experience, then the Tribes framework as a cross-check. Exact outcome parity with the real game is not required.

**Resources:** Ryzen 5 5600X3D and an RTX 4060 (8 GB). No speed target until a profile exists.

## Scope and feature flags

v1 is one tribe (Imperius), 1v1, an 11x11 land-only map, four unit types, and a nine-tech tree. Everything else sits behind a flag so it can return as a later milestone without a state-layout rewrite.

**In v1**

- Tribe: Imperius (starts with Organization and 5 stars)
- Units: Warrior, Rider, Archer, Defender, plus the Super Unit from level 5 up
- Skills: Dash, Escape, Fortify
- Villages and city capture, border growth, city walls, workshops, parks
- Explorer (resolved inside the level-2 reward, not a persistent unit)
- Fog of war, with full observability as a flag
- Veteran promotion

**Out of v1 (flagged)**

| Feature | Flag | Notes |
| --- | --- | --- |
| Roads and the Roads tech | `roads` | Removed from the tech list and from city connection rewards |
| Water, ports, naval units | `water` | No water techs, so the map is drylands only |
| Ruins | `ruins` | Excluded from map generation |
| Other skills (Persist, etc.) | none yet | Added per unit as needed |
| Other tribes | none yet | Needs per-tribe spawn modifiers |
| Lighthouses | lighthouses | Corner tiles are plain land in v1, and the Explorer's lighthouse bonus is ignored |

**Config values (not hard-coded):** turn cap (30, see Termination), score weights, `max_city_level` (unset means no cap), map size, full-observability flag, auto-recover (off by default).

## Data tables

All rules data lives in tables loaded at startup, not in code branches. Numbers below come from the owner's wiki extract. They live in data/units.json, techs.json, site\_actions.json, rewards.json and rules.json, loaded through Tables. Every entry appears exactly once, and reward menus cover every level from 2 up exactly once (Q3).

**Units**

| Unit | Cost (stars) | Health | Attack | Defense | Movement | Range | Skills | Unlocked by |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Warrior | 2 | 10 | 2 | 2 | 1 | 1 | Dash, Fortify | none (available at start) |
| Rider | 3 | 10 | 2 | 1 | 2 | 1 | Dash, Fortify, Escape | Riding |
| Archer | 3 | 10 | 2 | 1 | 1 | 2 | Dash, Fortify | Archery |
| Defender | 3 | 15 | 1 | 3 | 1 | 1 | Fortify | Strategy |
| Super Unit | reward only | 40 | 5 | 4 | 1 | 1 | Static | level 5+ reward |

The Super Unit cannot become a veteran and does not count toward any city's unit capacity.

The Super Unit's only skill is Static, which means it cannot promote. With no Dash, Escape, or Fortify it either moves or attacks each turn, and it gets no city defense bonus. A city wall's bonus also requires Fortify, so the Super Unit never gets it.

**Technologies**

Cost is `4 + (city count × tier)`, where city count is the player's current number of cities.

| Tech | Tier | Requires | Unlocks |
| --- | --- | --- | --- |
| Climbing | 1 | none | Mountain movement and defense bonus; metal visible |
| Hunting | 1 | none | Animal harvesting |
| Organization | 1 | none | Fruit harvesting; crops visible (Imperius starts with it) |
| Riding | 1 | none | Rider |
| Archery | 2 | Hunting | Archer; forest defense bonus |
| Forestry | 2 | Hunting | Lumber hut; clear forest |
| Farming | 2 | Organization | Farm on crops |
| Strategy | 2 | Organization | Defender |
| Mining | 2 | Climbing | Mine on metal |

**Tile actions (performed by the player)**

| Action | Cost (stars) | Reward | Needs |
| --- | --- | --- | --- |
| Harvest animal | 2 | +1 population | Hunting |
| Harvest fruit | 2 | +1 population | Organization |
| Lumber hut | 3 | +1 population | Forestry, on forest |
| Clear forest | 0 | +1 star | Forestry, on forest |
| Farm | 5 | +2 population | Farming, on crop |
| Mine | 5 | +2 population | Mining, on metal |

- Harvest animal and harvest fruit consume the animal or fruit. They need only the resource, so an animal under a lumber hut can still be hunted, and the hut stays (Q2).
- Lumber hut, farm, and mine are improvements and do not consume anything: the forest stays forest, a farm keeps its crop, and a mine keeps its metal. A tile holds at most one improvement, and improvements never change or disappear in v1 (owner overrule of M4 D4, Q2).
- Lumber hut needs a forest with no improvement. It can be built on a forest with an animal, which stays. The forest still stops movement and still gives the Archery defense bonus (D3).
- Clear forest needs a forest with no improvement. It turns the tile into field with no resource, so an animal on it is forfeited.
- A friendly unit on the tile does not block a tile action. An enemy unit does (D10).

**Level-up rewards**

The choice is forced and happens the moment population crosses the threshold. It can occur more than once in a turn.

| New level | Choices |
| --- | --- |
| 2 | Workshop, or Explorer |
| 3 | City Wall, or Resources (+5 stars) |
| 4 | Population Growth (+3 population), or Border Growth |
| 5 and up | Park, or Super Unit |

- Going from level n to n+1 needs n+1 more population. Cumulative totals: level 2 at 2, level 3 at 5, level 4 at 9, level 5 at 14.
- Border Growth can be taken once per city. It claims unowned tiles within radius 2, never village or city tiles, and is legal even when it claims nothing (D7).
- Parks stack. There is no level cap.
- Levels rise immediately: level goes up the moment the threshold is crossed, and the pending reward marks the open choice, so the menu depends only on the new level (D5).

## State definition

The sim holds full ground truth. Each player's observation is a function of that state (see Observation). `apply(state, action)` is a pure function of state and action, so the RNG state is part of the game state. The RNG is pcg32 (M2 C1). An illegal action changes nothing (C12).

| Scope | Fields |
| --- | --- |
| Global | turn number (a round counter from 0, incremented when player 1 ends, Q1), current player, RNG state, config, winner (if any) |
| Player | stars, researched techs, explored mask (monotonic, never shrinks), no stored score, since score is derived from state |
| Tile | terrain (field, forest, mountain), resource, improvement, owning city id (written at most once), occupying unit id |
| City | owner, tile, level, population, workshop flag, park count, wall flag, border-growth-used flag, pending reward, is-capital |
| Unit | id, type, owner, position, health, max health, origin city (nullable), kills, veteran flag, promotion-available flag, phase, last direction (one value, see Movement) |

**Unit phase****s**.

| Phase | Meaning | Legal next actions |
| --- | --- | --- |
| Fresh | Nothing done this turn | Move, attack, capture, recover |
| Moved | A pre-attack move was used | Attack (only with Dash) |
| Attacked | An attack was made | One move (only with Escape) |
| Done | Turn spent | None (promotion is still allowed) |

- Phase only advances within a turn and resets to Fresh at the start of the owner's turn.
- A melee unit that kills its target moves onto the vacated tile as part of the attack. That move does not use the Escape move.
- Newly trained units and a spawned Super Unit start Done (E6).
- Capture and recover move a unit from Fresh straight to Done.
- Escape also applies after a move-then-attack. After a kill-move the phase is Attacked, so a Rider can still use its Escape move (C11).

**Ownership rules**

- Tile ownership is by city, not by player. A city's owner can change on capture, and its tiles follow it.
- A tile's owning city never changes once set.
- City capture transfers all of the city's tiles, keeps its level, population and buildings, and orphans the previous owner's units from that city (origin city becomes null, and they never count toward capacity again).
- Village capture creates a level 1 city. It claims the village tile and the neighbors within radius 1 that are unclaimed. The new city's id is the number of existing cities, so ids are dense and never reused (F1).
- Unit capacity per city is `level + 1`, counting living units whose origin is that city (D8). A unit that captures a village or city migrates its origin to that city. Super Units never count.

## Combat and movement rules

### Combat

Formulas are taken from the wiki. Arithmetic is `f32`, evaluated in exactly the order written, with rounding that sends .5 up (Rust `f32::round` on non-negative values).

```latex
\begin{aligned}
\text{attackForce} &= \text{attacker.attack} \cdot \frac{\text{attacker.health}}{\text{attacker.maxHealth}} \\
\text{defenseForce} &= \text{defender.defense} \cdot \frac{\text{defender.health}}{\text{defender.maxHealth}} \cdot \text{defenseBonus} \\
\text{totalDamage} &= \text{attackForce} + \text{defenseForce} \\
\text{attackResult} &= \operatorname{round}\!\left(\frac{\text{attackForce}}{\text{totalDamage}} \cdot \text{attacker.attack} \cdot 4.5\right) \\
\text{defenseResult} &= \operatorname{round}\!\left(\frac{\text{defenseForce}}{\text{totalDamage}} \cdot \text{defender.defense} \cdot 4.5\right)
\end{aligned}
```

- `attackResult` is damage dealt to the defender. `defenseResult` is retaliation damage dealt to the attacker.
- **Defense bonus:** 1 by default, 1.5 for forest (needs Archery), mountain (needs Climbing), or a city (needs Fortify), and 4 for a city wall (also needs Fortify). The Super Unit has no Fortify, so it never gets a city or wall bonus. The defender's techs decide whether the terrain bonus applies. The city bonus applies only in the defender's own city, and the wall bonus replaces it rather than multiplying it (C8).
- **Retaliation** happens only if the defender survives and the attacker is within the defender's range. An Archer attacking from two tiles away takes no retaliation from a melee unit.
- A kill by attack or by retaliation counts toward veteran promotion, so both sides earn kills. Kills keep counting after promotion (C9).
- **Kill-move:** a melee attacker that kills moves onto the target's tile, unless the tile needs a tech the attacker's owner lacks (for example a mountain without Climbing). Ranged attackers never move on a kill. Melee means range 1; an Archer may attack at distance 1 or 2 (C5).
- **Veterans:** after 3 kills a unit becomes eligible. Promotion is an explicit action that raises max health by 5 and fully heals. It does not use the unit's turn and is legal in any phase. The Super Unit cannot be promoted.
- The formula, including retaliation, was checked against five results from the wiki's combat calculator (table below). All five match under \`f32\` arithmetic with half-up rounding. Example 1 is an exact .5 tie (4.5 rounds up to 5), so it rules out round-half-to-even.

**Combat golden tests** (from the wiki calculator; HP values are before and after one attack)

| # | Attacker | Atk HP | Defender | Def HP | Defense bonus | Atk final HP | Def final HP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Warrior | 10 | Warrior | 10 | 1 | 5 | 5 |
| 2 | Warrior | 6 | Warrior | 10 | 1 | 0 | 7 |
| 3 | Warrior | 10 | Defender | 15 | 1 | 2 | 11 |
| 4 | Rider (veteran, max HP 15) | 15 | Defender | 11 | 1.5 | 7 | 8 |
| 5 | Defender | 15 | Archer | 10 | 4 | 11 | 9 |

- Case 4 only works if the Rider's max health is 15, so the calculator was run on a veteran. The test must set max health explicitly.
- Case 2 covers a retaliation kill: the attacker dies while the defender survives.
- Case 1 is an exact .5 tie (attack and defense forces are both 2), so it pins down half-up rounding. Add a standalone unit test of the rounding function on .5 inputs.
- Both forces use each unit's health before the attack. All five cases match that reading.

### Movement

Source: the wiki's Movement page. Roads are out of scope.

- A unit reaches tiles within Chebyshev distance of its movement stat, subject to the rules below. Each step costs 1, cost rounds up, and fog tiles cannot be entered. Every step of the path uses the explored mask from before the move (C2).
- **Rough terrain:** a unit may move onto a forest or mountain tile but no further that turn. Mountains also need Climbing.
- **Zone of Control:** a unit that arrives on a tile adjacent to an enemy unit cannot move further that turn. It can still attack with Dash. The effect ends when the enemy unit dies. Only enemy units exert it, it applies on arrival, and never to the start tile (C4). An enemy on an unexplored tile still exerts it, so the movement mask can hint at a hidden unit; accepted for now (Q11).
- A unit that starts inside Zone of Control may leave it, and may even re-enter it.
- One unit per tile. Units cannot enter occupied tiles. Unoccupied cities and villages can be entered, so a capture can follow on the next turn (C1).
- Escape grants one move after an attack, with a full movement budget.
- Units cannot pass through friendly units.
- Last direction: one value per unit. A move (including a kill-move and an Escape move) sets it to the sign of the net displacement per axis; a ranged attack sets it toward the target. Recover, Promote and pushes leave it unchanged (C6).

### Forced unit spawn

When a Super Unit spawns on its city tile and a unit already stands there, the occupant is pushed. Direction rules, in order:

1. A friendly unit that previously moved is pushed the way it moved.
2. An enemy unit that previously moved is pushed opposite to its move.
3. A ranged unit is pushed the way of its last move or last attack.
4. A unit that has not moved is pushed toward the map center. On the exact center tile, it is pushed south.
5. If the target tile is occupied or impassable, try counterclockwise then clockwise, one tile at a time. If no free tile exists, the unit is removed from the game. This does not count as a kill.

A unit that is forcefully moved does not change its phase.

- "Previously moved" means ever. A unit's last direction is never reset between turns, and a unit that has never moved has no direction.
- Direction toward the center is the sign of the offset on each axis, so a unit on a diagonal is pushed diagonally.
- Probe order: try the intended direction, then alternate counterclockwise and clockwise one neighbor at a time (45 degrees, 45, 90, 90, 135, 135, then 180).
- The map center is (size - 1) / 2 on each axis, with integer division (E9).
- With one last direction per unit, a ranged unit that has a last direction is caught by rule 1 or 2, so an enemy Archer that only ever attacked is pushed opposite to its attack direction. Rule 3 therefore never fires; a ranged unit with no direction falls to rule 4 (D11).
- A pushed unit reveals fog for its owner at its new tile. It may be pushed onto an empty city or village tile (E8).

## Actions and legality

Every action has one actor kind. The sim exposes `legal_actions(state)` as structured values and `apply(state, action)`. Any encoding for the learner lives outside the sim.

| Actor | Action | Legal when | Effect |
| --- | --- | --- | --- |
| Unit | Move (destination tile) | Phase is Fresh, or Attacked with Escape. Destination is reachable under the movement rules. | Unit moves. Fresh goes to Moved. Attacked goes to Done. Explored mask updates. |
| Unit | Attack (target tile) | Phase is Fresh, or Moved with Dash. An enemy unit is within the unit's range (Chebyshev) on a tile the player has explored (C3). | Combat resolves. Phase becomes Attacked. |
| Unit | Capture | Phase is Fresh. The unit stands on a village or an enemy city. Any unit type can capture (F4). | Village becomes a level 1 city, or the city changes owner. Phase becomes Done. |
| Unit | Recover | Phase is Fresh. Legal at full health. | Heals 4 in friendly territory, 2 elsewhere. Phase becomes Done. |
| Unit | Promote | Promotion is available and the unit is not already a veteran. Any phase. | Max health +5, full heal. Phase unchanged. |
| City | Train (unit type) | The city belongs to the player. The city tile is empty. The unit is unlocked. Stars cover the cost. Supported units are below `level + 1`. | A new unit appears on the city tile in phase Done. |
| City | Choose reward (option) | The city has a pending reward. | Applies the reward. May cascade into another level-up. |
| Site (tile) | Harvest, lumber hut, clear forest, farm, mine | The tile is owned by the player's city. No enemy unit is on it (a friendly unit is fine). The tech is unlocked. The tile meets the action's needs (see Tile actions). Stars cover the cost. | Population goes to the **city that owns the tile**. Stars change per the tile action table. |
| Global | Research (tech) | Prerequisite is researched. Stars cover the cost. | Tech is added. |
| Global | End turn | Always legal when no reward is pending. | Ends the player's turn. |

**Forced rewards.** If any city has a pending reward, the only legal action is that city's Choose reward. Nothing else is legal until it resolves, including unit actions, Promote and End turn (D9). When several cities are pending, resolve in ascending city id. With roads out of v1, only one city can have a pending reward at a time, so this rule matters only once roads return.

**Capture notes**

- Capture must be a unit's first and only action of the turn, so a unit enters the tile one turn and captures the next. It has to survive in between.
- A tile occupied by an enemy unit cannot be entered, so a defender must be killed first.

**Level-up resolution:** the Explorer reward resolves its whole random walk inside the Choose reward action. See Turn flow for the walk rules.

## Turn flow, economy, and termination

### Start of a player's turn

1. Add income to the player's stars.
2. Reset every unit of that player to phase Fresh. Last-move memory is never reset.
3. The player acts until they choose End turn.

Income is added before units reset. Neither player gets income on their first turn (turn 0), only the starting stars (D1). Player 0 always moves first; the turn counter starts at 0 and goes up by 1 when player 1 ends (Q1, C10).

### Income

- Each city gives 1 star per turn per level.
- The capital gives 1 extra star per turn.
- Each workshop and each park gives 1 extra star per turn.

### Population and levels

- Population is stored per city and credited to the city that owns the tile acted on.
- Cumulative population needed for level L is `L(L+1)/2 - 1`, so 2, 5, 9, 14 for levels 2 to 5.
- Crossing a threshold sets a pending reward immediately. A reward such as +3 population can trigger another level-up.
- Starting population is 0 for both the capital and a village-converted city.

### Explorer

- Granted by choosing Explorer at level 2. It is not a unit, takes no tile, and never appears in the state.
- It makes 15 moves, then disappears. It heads for nearby fog and wanders only when none is in reach, as the walk rules below describe.
- It can enter mountains only with Climbing.
- Because it is random, the walk consumes the state's RNG, which keeps replays exact.
- Its vision is radius 1 around each tile it reaches, with no mountain bonus. It spawns on the city tile that granted it, ignores units and Zone of Control, and is blocked only by mountains when the owner lacks Climbing.

**Explorer walk.** Each of the 15 moves repeats these steps, taken from the wiki's description of the algorithm. Lighthouses are ignored in v1.

1. Scan for fog, meaning tiles the owner has not explored, that the explorer could uncover by standing on a tile within 4 steps. The explorer sees radius 1, so fog up to 5 tiles away counts.
2. Candidate tiles are those within 4 steps from which standing there uncovers at least one fog tile. The nearest candidates win.
3. Among equally near candidates, score each by the number of fog tiles it uncovers, capped at 4, then subtract 0.5 if the explorer already crossed that tile earlier in the walk. Each uncovered fog tile counts 1, so the penalty never creates a tie with a different count.
4. If the best score is still tied, pick one at random from the state's RNG, with candidates ordered row by row so replays are identical.
5. Move one step toward the winner. If several first steps lie on shortest paths, pick one at random from the RNG, in row-major order.
6. If no fog is within reach, step to a random adjacent passable tile. Tiles just crossed carry the penalty, so the explorer backtracks only when nothing else is open.

Steps 2 and 3 treat candidates as tiles the explorer can stand on, as confirmed by the owner.

- The visited set includes the starting tile, and the explorer reveals radius 1 at its start (E1).
- Random draws happen only on a tie (more than one candidate), in the order given, so a seed always gives the same walk (E2).
- Wandering prefers unvisited neighbors and falls back to any passable neighbor (E3).
- Candidates exclude the current tile. Distance is BFS over passable tiles; forests, fog and occupied tiles do not block (E4).
- All 15 moves are always taken, even when nothing is left to explore (E5).
- Every visited tile is already revealed, so it is never a candidate and the 0.5 penalty never changes a choice. The rule is kept as written (Q4).

### Starting state

- One capital city, one Warrior on the city tile, 5 stars, the Organization tech. One coin flip per game assigns the two capitals to the players (C10).
- Visible tiles: everything within Chebyshev distance 2 of the capital.
- Capital initial border radius is 1. Units see Chebyshev radius 1 by default. Units on a mountain see Chebyshev radius 2. Cities see every tile they own.

### Termination

- A player wins by capturing the other player's capital. The game ends at once; nothing else happens (F3).
- Otherwise the game ends when player 1 ends turn N - 1, so each player gets N turns. There is no income or unit reset after that. N is the turn cap, a config value of at least 1, set to 30 (open question 1, closed 2026-10-24) (F5).
- At the cap, the higher score wins and only exactly equal scores draw. There is no other elimination rule: a player with no units and no stars can still end turns (F9).
- Score is derived from ground-truth state using the formula in the Score section below.
- Reaching the cap is a true termination, not a truncation, so the learner must not bootstrap through it.

### Score

Source: the wiki's Score page, reduced to the v1 rules. Score is computed on demand from state and never stored. Weights live in config.

| Term | Points |
| --- | --- |
| Units alive | 5 per star of unit cost: Warrior 10, Rider 15, Archer 15, Defender 15. Super Unit 50. |
| Territory | 20 per tile owned by the player's cities |
| Exploration | 5 per tile in the player's explored mask |
| Cities | 100 per city, plus 50 per level above 1, plus 5 per population beyond the current level's threshold |
| Parks | 250 each |
| Technology | 100 × tier for each researched tech, including the starting tech |

Out of v1: monuments, temples, and ship upgrades.

- **City points.** The wiki scores each population at 5 and gives a level-up bonus of `50 - 5 × level`, so every level is worth 50 in total. A city's points are therefore `100 + 50 × (level - 1) + 5 × surplus population`. The bonus turns negative at level 10 and above, but the total per level stays 50, so the formula needs no special case.
- **Starting score is 515 for each player:** capital 100, territory 9 tiles = 180, exploration 25 tiles = 125, Warrior 10, Organization 100. The wiki's own check, a level 1 city with 9 tiles of territory, gives 280, which matches the first two terms.
- A captured city moves its city, tile, and park points to the new owner. Units of the captured city that lose their origin keep scoring for their owner.
- Confirmed by the owner: units score while alive, partial population scores, and starting techs score. Both players' scores are public.
- Score is a u32 derived on demand (F6). Surplus population is measured against the threshold of the city's current level (F7).
- Exploration counts the explored mask, so under full observability both players score 605 for it on an 11x11 map (F8).
- Test: the starting score above, and a scripted game where each term changes by exactly its weight.

## Map generation

v1 copies the Tribes framework's `LevelGenerator`, which the owner supplied. Its default setup is an 11x11 map with two tribes, matching v1. Where the Tribes generator and the wiki disagree on map generation, Tribes wins. Water is skipped, so the map is drylands, and ruins and lighthouses are skipped as well.

### Pipeline

All randomness comes from the game's own RNG. Java's `Random` is not replicated, so maps will not match Tribes seed for seed. On other map sizes, capitals use rows and columns 2 to size - 3; the smallest accepted size is 6, the first with two distinct capitals (Q16).

1. Every tile starts as field.
2. **Capital 1:** uniform over the 49 tiles with row and column both in 2 to 8, so at least 2 from the map edge.
3. **Capital 2:** for each of those tiles, take the Chebyshev distance to the nearest existing capital. Pick uniformly among the tiles with the largest such distance.
4. **Forest and mountain:** go through the non-capital tiles in row-major order. For each, draw `u` uniformly from 0 to 1. The tile becomes forest if `u` is below the forest probability, and mountain if `u` is above 1 minus the mountain probability. Otherwise it stays field.
5. **Village map:** mountain tiles and map-edge tiles can never hold a village. Each capital tile is a city, its 8 neighbors (ring 1) are marked inner, and the 16 tiles at distance 2 (ring 2) are marked outer. A tile inside two sites' rings takes the stronger inner mark, and marks never downgrade (C5).
6. **Villages:** repeatedly take the first tile in row-major order that is still unmarked. It becomes a village, and its ring 1 and ring 2 are marked inner and outer. Stop when no unmarked tile is left. Every tile placed this way becomes a village, and a village on a forest replaces the forest. Capital and village tiles are field with no resource (C2).
7. **Resources:** go through the tiles in row-major order. Village and city tiles get none. A resource can only appear on a tile marked inner or outer. A tile marked outer uses one third of the probability. Per terrain:
   - Field: roll fruit with probability `f × (1 - c / 2)`. If that fails, roll crop with probability `c × (1 - f / 2)`, where `f` and `c` are the fruit and crop probabilities.
   - Forest: roll animal.
   - Mountain: roll metal.
8. **Imperius guarantee:** if the 8 ring 1 tiles of a capital hold fewer than 2 fruit, repeat: pick one of the 8 uniformly and overwrite it with field plus fruit, until the count reaches 2.

Skipped stages: the land and water noise, tribe ownership flood fill (both players are Imperius, so it changes nothing), shallow water, ruins, and lighthouses.

### Parameters

Values come from `terrainProbs.json` in the Tribes repository, which the owner supplied. Every Imperius multiplier in that file is 1, so each probability equals its base value.

| Parameter | Value |
| --- | --- |
| Forest probability | 0.40 |
| Mountain probability | 0.15 |
| Fruit `f` and crop `c` | 0.5 each |
| Animal and metal | 0.5 each |
| Outer-ring factor | one third |

Terrain shares measured on this pipeline (20,000 seeds) are 36.1% forest, 14.5% mountain, and 49.4% field, against the wiki's 38, 14, and 48. They fall below the raw probabilities because villages replace forests and the Imperius guarantee overwrites ring tiles with field (C4).

**Resulting rates** (percent of tiles of that terrain)

| Terrain | Outcome | Inner (ring 1) | Outer (ring 2) |
| --- | --- | --- | --- |
| Field | Fruit | 37.5 | 12.5 |
| Field | Crop | 23.4 | 10.9 |
| Field | Empty | 39.1 | 76.6 |
| Forest | Animal | 50.0 | 16.7 |
| Forest | Empty | 50.0 | 83.3 |
| Mountain | Metal | 50.0 | 16.7 |
| Mountain | Empty | 50.0 | 83.3 |

- The Imperius guarantee binds often. Expected fruit in the 8 ring tiles is 1.35, so about 60% of capitals need at least one forced overwrite.
- The wiki gives Imperius 2.0x fruit and 0.5x animals, but the JSON gives it 1 for both. v1 copies the JSON, by the owner's decision. Applying the wiki multipliers through the same formula would give 75% fruit but only about 6% crop on inner field tiles, with the guarantee binding only about 19% of the time.
- Neither version reproduces the owner's in-game normalization of roughly 55% fruit and 27% crop on inner field tiles, because the Tribes formula is not a normalization.

### Known behavior, copied on purpose

These figures come from a simulation of this pipeline with the parameters above, and the statistical tests below should confirm them.

- **Capital separation** is 3 tiles about 2% of the time, 4 about 16%, 5 about 33%, and 6 about 49%.
- **Capital 2 is more peripheral.** It averages 3.0 tiles from the map center, against 2.3 for capital 1.
- **Villages:** about 6.7 per map, from 4 to 7. Each village counted for its nearer capital, the two counts differ by 2 or more in about 60% of maps, and the layout leans slightly toward the top-left because of the row-major scan.
- The pipeline has no validity checks.
- **Mitigation:** each game assigns players to the two capitals at random, and evaluation plays each seed twice with seats swapped.

### Tests

- Reproducible from the seed, bit for bit.
- Capitals are 2 or more from the edge, and every later capital is at maximum distance from the earlier ones.
- Villages are at least 3 from any city or village, never on the map edge or on a mountain, and no eligible (non-edge, non-mountain) tile remains unmarked. The tests replay the greedy placement in full (C3).
- Resources appear only within 2 tiles of a city or village, and the Imperius guarantee holds on every map.
- Over many seeds, capital separation, village counts, and terrain shares match the figures above within tolerance.

## Observation, action space, and ML interface

### Observation

`observe(state, player)` returns what that player may know. Explored tiles show live state, so the only hidden information is unexplored tiles, and a recurrent policy is probably unnecessary for v1.

- **Per-tile channels:** terrain, resource, improvement, explored flag, owning city's owner (mine, enemy, unowned), city and level, unit type and owner, health fraction, unit phase, veteran and promotion-available flags, and whether the tile can currently be harvested or built on.
- **Tech-gated visibility:** crops need Organization and metal needs Climbing, so the observation depends on the explored mask **and** the player's techs.
- **Unexplored tiles** are all zeros apart from the explored flag. Enemy units show health but not phase (J2). A terminal observation is still valid, from the observer's side, with every mask false (J3).
- **Global features:** stars, researched techs, income, turn number divided by the turn cap, and whether a reward is pending. The turn counter is required because the game ends by score at the cap. The turn counter starts at 0.
- Unbounded values (level, population, stars) are normalized and clipped when encoded. The observation is fixed-size float32: a channels-first board plus a global vector, for a map size fixed per Env (J2, J9). The exact layout is in sim-env and pinned by obs\_hash.
- Full observability is a config flag that sets the explored mask to all ones. The explored mask grows when a unit moves (radius 1, or radius 2 on a mountain) and whenever a city is founded, captured, or gains tiles, since cities see every tile they own.
- Both players' total scores are public, so the observation includes the opponent's score and the score difference. The opponent's stars, techs and income are not visible, only the score (G4).
- Planned, not built: the game's damage preview will be added as an observation feature, not an action. It partly reveals the defender's true defense bonus and so the enemy's techs, a deliberate exception to the hidden-tech rule. Adding it bumps obs\_hash.

### Action space

One environment step is one game action, and End turn is a global action. The learner picks in stages.

1. **Actor head:** logits over three slots per tile (unit, city, site) plus one global slot. A unit and a city never share a slot (J6).
2. **Action-type head:** conditioned on the actor, masked to legal types.
3. **Target head:** per-tile logits for move and attack. Enum-valued actions (train, reward, research) use small separate heads, only one of which is active at a time. Site actions need no target, because the tile is the actor; an enum head picks the site action. Capture, Recover, Promote and End turn have no argument head (J7).

- Masks at each stage come from `legal_actions`, and a pending reward restricts everything to that city.
- The joint log-probability sums only the heads that fired. Entropy is approximated by summing conditional entropies along the sampled path.
- A flat-index encoding is a stretch experiment built on the same legal-action list.

### Interface and training

- Single-environment Rust sim exposed through PyO3 (crates sim-env and sim-py), with vectorization on the Python side (a synchronous vector env). Revisit only if profiling shows the interface is the bottleneck (J8, K9). The Env runs in bot mode (the opponent's turns run inside step) or two-seat mode (the caller supplies every action).
- Algorithm: hand-written CleanRL-style PPO in PyTorch with the conditional actor, type and argument masks, since library PPOs do not support them (K1, open question 2 closed). gamma 0.999 (K4). Entropy is the sum of conditional entropies along the sampled path (K8).
- Opponents: scripted bots first, through a curriculum (random, then economy, then economy and rush); a stage advances on the training win rate (K7). Then self-play against a snapshot pool mixed with bots: 20% of episodes against a bot (economy or rush), otherwise a snapshot, the newest with probability 0.5 and a uniform pool entry otherwise. A snapshot every 1M steps, at most 40 in the pool. Snapshots sample their actions. The learner trains only on its own seat's transitions (L1 to L4).
- Rewards: terminal\_weight × (+1 win, -1 loss, 0 draw) plus shaping\_weight × (gamma × phi' - phi), with phi = (my score - opponent score) / 1000 and a terminal potential of 0. In bot mode a step's reward spans the opponent's turn (J4). The Rust Env defaults to shaping\_weight 0; the training config uses 1.0 (K3). Shaping changed behavior in practice, so its weight is an experiment, not a settled default. Self-play recomputes the same formula in Python, pinned by a test against the Rust env (L11).
- Every episode seed is replayable. Evaluation plays each seed twice with sides swapped; the learner's seat is seed parity (J10). Episodes truncate after 5,000 agent actions as a safety net, separate from the turn cap (J5).
- Seed ranges never overlap: training below 1e9, evaluation and the suite in \[1e9, 1.5e9), ratings and tournaments from 2e9 (K6, M11).
- The evaluation suite is the standing benchmark for every run: random, economy and rush at epsilon 0; economy and rush at epsilon 0.2; economy and rush under full observability; rush with sampled actions. 200 seeds, side-swapped. Adding a condition is a spec change (L6). Full observability is a diagnostic of distribution shift, not a target (L7).
- Reports always show the win rate by seat. First-player advantage is measured, not corrected (L10).
- Ratings are Bradley-Terry fits in Elo units from side-swapped matches. The random bot is anchored at 0, a draw counts half a win, and each pair that played gets one virtual 0.5-0.5 game as a prior (L5).
- Checkpoints store obs\_hash (observation layout, action space and enum orders) and rules\_hash (the data tables plus a hand-bumped RULES\_VERSION for rule changes made in code). A checkpoint refuses to load against a different obs\_hash and warns on a different rules\_hash (L9).

## Invariants and test plan

Determinism and testing are day-one requirements. Agent-written code is accepted only when these pass.

**Determinism**

- The same seed and the same action list always produce the same state, bit for bit.
- Every game can be written as a replay log (seed plus actions) and replayed to identical states.
- The RNG state is part of the game state, so cloning a state clones its future.

**Invariants checked after every action in randomized playouts**

- A tile's owning city, once set, never changes.
- Every unit and city sits on exactly one tile, and no tile holds two units.
- A unit's phase only advances within a turn, and all phases reset at the owner's turn start.
- The explored mask never shrinks.
- Health stays between 0 and max health, and dead units are removed.
- Stars never go negative.
- Supported units per city never exceed `level + 1`, except through the capture rules.
- Observations contain no information from unexplored tiles, and no crop or metal data without the matching tech.

**Mask correctness (property tests)**

- Every legal-action path through the factored heads is a legal action in the sim.
- Every legal action is reachable through the heads.
- With a pending reward, only that city's Choose reward is legal.
- Random legal playouts for thousands of seeds finish without panics or illegal states.

**Golden tests**

- Combat: a table of inputs and expected outputs from the wiki's worked examples, covering defense bonuses, veterans, retaliation, and rounding ties.
- Movement: reachable-tile sets for scripted boards, including Zone of Control, forest, mountain, and Rider double moves.
- Forced spawn: one scripted case per direction rule and the probe fallback.
- Level-up cascades: a +3 population reward that triggers a second level.
- Capture: village to city, and city transfer with tiles, level, and buildings kept.

**Tooling**

- A text renderer for the board, and a replay viewer driven by replay logs.
- Heuristic bots serve as both baselines and sim test drivers.
- Replay logs (version 1) store the seed, the actions and a state digest per step. The digest covers all state except the constant tables and identifies a state for verification; it is not a security hash (G1, G2).
- Bots: Random, Rush and Economy. They see a fog-restricted PlayerView, never the full state, and are stateless functions of the view and their own RNG stream (H1). They judge fights with defense bonus 1.0, since enemy techs are hidden (H2). When no reachable tile gains vision, a unit heads for the nearest explored tile that does (Q10). Rush takes the Explorer at level 2 and the Super Unit at level 5; Economy takes neither (H5). The bot rules in docs/MILESTONE\_8.md are the bots' spec, so changing them is a spec change (H6).
- Bot gate: Rush and Economy each beat Random in at least 80% of side-swapped games; a draw is not a win (H3). Rush beats Economy 84% of the time, outside the 25% to 75% target; accepted (Q13).

## Milestones and agent workflow

The sim is finished as a solid prototype before RL starts, which is the owner's choice. Each milestone ends with a gate that agents cannot skip.

1. **Repo and conventions.** Cargo workspace, lint and test commands, and a project instruction file for the coding agent. Gate: CI runs an empty test suite.
2. **Data tables and combat.** Unit, tech, and action tables as data, plus the combat function. Gate: golden combat tests pass.
3. **Map generation.** Seeded 11x11 generator with validity checks. Gate: output proportions match the spec over many seeds.
4. **State, movement, legality.** State structs, phases, reachable tiles, `legal_actions`, `apply`. Gate: movement golden tests and mask property tests pass.
5. **Economy and progression.** Income, population, levels, rewards, Explorer, techs, forced spawns. Gate: level-up and spawn golden tests pass.
6. **Capture, win, score.** Villages, city capture, termination, score. Gate: random playouts always terminate with a valid result.
7. **Tooling.** Text renderer, replay log, replay viewer. Gate: any game replays identically.
8. **Heuristic bots.** At least a rush bot and an economy bot. Gate: they play full games against each other, and the stats show both strategies matter.
9. **Python interface.** PyO3 bindings, observation and mask encoders, benchmark of steps per second. Gate: a random agent runs thousands of games from Python.
10. **RL.** PPO against the bots, then self-play. Flat-action baseline as a stretch experiment.

Status as of 2026-10-24: milestones 1 to 10 are done. Milestone 10's PPO agent wins 100% against random and economy and 97.8% against rush (400 games each). Milestone 11, beyond this list, adds the evaluation suite, checkpoint hashes, Elo ratings and snapshot-pool self-play; it is in progress. Each milestone's brief is docs/MILESTONE\_N.md.

**Working with agents**

- The owner writes or approves every rule. Agents implement, and any ambiguity they find goes back into the Open questions list instead of being guessed.
- Tests come first for each milestone, and the agent works until they pass.
- Agents run with limited permissions and limited scope, and one milestone at a time.
- The owner reviews diffs against this spec before merging.
- Questions to answer along the way: how precise the spec must be for an agent to succeed unattended, and where parallel agents help.

## Decision log and open questions

### Decisions

| Topic | Decision |
| --- | --- |
| Scope | 1v1, Imperius, 11x11 drylands, reduced tech tree |
| Fog of war | Explored tiles show live state. Full observability is a flag. |
| Vision | Units see radius 1, or radius 2 on a mountain. Cities see every tile they own. |
| Step semantics | One environment step is one game action. End turn is a global action. |
| Action structure | Factored heads with four actor slots: unit, city, site, global. Flat baseline is a stretch. |
| Unit state | Per-unit phase (Fresh, Moved, Attacked, Done) plus Dash and Escape as unit-type data |
| Escape | Applies after move-then-attack, with a full movement budget |
| Movement | No passing through friendly units. Unoccupied cities and villages can be entered. Zone of Control comes from enemy units only and stops a unit only when it arrives next to one (C1, C4). |
| Forced moves | A pushed unit keeps its phase. "Previously moved" means ever, and last direction is never reset. Toward-center uses the sign of the offset per axis. Probing alternates counterclockwise and clockwise. |
| Last direction | One value per unit, set by moves (including kill-moves and Escape), and by attacks for ranged units. Recover, Promote and pushes leave it unchanged (C6). |
| Impassable (forced spawn) | A tile the pushed unit's owner cannot enter, or off the map |
| Combat numerics | `f32`, half-up rounding, wiki formulas |
| Recover | Explicit action, legal only from Fresh, legal at full health. No auto-recover by default. |
| Promotion | Explicit action, any phase, does not use the turn |
| Super Unit | Only skill is Static (cannot promote). No Dash, so it moves or attacks. No Fortify, so no city or wall bonus. Forced spawn with push rules. Not counted toward capacity. |
| City and wall bonus | Both require the Fortify skill |
| Starting values | Population 0 for capital and converted villages. Capital border radius 1. |
| Turn start | Income is added first, then units reset |
| Pending rewards | Forced and immediate, may cascade. Simultaneous pending rewards resolve by ascending city id, which only matters once roads return. |
| Resource rates | Follows the Tribes generator's per-tile rolls with the values in terrainProbs.json: forest 0.40, mountain 0.15, and fruit, crop, animal, and metal 0.5, with the outer ring at one third. Imperius multipliers are 1 in that file. |
| Explorer | Vision radius 1, no mountain bonus. Walk follows the wiki's 15-move fog-seeking algorithm with a 0.5 backtrack penalty. Spawns on the granting city, ignores units and Zone of Control, blocked only by mountains without Climbing. Lighthouses ignored. |
| Roads | Out of v1, behind a flag |
| Water, ruins, lighthouses | Out of v1 |
| Level cap | None. Parks stack. |
| Mirrored maps | Rejected because they leak information under fog. Evaluate with swapped sides instead. |
| Win condition | Capital capture, or higher score at the turn cap |
| Order of work | Finish the sim prototype before RL |
| Score | Wiki formula with v1 terms only: units, territory, exploration, cities, parks, techs. See the Score section. Starting score is 515 per player. |
| Map generator | Copy the Tribes LevelGenerator pipeline with water and ruins skipped. Farthest-point capitals, independent terrain per tile, row-major greedy villages. The wiki's domain system is not used. |
| Seat assignment | One coin flip per game assigns the capitals; player 0 moves first. The RL env sets the learner's seat by seed parity, so evaluation over a seed range is side-swapped (C10, J10) |
| Map validity | No checks beyond the pipeline, as in Tribes. A connectivity check is unnecessary on all-land maps, and Climbing is available to both players. |
| Opponent information | Only the opponent's score is visible. Their stars and techs are hidden. |
| Imperius resource multipliers | The Tribes JSON values (1 for fruit and animal), not the wiki's 2.0x fruit and 0.5x animals |
| Turn counter | Starts at 0 and goes up when player 1 ends. Player 0 always moves first. One coin flip assigns the capitals (Q1, C10). |
| Turn cap | 30, a config value. Each player gets 30 turns. Open question 1 closed on 2026-10-24. |
| First-turn income | None on turn 0, only the starting stars (D1) |
| Improvements | Lumber hut, farm and mine do not consume the resource, and a tile holds one improvement. Harvests consume the animal or fruit. Clear forest forfeits an animal (M4 D4 overruled, Q2). |
| Levels | Rise the moment the threshold is crossed; the pending reward marks the open choice (D5) |
| Border Growth | Radius 2, unowned tiles only, never village or city tiles, legal when it claims nothing, once per city (D7) |
| Capture | Any unit type, only from Fresh. A captured city keeps level, population and buildings and orphans that city's units. Capital capture ends the game at once (F1 to F4). |
| Hidden Zone of Control | An enemy on an unexplored tile still exerts Zone of Control, so the move mask can hint at it. Kept for now (Q11). |
| Explorer details | Draws only on ties; all 15 moves always taken; the 0.5 penalty never changes a choice and is kept as written (E1 to E5, Q4) |
| Minimum map size | 6 (Q16) |
| Bots | Random, Rush, Economy on a fog-restricted view, stateless. Their brief rules are their spec. Rush beats Economy 84%, accepted (H1 to H6, Q10, Q13). |
| RL algorithm | Hand-written CleanRL-style PPO with conditional masks. Open question 2 closed (K1). |
| Reward | Terminal win/loss plus potential shaping on score difference / 1000, gamma 0.999, terminal potential 0. Env default shaping 0, training config 1.0; the weight is an experiment (J4, K3, K4). |
| Self-play | Snapshot pool plus 20% bot games; newest snapshot half the time; snapshot every 1M steps, at most 40; snapshots sample; learner trains on its own seat only (L1 to L4) |
| Evaluation | Fixed suite of 8 conditions, 200 side-swapped seeds, disjoint seed ranges, win rate by seat always, Elo with random at 0 (K6, L5 to L7, L10) |
| Versioning | Checkpoints carry obs\_hash and rules\_hash. obs\_hash mismatch refuses to load; rules changes in code bump RULES\_VERSION by hand (L9). |
| Damage preview | To be added later as an observation feature, not an action: a deliberate, documented exception to hidden techs, with an obs\_hash bump |

### Open questions

None open. Agents record new questions in docs/QUESTIONS.md; once the owner answers one, it is merged into this spec. Open question 1 (turn cap) closed at 30 on 2026-10-24; open question 2 (RL library) closed by milestone 10.
