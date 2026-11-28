# Polytopia Headless Sim: Spec

## Overview and goals

This spec is the single authoritative rulebook for the deterministic, headless Rust simulator of a reduced 1v1 Battle of Polytopia and its RL training stack. The owner makes every rules decision. Agents implement against this document and its tests.

**Sources of truth:** the owner's rulings and in-game values first, then the Polytopia wiki and the Tribes framework as cross-checks. Exact outcome parity with the real game is not required.

## Scope and rulesets

The simulator supports two drylands 1v1 Imperius rulesets selected by `config.ruleset` (`v1` by default, or `v2`):

- **v1:** 11x11 land-only map, Imperius vs. Imperius (starts with Organization and 5 stars), 4 trainable units (Warrior, Rider, Archer, Defender) plus the Super Unit, skills (Dash, Escape, Fortify, Static), 9 technologies, villages and city capture, border growth, city walls, workshops, parks, Explorer (resolved inside the level-2 reward), fog of war (with optional full observability), and veteran promotion.
- **v2:** extends v1 with 7 land technologies (Roads, Free Spirit, Trade, Chivalry, Mathematics, Construction, Smithery), 3 units (Swordsman, Catapult, Knight) and skills (Stiff, Persist), roads and city connections, 4 economy buildings (Sawmill, Windmill, Forge, Market), tile/unit actions (Burn forest, Disband, Destroy), 3 monuments (Park of Fortune, Gate of Power, Eye of God), corner lighthouses with an updated Explorer walk, and a 6-channel damage-preview observation (see **Game v2 rules** below). Under `v1`, every v1 rule, serialization, replay, and hash is unchanged.

**Config values (not hard-coded):** turn cap (30, see Termination), score weights, `max_city_level` (unset means no cap), `ruleset` (`v1` or `v2`), map size (minimum 6), `full_observability` flag, `auto_recover` (off by default).

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

**Unit phases**.

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

Source: the wiki's Movement page. Road movement applies under `ruleset = v2` (see **Game v2 rules**).

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

**Forced rewards.** If any city has a pending reward, the only legal action is that city's Choose reward. Nothing else is legal until it resolves, including unit actions, Promote and End turn (D9). When several cities are pending, resolve in ascending city id. Under v2, a pending reward blocks only its owner (chosen at the start of their turn); a captured city keeps its pending reward and the capturer must choose it at once (Q49).

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

Under v2, monuments add 400 points each and surplus population is floored at 0 (see **Game v2 rules**).

- **City points.** The wiki scores each population at 5 and gives a level-up bonus of `50 - 5 × level`, so every level is worth 50 in total. A city's points are therefore `100 + 50 × (level - 1) + 5 × surplus population`. The bonus turns negative at level 10 and above, but the total per level stays 50, so the formula needs no special case.
- **Starting score is 515 for each player:** capital 100, territory 9 tiles = 180, exploration 25 tiles = 125, Warrior 10, Organization 100. The wiki's own check, a level 1 city with 9 tiles of territory, gives 280, which matches the first two terms.
- A captured city moves its city, tile, and park points to the new owner. Units of the captured city that lose their origin keep scoring for their owner.
- Confirmed by the owner: units score while alive, partial population scores, and starting techs score. Both players' scores are public.
- Score is a u32 derived on demand (F6). Surplus population is measured against the threshold of the city's current level (F7).
- Exploration counts the explored mask, so under full observability both players score 605 for it on an 11x11 map (F8).
- Test: the starting score above, and a scripted game where each term changes by exactly its weight.

## Map generation

Map generation copies the Tribes framework's `LevelGenerator` on a drylands map (default 11x11, two Imperius players). Where the Tribes generator and the wiki disagree on map generation, Tribes wins. Under v2, the four corner tiles are set to fogged plain-field lighthouses with no resource (see **Lighthouses and the Explorer**).

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

`observe(state, player)` reads only `PlayerView` and returns what that player may know. Explored tiles show live state, so the only hidden information is unexplored tiles and opponent private state (stars, techs, income, enemy unit phases).

- **Per-tile channels (v1: 37; v2: 64):**
  - v1 prefix (channels 0–36): `explored`, terrain (3), resource (4), improvement (3), `village`, territory (mine/enemy), city (mine/enemy, normalized level, capital), own unit type (5), enemy unit type (5), health fraction, own unit phase (4), `veteran`, `can_promote`, `can_harvest`, `can_build`.
  - v2 suffix (channels 37–63): own/enemy v2 unit types (3+3), `road`, `city.connected`, economy buildings (4) and normalized `building.level`, `lighthouse`, monuments (3), v2 legality flags (`can_build_road`, `can_build_building`, `can_build_monument`, `can_destroy`), and 6 damage-preview channels (`preview.target_can_kill`, `preview.target_max_dmg`, `preview.target_min_retal`, `preview.target_def_bonus`, `preview.attacker_can_kill`, `preview.attacker_max_dmg`).
- **Tech-gated visibility:** crops need Organization and metal needs Climbing, so the observation depends on the explored mask **and** the player's techs.
- **Unexplored tiles** are all zeros apart from the explored flag. Enemy units show health and veteran status, not phase (J2). A terminal observation is still valid, from the observer's side, with every mask false (J3).
- **Global features (v1: 18; v2: 33):**
  - v1 prefix (0–17): normalized stars, income, turn fraction (`turn / turn_cap`), pending-reward flag, own score, opponent score, score difference, city count, unit count, and 9 v1 tech flags.
  - v2 suffix (18–32): 7 v2 tech flags, normalized kills (`min(kills, 10) / 10`), lighthouses discovered (`count / 4`), 3 monument-unlocked flags, and 3 monument-placed flags.
- Unbounded values (level, population, stars, scores) are normalized and clipped when encoded. The observation is fixed-size `float32`: a channels-first board `(C, H, W)` plus a global vector `(G,)`, pinned by `obs_hash` (J2, J9).
- Full observability is a config flag that sets the explored mask to all ones. The explored mask grows when a unit moves (radius 1, or radius 2 on a mountain) and whenever a city is founded, captured, or gains tiles, since cities see every tile they own.
- Both players' total scores are public. The opponent's stars, techs, and income are hidden (G4); under v2, the damage-preview channels expose the defender's true defense bonus on legal attacks as a deliberate, documented exception.

### Action space

One environment step is one game action, and `EndTurn` is a global action. The learner picks in three conditional stages:

1. **Actor head:** logits over `3 * size * size + 1` slots: three per tile (unit, city, site) plus one global slot. A unit and a city never share a slot (J6).
2. **Action-type head:** conditioned on the actor, masked to legal types (10 types in v1; 15 types in v2, appending `Disband`, `BuildRoad`, `BuildBuilding`, `BuildMonument`, `Destroy`).
3. **Argument head:** per-tile logits for `Move` and `Attack`; small enum heads for `Train`, `ChooseReward`, `Site`, `Research`, and (in v2) `BuildBuilding` and `BuildMonument`. `Capture`, `Recover`, `Promote`, `EndTurn`, `Disband`, `BuildRoad`, and `Destroy` take no argument (J7).

- Masks at each stage come from `legal_actions`, and a pending reward restricts everything to that city.
- The joint log-probability sums only the heads that fired. Entropy is approximated by summing conditional entropies along the sampled path.

### Interface and training

- Single-environment Rust sim exposed through PyO3 (`sim-env` and `sim-py`), vectorized in Python (`SyncVecEnv`). `Env` runs in bot mode (opponent turns execute inside `step`) or two-seat mode (the caller supplies every action). Both `ruleset = "v1"` and `ruleset = "v2"` are supported end to end.
- **Policy network (`PolicyNet`):** supports `arch="v1"` (standard Conv/ResNet trunk + shared attention key) and `arch="v2"` (CoordConv, pre-norm `GroupNorm` + SiLU + pooled-context/globals FiLM + residual-scaled `ResBlockV2`, separate `move`/`attack` keys with QK `LayerNorm` and 2D relative position bias in `ArgHeadV2`).
- **Algorithm:** hand-written CleanRL-style PPO in PyTorch with conditional actor, type, and argument masks (K1), `gamma = 0.999` (K4), and conditional-path entropy (K8). Linear LR annealing to zero (`anneal_lr = true`) is the default.
- **Policy transfer (`transfer.py`):** supports v1→v2 prefix warm-start (`warm_start_checkpoint`, copying the shared v1 channel/global/head prefixes into a v2 network) and temperature-scaled 3-stage factored KL distillation (`teacher_checkpoint`, `distill_coef`, `distill_steps`, `distill_temp`) with a linear decay schedule and a single teacher pass per rollout.
- **Curriculum and self-play:** curriculum training progresses through scripted bot stages (`random` → `economy` → `economy` + `rush`) on training win rate (K7). Self-play supports both the legacy snapshot pool (`rollout="interleaved"` or `"lockstep"`, `opponent_forward="per_snapshot"` or `"stacked"`, `p_bot` 0.4 in `selfplay.json`, `p_latest` 0.5, pool cap 40) and revised self-play (`rollout="stream"` barrier-free streaming rollouts with `compute_gae_segments`, `opponent_forward="auto"`, live two-sided self-play `p_self`, PFSP `"hard"`/`"variance"`, flexible `selfplay_bots`, pinned cross-arch `anchor_checkpoints`, win-rate-gated snapshots, shaping annealing, and paired seeds).
- **Rewards:** `terminal_weight × (+1 win, -1 loss, 0 draw) + shaping_weight × (gamma × phi' - phi)`, with `phi = (my score - opponent score) / 1000` and terminal potential 0. In bot mode a step's reward spans the opponent's turn (J4). The Rust `Env` defaults to `shaping_weight = 0`; training configs default to `1.0` (K3), with optional linear annealing (`shaping_anneal_steps`, `shaping_end`) in self-play. Self-play recomputes the same formula in Python, verified against Rust (L11).
- Every episode seed is replayable. Evaluation plays each seed twice with sides swapped; the learner's seat is seed parity (J10). Episodes truncate after 5,000 agent actions as a safety net, separate from the turn cap (J5).
- Seed ranges never overlap: training below `1e9`, evaluation and the suite in `[1e9, 1.5e9)`, ratings and tournaments from `2e9` (K6, M11).
- **Evaluation suite:** 8 fixed conditions (`random_e0`, `economy_e0`, `rush_e0`, `economy_e20`, `rush_e20`, `economy_fullobs`, `rush_fullobs`, `rush_e0_sampled`), 200 side-swapped seeds. Adding a condition is a spec change (L6). Full observability is a diagnostic of distribution shift, not a target (L7). Reports always show win rate by seat (L10).
- **Ratings:** Bradley-Terry fits in Elo units from side-swapped matches (`random` anchored at 0, draw = 0.5, one virtual 0.5-0.5 prior game per played pair, L5).
- **Spec hashes (`size = 11`):** checkpoints store `obs_hash` (refuses to load on mismatch) and `rules_hash` (warns on mismatch; covers `data/` plus `RULES_VERSION`, L9).
  - v1: `obs_hash = d5f9b1033cfd55ca`, `rules_hash = a8b52b91eae34503`
  - v2: `obs_hash = c15247decc21ae4f`, `rules_hash = ed271736a688e86d`

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

Milestones M1–M23 are implemented and merged on `main`:

- **M1–M8 (v1 simulator, tooling, bots):** data tables, combat, map generation, movement/legality, economy/Explorer/Super Unit, capture/win/score, renderer/replay/HTML viewer/CLI, `PlayerView` and scripted bots.
- **M9–M14 (v1 RL, evaluation, speed, scaling):** observation/action space, PyO3 bindings, CleanRL-style PPO, curriculum, checkpoint hashes, evaluation suite, Bradley-Terry Elo, snapshot-pool self-play, multi-seed experiment harness, interleaved/stacked rollouts, and the capacity ladder (`docs/RL_REPORT.md`).
- **M15–M20 (v2 simulator, bots, renderer, RL env):** `ruleset` flag and `data/v2/` tables, roads and city connections, economy buildings and `BurnForest`/`Disband`/`Destroy`, lighthouses/v2 Explorer/monuments, v2 units (`Swordsman`, `Catapult`, `Knight`) and skills (`Stiff`, `Persist`), v2 bots/renderer/CLI, and v2 64-ch/33-global observation with 6 damage-preview channels and 15 factored actions.
- **M16b (infrastructure):** rollout batching/sync fixes, `osx-arm64` pixi support, worktree-safe checks, paired statistics.
- **M21–M23 (v2 policy, transfer, self-play):** `arch="v2"` `PolicyNet`, v1→v2 prefix warm-start and 3-stage factored KL distillation, and revised self-play (PFSP, live two-sided self-play, barrier-free stream rollouts, cross-arch anchors, win-rate gating, shaping annealing, paired seeds).

Each milestone's brief is `docs/MILESTONE_N.md`. Current status lives in `docs/STATUS.md`, and the three-session test-first workflow is documented in `docs/WORKFLOW.md`.

## Decision log and open questions

### Decisions

| Topic | Decision |
| --- | --- |
| Scope | 1v1, Imperius, 11x11 drylands, reduced tech tree |
| Fog of war | Explored tiles show live state. Full observability is a flag. |
| Vision | Units see radius 1, or radius 2 on a mountain. Cities see every tile they own. |
| Step semantics | One environment step is one game action. End turn is a global action. |
| Action structure | Factored heads with four actor slots: unit, city, site, global (10 action types in v1, 15 in v2). |
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
| Pending rewards | Forced and immediate, may cascade. Simultaneous pending rewards resolve by ascending city id. Under v2, a pending reward blocks only its owner; a captured city's pending reward transfers to the capturer (Q49). |
| Resource rates | Follows the Tribes generator's per-tile rolls with the values in terrainProbs.json: forest 0.40, mountain 0.15, and fruit, crop, animal, and metal 0.5, with the outer ring at one third. Imperius multipliers are 1 in that file. |
| Explorer | Vision radius 1, no mountain bonus. Walk follows the wiki's 15-move fog-seeking algorithm with a 0.5 backtrack penalty. Spawns on the granting city, ignores units and Zone of Control, blocked only by mountains without Climbing. Lighthouses ignored. |
| Roads | Under `ruleset = v2` (see **Game v2 rules**) |
| Map scope | Drylands only (1v1 Imperius); corner lighthouses under `ruleset = v2` |
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
| Self-play | Legacy snapshot pool (`p_bot = 0.4` in `selfplay.json`, `p_latest = 0.5`, pool cap 40) and M23 revised self-play (`rollout="stream"`, live two-sided `p_self`, PFSP, pinned cross-arch anchors, win-rate gating, shaping annealing, paired seeds) |
| Evaluation | Fixed suite of 8 conditions, 200 side-swapped seeds, disjoint seed ranges, win rate by seat always, Elo with random at 0 (K6, L5 to L7, L10) |
| Versioning | Checkpoints carry `obs_hash` and `rules_hash`. `obs_hash` mismatch refuses to load; rules changes in code bump `RULES_VERSION` by hand (L9). |
| Damage preview | Implemented in v2 as 6 observation channels (`PlayerView::attack_preview`): a deliberate, documented exception to hidden techs (`obs_hash c15247decc21ae4f`). |
| Ruleset | Config value `v1` (default) or `v2`, left out of v1 JSON so v1 stays byte-identical. v2 data in `data/v2/`; both rulesets supported across sim, bots, CLI, `Env`/`PyEnv`, and RL training; attack is `f32` (Q42 to Q46). |

### Open questions

Open agent questions live in `docs/QUESTIONS.md`; answered entries are logged in `docs/DECISIONS.md` and merged into this spec.

# Game v2 rules

The v2 ruleset is the v1 spec above plus the additions below. Under `ruleset = v1` none of this applies. All v2 rules are implemented across `sim-core`, `bots`, `sim-tools`, `sim-env`, `sim-py`, and `polytopia_rl`.

## Scope

v2 adds seven land technologies, their units, buildings, and actions, three monuments, corner lighthouses, and the damage preview on the same 11x11 drylands 1v1 Imperius map.

- **Added in v2:** Roads, Free Spirit, Trade, Chivalry, Mathematics, Construction, Smithery; Swordsman, Catapult, Knight (with Stiff and Persist); road, sawmill, windmill, forge, market; burn forest, disband, destroy; Park of Fortune, Gate of Power, Eye of God; lighthouses on the four corners; 6-channel damage preview in observations.
- **Ruleset flag:** `config.ruleset` = `v1` (default) | `v2`. Under `v1` every rule, table entry, replay, and hash of v1 is unchanged; v2 table entries are ignored. The flag is part of `rules_hash`.

**Data and versioning** (Q42–Q47):

- v2 data lives in `data/v2/` (`units.json`, `techs.json`, `site_actions.json`, `buildings.json`, `monuments.json`, `rules.json`), loaded on top of the v1 tables via `Tables::from_json_v2`. Every v2 entry appears exactly once, a prerequisite always has a lower tier, and building adjacency names must exist.
- `ruleset` is omitted from serialized v1 config JSON so v1 replays and hashes remain byte-identical. For `size = 11`: v1 `rules_hash = a8b52b91eae34503` / `obs_hash = d5f9b1033cfd55ca`; v2 `rules_hash = ed271736a688e86d` / `obs_hash = c15247decc21ae4f`.
- Under v2, the state digest appends the ruleset byte, road flags, city connection links, economy buildings, lighthouses, monument state, and player kill counts, and hashes all 16 tech bits (Q42, Q47).
- Unit `attack` is `f32` in the tables and combat so the Knight's `3.5` is exact (Q45).

## Technologies

Seven new techs; the cost formula stays `4 + (city count × tier)` (owner, 2026-11-01).

| Tech | Tier | Requires | Unlocks |
| --- | --- | --- | --- |
| Roads | 2 | Riding | Road |
| Free Spirit | 2 | Riding | Disband |
| Trade | 3 | Roads | Market |
| Chivalry | 3 | Free Spirit | Knight; Destroy |
| Mathematics | 3 | Forestry | Catapult; Sawmill |
| Construction | 3 | Farming | Windmill; Burn forest |
| Smithery | 3 | Mining | Swordsman; Forge |

The Technology score term (100 × tier) applies to the new techs unchanged.

## Units

In-game values from the owner (2026-11-01/02).

| Unit | Cost (stars) | Health | Attack | Defense | Movement | Range | Skills | Unlocked by |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Swordsman | 5 | 15 | 3 | 3 | 1 | 1 | Dash, Fortify | Smithery |
| Catapult | 8 | 10 | 4 | 0 | 1 | 3 | Stiff | Mathematics |
| Knight | 8 | 10 | 3.5 | 1 | 3 | 1 | Dash, Fortify, Persist | Chivalry |

- **Stiff:** the unit never retaliates. With no Dash, a Catapult cannot attack after moving (owner).
- **Persist:** each kill by the unit's attack restores a full attack (the phase goes back to allowing an attack), with no limit. It does not restore movement, apart from the kill-move onto the target's tile (owner). Whether an Escape move remains available after a Persist chain is not relevant: the Knight has no Escape.
- **Ranged:** the Catapult attacks at Chebyshev distance up to 3 and never kill-moves (v1 rule for ranged units).
- **Score:** units score 5 per star of cost as in v1: Swordsman 25, Catapult 40, Knight 40.
- **Bots:** under v2, `RushBot` researches `Riding -> FreeSpirit -> Chivalry`, trains in priority order `Knight -> Rider -> Warrior`, and builds unlocked monuments; `EconomyBot` researches all 15 v2 techs, builds monuments and economy buildings (`Market`, `Forge`, `Sawmill`, `Windmill`), and trains `Swordsman -> Catapult -> Warrior` while units < `2 × cities`.

## Roads and city connections

A road is a building (Roads tech, 3 stars) that halves movement on roads and forms city connections (owner).

- **Where:** any tile except mountains (v2 has no water). It does not remove terrain features, and it may share a tile with another building or a city. In your own territory, the usual site-action rules apply (no enemy unit on the tile). In neutral territory, any explored, unoccupied tile, including under a neutral village. Never in enemy territory. (owner)
- **Lifetime:** roads cannot be destroyed. A road under a neutral village disappears when the village is captured. (owner)
- **Movement:** a step between two tiles that both have a road, or a road and a city, costs 0.5 movement and ignores the forest stop; Zone of Control still applies. Costs round up: a unit with 0.5 movement left can still enter a tile that costs 1 (or more). The final tile needs no road, and an obstacle on the final tile does not block the move. Roads in neutral territory work for both players; roads in enemy territory work for neither. (owner)
- **City connection:** a city is connected when roads link it to its owner's capital; the path may pass through the owner's cities. When a city becomes connected, it and the capital each gain +1 population; when the connection is lost, both lose it again. (owner)
- **Game over:** connections are not updated after the action that ends the game, so a capital capture changes no population through connections (owner, 2026-11-12, Q48).
- **Rewards on the other player's turn:** a road (even a neutral one the other player builds) can connect a city and level it up during the other player's turn. A pending reward blocks only its owner, who chooses it at the start of their turn; a city captured with a pending reward keeps it, and the capturer must choose it at once (owner, 2026-11-19, Q49).
- **Who acts:** building a road is a site action on the tile (including explored, unoccupied neutral tiles), encoded as `ActionType::BuildRoad` in the site actor family.

## Buildings and tile actions

The economy buildings take their level from the improvements next to them, and their output follows that level as it changes (owner, from namu.moe).

| Building | Tech | Cost | Placed | Effect per level | Cap |
| --- | --- | --- | --- | --- | --- |
| Sawmill | Mathematics | 5 | field next to a lumber hut | +1 population per adjacent lumber hut | one per city |
| Windmill | Construction | 5 | field next to a farm | +1 population per adjacent farm | one per city |
| Forge | Smithery | 5 | field next to a mine | +2 population per adjacent mine | one per city |
| Market | Trade | 5 | field next to a sawmill, windmill or forge | +1 star per turn per level of the adjacent sawmills, windmills and forges | 8 stars; one per city |

- **Adjacent** means the 8 surrounding tiles (owner, 2026-11-02).
- **Levels update:** when a neighbor is added, changes level or goes away, the building's level and its population or income change at once; the population goes to (or leaves) the city that owns the building's tile (owner).
- **Burn forest** (Construction, 3 stars): a forest tile with no improvement becomes field with crop (owner). An animal on it is forfeited, as with Clear forest (owner, 2026-11-02).
- **Disband** (Free Spirit): removes one of the player's units and refunds half its cost, rounded down (owner). Legal in any phase, like Promote, and it is not a kill (owner, 2026-11-02). A Super Unit counts as worth 10 stars and refunds 5 (owner, 2026-11-02).
- **Destroy** (Chivalry): removes any building except a road on the player's territory, with no refund; the population and score the building gave are removed too (owner). Destroying an improvement next to an economy building lowers that building's level (follows from the level rule).

## Monuments

v2 has three monuments; each is free, gives its city +3 population and its owner 400 score, and is placed on any field tile in one of the player's cities' territory that has no building. A player may place each monument once per game, after completing its task (owner, 2026-11-01/02).

| Monument | Task (unlocks it) |
| --- | --- |
| Park of Fortune | one of the player's cities reaches level 5 (by growth or capture) |
| Gate of Power | the player has killed 10 enemy units |
| Eye of God | the player has discovered all four lighthouses (active from the start) |

- Placing a monument is a site action on the target tile (owner, 2026-11-02). Kills count as in v1 (attack or retaliation, the Super Unit included) (owner, 2026-11-02).
- A monument is a building: Destroy removes it with its population and score, and it cannot be placed again (namu.moe: a destroyed monument cannot be rebuilt; owner, 2026-11-02).

## Lighthouses and the Explorer

A lighthouse stands on each of the map's four corner tiles (owner, 2026-11-01).

- Lighthouse tiles are always field, never carry a resource, village or city, and can be occupied by units (owner).
- They start fogged for both players, even when the capital's starting vision would cover them (owner).
- The first time a player removes the fog over a lighthouse, by a unit's vision or the Explorer, that player's capital gains +1 population. Each player gets this once per lighthouse (owner).
- **Explorer:** v2's Explorer no longer ignores lighthouses. New walk step 3b: a candidate whose uncovered fog tiles include a lighthouse gets +3.5, added after the cap of 4 on the fog count, so a candidate that uncovers one fog tile hiding a lighthouse (4.5) beats every candidate without one (owner, 2026-11-02).

## Population loss, score and the damage preview

- **Population loss:** a city's level never goes down. Its population may fall below the current level's threshold, even below zero, and must grow back past it before the next level-up. Population is lost when a building that gave it is destroyed or loses levels, and when a city connection is lost (owner, 2026-11-02).
- **Score:** v1's terms continue (units, territory, exploration, cities, parks, techs). New: monuments 400 each. Roads and the economy buildings score only through the population they give (owner, 2026-11-02). The city term's surplus population is never negative: below the threshold it counts as 0 (owner, 2026-11-02).
- **Damage preview:** an observation feature, not an action. `PlayerView::attack_preview(unit, target)` evaluates the combat formula using the defender's true defense bonus (`dealt`, `retaliation`, `defender_dies`, `defense_bonus`) for every legal attack—a deliberate, documented exception to the hidden-tech rule. In v2 observations (64 board channels, 33 globals, `obs_hash = c15247decc21ae4f`), previews are aggregated across legal attacks into 6 channels:
  - On each target tile: `preview.target_can_kill` (binary), `preview.target_max_dmg` (`max(dealt / target.hp)`), `preview.target_min_retal` (`min(retaliation / attacker.hp)`), `preview.target_def_bonus` (`defense_bonus / 4.0`).
  - On each attacker tile: `preview.attacker_can_kill` (binary), `preview.attacker_max_dmg` (`max(dealt / target.hp)`).

**Sources:** owner rulings and in-game values (authoritative); [namu.moe: The Battle of Polytopia](https://namu.moe/w/polytopia), [Pocket Gamer unit list](https://www.pocketgamer.com/the-battle-of-polytopia/the-battle-of-polytopia-cheats-and-tips-list-of-every-unit-and-tribe/), and [West Games damage calculator](https://west-games.com/polytopia-damage-calculator/) (secondary cross-checks).
