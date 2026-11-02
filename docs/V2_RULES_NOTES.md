# v2 rules: working notes (merged into the spec at milestone 15)

Owner rulings collected while drafting the v2 spec. Sources: the owner's in-game values and rulings
(authoritative), [namu.moe](https://namu.moe/w/polytopia) (secondary, partly an older game version).
The Polytopia Fandom wiki could not be read automatically (bot protection). Open points are at the end.

## Scope (owner, 2026-11-01)

Land only (water and naval units in v3). v1 stays runnable behind a ruleset flag. The v1 policy is
carried over to v2 by distillation or a warm start (decided in the transfer milestone).

## Technologies (owner, 2026-11-01)

Cost formula unchanged: `4 + cities x tier`.

| tech | tier | requires | unlocks |
| --- | --- | --- | --- |
| Roads | 2 | Riding | Road |
| Free Spirit | 2 | Riding | Disband |
| Trade | 3 | Roads | Market |
| Chivalry | 3 | Free Spirit | Knight, Destroy |
| Mathematics | 3 | Forestry | Catapult, Sawmill |
| Construction | 3 | Farming | Windmill, Burn forest |
| Smithery | 3 | Mining | Swordsman, Forge |

## Units (in-game values from the owner, 2026-11-01)

| unit | health | attack | defense | movement | range | cost | skills |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Swordsman | 15 | 3 | 3 | 1 | 1 | open | open |
| Catapult | 10 | 4 | 0 | 1 | 3 | open | open |
| Knight | 10 | 3.5 | open | 3 | 1 | open | open |

## Tile actions and buildings

- **Burn forest** (Construction, 3 stars): forest becomes field with crop (owner).
- **Road** (owner): a building, 3 stars. May share a tile with other buildings and vice versa. Any
  terrain except mountain (and water, ice, algae, which v2 does not have). Does not remove terrain
  features. May be built under a neutral village; it disappears when the village is captured. Cannot be
  built in, or used in, enemy territory; any player may build and use roads in neutral territory. Cannot
  be destroyed.
- **Road movement** (owner): a step between two tiles that both have a road (or a road and a city) costs
  0.5 movement and ignores the forest stop, but not Zone of Control. Costs round up: a unit with 0.5
  movement left can still enter a tile costing 1 (or 3). The final tile does not need a road, and an
  obstacle on the final tile does not block the move.
- **City connection** (owner): the capital is connected to another own city when roads link them; the
  path may pass through own cities.
- Sawmill, windmill, forge, market: namu.moe values, see "Open points".

## Monuments (owner, 2026-11-01)

v2 has three: Park of Fortune (task: a city reaches level 5), Gate of Power (task: kill 10 enemy units),
Eye of God (task: discover all four lighthouses). The other monuments' tasks are impractical in the v2
ruleset. Each monument: +3 population to its city, 400 score (namu.moe; to confirm).

## Lighthouses (owner, 2026-11-01)

A lighthouse stands on each of the four corners of the map. Lighthouse tiles are always fogged at the
start, even if they would be visible on turn 0. When a player first removes the fog over a lighthouse
(by a unit's vision or the Explorer), that player's capital gains 1 population.

## Open points (to ask the owner)

See the orchestrator's questions of 2026-11-01 in the session; answers go here.
