# v2 rules: working notes (merged into the spec on 2026-11-10)

Historical notes. The rulebook is `docs/SPEC.md` ("Game v2 rules") and the Claude Doc tab "v2 rules".

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
| Swordsman | 15 | 3 | 3 | 1 | 1 | 5 | Dash, Fortify |
| Catapult | 10 | 4 | 0 | 1 | 3 | 8 | Stiff |
| Knight | 10 | 3.5 | 1 | 3 | 1 | 8 | Dash, Fortify, Persist |

- **Stiff** (owner, 2026-11-02): never retaliates. With no Dash, a Catapult cannot attack after moving.
- **Persist** (owner, 2026-11-02): each kill restores a full attack, with no limit. It does not restore
  movement, apart from the kill-move onto the target's tile.

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
- **Road building in neutral territory** (owner, 2026-11-02): on any explored, unoccupied neutral tile.
- **Road movement in neutral territory** counts for both players (owner, 2026-11-02).
- **Connection reward** (owner, 2026-11-02): when a city becomes connected to the capital, both it and
  the capital gain +1 population (namu.moe, confirmed).
- **Sawmill, windmill, forge, market** (namu.moe, confirmed by the owner 2026-11-02): 5 stars each; one
  of each per city; placed next to what they count. Sawmill +1 population per adjacent lumber hut,
  windmill +1 per adjacent farm, forge +2 per adjacent mine; market +1 star per turn per level of the
  adjacent sawmills, windmills and forges, at most 8. A building's level is its adjacency count, and it
  updates when a neighbor is added, changes or goes away.
- **Disband** (Free Spirit, owner 2026-11-02): removes one of your units and refunds half its cost,
  rounded down; legal in any phase; not a kill. A Super Unit counts as worth 10 stars and refunds 5.
- **Destroy** (Chivalry, owner 2026-11-02): removes any building except a road on your territory, with no
  refund; the population and score it gave are removed too.

## Monuments (owner, 2026-11-01)

v2 has three: Park of Fortune (task: a city reaches level 5), Gate of Power (task: kill 10 enemy units),
Eye of God (task: discover all four lighthouses; active from the start). The other monuments' tasks are
impractical in the v2 ruleset. Each monument: free, +3 population to its city, 400 score; placed on any
field tile in a city's territory without a building; one of each per player per game (owner, 2026-11-02).

## Lighthouses (owner, 2026-11-01)

A lighthouse stands on each of the four corners of the map. Lighthouse tiles are always fogged at the
start, even if they would be visible on turn 0. When a player first removes the fog over a lighthouse
(by a unit's vision or the Explorer), that player's capital gains 1 population. Every player who reveals
a lighthouse gets this, once per lighthouse (owner, 2026-11-02).

Lighthouse tiles are always field, never carry a resource, village or city, and can be occupied by
units (owner, 2026-11-02).

**Explorer** (owner, 2026-11-02): v2's Explorer no longer ignores lighthouses. New step 3b of the walk:
a candidate whose uncovered fog tiles include a lighthouse gets a bonus a little larger than uncovering
three fog tiles (orchestrator's proposal: +3.5, so it never ties an integer count).

## Population loss (owner, 2026-11-02)

A city's level never goes down. Population can fall below the current level's threshold (even below
zero); the city must grow back past it before its next level-up. Population is lost when a building
that gave it is destroyed or loses adjacency levels, and when a road connection to the capital is lost
(the +1 of the connection is removed from both cities).

## Explorer lighthouse bonus (owner, 2026-11-02)

+3.5 per candidate whose uncovered fog tiles include a lighthouse, added after the cap of 4 on the fog
count; so a candidate uncovering 1 fog tile that hides a lighthouse scores 4.5 and beats every candidate
without one. Approved.

## Not in v2 (owner, 2026-11-02)

Water and naval units, ruins (random bonuses add variance; the owner leans against them), the other
monuments. Candidates for v3.

## Approved proposals (owner, 2026-11-02)

The ruleset flag `v1` | `v2` (part of `rules_hash`); building a road is a site action, including on
neutral tiles; "adjacent" = the 8 surrounding tiles; burn forest forfeits an animal; disband as above;
the bots at least handle the new units and actions legally (how much they use them: units milestone);
placing a monument is a site action, kills count as in v1, a destroyed monument cannot be placed again;
no separate score for roads and economy buildings, and surplus population is floored at 0 in the score.

The full v2 draft is the Claude Doc tab "v2 rules (draft)"
(https://claude.ai/artifact/3yVekWM5SAoUVxjZVmomTm#f868c7c8-c04b).

## Open points

None as of 2026-11-02.
