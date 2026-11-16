//! Movement and defense bonus (spec: "Movement"; docs/MILESTONE_3 "reachable_tiles",
//! "defense_bonus").

use crate::data::{Skill, Tech};
use crate::grid::{all_tiles, chebyshev, neighbors, Pos};
use crate::map::Terrain;
use crate::roads::road_usable;
use crate::state::{PlayerId, Ruleset, State, UnitId};

/// Whether `player`'s techs allow standing on `p`'s terrain: mountains need Climbing.
pub(crate) fn terrain_allows(state: &State, player: PlayerId, p: Pos) -> bool {
    state.tile(p).terrain != Terrain::Mountain
        || state.players[player as usize]
            .techs
            .contains(Tech::Climbing)
}

/// Tiles the unit could move to with its full movement budget, row-major, excluding its own
/// tile. Ignores the unit's phase.
///
/// Counted in half-points (docs/MILESTONE_16 "Road movement"): the budget is `2 x movement`; under
/// v2 a step between two tiles `road_usable` for the owner costs `2 x road_step_cost`, every other
/// step 2, floored at 0, and a step needs at least 1 left. Arriving next to an enemy, or on a
/// forest or mountain by a non-road step, stops the unit (the start tile is exempt). Under v1
/// every step is a normal step. The search keeps, per tile, the most budget left on arrival
/// without a stop; more budget there reaches a superset of tiles.
pub fn reachable_tiles(state: &State, unit: UnitId) -> Vec<Pos> {
    let Some(u) = state.unit(unit) else {
        return Vec::new();
    };
    let budget = 2 * state.tables.unit(u.kind).movement;
    let road_cost = match state.config.ruleset {
        Ruleset::V1 => None,
        Ruleset::V2 => Some((2.0 * state.tables.v2_rules().road_step_cost) as u32),
    };
    let enemies: Vec<Pos> = state
        .units
        .iter()
        .filter(|e| e.owner != u.owner)
        .map(|e| e.pos)
        .collect();

    let enterable = |q: Pos| {
        state.is_explored(u.owner, q)
            && state.tile(q).unit.is_none()
            && terrain_allows(state, u.owner, q)
    };
    let zoc = |q: Pos| enemies.iter().any(|&e| chebyshev(e, q) == 1);
    let rough = |q: Pos| matches!(state.tile(q).terrain, Terrain::Forest | Terrain::Mountain);

    let mut reached = vec![false; state.tiles.len()];
    // The most budget left on a non-stopping arrival at each tile.
    let mut best: Vec<Option<u32>> = vec![None; state.tiles.len()];
    best[state.index(u.pos)] = Some(budget);
    let mut stack = vec![(u.pos, budget)];
    while let Some((from, left)) = stack.pop() {
        if left < 1 || best[state.index(from)] != Some(left) {
            continue;
        }
        for q in neighbors(state.size, from) {
            if !enterable(q) {
                continue;
            }
            let road_step = road_cost
                .filter(|_| road_usable(state, u.owner, from) && road_usable(state, u.owner, q));
            let rest = left.saturating_sub(road_step.unwrap_or(2));
            let i = state.index(q);
            reached[i] = true;
            if zoc(q) || (rough(q) && road_step.is_none()) {
                continue;
            }
            if best[i].is_none_or(|b| rest > b) {
                best[i] = Some(rest);
                stack.push((q, rest));
            }
        }
    }
    reached[state.index(u.pos)] = false;

    all_tiles(state.size)
        .filter(|&q| reached[state.index(q)])
        .collect()
}

/// The defense bonus for `defender` on its current tile (largest applicable value, no stacking).
pub fn defense_bonus(state: &State, defender: UnitId) -> f32 {
    let u = state.unit(defender).expect("defender is alive");
    let techs = &state.players[u.owner as usize].techs;
    let rules = state.tables.rules();
    let tile = state.tile(u.pos);
    let mut bonus = 1.0f32;
    if (tile.terrain == Terrain::Forest && techs.contains(Tech::Archery))
        || (tile.terrain == Terrain::Mountain && techs.contains(Tech::Climbing))
    {
        bonus = bonus.max(rules.terrain_defense_bonus);
    }
    if let Some(cid) = tile.city {
        let city = state.city(cid);
        let fortify = state.tables.unit(u.kind).skills.contains(&Skill::Fortify);
        if city.owner == u.owner && fortify {
            bonus = bonus.max(if city.wall {
                rules.wall_defense_bonus
            } else {
                rules.city_defense_bonus
            });
        }
    }
    bonus
}
