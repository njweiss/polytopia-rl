//! Movement and defense bonus (spec: "Movement"; docs/MILESTONE_3 "reachable_tiles",
//! "defense_bonus").

use crate::data::{Skill, Tech};
use crate::grid::{all_tiles, chebyshev, neighbors, Pos};
use crate::map::Terrain;
use crate::state::{PlayerId, State, UnitId};

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
/// Breadth-first by step count. Whether a tile can be entered and whether it stops further
/// movement depend only on the tile, so the fewest steps to a tile leaves the most budget there.
pub fn reachable_tiles(state: &State, unit: UnitId) -> Vec<Pos> {
    let Some(u) = state.unit(unit) else {
        return Vec::new();
    };
    let budget = state.tables.unit(u.kind).movement;
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
    let stops = |q: Pos| {
        matches!(state.tile(q).terrain, Terrain::Forest | Terrain::Mountain)
            || enemies.iter().any(|&e| chebyshev(e, q) == 1)
    };

    let mut seen = vec![false; state.tiles.len()];
    seen[state.index(u.pos)] = true;
    let mut frontier = vec![u.pos];
    for _ in 0..budget {
        let mut next = Vec::new();
        for &from in &frontier {
            // The start tile is exempt from forest, mountain and ZoC stops.
            if from != u.pos && stops(from) {
                continue;
            }
            for q in neighbors(state.size, from) {
                let i = state.index(q);
                if !seen[i] && enterable(q) {
                    seen[i] = true;
                    next.push(q);
                }
            }
        }
        frontier = next;
    }
    seen[state.index(u.pos)] = false;

    all_tiles(state.size)
        .filter(|&q| seen[state.index(q)])
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
