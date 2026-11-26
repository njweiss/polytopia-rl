//! v2 economy buildings: levels, Market income and sync (docs/MILESTONE_17 "Building levels and Market income").

use crate::data::{Building, LevelFrom};
use crate::grid::{all_tiles, neighbors, Pos};
use crate::state::{Improvement, PlayerId, Ruleset, State};

fn improvement_name(imp: Improvement) -> &'static str {
    match imp {
        Improvement::LumberHut => "lumber_hut",
        Improvement::Farm => "farm",
        Improvement::Mine => "mine",
    }
}

fn building_name(b: Building) -> &'static str {
    match b {
        Building::Sawmill => "sawmill",
        Building::Windmill => "windmill",
        Building::Forge => "forge",
        Building::Market => "market",
    }
}

fn tile_matches_adjacent(state: &State, q: Pos, adjacent: &[String]) -> bool {
    let t = state.tile(q);
    t.improvement
        .is_some_and(|imp| adjacent.iter().any(|name| name == improvement_name(imp)))
        || t.building
            .is_some_and(|nb| adjacent.iter().any(|name| name == building_name(nb)))
}

/// Whether `pos` has at least one 8-adjacent friendly tile matching `building`'s `adjacent` list.
pub(crate) fn has_matching_adjacent(
    state: &State,
    owner: PlayerId,
    pos: Pos,
    building: Building,
) -> bool {
    let info = state.tables.building(building);
    neighbors(state.size, pos).any(|q| {
        let qt = state.tile(q);
        qt.owner_city.is_some_and(|c| state.city(c).owner == owner)
            && tile_matches_adjacent(state, q, &info.adjacent)
    })
}

/// Dynamic level of the economy building at `pos` (0 if unowned, empty, or under `V1`).
pub fn building_level(state: &State, pos: Pos) -> u32 {
    if state.config.ruleset != Ruleset::V2 {
        return 0;
    }
    let t = state.tile(pos);
    let (Some(b), Some(owner_city)) = (t.building, t.owner_city) else {
        return 0;
    };
    let owner = state.city(owner_city).owner;
    let info = state.tables.building(b);
    let mut total = 0u32;
    for q in neighbors(state.size, pos) {
        let qt = state.tile(q);
        if !qt.owner_city.is_some_and(|c| state.city(c).owner == owner) {
            continue;
        }
        match info.level_from {
            LevelFrom::Count => {
                if tile_matches_adjacent(state, q, &info.adjacent) {
                    total += 1;
                }
            }
            LevelFrom::Levels => {
                if qt
                    .building
                    .is_some_and(|nb| info.adjacent.iter().any(|name| name == building_name(nb)))
                {
                    total += building_level(state, q);
                }
            }
        }
    }
    total
}

/// Per-turn star income produced by a star-generating building at `pos` (capped by `max_output`).
pub fn market_income(state: &State, pos: Pos) -> u32 {
    if state.config.ruleset != Ruleset::V2 {
        return 0;
    }
    let Some(b) = state.tile(pos).building else {
        return 0;
    };
    let info = state.tables.building(b);
    if info.stars_per_level == 0 {
        return 0;
    }
    let raw = building_level(state, pos) * info.stars_per_level;
    match info.max_output {
        Some(cap) => raw.min(cap),
        None => raw,
    }
}

/// Syncs `tile.building_level` and owning-city population deltas for all buildings in two passes:
/// `LevelFrom::Count` first (row-major), then `LevelFrom::Levels` (row-major).
pub(crate) fn sync_buildings(state: &mut State) {
    for pass in [LevelFrom::Count, LevelFrom::Levels] {
        for pos in all_tiles(state.size) {
            let idx = state.index(pos);
            let Some(b) = state.tiles[idx].building else {
                continue;
            };
            let info = state.tables.building(b);
            if info.level_from != pass {
                continue;
            }
            let pop_per_level = info.population_per_level as i32;
            let old_level = state.tiles[idx].building_level;
            let new_level = building_level(state, pos) as u8;
            if new_level != old_level {
                if let Some(owner_city) = state.tiles[idx].owner_city {
                    let delta = (new_level as i32 - old_level as i32) * pop_per_level;
                    state.city_mut(owner_city).population += delta;
                }
                state.tiles[idx].building_level = new_level;
            }
        }
    }
}
