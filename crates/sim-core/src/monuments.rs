//! v2 lighthouses and monuments (docs/MILESTONE_18).

use crate::data::{MonumentTask, ALL_MONUMENTS};
use crate::grid::Pos;
use crate::roads::capital_of;
use crate::state::{PlayerId, Ruleset, State};

/// The four corner positions of a `size x size` map in row-major order.
pub fn corners(size: u8) -> [Pos; 4] {
    let last = size - 1;
    [
        Pos::new(0, 0),
        Pos::new(0, last),
        Pos::new(last, 0),
        Pos::new(last, last),
    ]
}

/// Index `0..=3` if `pos` is one of `corners(size)`, else `None`.
pub fn corner_index(size: u8, pos: Pos) -> Option<usize> {
    corners(size).iter().position(|&c| c == pos)
}

/// Triggers lighthouse discovery on a `false -> true` fog transition over a corner lighthouse
/// under `V2` while the game is non-terminal (docs/MILESTONE_18 M2, Q48).
pub(crate) fn discover_lighthouse(state: &mut State, player: PlayerId, at: Pos) {
    if state.config.ruleset != Ruleset::V2 || state.outcome.is_some() {
        return;
    }
    if !state.tile(at).lighthouse {
        return;
    }
    let Some(ci) = corner_index(state.size, at) else {
        return;
    };
    let bit = 1u8 << ci;
    if state.players[player as usize].lighthouses_discovered & bit != 0 {
        return;
    }
    state.players[player as usize].lighthouses_discovered |= bit;
    if let Some(cap) = capital_of(state, player) {
        let pop = state.tables.v2_rules().lighthouse_population as i32;
        state.city_mut(cap).population += pop;
    }
}

/// Permanently unlocks every monument whose table-defined task is satisfied for either player.
pub(crate) fn update_monument_unlocks(state: &mut State) {
    for p in 0..2usize {
        for m in ALL_MONUMENTS {
            if state.players[p].monuments_unlocked.contains(m) {
                continue;
            }
            let info = state.tables.monument(m);
            let done = match info.task {
                MonumentTask::CityLevel => state
                    .cities
                    .iter()
                    .any(|c| c.owner == p as u8 && c.level >= info.threshold),
                MonumentTask::Kills => state.players[p].kills >= info.threshold,
                MonumentTask::Lighthouses => {
                    state.players[p].lighthouses_discovered.count_ones() >= info.threshold
                }
            };
            if done {
                state.players[p].monuments_unlocked.insert(m);
            }
        }
    }
}
