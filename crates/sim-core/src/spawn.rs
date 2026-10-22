//! Super Unit spawn and the forced push of a city tile's occupant (spec: "Forced unit spawn";
//! docs/MILESTONE_5 "Super Unit spawn").

use crate::actions::{direction, reveal};
use crate::data::UnitType;
use crate::grid::Pos;
use crate::movement::terrain_allows;
use crate::state::{CityId, Dir, PlayerId, State, UnitId};

/// The eight directions clockwise as seen on the board (rows grow downward), starting north.
const RING: [Dir; 8] = [
    Dir { dr: -1, dc: 0 },
    Dir { dr: -1, dc: 1 },
    Dir { dr: 0, dc: 1 },
    Dir { dr: 1, dc: 1 },
    Dir { dr: 1, dc: 0 },
    Dir { dr: 1, dc: -1 },
    Dir { dr: 0, dc: -1 },
    Dir { dr: -1, dc: -1 },
];

/// Ring offsets from the intended direction: intended, 1 ccw, 1 cw, 2 ccw, 2 cw, 3 ccw, 3 cw,
/// opposite.
const PROBE_OFFSETS: [usize; 8] = [0, 7, 1, 6, 2, 5, 3, 4];

/// Pushes any occupant off `city`'s tile (or removes it when no neighbor is free), then creates
/// the Super Unit there. Returns the new unit's id. Used by `ChooseReward { SuperUnit }`.
pub fn spawn_super_unit(state: &mut State, city: CityId) -> UnitId {
    let c = state.city(city);
    let (pos, owner) = (c.pos, c.owner);
    if let Some(occupant) = state.tile(pos).unit {
        push(state, occupant, owner);
    }
    state.add_unit(UnitType::SuperUnit, owner, pos, Some(city))
}

/// The direction `occupant` is intended to be pushed when a Super Unit spawns on a city owned by
/// `city_owner` (before probing for a free tile).
pub fn push_direction(state: &State, occupant: UnitId, city_owner: PlayerId) -> Dir {
    let u = state.unit(occupant).expect("living occupant");
    match u.last_dir {
        Some(d) if u.owner == city_owner => d,
        Some(d) => Dir {
            dr: -d.dr,
            dc: -d.dc,
        },
        None => {
            let center = (state.size - 1) / 2;
            match direction(u.pos, Pos::new(center, center)) {
                Dir { dr: 0, dc: 0 } => Dir { dr: 1, dc: 0 },
                d => d,
            }
        }
    }
}

/// Moves the occupant to the first free probed neighbor, keeping everything but its position, or
/// removes it from the game (no kill credit) when every neighbor is blocked.
fn push(state: &mut State, occupant: UnitId, city_owner: PlayerId) {
    let intended = push_direction(state, occupant, city_owner);
    let start = RING
        .iter()
        .position(|&d| d == intended)
        .expect("push directions are ring directions");
    let u = state.unit(occupant).expect("living occupant");
    let (from, owner) = (u.pos, u.owner);
    let n = state.size as i32;
    let target = PROBE_OFFSETS.iter().find_map(|&k| {
        let d = RING[(start + k) % 8];
        let (r, c) = (from.row as i32 + d.dr as i32, from.col as i32 + d.dc as i32);
        if r < 0 || c < 0 || r >= n || c >= n {
            return None;
        }
        let q = Pos::new(r as u8, c as u8);
        let free = state.tile(q).unit.is_none() && terrain_allows(state, owner, q);
        free.then_some(q)
    });

    let fi = state.index(from);
    state.tiles[fi].unit = None;
    match target {
        Some(to) => {
            let ti = state.index(to);
            state.tiles[ti].unit = Some(occupant);
            state.unit_mut(occupant).pos = to;
            reveal(state, owner, to);
        }
        None => state.units.retain(|u| u.id != occupant),
    }
}
