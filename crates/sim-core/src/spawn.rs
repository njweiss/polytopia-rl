//! Super Unit spawn and the forced push of a city tile's occupant (spec: "Forced unit spawn";
//! docs/MILESTONE_5 "Super Unit spawn").

use crate::actions::reveal;
use crate::data::{Tech, UnitType};
use crate::grid::Pos;
use crate::map::Terrain;
use crate::state::{CityId, Dir, Phase, PlayerId, State, Unit, UnitId};

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

    let health = state.tables.unit(UnitType::SuperUnit).health;
    let id = UnitId(state.next_unit_id);
    state.next_unit_id += 1;
    // Ids only grow, so pushing keeps `state.units` in ascending id order.
    state.units.push(Unit {
        id,
        kind: UnitType::SuperUnit,
        owner,
        pos,
        health,
        max_health: health,
        origin: Some(city),
        kills: 0,
        veteran: false,
        promotion_available: false,
        phase: Phase::Done,
        last_dir: None,
    });
    let i = state.index(pos);
    state.tiles[i].unit = Some(id);
    id
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
            let center = (state.size as i32 - 1) / 2;
            let dr = (center - u.pos.row as i32).signum() as i8;
            let dc = (center - u.pos.col as i32).signum() as i8;
            if dr == 0 && dc == 0 {
                Dir { dr: 1, dc: 0 }
            } else {
                Dir { dr, dc }
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
    let climbing = state.players[owner as usize].techs.contains(Tech::Climbing);
    let n = state.size as i32;
    let target = PROBE_OFFSETS.iter().find_map(|&k| {
        let d = RING[(start + k) % 8];
        let (r, c) = (from.row as i32 + d.dr as i32, from.col as i32 + d.dc as i32);
        if r < 0 || c < 0 || r >= n || c >= n {
            return None;
        }
        let q = Pos::new(r as u8, c as u8);
        let tile = state.tile(q);
        let free = tile.unit.is_none() && (tile.terrain != Terrain::Mountain || climbing);
        free.then_some(q)
    });

    let fi = state.index(from);
    state.tiles[fi].unit = None;
    match target {
        Some(to) => {
            let ti = state.index(to);
            state.tiles[ti].unit = Some(occupant);
            state
                .units
                .iter_mut()
                .find(|u| u.id == occupant)
                .expect("living occupant")
                .pos = to;
            reveal(state, owner, to);
        }
        None => state.units.retain(|u| u.id != occupant),
    }
}
