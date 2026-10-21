//! Super Unit spawn and the forced push of a city tile's occupant (spec: "Forced unit spawn";
//! docs/MILESTONE_5 "Super Unit spawn").

use crate::state::{CityId, Dir, PlayerId, State, UnitId};

/// Pushes any occupant off `city`'s tile (or removes it when no neighbor is free), then creates
/// the Super Unit there. Returns the new unit's id. Used by `ChooseReward { SuperUnit }`.
pub fn spawn_super_unit(state: &mut State, city: CityId) -> UnitId {
    let _ = (state, city);
    unimplemented!("milestone 5: spawn_super_unit")
}

/// The direction `occupant` is intended to be pushed when a Super Unit spawns on a city owned by
/// `city_owner` (before probing for a free tile).
pub fn push_direction(state: &State, occupant: UnitId, city_owner: PlayerId) -> Dir {
    let _ = (state, occupant, city_owner);
    unimplemented!("milestone 5: push_direction")
}
