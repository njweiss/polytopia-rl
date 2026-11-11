//! Roads and city connections under v2 (docs/MILESTONE_16 "Road tiles", "City connections").

use crate::grid::Pos;
use crate::state::{CityId, PlayerId, State};

/// Whether `p` counts as a road for `player`: it has a road or a city, and is not in the other
/// player's territory.
pub fn road_usable(state: &State, player: PlayerId, p: Pos) -> bool {
    let _ = (state, player, p);
    unimplemented!("milestone 16: road_usable")
}

/// The path rule: a non-capital city is linked to its owner's capital by 8-adjacent tiles that are
/// all `road_usable` for its owner. Ignores `connected_to`.
pub fn is_connected(state: &State, city: CityId) -> bool {
    let _ = (state, city);
    unimplemented!("milestone 16: is_connected")
}
