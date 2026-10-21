//! The Explorer reward's random fog-seeking walk (spec: "Explorer"; docs/MILESTONE_5
//! "Explorer walk").

use crate::state::{CityId, State};

/// Moves the explorer makes before it disappears.
pub const EXPLORER_MOVES: u32 = 15;

/// Runs the whole walk from `city`'s tile for the city's owner: grows that player's explored mask
/// and draws from `state.rng` only on ties. Used by `ChooseReward { Explorer }`.
pub fn explorer_walk(state: &mut State, city: CityId) {
    let _ = (state, city);
    unimplemented!("milestone 5: explorer_walk")
}
