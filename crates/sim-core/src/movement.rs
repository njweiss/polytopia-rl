//! Movement and defense bonus (spec: "Movement"; docs/MILESTONE_3 "reachable_tiles",
//! "defense_bonus").

use crate::grid::Pos;
use crate::state::{State, UnitId};

/// Tiles the unit could move to with its full movement budget, row-major, excluding its own
/// tile. Ignores the unit's phase.
pub fn reachable_tiles(state: &State, unit: UnitId) -> Vec<Pos> {
    let _ = (state, unit);
    unimplemented!()
}

/// The defense bonus for `defender` on its current tile (largest applicable value, no stacking).
pub fn defense_bonus(state: &State, defender: UnitId) -> f32 {
    let _ = (state, defender);
    unimplemented!()
}
