//! Unit actions, legality and `apply` (docs/MILESTONE_3 "Move" .. "legal_actions and apply").

use crate::grid::Pos;
use crate::state::{State, UnitId};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Action {
    Move { unit: UnitId, to: Pos },
    Attack { unit: UnitId, target: Pos },
    Recover { unit: UnitId },
    Promote { unit: UnitId },
    EndTurn,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionError {
    Illegal,
}

/// The current player's legal actions in canonical order: units by ascending id; per unit Move
/// (row-major), Attack (row-major), Recover, Promote; then EndTurn. Empty once there is an outcome.
pub fn legal_actions(state: &State) -> Vec<Action> {
    let _ = state;
    unimplemented!()
}

/// Applies a legal action. Any action not in `legal_actions(state)` returns `Err(Illegal)` and
/// leaves `state` completely unchanged.
pub fn apply(state: &mut State, action: &Action) -> Result<(), ActionError> {
    let _ = (state, action);
    unimplemented!()
}
