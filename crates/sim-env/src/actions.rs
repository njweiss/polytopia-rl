//! The factored action space (docs/MILESTONE_9 "Factored action space").

use sim_core::actions::Action;
use sim_core::state::State;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionType {
    Move = 0,
    Attack,
    Capture,
    Recover,
    Promote,
    Train,
    ChooseReward,
    Site,
    Research,
    EndTurn,
}

pub const NUM_ACTION_TYPES: usize = 10;

impl ActionType {
    pub const ALL: [ActionType; NUM_ACTION_TYPES] = [
        ActionType::Move,
        ActionType::Attack,
        ActionType::Capture,
        ActionType::Recover,
        ActionType::Promote,
        ActionType::Train,
        ActionType::ChooseReward,
        ActionType::Site,
        ActionType::Research,
        ActionType::EndTurn,
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Factored {
    pub actor: usize,
    pub kind: ActionType,
    pub arg: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    /// `actor >= num_actors(size)`.
    ActorOutOfRange,
    /// A unit or city actor slot that holds no unit or city of `state.current`.
    NoActor,
    /// The kind does not belong to the actor's slot family.
    WrongKind,
    /// `arg >= arg_len(kind)`, or nonzero for a kind without an argument.
    BadArg,
}

/// `3 * size * size + 1`.
pub fn num_actors(size: u8) -> usize {
    let _ = size;
    unimplemented!("milestone 9")
}

pub fn arg_len(size: u8, kind: ActionType) -> usize {
    let _ = (size, kind);
    unimplemented!("milestone 9")
}

/// `state` is the state before the action.
pub fn encode_action(state: &State, action: &Action) -> Factored {
    let _ = (state, action);
    unimplemented!("milestone 9")
}

/// Identification only; legality is checked by `Env::step`.
pub fn decode_action(state: &State, f: &Factored) -> Result<Action, DecodeError> {
    let _ = (state, f);
    unimplemented!("milestone 9")
}

/// Built once from a legal-action list.
#[derive(Debug, Clone)]
pub struct ActionMasks {}

impl ActionMasks {
    pub fn new(state: &State, legal: &[Action]) -> ActionMasks {
        let _ = (state, legal);
        unimplemented!("milestone 9")
    }

    /// Length `num_actors`.
    pub fn actor(&self) -> &[bool] {
        unimplemented!("milestone 9")
    }

    pub fn kind(&self, actor: usize) -> [bool; NUM_ACTION_TYPES] {
        let _ = actor;
        unimplemented!("milestone 9")
    }

    /// Length `arg_len(kind)`; empty for kinds without an argument.
    pub fn arg(&self, actor: usize, kind: ActionType) -> &[bool] {
        let _ = (actor, kind);
        unimplemented!("milestone 9")
    }
}
