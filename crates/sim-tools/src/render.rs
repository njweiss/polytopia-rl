//! Text renderer (docs/MILESTONE_7 "Text renderer").

use sim_core::actions::Action;
use sim_core::grid::Pos;
use sim_core::state::{PlayerId, State};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum View {
    Omniscient,
    Player(PlayerId),
}

pub fn render(_state: &State, _view: View) -> String {
    unimplemented!()
}

pub fn describe_tile(_state: &State, _view: View, _pos: Pos) -> String {
    unimplemented!()
}

pub fn action_label(_action: &Action) -> String {
    unimplemented!()
}
