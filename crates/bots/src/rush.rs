//! The Rush bot (docs/MILESTONE_8 "Rush bot").

use sim_core::actions::Action;
use sim_core::rng::Rng;
use sim_core::view::PlayerView;

use crate::Bot;

pub struct RushBot;

impl Bot for RushBot {
    fn name(&self) -> &'static str {
        unimplemented!()
    }

    fn choose(&self, _view: &PlayerView, _rng: &mut Rng) -> Action {
        unimplemented!()
    }
}
