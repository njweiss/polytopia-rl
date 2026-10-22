//! The Economy bot (docs/MILESTONE_8 "Economy bot").

use sim_core::actions::Action;
use sim_core::rng::Rng;
use sim_core::view::PlayerView;

use crate::Bot;

pub struct EconomyBot;

impl Bot for EconomyBot {
    fn name(&self) -> &'static str {
        unimplemented!()
    }

    fn choose(&self, _view: &PlayerView, _rng: &mut Rng) -> Action {
        unimplemented!()
    }
}
