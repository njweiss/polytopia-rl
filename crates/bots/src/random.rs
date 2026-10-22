//! Uniform random legal action (docs/MILESTONE_8 "Random bot and epsilon wrapper").

use sim_core::actions::Action;
use sim_core::rng::Rng;
use sim_core::view::PlayerView;

use crate::Bot;

pub struct RandomBot;

impl Bot for RandomBot {
    fn name(&self) -> &'static str {
        unimplemented!()
    }

    fn choose(&self, _view: &PlayerView, _rng: &mut Rng) -> Action {
        unimplemented!()
    }
}
