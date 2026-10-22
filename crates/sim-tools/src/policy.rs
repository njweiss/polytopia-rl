//! Random action policy (docs/MILESTONE_7 "Random policy").

use sim_core::actions::Action;
use sim_core::rng::Rng;
use sim_core::state::State;

pub fn random_action(_state: &State, _rng: &mut Rng) -> Action {
    unimplemented!()
}
