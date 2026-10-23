//! Random action policy (docs/MILESTONE_7 "Random policy").

use bots::random_choice;
use sim_core::actions::{legal_actions, Action};
use sim_core::rng::Rng;
use sim_core::state::State;

/// `bots::random_choice` over `legal_actions(state)`.
pub fn random_action(state: &State, rng: &mut Rng) -> Action {
    random_choice(legal_actions(state), rng)
}
