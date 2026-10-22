//! Random action policy (docs/MILESTONE_7 "Random policy").

use sim_core::actions::{legal_actions, Action};
use sim_core::rng::Rng;
use sim_core::state::State;

/// `EndTurn` with probability 0.15 when other actions exist, else uniform over the others. With a
/// pending reward (`EndTurn` illegal) uniform over `legal` with a single `gen_range` (Q7).
pub fn random_action(state: &State, rng: &mut Rng) -> Action {
    let legal = legal_actions(state);
    assert!(!legal.is_empty(), "random_action on a terminal state");
    if !legal.contains(&Action::EndTurn) {
        return legal[rng.gen_range(legal.len() as u32) as usize];
    }
    let others: Vec<Action> = legal
        .into_iter()
        .filter(|a| *a != Action::EndTurn)
        .collect();
    if others.is_empty() {
        return Action::EndTurn;
    }
    if rng.gen_f32() < 0.15 {
        Action::EndTurn
    } else {
        others[rng.gen_range(others.len() as u32) as usize]
    }
}
