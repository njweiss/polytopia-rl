//! Uniform random legal action (docs/MILESTONE_8 "Random bot and epsilon wrapper").

use sim_core::actions::Action;
use sim_core::rng::Rng;
use sim_core::view::PlayerView;

use crate::Bot;

pub struct RandomBot;

impl Bot for RandomBot {
    fn name(&self) -> &'static str {
        "random"
    }

    fn choose(&self, view: &PlayerView, rng: &mut Rng) -> Action {
        random_choice(view.legal_actions(), rng)
    }
}

/// `EndTurn` with probability 0.15 when other actions exist (`gen_f32` first, then `gen_range`
/// over the others). With a pending reward (`EndTurn` illegal) a single `gen_range` over `legal`
/// (Q7). Shared with `sim_tools::policy::random_action`.
pub fn random_choice(legal: Vec<Action>, rng: &mut Rng) -> Action {
    assert!(!legal.is_empty(), "random_choice: no legal actions");
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
