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

    /// `sim_tools::policy::random_action` on `view.legal_actions()`: EndTurn with probability 0.15
    /// when other actions exist (`gen_f32` first, then `gen_range`); with EndTurn illegal a single
    /// `gen_range` over the legal list.
    fn choose(&self, view: &PlayerView, rng: &mut Rng) -> Action {
        let legal = view.legal_actions();
        assert!(!legal.is_empty(), "RandomBot: no legal actions");
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
}
