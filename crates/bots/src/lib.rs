//! Heuristic bots and the match runner (docs/MILESTONE_8). Bots read the game only through
//! `sim_core::view::PlayerView` and are pure functions of that view and their rng stream.

mod economy;
mod random;
mod runner;
mod rush;

use sim_core::actions::Action;
use sim_core::rng::Rng;
use sim_core::state::PlayerId;
use sim_core::view::PlayerView;

pub use economy::EconomyBot;
pub use random::RandomBot;
pub use runner::{
    play_game, run_match, CheckpointStats, GameResult, MatchStats, PlayerMeans, Snapshot,
};
pub use rush::RushBot;

/// Bots are stateless: a pure function of `(view, rng)`.
pub trait Bot {
    fn name(&self) -> &'static str;
    /// Must be in `view.legal_actions()`.
    fn choose(&self, view: &PlayerView, rng: &mut Rng) -> Action;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BotKind {
    Random,
    Rush,
    Economy,
}

impl BotKind {
    /// `"random"`, `"rush"` or `"economy"`.
    pub fn name(self) -> &'static str {
        unimplemented!()
    }
}

/// `epsilon` in `[0, 1]`; with `epsilon > 0` the bot is wrapped (docs/MILESTONE_8 "Random bot and
/// epsilon wrapper"), with `epsilon == 0` it is the bare bot.
pub fn make_bot(_kind: BotKind, _epsilon: f32) -> Box<dyn Bot> {
    unimplemented!()
}

/// `Rng::new(seed ^ SALT[player])`, `SALT = [0x9e3779b97f4a7c15, 0xc2b2ae3d27d4eb4f]`.
pub fn bot_rng(_seed: u64, _player: PlayerId) -> Rng {
    unimplemented!()
}
