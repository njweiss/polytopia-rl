//! Match running and statistics (docs/MILESTONE_8 "Match running and statistics").

use std::ops::Range;

use sim_core::map::MapParams;
use sim_core::state::{Config, Outcome};

use crate::BotKind;

/// Per-player counts at the first moment `state.turn == turn && state.current == 0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Snapshot {
    pub turn: u32,
    pub cities: [u32; 2],
    pub units: [u32; 2],
    pub stars: [u32; 2],
    pub score: [u32; 2],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GameResult {
    pub seed: u64,
    /// By player id.
    pub bots: [BotKind; 2],
    pub outcome: Outcome,
    pub turns: u32,
    pub scores: [u32; 2],
    pub captured_capital: bool,
    /// At turns 10, 20, 30 (those that were reached).
    pub snapshots: Vec<Snapshot>,
}

pub fn play_game(
    _seed: u64,
    _bots: [BotKind; 2],
    _epsilons: [f32; 2],
    _map: &MapParams,
    _config: &Config,
) -> GameResult {
    unimplemented!()
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlayerMeans {
    pub cities: f64,
    pub units: f64,
    pub stars: f64,
    pub score: f64,
}

/// Means over the games that reached `turn`.
#[derive(Debug, Clone, PartialEq)]
pub struct CheckpointStats {
    pub turn: u32,
    pub games: u32,
    pub a: PlayerMeans,
    pub b: PlayerMeans,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatchStats {
    pub a: BotKind,
    pub b: BotKind,
    pub games: u32,
    pub a_wins: u32,
    pub b_wins: u32,
    pub draws: u32,
    pub a_capture_wins: u32,
    pub b_capture_wins: u32,
    /// Mean of `a`'s final score minus `b`'s.
    pub mean_score_diff: f64,
    pub checkpoints: Vec<CheckpointStats>,
}

impl MatchStats {
    pub fn from_results(_a: BotKind, _b: BotKind, _results: &[GameResult]) -> MatchStats {
        unimplemented!()
    }

    /// `a_wins / games` (draws count as no win).
    pub fn win_rate_a(&self) -> f64 {
        unimplemented!()
    }

    /// `b_wins / games` (draws count as no win).
    pub fn win_rate_b(&self) -> f64 {
        unimplemented!()
    }
}

/// Plays each seed twice: `a` as player 0, then `b` as player 0, epsilons 0.
pub fn run_match(
    _a: BotKind,
    _b: BotKind,
    _seeds: Range<u64>,
    _map: &MapParams,
    _config: &Config,
) -> MatchStats {
    unimplemented!()
}
