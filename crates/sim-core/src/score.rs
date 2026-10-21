//! Score, derived on demand from state (spec: "Score"; docs/MILESTONE_6 "Score").

use crate::state::{PlayerId, State};

/// One player's score by term, each already multiplied by its weight.
#[derive(Debug, PartialEq, Eq)]
pub struct ScoreBreakdown {
    pub units: u32,
    pub territory: u32,
    pub exploration: u32,
    pub cities: u32,
    pub parks: u32,
    pub techs: u32,
}

impl ScoreBreakdown {
    pub fn total(&self) -> u32 {
        self.units + self.territory + self.exploration + self.cities + self.parks + self.techs
    }
}

pub fn score_breakdown(_state: &State, _player: PlayerId) -> ScoreBreakdown {
    unimplemented!("milestone 6: score_breakdown")
}

/// `score_breakdown(state, player).total()`.
pub fn score(_state: &State, _player: PlayerId) -> u32 {
    unimplemented!("milestone 6: score")
}
