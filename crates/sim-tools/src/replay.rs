//! Replay log: record, serialize, replay with verification (docs/MILESTONE_7 "Replay log").

use serde::{Deserialize, Serialize};
use sim_core::actions::Action;
use sim_core::map::MapParams;
use sim_core::state::{Config, Outcome, State};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayLog {
    /// Always 1.
    pub version: u32,
    pub seed: u64,
    pub map_params: MapParams,
    pub config: Config,
    pub actions: Vec<Action>,
    /// `{:016x}` digests; `len == actions.len() + 1` (index 0 = the initial state).
    pub digests: Vec<String>,
    /// Outcome of the final state (`None` if unfinished).
    pub outcome: Option<Outcome>,
}

#[derive(Debug, PartialEq)]
pub enum ReplayError {
    Json(String),
    BadVersion(u32),
    LengthMismatch,
    /// Index of the offending action.
    IllegalAction {
        index: usize,
    },
    /// 0 = the initial state, i + 1 = after action i.
    DigestMismatch {
        index: usize,
    },
    OutcomeMismatch,
}

impl ReplayLog {
    pub fn to_json(&self) -> String {
        unimplemented!()
    }

    pub fn from_json(_s: &str) -> Result<ReplayLog, ReplayError> {
        unimplemented!()
    }
}

pub fn record_game(
    _seed: u64,
    _map_params: &MapParams,
    _config: &Config,
    _max_actions: usize,
    _policy: &mut dyn FnMut(&State) -> Action,
) -> ReplayLog {
    unimplemented!()
}

pub fn replay(_log: &ReplayLog) -> Result<State, ReplayError> {
    unimplemented!()
}

pub fn replay_each(_log: &ReplayLog, _f: &mut dyn FnMut(usize, &State)) -> Result<(), ReplayError> {
    unimplemented!()
}
