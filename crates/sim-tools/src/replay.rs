//! Replay log: record, serialize, replay with verification (docs/MILESTONE_7 "Replay log").

use serde::{Deserialize, Serialize};
use sim_core::actions::{apply, Action};
use sim_core::map::MapParams;
use sim_core::setup::new_game_from_seed;
use sim_core::state::{Config, Outcome, State};

use crate::digest::state_digest;

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
        serde_json::to_string(self).expect("a replay log always serializes")
    }

    pub fn from_json(s: &str) -> Result<ReplayLog, ReplayError> {
        serde_json::from_str(s).map_err(|e| ReplayError::Json(e.to_string()))
    }
}

fn hex(state: &State) -> String {
    format!("{:016x}", state_digest(state))
}

/// Plays `policy` from the seeded start until the game is terminal or `max_actions` were applied.
/// Panics if the policy returns an illegal action.
pub fn record_game(
    seed: u64,
    map_params: &MapParams,
    config: &Config,
    max_actions: usize,
    policy: &mut dyn FnMut(&State) -> Action,
) -> ReplayLog {
    let mut state = new_game_from_seed(seed, map_params, config);
    let mut actions = Vec::new();
    let mut digests = vec![hex(&state)];
    while !state.is_terminal() && actions.len() < max_actions {
        let action = policy(&state);
        apply(&mut state, &action).expect("the policy returned an illegal action");
        actions.push(action);
        digests.push(hex(&state));
    }
    ReplayLog {
        version: 1,
        seed,
        map_params: map_params.clone(),
        config: config.clone(),
        actions,
        digests,
        outcome: state.outcome,
    }
}

pub fn replay(log: &ReplayLog) -> Result<State, ReplayError> {
    run(log, &mut |_, _| {})
}

pub fn replay_each(log: &ReplayLog, f: &mut dyn FnMut(usize, &State)) -> Result<(), ReplayError> {
    run(log, f).map(|_| ())
}

/// Verifies the log in the brief's error priority, calling `f` on each state once its digest has
/// been checked, and returns the final state.
fn run(log: &ReplayLog, f: &mut dyn FnMut(usize, &State)) -> Result<State, ReplayError> {
    if log.version != 1 {
        return Err(ReplayError::BadVersion(log.version));
    }
    if log.digests.len() != log.actions.len() + 1 {
        return Err(ReplayError::LengthMismatch);
    }
    let mut state = new_game_from_seed(log.seed, &log.map_params, &log.config);
    if hex(&state) != log.digests[0] {
        return Err(ReplayError::DigestMismatch { index: 0 });
    }
    f(0, &state);
    for (i, action) in log.actions.iter().enumerate() {
        apply(&mut state, action).map_err(|_| ReplayError::IllegalAction { index: i })?;
        if hex(&state) != log.digests[i + 1] {
            return Err(ReplayError::DigestMismatch { index: i + 1 });
        }
        f(i + 1, &state);
    }
    if log.outcome != state.outcome {
        return Err(ReplayError::OutcomeMismatch);
    }
    Ok(state)
}
