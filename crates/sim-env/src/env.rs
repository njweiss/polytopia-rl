//! The single-environment API (docs/MILESTONE_9 "Env", "Reward", "Replay").

use bots::BotKind;
use sim_core::actions::Action;
use sim_core::map::MapParams;
use sim_core::state::{Config, PlayerId, State};
use sim_core::view::PlayerView;

use crate::actions::{ActionMasks, DecodeError, Factored};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Opponent {
    /// The caller supplies every action.
    TwoSeat,
    /// The opponent's turns run inside `step`.
    Bot { kind: BotKind, epsilon: f32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentSeat {
    Fixed(PlayerId),
    /// `seed % 2`.
    SeedParity,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RewardConfig {
    pub terminal_weight: f32,
    pub shaping_weight: f32,
    pub gamma: f32,
    pub potential_scale: f32,
}

impl Default for RewardConfig {
    fn default() -> RewardConfig {
        RewardConfig {
            terminal_weight: 1.0,
            shaping_weight: 0.0,
            gamma: 1.0,
            potential_scale: 1000.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnvConfig {
    pub map: MapParams,
    pub config: Config,
    pub opponent: Opponent,
    pub agent_seat: AgentSeat,
    pub reward: RewardConfig,
    /// Default 5000.
    pub max_agent_actions: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepOut {
    pub reward: f32,
    pub rewards: [f32; 2],
    pub terminated: bool,
    pub truncated: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvError {
    NotReset,
    Finished,
    Illegal,
    BadIndex,
    Decode(DecodeError),
}

pub struct Env {}

impl Env {
    /// Asserts that `cfg.map.size` is accepted by `MapParams`.
    pub fn new(cfg: EnvConfig) -> Env {
        let _ = cfg;
        unimplemented!("milestone 9")
    }

    pub fn reset(&mut self, seed: u64) {
        let _ = seed;
        unimplemented!("milestone 9")
    }

    /// For tests and tools only; Python never sees it.
    pub fn state(&self) -> &State {
        unimplemented!("milestone 9")
    }

    pub fn seed(&self) -> u64 {
        unimplemented!("milestone 9")
    }

    pub fn agent_seat(&self) -> PlayerId {
        unimplemented!("milestone 9")
    }

    /// Bot mode: the agent seat. TwoSeat: `state.current`, or the last actor once terminal.
    pub fn observer(&self) -> PlayerId {
        unimplemented!("milestone 9")
    }

    /// `PlayerView::new(state, observer())`.
    pub fn view(&self) -> PlayerView<'_> {
        unimplemented!("milestone 9")
    }

    /// Empty when terminal.
    pub fn legal_actions(&self) -> &[Action] {
        unimplemented!("milestone 9")
    }

    pub fn masks(&self) -> &ActionMasks {
        unimplemented!("milestone 9")
    }

    pub fn step(&mut self, action: &Action) -> Result<StepOut, EnvError> {
        let _ = action;
        unimplemented!("milestone 9")
    }

    /// `index` into `legal_actions()`.
    pub fn step_legal(&mut self, index: usize) -> Result<StepOut, EnvError> {
        let _ = index;
        unimplemented!("milestone 9")
    }

    pub fn step_factored(&mut self, f: &Factored) -> Result<StepOut, EnvError> {
        let _ = f;
        unimplemented!("milestone 9")
    }

    /// Every applied action, the opponent's included, in order.
    pub fn action_log(&self) -> &[(PlayerId, Action)] {
        unimplemented!("milestone 9")
    }

    pub fn is_terminal(&self) -> bool {
        unimplemented!("milestone 9")
    }
}
