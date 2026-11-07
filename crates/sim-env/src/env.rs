//! The single-environment API (docs/MILESTONE_9 "Env", "Reward", "Replay").

use bots::{bot_rng, make_bot, Bot, BotKind};
use sim_core::actions::{apply, legal_actions, Action};
use sim_core::map::MapParams;
use sim_core::rng::Rng;
use sim_core::score::score;
use sim_core::setup::new_game_from_seed;
use sim_core::state::{Config, Outcome, PlayerId, State};
use sim_core::view::PlayerView;
use sim_tools::replay::record_game;

use crate::actions::{decode_action, encode_action, ActionMasks, DecodeError, Factored};

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

pub struct Env {
    cfg: EnvConfig,
    /// `None` in two-seat mode.
    opponent: Option<Box<dyn Bot>>,
    /// `None` before the first `reset`.
    episode: Option<Episode>,
}

struct Episode {
    seed: u64,
    agent: PlayerId,
    state: State,
    /// The opponent seat's `bot_rng`, one for the whole episode.
    opponent_rng: Rng,
    /// `bot_rng(seed, seat)` per seat, advanced only by `Env::bot_action`.
    bot_rngs: [Rng; 2],
    log: Vec<(PlayerId, Action)>,
    legal: Vec<Action>,
    masks: ActionMasks,
    agent_actions: u32,
    truncated: bool,
    /// The player who made the caller's last action.
    last_actor: PlayerId,
}

impl Episode {
    fn refresh(&mut self) {
        self.legal = legal_actions(&self.state);
        self.masks = ActionMasks::new(&self.state, &self.legal);
    }
}

/// `phi_s` for both seats: `(me - opp) / potential_scale`, 0 when terminal.
fn potentials(s: &State, r: &RewardConfig) -> [f32; 2] {
    if s.is_terminal() {
        return [0.0; 2];
    }
    let scores = [score(s, 0) as f32, score(s, 1) as f32];
    [0, 1].map(|seat| (scores[seat] - scores[1 - seat]) / r.potential_scale)
}

impl Env {
    /// Panics if the map is below `MapParams::MIN_SIZE`, or if `cfg.config.ruleset` is `V2` (refused
    /// until milestone 20, docs/MILESTONE_15 "Env").
    pub fn new(cfg: EnvConfig) -> Env {
        assert!(
            cfg.map.size >= MapParams::MIN_SIZE,
            "Env::new: map size {} is below MapParams::MIN_SIZE",
            cfg.map.size
        );
        let opponent = match cfg.opponent {
            Opponent::TwoSeat => None,
            Opponent::Bot { kind, epsilon } => Some(make_bot(kind, epsilon)),
        };
        Env {
            cfg,
            opponent,
            episode: None,
        }
    }

    pub fn reset(&mut self, seed: u64) {
        let agent = match self.cfg.agent_seat {
            AgentSeat::Fixed(p) => p,
            AgentSeat::SeedParity => (seed % 2) as PlayerId,
        };
        let state = new_game_from_seed(seed, &self.cfg.map, &self.cfg.config);
        // Filled by `refresh` once the opponent has moved.
        let masks = ActionMasks::new(&state, &[]);
        self.episode = Some(Episode {
            seed,
            agent,
            state,
            opponent_rng: bot_rng(seed, 1 - agent),
            bot_rngs: [bot_rng(seed, 0), bot_rng(seed, 1)],
            log: Vec::new(),
            legal: Vec::new(),
            masks,
            agent_actions: 0,
            truncated: false,
            last_actor: 0,
        });
        self.run_opponent();
        self.ep_mut().refresh();
    }

    /// Bot mode: plays the opponent until the agent is to move or the game is over.
    fn run_opponent(&mut self) {
        let Some(bot) = &self.opponent else {
            return;
        };
        let ep = self.episode.as_mut().expect("reset");
        while !ep.state.is_terminal() && ep.state.current != ep.agent {
            let me = ep.state.current;
            let a = bot.choose(&PlayerView::new(&ep.state, me), &mut ep.opponent_rng);
            apply(&mut ep.state, &a).expect("bots choose legal actions");
            ep.log.push((me, a));
        }
    }

    /// False until the first `reset`; every accessor below panics until then.
    pub fn is_reset(&self) -> bool {
        self.episode.is_some()
    }

    fn ep(&self) -> &Episode {
        self.episode.as_ref().expect("Env: call reset first")
    }

    fn ep_mut(&mut self) -> &mut Episode {
        self.episode.as_mut().expect("Env: call reset first")
    }

    /// For tests and tools only; Python never sees it.
    pub fn state(&self) -> &State {
        &self.ep().state
    }

    pub fn seed(&self) -> u64 {
        self.ep().seed
    }

    pub fn agent_seat(&self) -> PlayerId {
        self.ep().agent
    }

    /// Bot mode: the agent seat. TwoSeat: `state.current`, or the last actor once terminal.
    pub fn observer(&self) -> PlayerId {
        let ep = self.ep();
        if self.opponent.is_some() {
            ep.agent
        } else if ep.state.is_terminal() {
            ep.last_actor
        } else {
            ep.state.current
        }
    }

    /// `PlayerView::new(state, observer())`.
    pub fn view(&self) -> PlayerView<'_> {
        PlayerView::new(&self.ep().state, self.observer())
    }

    /// Empty when terminal.
    pub fn legal_actions(&self) -> &[Action] {
        &self.ep().legal
    }

    pub fn masks(&self) -> &ActionMasks {
        &self.ep().masks
    }

    /// The caller's successful steps since `reset`.
    pub fn agent_actions(&self) -> u32 {
        self.ep().agent_actions
    }

    fn check_running(&self) -> Result<&Episode, EnvError> {
        let ep = self.episode.as_ref().ok_or(EnvError::NotReset)?;
        if ep.state.is_terminal() || ep.truncated {
            return Err(EnvError::Finished);
        }
        Ok(ep)
    }

    pub fn step(&mut self, action: &Action) -> Result<StepOut, EnvError> {
        let ep = self.check_running()?;
        if !ep.legal.contains(action) {
            return Err(EnvError::Illegal);
        }
        let r = self.cfg.reward;
        let before = potentials(&ep.state, &r);

        let ep = self.ep_mut();
        let actor = ep.state.current;
        apply(&mut ep.state, action).expect("checked against the legal list");
        ep.log.push((actor, *action));
        ep.last_actor = actor;
        ep.agent_actions += 1;
        self.run_opponent();

        let max = self.cfg.max_agent_actions;
        let two_seat = self.opponent.is_none();
        let ep = self.ep_mut();
        ep.refresh();
        let terminated = ep.state.is_terminal();
        ep.truncated = !terminated && ep.agent_actions >= max;
        let after = potentials(&ep.state, &r);
        let rewards = [0u8, 1].map(|seat| {
            let t = match ep.state.outcome {
                Some(Outcome::Win(w)) if w == seat => 1.0,
                Some(Outcome::Win(_)) => -1.0,
                _ => 0.0,
            };
            let s = seat as usize;
            r.terminal_weight * t + r.shaping_weight * (r.gamma * after[s] - before[s])
        });
        let seat = if two_seat { actor } else { ep.agent };
        Ok(StepOut {
            reward: rewards[seat as usize],
            rewards,
            terminated,
            truncated: ep.truncated,
        })
    }

    /// `index` into `legal_actions()`.
    pub fn step_legal(&mut self, index: usize) -> Result<StepOut, EnvError> {
        let ep = self.check_running()?;
        let a = *ep.legal.get(index).ok_or(EnvError::BadIndex)?;
        self.step(&a)
    }

    pub fn step_factored(&mut self, f: &Factored) -> Result<StepOut, EnvError> {
        let ep = self.check_running()?;
        let a = decode_action(&ep.state, f).map_err(EnvError::Decode)?;
        self.step(&a)
    }

    /// The action bot `kind` (epsilon 0) chooses for the player to move, from that player's
    /// `PlayerView`, encoded against the current state. Each seat draws from its own rng stream
    /// `bot_rng(seed, seat)`, created at `reset` and advanced only by this call, so driving a seat
    /// with it reproduces that seat's moves in `bots::play_game`. `Finished` once the episode is
    /// over. A test and tournament hook: the env's own opponent is not affected.
    pub fn bot_action(&mut self, kind: BotKind) -> Result<Factored, EnvError> {
        self.check_running()?;
        let ep = self.ep_mut();
        let me = ep.state.current;
        let a = make_bot(kind, 0.0).choose(
            &PlayerView::new(&ep.state, me),
            &mut ep.bot_rngs[me as usize],
        );
        Ok(encode_action(&ep.state, &a))
    }

    /// Every applied action, the opponent's included, in order.
    pub fn action_log(&self) -> &[(PlayerId, Action)] {
        &self.ep().log
    }

    pub fn is_terminal(&self) -> bool {
        self.ep().state.is_terminal()
    }

    /// The milestone 7 replay log (`sim_tools::replay`) of the episode so far, as JSON.
    pub fn replay_json(&self) -> String {
        let ep = self.ep();
        let mut next = ep.log.iter().map(|(_, a)| *a);
        record_game(
            ep.seed,
            &self.cfg.map,
            &self.cfg.config,
            ep.log.len(),
            &mut |_: &State| next.next().expect("one logged action per call"),
        )
        .to_json()
    }
}
