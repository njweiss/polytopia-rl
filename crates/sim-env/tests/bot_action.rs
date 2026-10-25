//! Milestone 11 test hook `Env::bot_action` (docs/QUESTIONS.md Q23): the scripted policy "equal to
//! the Rust bot" that the suite and self-play tests drive through Python.
//!
//! Expected games come from `bots::bot_policy` (milestone 8, one rng stream per seat from
//! `bot_rng(seed, seat)`) applied to a plain `sim_core` state, independently of `sim_env`.

mod common;

use bots::{bot_policy, BotKind};
use common::*;
use sim_core::actions::{apply, Action};
use sim_core::map::MapParams;
use sim_core::setup::new_game_from_seed;
use sim_core::state::PlayerId;
use sim_env::actions::decode_action;
use sim_env::env::{AgentSeat, Env, EnvError, Opponent};

const CAP: u32 = 10;

/// The action log of a bot-vs-bot game from the milestone 8 runner's policy.
fn expected_log(seed: u64, bots: [BotKind; 2]) -> Vec<(PlayerId, Action)> {
    let mut policy = bot_policy(seed, bots, [0.0, 0.0]);
    let mut s = new_game_from_seed(seed, &MapParams::default(), &cfg(CAP));
    let mut log = Vec::new();
    while !s.is_terminal() {
        let me = s.current;
        let a = policy(&s);
        apply(&mut s, &a).expect("bots choose legal actions");
        log.push((me, a));
    }
    log
}

#[test]
fn two_seat_game_driven_by_bot_action_equals_the_runner() {
    let pairs = [
        [BotKind::Rush, BotKind::Economy],
        [BotKind::Random, BotKind::Rush],
        [BotKind::Economy, BotKind::Random],
    ];
    for (i, bots) in pairs.into_iter().enumerate() {
        let seed = 40 + i as u64;
        let mut env = Env::new(env_cfg(Opponent::TwoSeat, AgentSeat::Fixed(0), CAP));
        env.reset(seed);
        while !env.is_terminal() {
            let kind = bots[env.state().current as usize];
            let f = env.bot_action(kind).expect("running");
            // The factored action decodes to a legal action of the player to move.
            let a = decode_action(env.state(), &f).expect("decodes");
            assert!(env.legal_actions().contains(&a));
            env.step_factored(&f).expect("legal");
        }
        assert_eq!(env.action_log(), &expected_log(seed, bots)[..], "{bots:?}");
    }
}

#[test]
fn bot_action_for_the_agent_in_bot_mode_mirrors_the_runner() {
    // Agent seat 1 played by Economy through the hook, against the env's own Rush opponent.
    for seed in [7u64, 8] {
        let mut env = Env::new(env_cfg(bot(BotKind::Rush, 0.0), AgentSeat::Fixed(1), CAP));
        env.reset(seed);
        while !env.is_terminal() {
            let f = env.bot_action(BotKind::Economy).expect("running");
            env.step_factored(&f).expect("legal");
        }
        let want = expected_log(seed, [BotKind::Rush, BotKind::Economy]);
        assert_eq!(env.action_log(), &want[..], "seed {seed}");
    }
}

#[test]
fn bot_action_does_not_change_the_game() {
    let mut env = Env::new(env_cfg(Opponent::TwoSeat, AgentSeat::Fixed(0), CAP));
    env.reset(3);
    // Economy and Rush ignore their rng, so asking twice gives the same answer.
    let before = env.legal_actions().to_vec();
    let a = env.bot_action(BotKind::Economy).unwrap();
    let b = env.bot_action(BotKind::Economy).unwrap();
    env.bot_action(BotKind::Rush).unwrap();
    assert_eq!(a, b);
    assert_eq!(env.legal_actions(), &before[..]);
    assert!(env.action_log().is_empty());
    assert_eq!(env.agent_actions(), 0);
}

#[test]
fn bot_action_after_the_end_is_finished() {
    let mut env = Env::new(env_cfg(Opponent::TwoSeat, AgentSeat::Fixed(0), 1));
    env.reset(5);
    while !env.is_terminal() {
        let f = env.bot_action(BotKind::Rush).unwrap();
        env.step_factored(&f).unwrap();
    }
    assert_eq!(env.bot_action(BotKind::Rush), Err(EnvError::Finished));
}
