//! Milestone 9 "Tests" item 9: determinism and replay. Same seed and agent policy give identical
//! observations, rewards and logs; the opponent's draws never touch the game rng (after every
//! step, replaying the action log with `sim_core::actions::apply` reproduces the env's state bit
//! for bit, rng included); the action log turns into a milestone 7 replay log that
//! `sim_tools::replay::replay` verifies and that reproduces the final state.

mod common;

use bots::BotKind;
use common::*;
use sim_core::actions::{apply, Action};
use sim_core::map::MapParams;
use sim_core::rng::Rng;
use sim_core::setup::new_game_from_seed;
use sim_core::state::{PlayerId, State};
use sim_env::env::{AgentSeat, Env, Opponent, StepOut};
use sim_env::obs::{encode_board, encode_globals, obs_spec};
use sim_tools::replay::{record_game, replay, ReplayLog};

/// Board and globals of the env's current observation, as bits.
fn obs(e: &Env) -> Vec<u32> {
    let spec = obs_spec(11);
    let mut b = vec![0.0f32; spec.channels * spec.height * spec.width];
    let mut g = vec![0.0f32; spec.globals];
    encode_board(&e.view(), &mut b);
    encode_globals(&e.view(), &mut g);
    b.iter().chain(&g).map(|x| x.to_bits()).collect()
}

/// A full episode trace: observation, legal count and step output per step, then the log.
type Trace = (Vec<(Vec<u32>, usize, StepOut)>, Vec<(PlayerId, Action)>);

fn trace(opponent: Opponent, seed: u64, policy_seed: u64) -> Trace {
    let mut e = Env::new(env_cfg(opponent, AgentSeat::SeedParity, 10));
    e.reset(seed);
    let mut rng = Rng::new(policy_seed);
    let mut steps = Vec::new();
    while !e.is_terminal() {
        let o = obs(&e);
        let n = e.legal_actions().len();
        let i = random_index(e.legal_actions(), &mut rng);
        steps.push((o, n, e.step_legal(i).unwrap()));
    }
    (steps, e.action_log().to_vec())
}

#[test]
fn same_seed_and_policy_give_identical_episodes() {
    for seed in 0..3u64 {
        for opponent in [
            Opponent::TwoSeat,
            bot(BotKind::Random, 0.0),
            bot(BotKind::Rush, 0.3),
            bot(BotKind::Economy, 0.1),
        ] {
            let a = trace(opponent, seed, 40 + seed);
            let b = trace(opponent, seed, 40 + seed);
            assert!(a == b, "seed {seed} {opponent:?}: episodes differ");
            assert!(!a.0.is_empty());
        }
    }
}

#[test]
fn reusing_one_env_gives_the_same_episode() {
    let mut e = Env::new(env_cfg(bot(BotKind::Rush, 0.5), AgentSeat::Fixed(1), 10));
    let mut logs = Vec::new();
    for _ in 0..2 {
        e.reset(17);
        let mut rng = Rng::new(3);
        while !e.is_terminal() {
            let i = random_index(e.legal_actions(), &mut rng);
            e.step_legal(i).unwrap();
        }
        logs.push((e.action_log().to_vec(), e.state().clone()));
    }
    assert!(
        logs[0] == logs[1],
        "the opponent rng was not reset with the episode"
    );
}

fn replay_from_log(seed: u64, turn_cap: u32, log: &[(PlayerId, Action)]) -> State {
    let mut s = new_game_from_seed(seed, &MapParams::default(), &cfg(turn_cap));
    for (_, a) in log {
        apply(&mut s, a).expect("logged actions replay");
    }
    s
}

#[test]
fn the_opponent_never_draws_from_the_game_rng() {
    for seed in 0..4u64 {
        for eps in [0.0, 0.5, 1.0] {
            for kind in KINDS {
                let mut e = Env::new(env_cfg(bot(kind, eps), AgentSeat::SeedParity, 8));
                e.reset(seed);
                assert_eq!(&replay_from_log(seed, 8, e.action_log()), e.state());
                let mut rng = Rng::new(seed);
                while !e.is_terminal() {
                    let i = random_index(e.legal_actions(), &mut rng);
                    e.step_legal(i).unwrap();
                    let replayed = replay_from_log(seed, 8, e.action_log());
                    assert!(
                        replayed == *e.state(),
                        "seed {seed} {kind:?} eps {eps}: state differs from the log's replay"
                    );
                }
            }
        }
    }
}

/// A milestone 7 replay log of the env's episode, recorded by feeding the env's action log back.
fn replay_log_of(e: &Env, turn_cap: u32) -> ReplayLog {
    let actions: Vec<Action> = e.action_log().iter().map(|(_, a)| *a).collect();
    let mut next = actions.iter();
    let log = record_game(
        e.seed(),
        &MapParams::default(),
        &cfg(turn_cap),
        actions.len(),
        &mut |_: &State| *next.next().expect("one action per call"),
    );
    assert_eq!(log.actions, actions);
    log
}

#[test]
fn the_action_log_replays_with_the_milestone_7_verifier() {
    for seed in 0..4u64 {
        for opponent in [
            Opponent::TwoSeat,
            bot(BotKind::Rush, 0.2),
            bot(BotKind::Random, 0.0),
        ] {
            for seat in [AgentSeat::Fixed(0), AgentSeat::Fixed(1)] {
                let mut e = Env::new(env_cfg(opponent, seat, 10));
                e.reset(seed);
                let mut rng = Rng::new(seed + 5);
                while !e.is_terminal() {
                    let i = random_index(e.legal_actions(), &mut rng);
                    e.step_legal(i).unwrap();
                }
                let log = replay_log_of(&e, 10);
                assert_eq!(log.outcome, e.state().outcome);
                let json = log.to_json();
                let back = ReplayLog::from_json(&json).expect("round trip");
                let fin = replay(&back).expect("the milestone 7 verifier accepts the log");
                assert!(fin == *e.state(), "seed {seed} {opponent:?} {seat:?}");
                if opponent != Opponent::TwoSeat {
                    let seat = e.agent_seat();
                    assert!(
                        e.action_log().iter().any(|(pl, _)| *pl != seat),
                        "the log has the opponent's actions"
                    );
                }
            }
        }
    }
}
