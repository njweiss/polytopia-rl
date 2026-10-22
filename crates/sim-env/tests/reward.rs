//! Milestone 9 "Tests" item 8: rewards. Goldens are hand-computed from the brief's formula
//! `rewards[s] = terminal_weight * t_s + shaping_weight * (gamma * phi_s(after) - phi_s(before))`
//! with `phi_s = (score(s) - score(1 - s)) / potential_scale` (0 when terminal), on scripted
//! two-seat games: both players start at 515 (spec "Score"), and researching a tier-1 tech at
//! turn 0 costs exactly the 5 starting stars (4 + 1 city x tier 1) and adds 100.
//! Property tests compare every step with `common::my_rewards`, computed from the states
//! before and after the step.

mod common;

use bots::BotKind;
use common::*;
use sim_core::actions::{legal_actions, Action};
use sim_core::data::Tech;
use sim_core::rng::Rng;
use sim_core::score::score;
use sim_core::state::{Outcome, State};
use sim_env::env::{AgentSeat, Env, EnvConfig, Opponent, RewardConfig, StepOut};

fn shaped(gamma: f32) -> RewardConfig {
    RewardConfig {
        terminal_weight: 1.0,
        shaping_weight: 1.0,
        gamma,
        potential_scale: 1000.0,
    }
}

fn two_seat(turn_cap: u32, reward: RewardConfig, seed: u64) -> Env {
    let mut c = env_cfg(Opponent::TwoSeat, AgentSeat::Fixed(0), turn_cap);
    c.reward = reward;
    let mut e = Env::new(c);
    e.reset(seed);
    assert_eq!(
        [score(e.state(), 0), score(e.state(), 1)],
        [515, 515],
        "test premise"
    );
    e
}

fn research(e: &mut Env, tech: Tech) -> StepOut {
    let a = Action::Research { tech };
    assert!(legal_actions(e.state()).contains(&a), "test premise: {a:?}");
    e.step(&a).unwrap()
}

fn assert_out(out: StepOut, reward: f32, rewards: [f32; 2], terminated: bool, what: &str) {
    assert_approx(out.reward, reward, &format!("{what}: reward"));
    assert_approx(out.rewards[0], rewards[0], &format!("{what}: rewards[0]"));
    assert_approx(out.rewards[1], rewards[1], &format!("{what}: rewards[1]"));
    assert_eq!(out.terminated, terminated, "{what}: terminated");
    assert!(!out.truncated, "{what}: truncated");
}

#[test]
fn default_weights_reward_only_the_end() {
    assert_eq!(
        RewardConfig::default(),
        RewardConfig {
            terminal_weight: 1.0,
            shaping_weight: 0.0,
            gamma: 1.0,
            potential_scale: 1000.0
        }
    );
    // Player 0 researches (615 vs 515), then both end turn at cap 1: Win(0).
    let mut e = two_seat(1, RewardConfig::default(), 4);
    assert_out(
        research(&mut e, Tech::Climbing),
        0.0,
        [0.0, 0.0],
        false,
        "research",
    );
    assert_out(
        e.step(&Action::EndTurn).unwrap(),
        0.0,
        [0.0, 0.0],
        false,
        "end 0",
    );
    // The last step is player 1's: reward = rewards[1] = -1.
    assert_out(
        e.step(&Action::EndTurn).unwrap(),
        -1.0,
        [1.0, -1.0],
        true,
        "end 1",
    );
    assert_eq!(e.state().outcome, Some(Outcome::Win(0)));
}

#[test]
fn win_for_player_0_with_shaping_and_gamma() {
    let mut e = two_seat(1, shaped(0.9), 6);
    // phi_0: 0 -> 0.1. r0 = 0.9 x 0.1 - 0 = 0.09, r1 = -0.09.
    assert_out(
        research(&mut e, Tech::Climbing),
        0.09,
        [0.09, -0.09],
        false,
        "research",
    );
    // No score change: r0 = 0.9 x 0.1 - 0.1 = -0.01, r1 = +0.01. Actor: player 0.
    assert_out(
        e.step(&Action::EndTurn).unwrap(),
        -0.01,
        [-0.01, 0.01],
        false,
        "end 0",
    );
    // Terminal, Win(0): phi(after) = 0. r0 = 1 + (0 - 0.1) = 0.9, r1 = -1 + (0 + 0.1) = -0.9.
    assert_out(
        e.step(&Action::EndTurn).unwrap(),
        -0.9,
        [0.9, -0.9],
        true,
        "end 1",
    );
}

#[test]
fn win_for_player_1_has_the_mirrored_sign() {
    let mut e = two_seat(1, shaped(0.9), 6);
    assert_out(
        e.step(&Action::EndTurn).unwrap(),
        0.0,
        [0.0, 0.0],
        false,
        "end 0",
    );
    // Player 1 researches: phi_1 0 -> 0.1. r1 = 0.09, r0 = -0.09; actor player 1.
    assert_out(
        research(&mut e, Tech::Hunting),
        0.09,
        [-0.09, 0.09],
        false,
        "research",
    );
    // Terminal, Win(1): r1 = 1 - 0.1 = 0.9, r0 = -1 + 0.1 = -0.9; actor player 1.
    assert_out(
        e.step(&Action::EndTurn).unwrap(),
        0.9,
        [-0.9, 0.9],
        true,
        "end 1",
    );
    assert_eq!(e.state().outcome, Some(Outcome::Win(1)));
}

#[test]
fn draw_rewards_zero_terminal_and_drops_the_potential() {
    let mut e = two_seat(1, shaped(1.0), 8);
    assert_out(
        research(&mut e, Tech::Riding),
        0.1,
        [0.1, -0.1],
        false,
        "research 0",
    );
    assert_out(
        e.step(&Action::EndTurn).unwrap(),
        0.0,
        [0.0, 0.0],
        false,
        "end 0",
    );
    // 615 vs 615: phi 0.1 -> 0 for player 0. r0 = -0.1, r1 = 0.1.
    assert_out(
        research(&mut e, Tech::Hunting),
        0.1,
        [-0.1, 0.1],
        false,
        "research 1",
    );
    // Draw at the cap: t = 0 and phi(before) = phi(after) = 0.
    assert_out(
        e.step(&Action::EndTurn).unwrap(),
        0.0,
        [0.0, 0.0],
        true,
        "end 1",
    );
    assert_eq!(e.state().outcome, Some(Outcome::Draw));
}

#[test]
fn terminal_weight_scales_the_end_signal() {
    let r = RewardConfig {
        terminal_weight: 2.5,
        ..RewardConfig::default()
    };
    let mut e = two_seat(1, r, 4);
    research(&mut e, Tech::Climbing);
    e.step(&Action::EndTurn).unwrap();
    assert_out(
        e.step(&Action::EndTurn).unwrap(),
        -2.5,
        [2.5, -2.5],
        true,
        "end 1",
    );
}

#[test]
fn potential_scale_divides_the_score_difference() {
    let r = RewardConfig {
        terminal_weight: 0.0,
        shaping_weight: 2.0,
        gamma: 1.0,
        potential_scale: 400.0,
    };
    let mut e = two_seat(1, r, 4);
    // 2 x (100 / 400) = 0.5.
    assert_out(
        research(&mut e, Tech::Climbing),
        0.5,
        [0.5, -0.5],
        false,
        "research",
    );
    // Terminal Win(0) with terminal_weight 0: 2 x (0 - 0.25) = -0.5 for player 0.
    e.step(&Action::EndTurn).unwrap();
    assert_out(
        e.step(&Action::EndTurn).unwrap(),
        0.5,
        [-0.5, 0.5],
        true,
        "end 1",
    );
}

/// Plays a whole episode with a random agent, checking every step against `my_rewards`.
/// Returns `(steps, steps whose shaping delta came only from the opponent's actions)`.
fn check_episode(c: EnvConfig, seed: u64) -> (usize, usize) {
    let r = c.reward;
    let two_seat = c.opponent == Opponent::TwoSeat;
    let mut e = Env::new(c);
    e.reset(seed);
    let mut rng = Rng::new(seed ^ 0x55);
    let (mut steps, mut from_opponent) = (0, 0);
    while !e.is_terminal() {
        let before: State = e.state().clone();
        let actor = before.current;
        let i = random_index(e.legal_actions(), &mut rng);
        let a = e.legal_actions()[i];
        let out = e.step_legal(i).unwrap();
        let want = my_rewards(&before, e.state(), &r);
        let ctx = format!("seed {seed} step {steps} {a:?}");
        assert_approx(out.rewards[0], want[0], &ctx);
        assert_approx(out.rewards[1], want[1], &ctx);
        let seat = if two_seat { actor } else { e.agent_seat() };
        assert_eq!(
            out.reward, out.rewards[seat as usize],
            "{ctx}: reward picks the seat"
        );
        if !two_seat && a == Action::EndTurn && !e.is_terminal() && want[0].abs() > 1e-4 {
            from_opponent += 1;
        }
        steps += 1;
    }
    (steps, from_opponent)
}

#[test]
fn every_step_matches_the_formula_in_bot_mode() {
    let mut from_opponent = 0;
    for seed in 0..6u64 {
        for (kind, gamma) in [
            (BotKind::Rush, 1.0),
            (BotKind::Economy, 0.95),
            (BotKind::Random, 0.8),
        ] {
            let mut c = env_cfg(bot(kind, 0.1), AgentSeat::SeedParity, 12);
            c.reward = RewardConfig {
                terminal_weight: 1.5,
                shaping_weight: 0.7,
                gamma,
                potential_scale: 250.0,
            };
            from_opponent += check_episode(c, seed).1;
        }
    }
    // An agent EndTurn changes no score, so these deltas are the opponent's turn alone.
    assert!(
        from_opponent > 30,
        "opponent turns in the delta: {from_opponent}"
    );
}

#[test]
fn every_step_matches_the_formula_in_two_seat_mode() {
    for seed in 0..4u64 {
        let mut c = env_cfg(Opponent::TwoSeat, AgentSeat::Fixed(1), 8);
        c.reward = shaped(0.9);
        check_episode(c, seed);
    }
}

#[test]
fn zero_sum_at_gamma_one() {
    for seed in 0..4u64 {
        for opponent in [Opponent::TwoSeat, bot(BotKind::Rush, 0.2)] {
            let mut c = env_cfg(opponent, AgentSeat::SeedParity, 10);
            c.reward = shaped(1.0);
            let mut e = Env::new(c);
            e.reset(seed);
            let mut rng = Rng::new(seed);
            while !e.is_terminal() {
                let i = random_index(e.legal_actions(), &mut rng);
                let out = e.step_legal(i).unwrap();
                assert!(
                    (out.rewards[0] + out.rewards[1]).abs() <= 1e-6,
                    "{:?}",
                    out.rewards
                );
            }
        }
    }
}

#[test]
fn default_config_is_zero_until_the_last_step() {
    for seed in 0..6u64 {
        let mut e = Env::new(env_cfg(bot(BotKind::Rush, 0.0), AgentSeat::SeedParity, 15));
        e.reset(seed);
        let agent = e.agent_seat();
        let mut rng = Rng::new(seed + 1);
        let mut last = None;
        while !e.is_terminal() {
            if let Some(out) = last {
                let out: StepOut = out;
                assert_eq!(
                    out.rewards,
                    [0.0, 0.0],
                    "seed {seed}: nonzero before the end"
                );
            }
            let i = random_index(e.legal_actions(), &mut rng);
            last = Some(e.step_legal(i).unwrap());
        }
        let out = last.unwrap();
        let t = terminal_sign(e.state(), agent);
        assert_eq!(out.reward, t, "seed {seed}");
        assert_eq!(out.rewards[agent as usize], t);
        assert_eq!(out.rewards[1 - agent as usize], -t);
    }
}
