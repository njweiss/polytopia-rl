//! Milestone 7: the random policy (docs/MILESTONE_7 "Random policy", G6, "Tests" item 5). The
//! exact draw order is checked against `common::my_random_action`, a replica written from the
//! brief; the statistical tests use fixed seeds and about 5 sigma tolerances.

mod common;

use common::*;
use sim_core::actions::{apply, legal_actions, Action};
use sim_core::data::UnitType;
use sim_core::map::MapParams;
use sim_core::rng::Rng;
use sim_core::setup::new_game_from_seed;
use sim_core::state::{Config, Outcome, Phase, State};
use sim_core::testing::StateBuilder;
use sim_tools::policy::random_action;
use sim_tools::replay::record_game;

/// Plays with `random_action` and checks every pick against the replica, including the rng state
/// after the draw.
fn play_checked(seed: u64, config: &Config, policy_seed: u64, max_actions: usize) -> Vec<Action> {
    let mut s = new_game_from_seed(seed, &MapParams::default(), config);
    let mut rng = Rng::new(policy_seed);
    let mut mine = Rng::new(policy_seed);
    let mut actions = Vec::new();
    while !s.is_terminal() && actions.len() < max_actions {
        let a = random_action(&s, &mut rng);
        let b = my_random_action(&s, &mut mine);
        assert_eq!(a, b, "seed {seed} step {}", actions.len());
        assert_eq!(rng, mine, "seed {seed} step {}: draws", actions.len());
        assert!(legal_actions(&s).contains(&a), "{a:?} is not legal");
        apply(&mut s, &a).unwrap();
        actions.push(a);
    }
    actions
}

#[test]
fn picks_match_the_brief_draw_order_and_are_legal() {
    for seed in 0..10u64 {
        play_checked(seed, &cfg(4, seed % 2 == 1), seed + 100, 3000);
    }
}

#[test]
fn same_seed_same_sequence() {
    let a = play_checked(3, &cfg(5, false), 9, 3000);
    let b = play_checked(3, &cfg(5, false), 9, 3000);
    assert_eq!(a, b);
    let c = play_checked(3, &cfg(5, false), 10, 3000);
    assert_ne!(a, c, "a different policy seed should give a different game");
}

#[test]
fn only_end_turn_legal_returns_end_turn_without_drawing() {
    // No units, no cities, no stars: EndTurn is the only legal action.
    let s = StateBuilder::new(4).build();
    assert_eq!(legal_actions(&s), vec![Action::EndTurn]);
    let mut rng = Rng::new(5);
    for _ in 0..10 {
        assert_eq!(random_action(&s, &mut rng), Action::EndTurn);
    }
    assert_eq!(rng, Rng::new(5), "no draw when EndTurn is the only choice");
}

/// Q7 (assumption): a pending reward makes `EndTurn` illegal, so the policy picks uniformly among
/// the legal `ChooseReward` actions with one `gen_range` draw and no `gen_f32`.
#[test]
fn pending_reward_never_returns_end_turn() {
    let s = StateBuilder::new(5)
        .city_with(0, p(2, 2), |c| {
            c.level = 2;
            c.population = 2;
            c.pending_reward = true;
        })
        .build();
    let legal = legal_actions(&s);
    assert_eq!(legal.len(), 2, "{legal:?}");
    assert!(!legal.contains(&Action::EndTurn));
    let mut seen = [0usize; 2];
    for k in 0..400u64 {
        let mut rng = Rng::new(k);
        let mut probe = Rng::new(k);
        let i = probe.gen_range(2) as usize;
        let a = random_action(&s, &mut rng);
        assert_eq!(a, legal[i], "k {k}");
        assert_eq!(rng, probe, "k {k}: exactly one gen_range draw");
        seen[i] += 1;
    }
    assert!(seen[0] > 0 && seen[1] > 0);
}

/// Long default games pass through many level-ups; every pick must stay legal.
#[test]
fn default_games_with_rewards_stay_legal() {
    for seed in 0..4u64 {
        let actions = play_checked(seed, &Config::default(), seed ^ 0x55, 5000);
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, Action::ChooseReward { .. })),
            "seed {seed}: no reward was chosen"
        );
    }
}

#[test]
fn single_other_action_costs_one_draw_on_end_turn_and_two_otherwise() {
    // A P0 Rider that already attacked (Escape: one move left) on a 2x2 board where only (0,1) is
    // explored besides its own tile, no stars: Move to (0,1) and EndTurn are the only actions.
    let s = StateBuilder::new(2)
        .unit_with(0, UnitType::Rider, p(0, 0), |u| u.phase = Phase::Attacked)
        .explore(0, &[p(0, 0), p(0, 1)])
        .build();
    let legal = legal_actions(&s);
    assert_eq!(legal.len(), 2, "{legal:?}");
    let other = legal[0];
    for k in 0..200u64 {
        let mut rng = Rng::new(k);
        let mut probe = Rng::new(k);
        let u = probe.gen_f32();
        let a = random_action(&s, &mut rng);
        if u < 0.15 {
            assert_eq!(a, Action::EndTurn, "k {k}");
        } else {
            assert_eq!(a, other, "k {k}");
            assert_eq!(probe.gen_range(1), 0);
        }
        assert_eq!(rng, probe, "k {k}: draw count");
    }
}

#[test]
fn panics_on_a_terminal_state() {
    let mut s = StateBuilder::new(4).build();
    // Works on the live state first (so a stub that always panics fails this test).
    assert_eq!(random_action(&s, &mut Rng::new(0)), Action::EndTurn);
    s.outcome = Some(Outcome::Draw);
    let r = std::panic::catch_unwind(|| random_action(&s, &mut Rng::new(0)));
    assert!(r.is_err(), "random_action must panic on a terminal state");
}

/// Expected frequencies: EndTurn 0.15 exactly when other actions exist (brief), each other action
/// 0.85 / m. With N = 20000 draws, sigma(EndTurn) = sqrt(0.15 * 0.85 / N) = 0.00252, so 5 sigma is
/// 0.0126; per other action sigma = sqrt(q (1 - q) / N) with q = 0.85 / m.
#[test]
fn end_turn_frequency_is_0_15_and_others_are_uniform() {
    let s = new_game_from_seed(0, &MapParams::default(), &Config::default());
    let legal = legal_actions(&s);
    let others: Vec<Action> = legal
        .iter()
        .copied()
        .filter(|a| *a != Action::EndTurn)
        .collect();
    assert!(others.len() >= 2, "{legal:?}");
    let n = 20_000usize;
    let mut rng = Rng::new(2026);
    let mut end = 0usize;
    let mut counts = vec![0usize; others.len()];
    for _ in 0..n {
        let a = random_action(&s, &mut rng);
        if a == Action::EndTurn {
            end += 1;
        } else {
            let i = others.iter().position(|o| *o == a).expect("legal");
            counts[i] += 1;
        }
    }
    let f = end as f64 / n as f64;
    assert!((f - 0.15).abs() < 0.0126, "EndTurn frequency {f}");
    let q = 0.85 / others.len() as f64;
    let tol = 5.0 * (q * (1.0 - q) / n as f64).sqrt();
    for (a, c) in others.iter().zip(&counts) {
        let fa = *c as f64 / n as f64;
        assert!((fa - q).abs() < tol, "{a:?}: {fa} vs {q}");
    }
}

#[test]
fn a_long_game_has_both_end_turns_and_other_actions() {
    let actions = play_checked(1, &cfg(50, false), 3, 400);
    let ends = actions.iter().filter(|a| **a == Action::EndTurn).count();
    assert!(ends > 0, "no EndTurn");
    assert!(ends < actions.len(), "only EndTurn");
    assert!(actions.iter().any(|a| matches!(a, Action::Move { .. })));
}

#[test]
fn small_turn_caps_terminate_well_within_max_actions() {
    for seed in 0..20u64 {
        let turn_cap = 1 + (seed as u32 % 5);
        let mut rng = Rng::new(seed + 500);
        let log = record_game(
            seed,
            &MapParams::default(),
            &cfg(turn_cap, seed % 2 == 0),
            5000,
            &mut |s: &State| random_action(s, &mut rng),
        );
        assert!(log.outcome.is_some(), "seed {seed} did not finish");
        assert!(
            log.actions.len() < 1000,
            "seed {seed}: {} actions for turn cap {turn_cap}",
            log.actions.len()
        );
    }
}
