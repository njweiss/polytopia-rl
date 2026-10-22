//! Milestone 9 "Tests" item 3: global-vector goldens. Every expected value is computed by hand
//! from the brief's normalizers and the spec's score table (each scripted score is also checked
//! against `sim_core::score::score`, so a wrong board fails loudly as a test-board error).

mod common;

use common::*;
use sim_core::data::{Tech, UnitType};
use sim_core::map::MapParams;
use sim_core::score::score;
use sim_core::setup::new_game_from_seed;
use sim_core::state::{Config, State};
use sim_core::testing::StateBuilder;
use sim_core::view::PlayerView;
use sim_env::obs::encode_globals;

fn globals(s: &State, v: u8) -> Vec<f32> {
    let mut out = vec![0.0f32; gl::COUNT];
    encode_globals(&PlayerView::new(s, v), &mut out);
    out
}

/// `[stars, income, turn, pending, my_score, opp_score, score_diff, my_cities, my_units]` then
/// the nine tech flags.
fn vector(head: [f32; 9], techs: &[Tech]) -> Vec<f32> {
    let mut v = head.to_vec();
    for t in TECHS {
        v.push(if techs.contains(&t) { 1.0 } else { 0.0 });
    }
    v
}

#[test]
fn starting_state_for_both_seats() {
    // Spec "Starting state": 5 stars, Organization, one level-1 capital (income 1 + 1), one
    // Warrior, score 515 each (spec "Score").
    for seed in 0..4 {
        let s = new_game_from_seed(seed, &MapParams::default(), &cfg(30));
        let want = vector(
            [0.1, 0.1, 0.0, 0.0, 0.2575, 0.2575, 0.0, 0.125, 0.0625],
            &[Tech::Organization],
        );
        for v in [0u8, 1] {
            assert_slices_approx(&globals(&s, v), &want, &format!("seed {seed} viewer {v}"));
        }
    }
}

/// 8x8, nothing explored. Player 0: a level-1 capital and a Warrior, 51 stars (above the cap).
/// Player 1: a level-1 capital, a Warrior, Climbing and Archery.
fn behind() -> StateBuilder {
    StateBuilder::new(8)
        .config(cfg(20))
        .turn(7)
        .stars(0, 51)
        .city_with(0, p(1, 1), |c| c.is_capital = true)
        .city_with(1, p(5, 5), |c| c.is_capital = true)
        .unit(0, UnitType::Warrior, p(1, 1))
        .unit(1, UnitType::Warrior, p(5, 5))
        .techs(1, &[Tech::Climbing, Tech::Archery])
}

#[test]
fn scripted_negative_score_diff_and_star_clip() {
    let s = behind().build();
    // Player 0: territory 9 x 20 + city 100 + Warrior 10 = 290.
    // Player 1: 290 + Climbing 100 + Archery 200 = 590.
    assert_eq!((score(&s, 0), score(&s, 1)), (290, 590), "test board");
    let want0 = vector(
        [
            1.0,   // 51 stars, clipped at 50
            0.1,   // income 2 / 20
            0.35,  // turn 7 / cap 20
            0.0,   // no pending reward
            0.145, // 290 / 2000
            0.295, // 590 / 2000
            -0.3,  // (290 - 590) / 1000
            0.125, // 1 city / 8
            0.0625,
        ],
        &[],
    );
    assert_slices_approx(&globals(&s, 0), &want0, "viewer 0");
    let want1 = vector(
        [0.0, 0.1, 0.35, 0.0, 0.295, 0.145, 0.3, 0.125, 0.0625],
        &[Tech::Climbing, Tech::Archery],
    );
    assert_slices_approx(&globals(&s, 1), &want1, "viewer 1");
}

#[test]
fn opponent_stars_and_techs_have_no_effect() {
    let s = behind().build();
    let mut twin = s.clone();
    twin.players[1].stars = 37;
    // Hunting has Climbing's tier, so player 1's public score is unchanged.
    twin.players[1].techs = sim_core::state::TechSet::from_slice(&[Tech::Hunting, Tech::Archery]);
    assert_eq!(score(&twin, 1), score(&s, 1), "test twin");
    let a: Vec<u32> = globals(&s, 0).iter().map(|x| x.to_bits()).collect();
    let b: Vec<u32> = globals(&twin, 0).iter().map(|x| x.to_bits()).collect();
    assert_eq!(a, b);
}

/// 11x11. Player 0: nine cities 3 apart (one capital, one at level 15), 17 Warriors, every
/// tech, exactly 50 stars, a pending reward. Player 1: one level-1 city in the corner.
fn ahead() -> State {
    let mut b = StateBuilder::new(11).stars(0, 50).techs(0, &TECHS);
    for (i, &(r, c)) in [
        (1, 1),
        (1, 4),
        (1, 7),
        (4, 1),
        (4, 4),
        (4, 7),
        (7, 1),
        (7, 4),
        (7, 7),
    ]
    .iter()
    .enumerate()
    {
        b = b.city_with(0, p(r, c), |city| {
            city.is_capital = i == 0;
            if i == 1 {
                city.level = 15;
                city.pending_reward = true;
            }
        });
    }
    b = b.city(1, p(10, 10));
    for c in 0..11 {
        b = b.unit(0, UnitType::Warrior, p(0, c));
    }
    for c in 0..6 {
        b = b.unit(0, UnitType::Warrior, p(10, c));
    }
    b.build()
}

#[test]
fn scripted_clips_and_clamps() {
    let s = ahead();
    // Player 0: territory 81 x 20 = 1620; cities 9 x 100 + 14 x 50 = 1600; Warriors 170;
    // techs 100 x (1+1+1+1+2+2+2+2+2) = 1400. Player 1: 4 tiles x 20 + 100 = 180.
    assert_eq!((score(&s, 0), score(&s, 1)), (4790, 180), "test board");
    let want0 = vector(
        [
            1.0,  // exactly 50 stars
            1.0,  // income 8 + 15 + 1 = 24, clipped at 20
            0.0,  // turn 0
            1.0,  // own pending reward
            1.0,  // 4790 clipped at 2000
            0.09, // 180 / 2000
            1.0,  // (4790 - 180) / 1000 clamped to 1
            1.0,  // 9 cities clipped at 8
            1.0,  // 17 units clipped at 16
        ],
        &TECHS,
    );
    assert_slices_approx(&globals(&s, 0), &want0, "viewer 0");
    let want1 = vector(
        [
            0.0, 0.05, // income 1 / 20
            0.0, 0.0, // the pending reward is not the viewer's
            0.09, 1.0, -1.0, // clamped
            0.125, 0.0,
        ],
        &[],
    );
    assert_slices_approx(&globals(&s, 1), &want1, "viewer 1");
}

#[test]
fn income_exactly_at_the_cap() {
    // One level-19 capital: 19 + 1 = 20 stars per turn.
    let s = StateBuilder::new(8)
        .city_with(0, p(3, 3), |c| {
            c.level = 19;
            c.is_capital = true;
        })
        .build();
    assert_approx(globals(&s, 0)[gl::INCOME], 1.0, "income 20 / 20");
    let s = StateBuilder::new(8)
        .city_with(0, p(3, 3), |c| c.level = 13)
        .build();
    assert_approx(globals(&s, 0)[gl::INCOME], 0.65, "income 13 / 20");
}

#[test]
fn turn_fraction_uses_the_configured_cap() {
    for (turn, cap, want) in [
        (0, 30, 0.0),
        (15, 30, 0.5),
        (29, 30, 29.0 / 30.0),
        (3, 4, 0.75),
    ] {
        let s = StateBuilder::new(6)
            .config(Config {
                turn_cap: cap,
                ..Config::default()
            })
            .turn(turn)
            .build();
        assert_approx(globals(&s, 0)[gl::TURN], want, &format!("{turn}/{cap}"));
    }
}

#[test]
fn matches_the_independent_replica_on_playout_states() {
    let mut n = 0;
    for s in corpus(0..8, 7) {
        for v in [0u8, 1] {
            assert_slices_approx(
                &globals(&s, v),
                &my_encode_globals(&s, v),
                &format!("turn {} viewer {v}", s.turn),
            );
            n += 1;
        }
    }
    assert!(n > 500, "corpus too small: {n}");
}
