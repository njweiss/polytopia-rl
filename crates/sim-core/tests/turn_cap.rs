//! Milestone 6: turn-cap termination in `EndTurn` (docs/MILESTONE_6 "Turn cap", F5, "Tests"
//! item 6; spec "Termination"). Expected states are written by hand from the brief: when player 1
//! ends turn `turn_cap - 1`, `turn` becomes `turn_cap`, `current` 0, and the outcome is decided by
//! score with no income and no unit reset. Scores use the spec's numbers: 515 at the start, +100
//! for a tier 1 tech.

mod common;

use common::*;
use sim_core::actions::{apply, legal_actions, Action, ActionError};
use sim_core::data::{Tech, UnitType};
use sim_core::map::MapParams;
use sim_core::setup::new_game_from_seed;
use sim_core::state::{CityId, Config, Outcome, Phase, State, UnitId};
use sim_core::testing::StateBuilder;

fn game(seed: u64, turn_cap: u32) -> State {
    let cfg = Config {
        turn_cap,
        ..Config::default()
    };
    new_game_from_seed(seed, &MapParams::default(), &cfg)
}

fn ok(s: &mut State, a: Action) {
    assert!(
        legal_actions(s).contains(&a),
        "{a:?} should be legal; legal = {:?}",
        legal_actions(s)
    );
    assert_eq!(apply(s, &a), Ok(()), "{a:?}");
}

fn end_turn(s: &mut State) {
    ok(s, Action::EndTurn);
}

/// The final state's checks: no actions, and `apply` refuses everything without a change.
fn assert_final(s: &State) {
    assert!(s.is_terminal());
    assert!(legal_actions(s).is_empty());
    for a in every_action_kind(s) {
        let mut t = s.clone();
        assert_eq!(apply(&mut t, &a), Err(ActionError::Illegal), "{a:?}");
        assert_eq!(t, *s, "{a:?} changed a final state");
    }
}

/// With `turn_cap = 2` each player plays turns 0 and 1; the game ends exactly when player 1
/// ends turn 1. Equal scores (515 each, nobody did anything that scores) are a draw. Player 0's
/// Warrior recovered on turn 1 (phase Done) and is not reset; player 0's stars (5 + 2 income on
/// turn 1) get no income at the end.
#[test]
fn the_game_ends_when_player_1_ends_the_last_turn() {
    for seed in 0..10 {
        let mut s = game(seed, 2);
        end_turn(&mut s);
        end_turn(&mut s);
        assert_eq!((s.turn, s.current, s.outcome), (1, 0, None), "seed {seed}");
        ok(&mut s, Action::Recover { unit: UnitId(0) });
        end_turn(&mut s);
        assert_eq!((s.turn, s.current, s.outcome), (1, 1, None), "seed {seed}");
        assert!(!s.is_terminal());
        assert_eq!(s.players[0].stars, 7, "seed {seed}");

        let before = s.clone();
        end_turn(&mut s);
        let mut want = before.clone();
        (want.turn, want.current, want.outcome) = (2, 0, Some(Outcome::Draw));
        assert_eq!(s, want, "seed {seed}");
        assert_eq!(s.unit(UnitId(0)).unwrap().phase, Phase::Done);
        assert_final(&s);
    }
}

#[test]
fn a_cap_of_1_ends_after_one_turn_each() {
    let mut s = game(1, 1);
    end_turn(&mut s);
    assert_eq!((s.turn, s.current, s.outcome), (0, 1, None));
    end_turn(&mut s);
    assert_eq!((s.turn, s.current, s.outcome), (1, 0, Some(Outcome::Draw)));
    assert_final(&s);
}

/// Researching Hunting (cost 4 + 1 * 1 = 5, all of the starting 5 stars) adds 100 to the score.
#[test]
fn the_higher_score_wins_at_the_cap() {
    let hunting = Action::Research {
        tech: Tech::Hunting,
    };
    // Player 0 is ahead, 615 to 515.
    let mut s = game(2, 1);
    ok(&mut s, hunting);
    end_turn(&mut s);
    end_turn(&mut s);
    assert_eq!(s.outcome, Some(Outcome::Win(0)));
    // Player 1 is ahead, 515 to 615.
    let mut s = game(2, 1);
    end_turn(&mut s);
    ok(&mut s, hunting);
    end_turn(&mut s);
    assert_eq!(s.outcome, Some(Outcome::Win(1)));
    // Both research it: 615 each, a draw.
    let mut s = game(2, 1);
    ok(&mut s, hunting);
    end_turn(&mut s);
    ok(&mut s, hunting);
    end_turn(&mut s);
    assert_eq!(s.outcome, Some(Outcome::Draw));
}

/// A small lead is enough: player 1 has one more explored tile (5 points) on a scripted board
/// with the default weights, and the cap is read from the state's config.
#[test]
fn a_five_point_lead_wins() {
    let board = |extra: bool| {
        let b = StateBuilder::new(5)
            .city(0, p(0, 0))
            .city(1, p(4, 4))
            .explore(0, &[p(2, 2)])
            .explore(1, &[p(2, 2)])
            .config(Config {
                turn_cap: 7,
                ..Config::default()
            })
            .turn(6)
            .current(1);
        if extra {
            b.explore(1, &[p(2, 3)]).build()
        } else {
            b.build()
        }
    };
    let mut s = board(true);
    end_turn(&mut s);
    assert_eq!((s.turn, s.current), (7, 0));
    assert_eq!(s.outcome, Some(Outcome::Win(1)));
    let mut s = board(false);
    end_turn(&mut s);
    assert_eq!(s.outcome, Some(Outcome::Draw));
}

/// Player 0 ending the last turn does not end the game; only player 1's EndTurn does.
#[test]
fn player_0_ending_the_last_turn_does_not_end_the_game() {
    let mut s = StateBuilder::new(5)
        .city(0, p(0, 0))
        .city(1, p(4, 4))
        .turn(9)
        .config(Config {
            turn_cap: 10,
            ..Config::default()
        })
        .build();
    end_turn(&mut s);
    assert_eq!((s.turn, s.current, s.outcome), (9, 1, None));
    end_turn(&mut s);
    assert_eq!((s.turn, s.current), (10, 0));
    assert!(s.is_terminal());
}

/// Before the cap, EndTurn is milestone 4's: the next player gets income (capital level 1 + 1)
/// and its units become Fresh.
#[test]
fn end_turn_before_the_cap_is_unchanged() {
    let mut s = StateBuilder::new(5)
        .city_with(0, p(0, 0), |c| c.is_capital = true)
        .city_with(1, p(4, 4), |c| c.is_capital = true)
        .unit_with(0, UnitType::Warrior, p(1, 1), |u| u.phase = Phase::Done)
        .unit_with(1, UnitType::Warrior, p(3, 3), |u| u.phase = Phase::Done)
        .turn(1)
        .current(1)
        .config(Config {
            turn_cap: 3,
            ..Config::default()
        })
        .build();
    let before = s.clone();
    end_turn(&mut s);
    let mut want = before.clone();
    (want.turn, want.current) = (2, 0);
    want.players[0].stars += 2;
    want.units[0].phase = Phase::Fresh;
    assert_eq!(s, want);
    assert!(!s.is_terminal());

    // The default cap is 30: turn 29 is the last one.
    let mut s = StateBuilder::new(5)
        .city_with(0, p(0, 0), |c| c.is_capital = true)
        .city_with(1, p(4, 4), |c| c.is_capital = true)
        .turn(28)
        .current(1)
        .build();
    end_turn(&mut s);
    assert_eq!((s.turn, s.current, s.outcome), (29, 0, None));
    end_turn(&mut s);
    end_turn(&mut s);
    assert_eq!((s.turn, s.current), (30, 0));
    assert!(s.is_terminal());
}

/// A capital capture on the final turn wins for the capturer even with the lower score, and the
/// cap is never evaluated.
#[test]
fn a_capital_capture_on_the_last_turn_beats_the_score() {
    for (current, capital) in [(1u8, p(0, 0)), (0, p(4, 4))] {
        let other = 1 - current;
        let mut s = StateBuilder::new(5)
            .city_with(0, p(0, 0), |c| c.is_capital = true)
            .city_with(1, p(4, 4), |c| c.is_capital = true)
            .unit(current, UnitType::Warrior, capital)
            .techs(other, &sim_core::data::ALL_TECHS)
            .turn(2)
            .current(current)
            .config(Config {
                turn_cap: 3,
                ..Config::default()
            })
            .explore_all()
            .build();
        // The other player is far ahead on score (all nine techs).
        assert!(exp_score(&s, other) > exp_score(&s, current) + 1000);
        ok(&mut s, Action::Capture { unit: UnitId(0) });
        assert_eq!(s.outcome, Some(Outcome::Win(current)), "player {current}");
        assert_eq!((s.turn, s.current), (2, current));
        assert_eq!(
            s.cities.iter().find(|c| c.pos == capital).unwrap().id,
            CityId(u32::from(other))
        );
        assert_final(&s);
    }
}

#[test]
#[should_panic]
fn new_game_rejects_a_zero_turn_cap() {
    game(0, 0);
}
