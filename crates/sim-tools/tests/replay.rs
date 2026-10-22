//! Milestone 7: replay log (docs/MILESTONE_7 "Replay log", G2, "Tests" item 4). Expected logs and
//! states come from `common`: an independent re-simulation with plain `sim_core` calls, the
//! policy replica and the reference digest encoder.

mod common;

use common::*;
use sim_core::actions::{apply, legal_actions, Action};
use sim_core::grid::Pos;
use sim_core::map::MapParams;
use sim_core::rng::Rng;
use sim_core::setup::new_game_from_seed;
use sim_core::state::{Config, Outcome, State, UnitId};
use sim_tools::replay::{record_game, replay, replay_each, ReplayError, ReplayLog};

/// Records with the policy replica, so the log can be compared with `play_independent`.
fn record(seed: u64, config: &Config, max_actions: usize, policy_seed: u64) -> ReplayLog {
    let mut rng = Rng::new(policy_seed);
    record_game(
        seed,
        &MapParams::default(),
        config,
        max_actions,
        &mut |s: &State| my_random_action(s, &mut rng),
    )
}

/// A finished game used by the tamper tests.
fn finished_log() -> ReplayLog {
    let log = record(5, &cfg(3, false), 5000, 77);
    assert!(log.outcome.is_some());
    log
}

#[test]
fn record_game_matches_an_independent_game_for_many_seeds() {
    for seed in 0..16u64 {
        let config = cfg(2 + (seed as u32 % 4), seed % 2 == 0);
        let policy_seed = seed.wrapping_mul(31) + 7;
        let log = record(seed, &config, 5000, policy_seed);
        let (states, actions) =
            play_independent(seed, &MapParams::default(), &config, 5000, policy_seed);
        assert!(states.last().unwrap().is_terminal(), "seed {seed}");
        let expected = expected_log(seed, &MapParams::default(), &config, &states, &actions);
        assert_eq!(log, expected, "seed {seed}");
        for d in &log.digests {
            assert_eq!(d.len(), 16, "{d}");
            assert!(
                d.chars()
                    .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)),
                "{d}"
            );
        }
    }
}

#[test]
fn record_game_stops_at_max_actions() {
    let config = cfg(50, false);
    let log = record(3, &config, 10, 1);
    let (states, actions) = play_independent(3, &MapParams::default(), &config, 10, 1);
    assert_eq!(log.actions.len(), 10);
    assert_eq!(log.digests.len(), 11);
    assert_eq!(log.outcome, None);
    assert_eq!(
        log,
        expected_log(3, &MapParams::default(), &config, &states, &actions)
    );

    let empty = record(3, &config, 0, 1);
    assert!(empty.actions.is_empty());
    assert_eq!(
        empty.digests,
        vec![my_digest_hex(&new_game_from_seed(
            3,
            &MapParams::default(),
            &config
        ))]
    );
    assert_eq!(empty.outcome, None);
    assert_eq!(empty.version, 1);
}

#[test]
fn record_game_uses_the_given_map_params() {
    let params = MapParams {
        size: 9,
        forest_prob: 0.2,
        ..MapParams::default()
    };
    let config = cfg(2, false);
    let mut rng = Rng::new(4);
    let log = record_game(11, &params, &config, 5000, &mut |s: &State| {
        my_random_action(s, &mut rng)
    });
    let (states, actions) = play_independent(11, &params, &config, 5000, 4);
    assert_eq!(log, expected_log(11, &params, &config, &states, &actions));
    assert_eq!(replay(&log).unwrap(), *states.last().unwrap());
}

#[test]
fn record_game_panics_on_an_illegal_policy_action() {
    // A legal policy works (so a stub that always panics fails this test).
    let ok = record_game(
        1,
        &MapParams::default(),
        &cfg(5, false),
        3,
        &mut |s: &State| legal_actions(s)[0],
    );
    assert_eq!(ok.actions.len(), 3);
    let r = std::panic::catch_unwind(|| {
        record_game(1, &MapParams::default(), &cfg(5, false), 100, &mut |_| {
            Action::Move {
                unit: UnitId(999),
                to: Pos::new(0, 0),
            }
        })
    });
    assert!(r.is_err(), "an illegal policy action must panic");
}

#[test]
fn json_round_trip_and_byte_determinism() {
    for seed in 0..6u64 {
        let log = record(seed, &cfg(3, seed % 2 == 1), 5000, seed + 50);
        let json = log.to_json();
        assert_eq!(ReplayLog::from_json(&json), Ok(log.clone()));
        assert_eq!(log.clone().to_json(), json, "same log, same bytes");
        // Recorded again from scratch: same bytes.
        assert_eq!(
            record(seed, &cfg(3, seed % 2 == 1), 5000, seed + 50).to_json(),
            json
        );
        // Compact (no whitespace outside strings; none of the strings contain spaces) and
        // fields in declaration order.
        assert!(!json.contains(' ') && !json.contains('\n'), "not compact");
        let keys = [
            "\"version\"",
            "\"seed\"",
            "\"map_params\"",
            "\"config\"",
            "\"actions\"",
            "\"digests\"",
            "\"outcome\"",
        ];
        let at: Vec<usize> = keys.iter().map(|k| json.find(k).expect(k)).collect();
        assert!(at.windows(2).all(|w| w[0] < w[1]), "field order: {at:?}");
        // The JSON is the derived serde representation of the log.
        let derived: ReplayLog = serde_json::from_str(&json).expect("derived representation");
        assert_eq!(derived, log);
    }
}

#[test]
fn replay_returns_the_final_state_of_the_game() {
    for seed in 20..30u64 {
        let config = cfg(2 + (seed as u32 % 3), seed % 3 == 0);
        let mut rng = Rng::new(seed);
        let mut last: Option<State> = None;
        let log = record_game(
            seed,
            &MapParams::default(),
            &config,
            5000,
            &mut |s: &State| {
                let a = my_random_action(s, &mut rng);
                let mut t = s.clone();
                apply(&mut t, &a).unwrap();
                last = Some(t);
                a
            },
        );
        let (states, _) = play_independent(seed, &MapParams::default(), &config, 5000, seed);
        let fin = replay(&log).expect("valid log");
        assert_eq!(fin, *states.last().unwrap(), "seed {seed}");
        assert_eq!(
            Some(fin),
            last,
            "seed {seed}: the state record_game ended with"
        );
    }
}

#[test]
fn replay_each_visits_every_state_in_order() {
    for seed in 40..46u64 {
        let log = record(seed, &cfg(3, seed % 2 == 0), 5000, seed * 3);
        let expected = resimulate(&log);
        let mut seen: Vec<(usize, State)> = Vec::new();
        replay_each(&log, &mut |i, s| seen.push((i, s.clone()))).expect("valid log");
        assert_eq!(seen.len(), log.actions.len() + 1);
        for (k, (i, s)) in seen.iter().enumerate() {
            assert_eq!(*i, k);
            assert_eq!(*s, expected[k], "seed {seed} step {k}");
        }
    }
}

#[test]
fn unfinished_logs_replay_too() {
    let log = record(8, &cfg(50, false), 37, 2);
    assert_eq!(log.outcome, None);
    let expected = resimulate(&log);
    assert_eq!(replay(&log), Ok(expected.last().unwrap().clone()));
}

// --- tampering -----------------------------------------------------------------------------------

/// First index `k` and a replacement action that is legal in state `k` but differs from
/// `actions[k]` and leads to a different state.
fn legal_substitute(log: &ReplayLog) -> (usize, Action) {
    let states = resimulate(log);
    for (k, a) in log.actions.iter().enumerate() {
        for b in legal_actions(&states[k]) {
            if b != *a {
                let mut t = states[k].clone();
                apply(&mut t, &b).unwrap();
                if my_digest(&t) != my_digest(&states[k + 1]) {
                    return (k, b);
                }
            }
        }
    }
    panic!("no substitute found");
}

#[test]
fn edited_action_that_becomes_illegal() {
    let log = finished_log();
    for k in [0, 1, log.actions.len() / 2, log.actions.len() - 1] {
        let mut t = log.clone();
        t.actions[k] = Action::Move {
            unit: UnitId(999),
            to: Pos::new(0, 0),
        };
        assert_eq!(
            replay(&t),
            Err(ReplayError::IllegalAction { index: k }),
            "k {k}"
        );
    }
}

#[test]
fn edited_action_that_stays_legal() {
    let log = finished_log();
    let (k, b) = legal_substitute(&log);
    let mut t = log.clone();
    t.actions[k] = b;
    assert_eq!(
        replay(&t),
        Err(ReplayError::DigestMismatch { index: k + 1 })
    );
}

#[test]
fn changed_seed_fails_on_the_initial_digest() {
    let log = finished_log();
    let mut t = log.clone();
    t.seed += 1;
    assert_ne!(
        my_digest(&new_game_from_seed(t.seed, &t.map_params, &t.config)),
        my_digest(&new_game_from_seed(log.seed, &log.map_params, &log.config))
    );
    assert_eq!(replay(&t), Err(ReplayError::DigestMismatch { index: 0 }));
}

#[test]
fn changed_config_or_map_params_fail_on_the_initial_digest() {
    let log = finished_log();
    let mut t = log.clone();
    t.config.turn_cap += 1;
    assert_eq!(replay(&t), Err(ReplayError::DigestMismatch { index: 0 }));
    let mut t = log.clone();
    t.map_params.forest_prob = 0.9;
    assert_eq!(replay(&t), Err(ReplayError::DigestMismatch { index: 0 }));
}

#[test]
fn altered_digest_strings() {
    let log = finished_log();
    let n = log.actions.len();
    for j in [0, 1, n / 2, n] {
        let mut t = log.clone();
        t.digests[j] = "0123456789abcdef".to_string();
        assert_ne!(log.digests[j], t.digests[j]);
        assert_eq!(
            replay(&t),
            Err(ReplayError::DigestMismatch { index: j }),
            "j {j}"
        );
    }
}

#[test]
fn digest_count_must_be_actions_plus_one() {
    let log = finished_log();
    let mut t = log.clone();
    t.digests.pop();
    assert_eq!(replay(&t), Err(ReplayError::LengthMismatch));
    let mut t = log.clone();
    t.digests.push(log.digests[0].clone());
    assert_eq!(replay(&t), Err(ReplayError::LengthMismatch));
    let mut t = log.clone();
    t.digests.clear();
    assert_eq!(replay(&t), Err(ReplayError::LengthMismatch));
}

#[test]
fn version_must_be_1() {
    let mut t = finished_log();
    t.version = 2;
    assert_eq!(replay(&t), Err(ReplayError::BadVersion(2)));
    t.version = 0;
    assert_eq!(replay(&t), Err(ReplayError::BadVersion(0)));
}

#[test]
fn altered_outcome() {
    let log = finished_log();
    for o in [
        None,
        Some(Outcome::Draw),
        Some(Outcome::Win(0)),
        Some(Outcome::Win(1)),
    ] {
        if o == log.outcome {
            continue;
        }
        let mut t = log.clone();
        t.outcome = o;
        assert_eq!(replay(&t), Err(ReplayError::OutcomeMismatch), "{o:?}");
    }
    // A finished game cut short: the prefix replays, but its final state has no outcome.
    let mut t = log.clone();
    t.actions.pop();
    t.digests.pop();
    assert_eq!(replay(&t), Err(ReplayError::OutcomeMismatch));
    t.outcome = None;
    assert!(replay(&t).is_ok());
}

#[test]
fn action_after_the_end_is_illegal() {
    let log = finished_log();
    let n = log.actions.len();
    let mut t = log.clone();
    t.actions.push(Action::EndTurn);
    t.digests.push(log.digests[n].clone());
    assert_eq!(replay(&t), Err(ReplayError::IllegalAction { index: n }));
}

#[test]
fn error_priority() {
    let log = finished_log();
    let n = log.actions.len();
    let illegal = Action::Move {
        unit: UnitId(999),
        to: Pos::new(0, 0),
    };

    // BadVersion before LengthMismatch.
    let mut t = log.clone();
    t.version = 2;
    t.digests.pop();
    assert_eq!(replay(&t), Err(ReplayError::BadVersion(2)));

    // LengthMismatch before the initial digest and any action.
    let mut t = log.clone();
    t.digests.pop();
    t.digests[0] = "0000000000000000".into();
    t.actions[0] = illegal;
    assert_eq!(replay(&t), Err(ReplayError::LengthMismatch));

    // Initial digest before an illegal first action.
    let mut t = log.clone();
    t.digests[0] = "0000000000000000".into();
    t.actions[0] = illegal;
    assert_eq!(replay(&t), Err(ReplayError::DigestMismatch { index: 0 }));

    // Per action: IllegalAction{k} before DigestMismatch{k+1}.
    let k = n / 2;
    let mut t = log.clone();
    t.actions[k] = illegal;
    t.digests[k + 1] = "0000000000000000".into();
    assert_eq!(replay(&t), Err(ReplayError::IllegalAction { index: k }));

    // Earlier digest mismatch before a later illegal action.
    let mut t = log.clone();
    t.digests[2] = "0000000000000000".into();
    t.actions[k] = illegal;
    assert_eq!(replay(&t), Err(ReplayError::DigestMismatch { index: 2 }));

    // Earlier illegal action before a later digest mismatch.
    let mut t = log.clone();
    t.actions[1] = illegal;
    t.digests[k] = "0000000000000000".into();
    assert_eq!(replay(&t), Err(ReplayError::IllegalAction { index: 1 }));

    // Everything before OutcomeMismatch.
    let mut t = log.clone();
    t.outcome = None;
    t.digests[n] = "0000000000000000".into();
    assert_eq!(replay(&t), Err(ReplayError::DigestMismatch { index: n }));
    let mut t = log.clone();
    t.outcome = None;
    t.actions[n - 1] = illegal;
    assert_eq!(replay(&t), Err(ReplayError::IllegalAction { index: n - 1 }));
}

#[test]
fn replay_each_reports_the_same_errors_and_stops_there() {
    let log = finished_log();
    let k = log.actions.len() / 2;
    let mut t = log.clone();
    t.digests[k] = "0000000000000000".into();
    let mut visited = Vec::new();
    let r = replay_each(&t, &mut |i, _| visited.push(i));
    assert_eq!(r, Err(ReplayError::DigestMismatch { index: k }));
    assert_eq!(replay(&t), Err(ReplayError::DigestMismatch { index: k }));
    // The states before the mismatch were visited in order, and nothing after it (whether the
    // mismatching state itself is visited before its check is left open).
    assert!(
        visited == (0..k).collect::<Vec<_>>() || visited == (0..=k).collect::<Vec<_>>(),
        "{visited:?}"
    );

    let mut t = log.clone();
    t.version = 3;
    assert_eq!(
        replay_each(&t, &mut |_, _| {}),
        Err(ReplayError::BadVersion(3))
    );

    let mut t = log.clone();
    t.outcome = None;
    assert_eq!(
        replay_each(&t, &mut |_, _| {}),
        Err(ReplayError::OutcomeMismatch)
    );
}

#[test]
fn garbage_json_is_a_json_error() {
    for s in [
        "",
        "not json",
        "{",
        "[]",
        "{\"version\":1}",
        "{\"version\":\"one\",\"seed\":1}",
    ] {
        assert!(
            matches!(ReplayLog::from_json(s), Err(ReplayError::Json(_))),
            "{s:?}"
        );
    }
    // A valid log with an unknown action variant.
    let json = finished_log()
        .to_json()
        .replacen("\"EndTurn\"", "\"Pass\"", 1);
    assert!(matches!(
        ReplayLog::from_json(&json),
        Err(ReplayError::Json(_))
    ));
}
