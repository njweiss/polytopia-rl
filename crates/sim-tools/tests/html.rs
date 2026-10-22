//! Milestone 7: `frames_json` and the HTML replay page (docs/MILESTONE_7 "HTML page", "Tests"
//! item 7). Frame values are compared with independently re-simulated states; scores come from
//! `sim_core::score::score` (milestone 6, not under test here). Only the keys the brief names are
//! checked; tile keys and the encoding of `kind`/`phase` are left to the implementer.

mod common;

use std::collections::BTreeMap;

use common::*;
use serde_json::Value;
use sim_core::map::MapParams;
use sim_core::rng::Rng;
use sim_core::score::score;
use sim_core::state::{Config, Outcome, State};
use sim_tools::html::{frames_json, html_page};
use sim_tools::policy::random_action;
use sim_tools::render::action_label;
use sim_tools::replay::{record_game, ReplayError, ReplayLog};

fn record(seed: u64, config: &Config, policy_seed: u64) -> ReplayLog {
    let mut rng = Rng::new(policy_seed);
    record_game(
        seed,
        &MapParams::default(),
        config,
        5000,
        &mut |s: &State| my_random_action(s, &mut rng),
    )
}

fn int(v: &Value, key: &str) -> u64 {
    v.get(key)
        .and_then(Value::as_u64)
        .unwrap_or_else(|| panic!("missing integer {key:?} in {v}"))
}

fn boolean(v: &Value, key: &str) -> bool {
    v.get(key)
        .and_then(Value::as_bool)
        .unwrap_or_else(|| panic!("missing bool {key:?} in {v}"))
}

fn check_frame(frame: &Value, s: &State, what: &str) {
    assert_eq!(int(frame, "turn"), s.turn as u64, "{what}: turn");
    assert_eq!(int(frame, "current"), s.current as u64, "{what}: current");
    let outcome = match s.outcome {
        None => Value::Null,
        Some(Outcome::Win(0)) => Value::from("p0"),
        Some(Outcome::Win(_)) => Value::from("p1"),
        Some(Outcome::Draw) => Value::from("draw"),
    };
    assert_eq!(frame.get("outcome"), Some(&outcome), "{what}: outcome");

    let players = frame["players"].as_array().expect("players array");
    assert_eq!(players.len(), 2, "{what}");
    for (i, pl) in players.iter().enumerate() {
        let id = i as u8;
        assert_eq!(
            int(pl, "stars"),
            s.players[i].stars as u64,
            "{what}: P{i} stars"
        );
        assert_eq!(int(pl, "score"), score(s, id) as u64, "{what}: P{i} score");
        let cities = s.cities.iter().filter(|c| c.owner == id).count() as u64;
        let units = s.units.iter().filter(|u| u.owner == id).count() as u64;
        assert_eq!(int(pl, "cities"), cities, "{what}: P{i} cities");
        assert_eq!(int(pl, "units"), units, "{what}: P{i} units");
    }

    let tiles = frame["tiles"].as_array().expect("tiles array");
    assert_eq!(tiles.len(), s.tiles.len(), "{what}: tiles");
    assert!(
        tiles.iter().all(Value::is_object),
        "{what}: tiles are objects"
    );

    let units = frame["units"].as_array().expect("units array");
    assert_eq!(units.len(), s.units.len(), "{what}: unit count");
    let by_id: BTreeMap<u64, &Value> = units.iter().map(|u| (int(u, "id"), u)).collect();
    assert_eq!(by_id.len(), units.len(), "{what}: unit ids unique");
    for u in &s.units {
        let f = by_id
            .get(&(u.id.0 as u64))
            .unwrap_or_else(|| panic!("{what}: unit {} missing", u.id.0));
        assert_eq!(int(f, "owner"), u.owner as u64, "{what}: unit {}", u.id.0);
        assert_eq!(int(f, "row"), u.pos.row as u64, "{what}: unit {}", u.id.0);
        assert_eq!(int(f, "col"), u.pos.col as u64, "{what}: unit {}", u.id.0);
        assert_eq!(int(f, "health"), u.health as u64, "{what}: unit {}", u.id.0);
        assert_eq!(
            int(f, "max_health"),
            u.max_health as u64,
            "{what}: unit {}",
            u.id.0
        );
        assert_eq!(boolean(f, "veteran"), u.veteran, "{what}: unit {}", u.id.0);
        assert!(
            f.get("kind").is_some() && f.get("phase").is_some(),
            "{what}"
        );
    }

    let cities = frame["cities"].as_array().expect("cities array");
    assert_eq!(cities.len(), s.cities.len(), "{what}: city count");
    let by_id: BTreeMap<u64, &Value> = cities.iter().map(|c| (int(c, "id"), c)).collect();
    for c in &s.cities {
        let f = by_id
            .get(&(c.id.0 as u64))
            .unwrap_or_else(|| panic!("{what}: city {} missing", c.id.0));
        assert_eq!(int(f, "owner"), c.owner as u64, "{what}: city {}", c.id.0);
        assert_eq!(int(f, "row"), c.pos.row as u64, "{what}: city {}", c.id.0);
        assert_eq!(int(f, "col"), c.pos.col as u64, "{what}: city {}", c.id.0);
        assert_eq!(int(f, "level"), c.level as u64, "{what}: city {}", c.id.0);
        assert_eq!(
            boolean(f, "capital"),
            c.is_capital,
            "{what}: city {}",
            c.id.0
        );
    }
}

#[test]
fn frames_json_has_one_frame_per_state_and_one_label_per_action() {
    for seed in 0..4u64 {
        let log = record(seed, &cfg(3, seed % 2 == 0), seed + 9);
        let states = resimulate(&log);
        let n = log.actions.len();
        let v: Value = serde_json::from_str(&frames_json(&log).unwrap()).expect("valid JSON");
        assert_eq!(int(&v, "size"), 11);
        let labels = v["labels"].as_array().expect("labels");
        assert_eq!(labels.len(), n, "seed {seed}");
        for (i, l) in labels.iter().enumerate() {
            assert_eq!(l.as_str(), Some(action_label(&log.actions[i]).as_str()));
        }
        let frames = v["frames"].as_array().expect("frames");
        assert_eq!(frames.len(), n + 1, "seed {seed}");
        check_frame(&frames[0], &states[0], &format!("seed {seed} first"));
        check_frame(&frames[n], &states[n], &format!("seed {seed} last"));
        for k in [1, n / 2] {
            check_frame(&frames[k], &states[k], &format!("seed {seed} frame {k}"));
        }
    }
}

#[test]
fn frames_of_a_finished_game_show_the_outcome() {
    // Seeds chosen to cover a non-null outcome; every finished game has one.
    let log = record(2, &cfg(2, false), 1);
    assert!(log.outcome.is_some());
    let v: Value = serde_json::from_str(&frames_json(&log).unwrap()).unwrap();
    let frames = v["frames"].as_array().unwrap();
    assert!(frames[0]["outcome"].is_null());
    assert!(!frames.last().unwrap()["outcome"].is_null());
}

#[test]
fn frames_json_verifies_the_log() {
    let mut log = record(1, &cfg(2, false), 4);
    log.digests[1] = "0000000000000000".into();
    assert_eq!(
        frames_json(&log),
        Err(ReplayError::DigestMismatch { index: 1 })
    );
    assert_eq!(
        html_page(&log),
        Err(ReplayError::DigestMismatch { index: 1 })
    );
}

#[test]
fn page_is_self_contained_and_embeds_the_frames() {
    let log = record(6, &cfg(3, false), 6);
    let page = html_page(&log).unwrap();
    let frames = frames_json(&log).unwrap();
    assert!(
        page.contains(&format!("const REPLAY = {frames};")),
        "the page must embed `const REPLAY = <frames_json>;`"
    );
    assert!(page.contains("<input type=\"range\""), "slider");
    assert!(!page.contains("http://") && !page.contains("https://"));
    assert!(
        page.contains("<script") && page.contains("<style"),
        "inline JS and CSS"
    );
    assert!(
        !page.contains("<script src") && !page.contains("<link"),
        "no external files"
    );
    // Deterministic.
    assert_eq!(html_page(&log).unwrap(), page);
}

#[test]
fn page_for_a_50_turn_game_stays_under_5_mb() {
    let mut rng = Rng::new(1 ^ POLICY_SEED_XOR);
    let log = record_game(
        1,
        &MapParams::default(),
        &Config::default(),
        5000,
        &mut |s: &State| random_action(s, &mut rng),
    );
    let page = html_page(&log).unwrap();
    assert!(
        page.len() < 5 * 1024 * 1024,
        "{} bytes for {} actions",
        page.len(),
        log.actions.len()
    );
}
