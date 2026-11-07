//! Milestone 15 "Tests" item 3 (digest and replay side of the ruleset flag, docs/MILESTONE_15
//! "Ruleset flag"): the digest appends `ruleset` as one byte (`V2` = 1) after the `Config` bytes
//! only under v2, so the v2 digest of a state is the FNV-1a of the v1 byte stream
//! (`common::digest_bytes`, the milestone 7 reference encoder) with that byte inserted. The
//! `Config` bytes end at offset 55: size 1 + turn 4 + current 1 + two rng outputs 8 +
//! full_observability 1 + turn_cap 4 + nine score weights 36.
//!
//! v2 techs in the digest: docs/QUESTIONS.md Q42 (the `TechSet` is its whole `u16`, bits 9 to 15
//! included).

mod common;

use std::sync::Arc;

use common::*;
use sim_core::actions::Action;
use sim_core::data::{Tables, Tech, UnitType};
use sim_core::map::{MapParams, Terrain};
use sim_core::rng::Rng;
use sim_core::setup::new_game_from_seed;
use sim_core::state::{Config, Ruleset, State};
use sim_core::testing::StateBuilder;
use sim_tools::digest::state_digest;
use sim_tools::replay::{record_game, replay, ReplayError, ReplayLog};

const CONFIG_END: usize = 55;

const NEW_TECHS: [Tech; 7] = [
    Tech::Roads,
    Tech::FreeSpirit,
    Tech::Trade,
    Tech::Chivalry,
    Tech::Mathematics,
    Tech::Construction,
    Tech::Smithery,
];

fn v2_config(turn_cap: u32) -> Config {
    Config {
        turn_cap,
        ruleset: Ruleset::V2,
        ..Config::default()
    }
}

/// The brief's v2 digest: the v1 stream with the ruleset byte inserted after the `Config`.
fn expected_v2_digest(s: &State) -> u64 {
    let mut v1 = s.clone();
    v1.config.ruleset = Ruleset::V1;
    let mut bytes = digest_bytes(&v1);
    // Sanity on the offset: the bytes just before it are `tech_per_tier`, little-endian.
    assert_eq!(
        bytes[CONFIG_END - 4..CONFIG_END],
        s.config.score_weights.tech_per_tier.to_le_bytes()
    );
    bytes.insert(CONFIG_END, 1);
    fnv1a64(&bytes)
}

fn scripted() -> Vec<State> {
    let a = StateBuilder::new(5)
        .terrain(p(0, 1), Terrain::Forest)
        .city(0, p(1, 1))
        .city(1, p(3, 3))
        .unit(0, UnitType::Warrior, p(1, 1))
        .unit(1, UnitType::Rider, p(2, 3))
        .techs(0, &[Tech::Organization, Tech::Riding])
        .stars(1, 9)
        .turn(3)
        .current(1)
        .rng(Rng::new(7))
        .build();
    let mut b = a.clone();
    b.config.score_weights.tech_per_tier = 77;
    b.config.full_observability = true;
    let mut out = vec![a, b, StateBuilder::new(3).build()];
    for seed in 0..4 {
        out.push(new_game_from_seed(
            seed,
            &MapParams::default(),
            &Config::default(),
        ));
    }
    out
}

#[test]
fn the_v1_digest_has_no_ruleset_byte() {
    for s in scripted() {
        assert_eq!(s.config.ruleset, Ruleset::V1);
        assert_eq!(state_digest(&s), my_digest(&s));
    }
}

#[test]
fn the_v2_digest_is_the_v1_stream_plus_one_byte_after_the_config() {
    for s in scripted() {
        let mut v2 = s.clone();
        v2.config.ruleset = Ruleset::V2;
        assert_eq!(state_digest(&v2), expected_v2_digest(&v2));
        assert_ne!(state_digest(&v2), state_digest(&s));
        // The tables play no part (as in v1).
        let mut t = v2.clone();
        t.tables = Arc::new(Tables::load_for(Ruleset::V2).unwrap());
        assert_eq!(state_digest(&t), state_digest(&v2));
    }
    // A real v2 game start too.
    for seed in 0..4 {
        let s = new_game_from_seed(seed, &MapParams::default(), &v2_config(30));
        assert_eq!(state_digest(&s), expected_v2_digest(&s), "seed {seed}");
    }
}

#[test]
fn v2_techs_are_part_of_the_digest() {
    // Q42: each v2 tech changes the digest, and no two single-tech sets collide.
    let base = StateBuilder::new(4)
        .config(v2_config(30))
        .techs(0, &[Tech::Organization])
        .build();
    let mut seen = vec![state_digest(&base)];
    for t in NEW_TECHS {
        for player in 0..2 {
            let mut s = base.clone();
            s.players[player].techs.insert(t);
            let d = state_digest(&s);
            assert!(!seen.contains(&d), "{t:?} for player {player}");
            seen.push(d);
        }
    }
}

/// Random v2 games recorded with the milestone 7 random policy replica.
fn v2_logs() -> Vec<ReplayLog> {
    (0..6u64)
        .map(|seed| {
            let mut rng = Rng::new(seed ^ POLICY_SEED_XOR);
            record_game(
                seed,
                &MapParams::default(),
                &v2_config(12),
                3000,
                &mut |s: &State| my_random_action(s, &mut rng),
            )
        })
        .collect()
}

#[test]
fn v2_replay_files_carry_the_ruleset_and_verify() {
    let mut v2_research = 0;
    for log in v2_logs() {
        let json = log.to_json();
        assert!(json.contains(r#""ruleset":"v2""#), "{}", &json[..300]);
        let back = ReplayLog::from_json(&json).unwrap();
        assert_eq!(back, log);
        assert_eq!(back.config.ruleset, Ruleset::V2);
        replay(&back).expect("a v2 log verifies");
        v2_research += log
            .actions
            .iter()
            .filter(|a| matches!(a, Action::Research { tech } if NEW_TECHS.contains(tech)))
            .count();
        // Every digest of the log is the brief's v2 digest of the re-simulated state.
        let states = resimulate(&log);
        for (i, s) in states.iter().enumerate() {
            if s.outcome.is_none() {
                assert_eq!(log.digests[i], format!("{:016x}", expected_v2_digest(s)));
            }
        }
        // The same log claimed as v1 fails at the initial digest.
        let mut as_v1 = log.clone();
        as_v1.config.ruleset = Ruleset::V1;
        assert!(!as_v1.to_json().contains("ruleset"));
        assert_eq!(
            replay(&as_v1).unwrap_err(),
            ReplayError::DigestMismatch { index: 0 }
        );
    }
    assert!(v2_research >= 3, "only {v2_research} v2 researches");
}
