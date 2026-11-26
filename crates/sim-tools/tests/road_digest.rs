//! Milestone 16 "Tests" item 6 (docs/MILESTONE_16 "Digest"): under v2 each tile record gets a
//! trailing `road` byte and each city record a trailing `connected_to` option (0, or 1 then the
//! city id as `u32` little-endian, the digest's encoding of options); v1 digests are unchanged and
//! never see the new fields.
//!
//! `expected_v2_bytes` builds the v2 stream from the milestone 7 reference encoder
//! (`common::digest_bytes`, the v1 stream) by inserting the ruleset byte after the `Config` (offset
//! 55, docs/MILESTONE_15) and the new fields at the end of each tile and city record. Record sizes
//! come from docs/MILESTONE_7 "State digest": a tile is terrain 1, resource and improvement
//! `Option<u8>` (1 or 2), village 1, city, owner city and unit `Option<u32>` (1 or 5); a unit is
//! 4+1+1+2+4+4, origin (1 or 5), 4+1+1+1, last direction (1 or 3); a city is 24 bytes.

mod common;

use common::*;
use sim_core::actions::{apply, legal_actions, Action};
use sim_core::data::{Tech, UnitType};
use sim_core::map::{MapParams, Terrain};
use sim_core::rng::Rng;
use sim_core::setup::new_game_from_seed;
use sim_core::state::{CityId, Config, Ruleset, State};
use sim_core::testing::StateBuilder;
use sim_tools::digest::state_digest;

const CONFIG_END: usize = 55;

fn v2_config() -> Config {
    Config {
        ruleset: Ruleset::V2,
        ..Config::default()
    }
}

fn opt_len<T>(v: &Option<T>, payload: usize) -> usize {
    if v.is_some() {
        1 + payload
    } else {
        1
    }
}

/// The brief's v2 byte stream of `s`.
fn expected_v2_bytes(s: &State) -> Vec<u8> {
    let mut v1 = s.clone();
    v1.config.ruleset = Ruleset::V1;
    let b = digest_bytes(&v1);
    let mut out = b[..CONFIG_END].to_vec();
    out.push(1);
    let mut i = CONFIG_END;
    let copy = |out: &mut Vec<u8>, i: &mut usize, n: usize| {
        out.extend_from_slice(&b[*i..*i + n]);
        *i += n;
    };
    copy(&mut out, &mut i, 4);
    for t in &s.tiles {
        let n = 1
            + opt_len(&t.resource, 1)
            + opt_len(&t.improvement, 1)
            + 1
            + opt_len(&t.city, 4)
            + opt_len(&t.owner_city, 4)
            + opt_len(&t.unit, 4);
        copy(&mut out, &mut i, n);
        out.push(t.road as u8);
        match t.building {
            None => out.push(0),
            Some(b) => {
                out.push(1);
                out.push(b as u8);
            }
        }
        out.push(t.building_level);
    }
    let units: usize = s
        .units
        .iter()
        .map(|u| 16 + opt_len(&u.origin, 4) + 7 + if u.last_dir.is_some() { 3 } else { 1 })
        .sum();
    copy(&mut out, &mut i, 4 + units);
    copy(&mut out, &mut i, 4);
    for c in &s.cities {
        // Sanity on the walk: each city record starts with its id.
        assert_eq!(b[i..i + 4], c.id.0.to_le_bytes(), "record offsets");
        copy(&mut out, &mut i, 24);
        match c.connected_to {
            None => out.push(0),
            Some(k) => {
                out.push(1);
                out.extend_from_slice(&k.0.to_le_bytes());
            }
        }
    }
    let rest = b.len() - i;
    copy(&mut out, &mut i, rest);
    out
}

fn expected_v2_digest(s: &State) -> u64 {
    fnv1a64(&expected_v2_bytes(s))
}

/// A 6x6 board with options of every kind set and unset: resources, an improvement, cities,
/// territory, units with and without origin and direction, roads, and city 2 connected.
fn scripted(config: Config) -> State {
    let mut s = StateBuilder::new(6)
        .terrain(p(0, 1), Terrain::Forest)
        .city_with(0, p(1, 1), |c| c.is_capital = true)
        .city_with(1, p(4, 4), |c| c.is_capital = true)
        .city_with(0, p(1, 3), |c| {
            c.connected_to = Some(CityId(0));
            c.population = 1;
        })
        .village(p(3, 0))
        .unit(0, UnitType::Warrior, p(1, 1))
        .unit_with(1, UnitType::Rider, p(2, 3), |u| {
            u.origin = Some(CityId(1));
            u.last_dir = Some(sim_core::state::Dir { dr: 1, dc: -1 });
        })
        .techs(0, &[Tech::Riding, Tech::Roads])
        .stars(0, 7)
        .turn(2)
        .rng(Rng::new(16))
        .config(config)
        .build();
    for q in [p(1, 2), p(3, 0), p(5, 5)] {
        s.tiles[q.row as usize * 6 + q.col as usize].road = true;
    }
    s
}

/// Under v1 the new fields never reach the digest; under v2 they do.
#[test]
fn the_new_fields_are_digested_under_v2_only() {
    let mut s = scripted(Config::default());
    // A v1 state cannot hold these, but if it did the digest would still ignore them.
    assert_eq!(state_digest(&s), my_digest(&s));
    let with = state_digest(&s);
    for t in &mut s.tiles {
        t.road = false;
    }
    for c in &mut s.cities {
        c.connected_to = None;
    }
    assert_eq!(state_digest(&s), with);
    assert_eq!(state_digest(&s), my_digest(&s));
    for seed in 0..4 {
        let g = new_game_from_seed(seed, &MapParams::default(), &Config::default());
        assert_eq!(state_digest(&g), my_digest(&g), "seed {seed}");
    }
    let v = scripted(v2_config());
    let mut w = v.clone();
    w.tiles[0].road = !w.tiles[0].road;
    assert_ne!(state_digest(&v), state_digest(&w));
}

#[test]
fn the_v2_digest_appends_road_and_connected_to() {
    let s = scripted(v2_config());
    assert_eq!(state_digest(&s), expected_v2_digest(&s));
    // Without the new fields set, the records still carry them (as 0 bytes).
    let mut t = s.clone();
    for x in &mut t.tiles {
        x.road = false;
    }
    for c in &mut t.cities {
        c.connected_to = None;
    }
    assert_eq!(state_digest(&t), expected_v2_digest(&t));
    for seed in 0..4 {
        let g = new_game_from_seed(seed, &MapParams::default(), &v2_config());
        assert_eq!(state_digest(&g), expected_v2_digest(&g), "seed {seed}");
    }
}

#[test]
fn a_road_or_a_connection_changes_the_v2_digest() {
    let s = scripted(v2_config());
    let base = state_digest(&s);
    let mut seen = vec![base];
    let mut variants: Vec<State> = Vec::new();
    for i in [0usize, 9, 35] {
        let mut t = s.clone();
        t.tiles[i].road = !t.tiles[i].road;
        variants.push(t);
    }
    let mut t = s.clone();
    t.cities[2].connected_to = None;
    variants.push(t);
    let mut t = s.clone();
    t.cities[2].connected_to = Some(CityId(1));
    variants.push(t);
    let mut t = s.clone();
    t.cities[1].connected_to = Some(CityId(0));
    variants.push(t);
    for (k, v) in variants.iter().enumerate() {
        let d = state_digest(v);
        assert_eq!(d, expected_v2_digest(v), "variant {k}");
        assert!(!seen.contains(&d), "variant {k} collides");
        seen.push(d);
    }
}

/// Random v2 games with roads: every state's digest is the brief's.
#[test]
fn v2_playout_digests_follow_the_brief() {
    let mut roads = 0;
    for seed in 0..4u64 {
        let mut s = new_game_from_seed(
            seed,
            &MapParams::default(),
            &Config {
                turn_cap: 15,
                ..v2_config()
            },
        );
        for pl in &mut s.players {
            pl.techs.insert(Tech::Riding);
            pl.techs.insert(Tech::Roads);
            pl.stars += 30;
        }
        let mut rng = Rng::new(seed ^ 0x16_d1);
        let mut steps = 0;
        while !s.is_terminal() && steps < 1500 {
            assert_eq!(state_digest(&s), expected_v2_digest(&s), "seed {seed}");
            let legal = legal_actions(&s);
            let built: Vec<Action> = legal
                .iter()
                .copied()
                .filter(|a| matches!(a, Action::BuildRoad { .. }))
                .collect();
            let a = if !built.is_empty() && rng.gen_range(2) == 0 {
                roads += 1;
                built[rng.gen_range(built.len() as u32) as usize]
            } else {
                legal[rng.gen_range(legal.len() as u32) as usize]
            };
            apply(&mut s, &a).unwrap();
            steps += 1;
        }
    }
    assert!(roads > 50, "only {roads} roads");
}
