//! Milestone 17 digest tests (item 8): under `V2`, each tile record appends `building`
//! (`Option<u8>`) and `building_level` (`u8`) after `road`, and negative `City.population` uses
//! `to_le_bytes()`. Under `V1`, `building`, `building_level` are ignored and digests are unchanged.

mod common;

use common::*;
use sim_core::data::Building;
use sim_core::map::MapParams;
use sim_core::rng::Rng;
use sim_core::setup::new_game_from_seed;
use sim_core::state::{Config, Ruleset, State};
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

/// Independent M17 v2 byte-stream encoder.
fn expected_m17_v2_digest(s: &State) -> u64 {
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
            Some(bld) => {
                out.push(1);
                out.push(bld as u8);
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
    fnv1a64(&out)
}

#[test]
fn v1_digest_ignores_building_fields_even_if_synthetically_set() {
    let mut s = new_game_from_seed(7, &MapParams::default(), &Config::default());
    let base = state_digest(&s);
    assert_eq!(base, my_digest(&s));
    s.tiles[0].building = Some(Building::Sawmill);
    s.tiles[0].building_level = 3;
    assert_eq!(state_digest(&s), base);
}

#[test]
fn v2_digest_distinguishes_building_building_level_and_negative_population() {
    let base = StateBuilder::new(5)
        .city_with(0, p(2, 2), |c| c.is_capital = true)
        .rng(Rng::new(17))
        .config(v2_config())
        .build();
    assert_eq!(state_digest(&base), expected_m17_v2_digest(&base));

    let mut variants = Vec::new();
    for b in [
        Building::Sawmill,
        Building::Windmill,
        Building::Forge,
        Building::Market,
    ] {
        let mut t = base.clone();
        t.tiles[0].building = Some(b);
        variants.push(t);
    }
    let mut lvl = base.clone();
    lvl.tiles[0].building = Some(Building::Forge);
    lvl.tiles[0].building_level = 3;
    variants.push(lvl);

    for pop in [-1, -5, 1] {
        let mut t = base.clone();
        t.cities[0].population = pop;
        variants.push(t);
    }

    let mut seen = vec![state_digest(&base)];
    for (idx, v) in variants.iter().enumerate() {
        let d = state_digest(v);
        assert_eq!(d, expected_m17_v2_digest(v), "variant {idx}");
        assert!(!seen.contains(&d), "variant {idx} collided");
        seen.push(d);
    }
}
