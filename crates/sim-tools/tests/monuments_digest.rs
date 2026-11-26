//! Milestone 18 digest tests (item 7): under `V2`, each tile record appends `lighthouse` (`bool`)
//! and `monument` (`Option<u8>`) after `building_level`, and each player record appends `kills`
//! (`u32`), `lighthouses_discovered` (`u8`), `monuments_unlocked` (`u8`), and `monuments_placed`
//! (`u8`) after `explored`. Under `V1`, none of these fields are written and digests are unchanged.

mod common;

use common::*;
use sim_core::data::{Monument, ALL_MONUMENTS};
use sim_core::map::MapParams;
use sim_core::rng::Rng;
use sim_core::setup::new_game_from_seed;
use sim_core::state::{Config, MonumentSet, Ruleset, State};
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

/// Independent M18 v2 byte-stream encoder.
fn expected_m18_v2_digest(s: &State) -> u64 {
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
        out.push(t.lighthouse as u8);
        match t.monument {
            None => out.push(0),
            Some(m) => {
                out.push(1);
                out.push(m as u8);
            }
        }
    }
    let units: usize = s
        .units
        .iter()
        .map(|u| 16 + opt_len(&u.origin, 4) + 7 + if u.last_dir.is_some() { 3 } else { 1 })
        .sum();
    copy(&mut out, &mut i, 4 + units);
    copy(&mut out, &mut i, 4);
    for c in &s.cities {
        assert_eq!(b[i..i + 4], c.id.0.to_le_bytes(), "city offsets");
        copy(&mut out, &mut i, 24);
        match c.connected_to {
            None => out.push(0),
            Some(k) => {
                out.push(1);
                out.extend_from_slice(&k.0.to_le_bytes());
            }
        }
    }
    for pl in &s.players {
        copy(&mut out, &mut i, 4 + 2 + 4 + pl.explored.len());
        out.extend_from_slice(&pl.kills.to_le_bytes());
        out.push(pl.lighthouses_discovered);
        out.push(pl.monuments_unlocked.bits());
        out.push(pl.monuments_placed.bits());
    }
    assert_eq!(b[i..i + 4], s.next_unit_id.to_le_bytes(), "player offsets");
    let rest = b.len() - i;
    copy(&mut out, &mut i, rest);
    fnv1a64(&out)
}

#[test]
fn v1_digest_ignores_m18_tile_and_player_fields_even_if_synthetically_set() {
    let mut s = new_game_from_seed(11, &MapParams::default(), &Config::default());
    let base = state_digest(&s);
    assert_eq!(base, my_digest(&s));

    s.tiles[0].lighthouse = true;
    s.tiles[1].monument = Some(Monument::ParkOfFortune);
    s.players[0].kills = 12;
    s.players[0].lighthouses_discovered = 0b1010;
    s.players[0].monuments_unlocked = MonumentSet::from_slice(&ALL_MONUMENTS);
    s.players[0].monuments_placed = MonumentSet::from_slice(&[Monument::GateOfPower]);
    s.players[1].kills = 3;
    s.players[1].lighthouses_discovered = 0b0101;
    s.players[1].monuments_unlocked = MonumentSet::from_slice(&[Monument::EyeOfGod]);
    s.players[1].monuments_placed = MonumentSet::from_slice(&[Monument::EyeOfGod]);

    assert_eq!(state_digest(&s), base);
    assert_eq!(state_digest(&s), my_digest(&s));
}

#[test]
fn v2_digest_distinguishes_all_m18_tile_and_player_fields() {
    let base = StateBuilder::new(5)
        .city_with(0, p(2, 2), |c| c.is_capital = true)
        .city_with(1, p(4, 4), |c| c.is_capital = true)
        .rng(Rng::new(18))
        .config(v2_config())
        .build();
    assert_eq!(state_digest(&base), expected_m18_v2_digest(&base));

    let mut variants = Vec::new();
    // Tile lighthouse.
    let mut lh = base.clone();
    lh.tiles[0].lighthouse = true;
    variants.push(lh);

    // Tile monument (each of the 3 variants).
    for m in ALL_MONUMENTS {
        let mut t = base.clone();
        t.tiles[1].monument = Some(m);
        variants.push(t);
    }

    // Player fields (for both player 0 and player 1).
    for pl in 0..2 {
        let mut k = base.clone();
        k.players[pl].kills = 1;
        variants.push(k);

        let mut d = base.clone();
        d.players[pl].lighthouses_discovered = 1 << pl;
        variants.push(d);

        let mut u = base.clone();
        u.players[pl]
            .monuments_unlocked
            .insert(Monument::ParkOfFortune);
        variants.push(u);

        let mut p_set = base.clone();
        p_set.players[pl]
            .monuments_placed
            .insert(Monument::GateOfPower);
        variants.push(p_set);
    }

    let mut seen = vec![state_digest(&base)];
    for (idx, v) in variants.iter().enumerate() {
        let d = state_digest(v);
        assert_eq!(d, expected_m18_v2_digest(v), "variant {idx}");
        assert!(!seen.contains(&d), "variant {idx} collided");
        seen.push(d);
    }

    // Real v2 new_game starts match too.
    for seed in 0..4u64 {
        let g = new_game_from_seed(seed, &MapParams::default(), &v2_config());
        assert_eq!(state_digest(&g), expected_m18_v2_digest(&g), "seed {seed}");
    }
}
