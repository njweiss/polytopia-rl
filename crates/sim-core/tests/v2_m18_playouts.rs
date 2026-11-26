//! Milestone 18 playout tests (item 7): multi-seed randomized v2 playouts exercising corner
//! lighthouses, Explorer lighthouse bonus, monument unlocks/placement/destroy, invariants, and
//! replay determinism.

mod common;

use common::{all_pos, exp_cum};
use sim_core::actions::{apply, legal_actions, Action};
use sim_core::data::{Monument, Reward, Tech, ALL_MONUMENTS};
use sim_core::map::{MapParams, Terrain};
use sim_core::monuments::{corner_index, corners};
use sim_core::rng::Rng;
use sim_core::score::{score, score_breakdown};
use sim_core::setup::new_game_from_seed;
use sim_core::state::{Config, MonumentSet, Ruleset, State};

const ALL_V2_TECHS: [Tech; 16] = [
    Tech::Climbing,
    Tech::Hunting,
    Tech::Organization,
    Tech::Riding,
    Tech::Archery,
    Tech::Forestry,
    Tech::Farming,
    Tech::Strategy,
    Tech::Mining,
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

fn pick_m18(s: &State, legal: &[Action], rng: &mut Rng) -> Action {
    // Strongly prefer BuildMonument, Explorer, Destroy on monument, and Attack/Move to exercise
    // lighthouses, kills, and monuments quickly.
    let priority: Vec<Action> = legal
        .iter()
        .copied()
        .filter(|a| match *a {
            Action::BuildMonument { .. }
            | Action::ChooseReward {
                reward: Reward::Explorer | Reward::PopulationGrowth,
                ..
            }
            | Action::Attack { .. }
            | Action::Train { .. }
            | Action::Capture { .. } => true,
            Action::Destroy { at } => s.tile(at).monument.is_some(),
            _ => false,
        })
        .collect();
    if !priority.is_empty() && rng.gen_range(2) == 0 {
        return priority[rng.gen_range(priority.len() as u32) as usize];
    }
    let others: Vec<Action> = legal
        .iter()
        .copied()
        .filter(|a| *a != Action::EndTurn)
        .collect();
    if others.is_empty() || (legal.contains(&Action::EndTurn) && rng.gen_range(8) == 0) {
        return Action::EndTurn;
    }
    others[rng.gen_range(others.len() as u32) as usize]
}

fn check_m18_invariants(prev: &State, s: &State, ctx: &str) {
    let c4 = corners(s.size);
    for q in all_pos(s.size) {
        let t = s.tile(q);
        let is_corner = corner_index(s.size, q).is_some();
        assert_eq!(t.lighthouse, is_corner, "{ctx}: lighthouse only on corners");
        if t.lighthouse {
            assert_eq!(t.terrain, Terrain::Field, "{ctx}: lighthouse terrain");
            assert_eq!(t.resource, None, "{ctx}: lighthouse resource");
            assert!(!t.road, "{ctx}: no road on lighthouse");
            assert!(t.building.is_none(), "{ctx}: no building on lighthouse");
            assert!(t.monument.is_none(), "{ctx}: no monument on lighthouse");
        }
        if t.monument.is_some() {
            assert_eq!(t.terrain, Terrain::Field, "{ctx}: monument on field");
            assert!(t.owner_city.is_some(), "{ctx}: monument owned");
            assert!(t.city.is_none(), "{ctx}: no monument on city");
            assert!(t.improvement.is_none(), "{ctx}: no monument on improvement");
            assert!(t.building.is_none(), "{ctx}: no monument on building");
            assert!(t.resource.is_none(), "{ctx}: no monument on resource");
        }
    }

    for pl in 0..2usize {
        let p0 = &prev.players[pl];
        let p1 = &s.players[pl];
        // Monotonic masks/counters.
        assert!(p1.kills >= p0.kills, "{ctx}: kills shrank");
        assert_eq!(
            p1.lighthouses_discovered & !0b1111,
            0,
            "{ctx}: high bits in lighthouses_discovered"
        );
        assert_eq!(
            p1.lighthouses_discovered & p0.lighthouses_discovered,
            p0.lighthouses_discovered,
            "{ctx}: lighthouses_discovered shrank"
        );
        assert_eq!(
            p1.monuments_unlocked.bits() & p0.monuments_unlocked.bits(),
            p0.monuments_unlocked.bits(),
            "{ctx}: monuments_unlocked shrank"
        );
        assert_eq!(
            p1.monuments_placed.bits() & p0.monuments_placed.bits(),
            p0.monuments_placed.bits(),
            "{ctx}: monuments_placed shrank"
        );
        assert_eq!(
            p1.monuments_placed.bits() & !p1.monuments_unlocked.bits(),
            0,
            "{ctx}: placed without unlocked"
        );

        // Non-terminal sync invariants.
        if s.outcome.is_none() {
            for (idx, &q) in c4.iter().enumerate() {
                let bit = 1u8 << idx;
                if s.is_explored(pl as u8, q) {
                    assert_ne!(
                        p1.lighthouses_discovered & bit,
                        0,
                        "{ctx}: explored corner {idx} not discovered for P{pl}"
                    );
                } else {
                    assert_eq!(
                        p1.lighthouses_discovered & bit,
                        0,
                        "{ctx}: unexplored corner {idx} marked discovered for P{pl}"
                    );
                }
            }
            if s.cities.iter().any(|c| c.owner == pl as u8 && c.level >= 5) {
                assert!(
                    p1.monuments_unlocked.contains(Monument::ParkOfFortune),
                    "{ctx}: missed ParkOfFortune unlock"
                );
            }
            if p1.kills >= 10 {
                assert!(
                    p1.monuments_unlocked.contains(Monument::GateOfPower),
                    "{ctx}: missed GateOfPower unlock"
                );
            }
            if p1.lighthouses_discovered.count_ones() >= 4 {
                assert!(
                    p1.monuments_unlocked.contains(Monument::EyeOfGod),
                    "{ctx}: missed EyeOfGod unlock"
                );
            }
        }

        // Monument score matches tiles owned by pl's cities.
        let mon_tiles = s
            .tiles
            .iter()
            .filter(|t| {
                t.monument.is_some()
                    && t.owner_city
                        .is_some_and(|cid| s.city(cid).owner == pl as u8)
            })
            .count() as u32;
        let bd = score_breakdown(s, pl as u8);
        assert_eq!(bd.monuments, mon_tiles * 400, "{ctx}: monument score");
        assert_eq!(score(s, pl as u8), bd.total(), "{ctx}: score total");
    }

    for c in &s.cities {
        if !c.pending_reward {
            assert!(
                c.population < exp_cum(c.level + 1) as i32,
                "{ctx}: missed level up {c:?}"
            );
        }
    }
}

#[test]
fn random_v2_playouts_exercise_lighthouses_monuments_and_replay_identically() {
    let mut lighthouses_found = 0usize;
    let mut explorers_picked = 0usize;
    let mut monuments_built = [0usize; 3];
    let mut monuments_destroyed = 0usize;

    for seed in 0..24u64 {
        let mut s = new_game_from_seed(seed, &MapParams::default(), &v2_config(18));
        let c4 = corners(s.size);
        for pl in &mut s.players {
            for t in ALL_V2_TECHS {
                pl.techs.insert(t);
            }
            pl.stars += 50;
            // Start with 9 kills so a single combat kill organically unlocks GateOfPower.
            pl.kills = 9;
            // On a third of the seeds, pre-explore 3 corners (matching lighthouses_discovered) so
            // finding the 4th organically unlocks EyeOfGod in playouts too.
            if seed % 3 == 0 {
                for (ci, &q) in c4[..3].iter().enumerate() {
                    let idx = q.row as usize * s.size as usize + q.col as usize;
                    pl.explored[idx] = true;
                    pl.lighthouses_discovered |= 1 << ci;
                }
            }
        }
        // Start capitals at level 1 pop 1 (so Explorer at level 2 shows up immediately),
        // and on half the seeds at level 4 pop 12 so ParkOfFortune unlocks rapidly.
        for c in &mut s.cities {
            if seed % 2 == 0 {
                c.level = 1;
                c.population = 1;
            } else {
                c.level = 4;
                c.population = 12;
            }
        }
        // Sanity: no monuments unlocked yet at start.
        assert_eq!(s.players[0].monuments_unlocked, MonumentSet::new());
        assert_eq!(s.players[1].monuments_unlocked, MonumentSet::new());

        let start = s.clone();
        let mut rng = Rng::new(seed ^ 0x18_cafe);
        let mut log = Vec::new();
        let mut states = Vec::new();

        while !s.is_terminal() && log.len() < 1500 {
            let legal = legal_actions(&s);
            let a = pick_m18(&s, &legal, &mut rng);
            let prev = s.clone();
            let had_mon_at_destroy = match a {
                Action::Destroy { at } => prev.tile(at).monument.is_some(),
                _ => false,
            };
            apply(&mut s, &a).unwrap();
            let ctx = format!("seed {seed} step {}", log.len());
            check_m18_invariants(&prev, &s, &ctx);

            for pl in 0..2 {
                let newly = (s.players[pl].lighthouses_discovered
                    & !prev.players[pl].lighthouses_discovered)
                    .count_ones() as usize;
                lighthouses_found += newly;
            }
            match a {
                Action::BuildMonument { monument, .. } => {
                    let idx = ALL_MONUMENTS.iter().position(|&m| m == monument).unwrap();
                    monuments_built[idx] += 1;
                }
                Action::ChooseReward {
                    reward: Reward::Explorer,
                    ..
                } => explorers_picked += 1,
                Action::Destroy { .. } if had_mon_at_destroy => monuments_destroyed += 1,
                _ => {}
            }
            log.push(a);
            states.push(s.clone());
        }

        // Exact replay determinism check.
        let mut r = start;
        for (i, a) in log.iter().enumerate() {
            apply(&mut r, a).unwrap();
            assert_eq!(r, states[i], "seed {seed}: replay mismatch at step {i}");
        }
    }

    assert!(
        lighthouses_found >= 10,
        "only {lighthouses_found} lighthouses discovered"
    );
    assert!(
        explorers_picked >= 4,
        "only {explorers_picked} Explorers picked"
    );
    assert!(
        monuments_built[0] >= 3,
        "only {} ParkOfFortune built",
        monuments_built[0]
    );
    assert!(
        monuments_built[1] >= 1,
        "expected GateOfPower built: {monuments_built:?}"
    );
    assert!(
        monuments_built[2] >= 1,
        "expected EyeOfGod built: {monuments_built:?}"
    );
    assert!(
        monuments_destroyed >= 1,
        "only {monuments_destroyed} monuments destroyed"
    );
}
