//! `new_game` field by field (docs/MILESTONE_3 "new_game"; spec "Starting state"). The map comes
//! from milestone 2's `generate_map`; the expected seat assignment is replayed here with one
//! `gen_range(2)` draw on the same RNG. The turn counter starts at 0 (C10; owner decision
//! recorded in docs/QUESTIONS.md Q1).

mod common;

use common::*;
use sim_core::data::{Tables, Tech, UnitType, ALL_TECHS};
use sim_core::map::{generate_map, GeneratedMap, MapParams, Site};
use sim_core::rng::Rng;
use sim_core::setup::{new_game, new_game_from_seed};
use sim_core::state::{CityId, Config, Phase, State, UnitId};

const SEEDS: u64 = 300;

/// (map, rng after generation, expected seats [player 0 capital, player 1 capital], rng after flip)
fn replay(seed: u64) -> (GeneratedMap, Rng, [sim_core::grid::Pos; 2], Rng) {
    let mut rng = Rng::new(seed);
    let map = generate_map(&mut rng, &MapParams::default());
    let mut after = rng.clone();
    let flip = after.gen_range(2) as usize;
    let seats = [map.capitals[flip], map.capitals[1 - flip]];
    (map, rng, seats, after)
}

fn start(seed: u64, config: &Config) -> (State, GeneratedMap, [sim_core::grid::Pos; 2], Rng) {
    let (map, rng, seats, after) = replay(seed);
    (new_game(&map, config, rng), map, seats, after)
}

#[test]
fn globals_and_rng() {
    for seed in 0..SEEDS {
        let cfg = Config::default();
        let (s, _, _, after) = start(seed, &cfg);
        assert_eq!(s.size, 11);
        assert_eq!(s.turn, 0, "seed {seed}: the turn counter starts at 0");
        assert_eq!(s.current, 0);
        assert_eq!(s.outcome, None);
        assert_eq!(s.config, cfg);
        assert_eq!(
            s.rng, after,
            "seed {seed}: exactly one draw (the flip) after the map"
        );
        assert_eq!(*s.tables, Tables::load().unwrap());
        assert_eq!(s.next_unit_id, 2);
        assert_eq!(s.tiles.len(), 121);
    }
}

#[test]
fn coin_flip_assigns_capitals_and_both_outcomes_occur() {
    let mut flipped = [0u32; 2];
    for seed in 0..SEEDS {
        let (s, map, seats, _) = start(seed, &Config::default());
        assert_eq!(s.cities[0].pos, seats[0], "seed {seed}");
        assert_eq!(s.cities[1].pos, seats[1], "seed {seed}");
        flipped[(seats[0] == map.capitals[1]) as usize] += 1;
    }
    // Fair coin over 300 seeds: each side ~150 +- 8.7 (1 sigma); 5 sigma lower bound ~107.
    assert!(flipped[0] > 100 && flipped[1] > 100, "{flipped:?}");
}

#[test]
fn tiles_copy_the_map_and_record_cities_territory_and_units() {
    for seed in 0..SEEDS {
        let (s, map, seats, _) = start(seed, &Config::default());
        for q in all_pos(11) {
            let t = s.tile(q);
            let m = map.tile(q);
            assert_eq!(t.terrain, m.terrain, "seed {seed} {q:?}");
            assert_eq!(t.resource, m.resource, "seed {seed} {q:?}");
            assert_eq!(t.improvement, None);
            assert_eq!(t.village, m.site == Site::Village, "seed {seed} {q:?}");
            let city = (0..2).find(|&i| seats[i] == q).map(|i| CityId(i as u32));
            assert_eq!(t.city, city, "seed {seed} {q:?}");
            let owner = (0..2)
                .find(|&i| cheb(seats[i], q) <= 1)
                .map(|i| CityId(i as u32));
            assert_eq!(t.owner_city, owner, "seed {seed} {q:?}: border radius 1");
            let occupant = (0..2).find(|&i| seats[i] == q).map(|i| UnitId(i as u32));
            assert_eq!(t.unit, occupant, "seed {seed} {q:?}");
        }
    }
}

#[test]
fn cities_are_level_one_capitals() {
    for seed in 0..SEEDS {
        let (s, _, seats, _) = start(seed, &Config::default());
        assert_eq!(s.cities.len(), 2);
        for (i, c) in s.cities.iter().enumerate() {
            assert_eq!(c.id, CityId(i as u32));
            assert_eq!(c.owner, i as u8);
            assert_eq!(c.pos, seats[i]);
            assert_eq!(c.level, 1);
            assert_eq!(c.population, 0);
            assert!(c.is_capital);
            assert!(!c.workshop);
            assert_eq!(c.parks, 0);
            assert!(!c.wall);
            assert!(!c.border_growth_used);
            assert!(!c.pending_reward);
            assert_eq!(s.city(c.id), c);
        }
    }
}

#[test]
fn each_player_starts_with_one_warrior_on_the_capital() {
    for seed in 0..SEEDS {
        let (s, _, seats, _) = start(seed, &Config::default());
        assert_eq!(s.units.len(), 2);
        for (i, u) in s.units.iter().enumerate() {
            assert_eq!(u.id, UnitId(i as u32));
            assert_eq!(u.kind, UnitType::Warrior);
            assert_eq!(u.owner, i as u8);
            assert_eq!(u.pos, seats[i]);
            assert_eq!(u.max_health, 10, "Warrior table health");
            assert_eq!(u.health, 10);
            assert_eq!(u.origin, Some(CityId(i as u32)));
            assert_eq!(u.kills, 0);
            assert!(!u.veteran);
            assert!(!u.promotion_available);
            assert_eq!(u.phase, Phase::Fresh);
            assert_eq!(u.last_dir, None);
            assert_eq!(s.unit(u.id), Some(u));
        }
    }
}

#[test]
fn players_start_with_five_stars_organization_and_25_explored_tiles() {
    for seed in 0..SEEDS {
        let (s, _, seats, _) = start(seed, &Config::default());
        for i in 0..2u8 {
            let pl = &s.players[i as usize];
            assert_eq!(pl.stars, 5);
            for t in ALL_TECHS {
                assert_eq!(pl.techs.contains(t), t == Tech::Organization, "{t:?}");
            }
            let want: Vec<_> = all_pos(11)
                .into_iter()
                .filter(|&q| cheb(q, seats[i as usize]) <= 2)
                .collect();
            assert_eq!(want.len(), 25);
            assert_eq!(explored_set(&s, i), want, "seed {seed} player {i}");
            assert_eq!(pl.explored.len(), 121);
        }
    }
}

#[test]
fn full_observability_explores_everything_for_both() {
    let cfg = Config {
        full_observability: true,
        ..Config::default()
    };
    for seed in 0..50 {
        let (s, _, _, _) = start(seed, &cfg);
        assert_eq!(s.config, cfg);
        for i in 0..2 {
            assert!(s.players[i].explored.iter().all(|&e| e), "seed {seed}");
        }
    }
}

#[test]
fn new_game_from_seed_is_the_composition_and_deterministic() {
    for seed in 0..100 {
        let cfg = Config::default();
        let (want, _, _, _) = start(seed, &cfg);
        let a = new_game_from_seed(seed, &MapParams::default(), &cfg);
        let b = new_game_from_seed(seed, &MapParams::default(), &cfg);
        assert_eq!(a, want, "seed {seed}");
        assert_eq!(a, b, "seed {seed}");
    }
    let distinct = (0..50)
        .map(|seed| new_game_from_seed(seed, &MapParams::default(), &Config::default()))
        .collect::<Vec<_>>();
    for i in 1..distinct.len() {
        assert_ne!(distinct[i], distinct[i - 1], "seeds {} and {}", i - 1, i);
    }
}
