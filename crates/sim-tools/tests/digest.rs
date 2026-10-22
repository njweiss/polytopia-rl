//! Milestone 7: `state_digest` (docs/MILESTONE_7 "State digest", G1, "Tests" item 2). The exact
//! value is checked against `common::my_digest`, an independent FNV-1a encoder of the brief's byte
//! stream (itself checked against published FNV-1a test vectors). The remaining tests are
//! properties that hold for any correct digest.

mod common;

use std::sync::Arc;

use common::*;
use sim_core::data::{Tables, Tech, UnitType};
use sim_core::map::{MapParams, Resource, Terrain};
use sim_core::rng::Rng;
use sim_core::setup::new_game_from_seed;
use sim_core::state::{
    City, CityId, Config, Dir, Improvement, Outcome, Phase, PlayerState, ScoreWeights, State,
    TechSet, TileState, Unit, UnitId,
};
use sim_core::testing::StateBuilder;
use sim_tools::digest::state_digest;

/// A small board where every `Option` and flag in the state has a non-default value somewhere.
fn rich_state() -> State {
    let mut s = StateBuilder::new(5)
        .terrain(p(0, 1), Terrain::Forest)
        .resource(p(0, 1), Resource::Animal)
        .terrain(p(4, 4), Terrain::Mountain)
        .resource(p(4, 4), Resource::Metal)
        .improvement(p(4, 4), Improvement::Mine)
        .resource(p(1, 0), Resource::Fruit)
        .village(p(0, 4))
        .city_with(0, p(1, 1), |c| {
            c.is_capital = true;
            c.level = 3;
            c.population = 6;
            c.workshop = true;
            c.wall = true;
        })
        .city_with(1, p(3, 3), |c| {
            c.level = 5;
            c.population = 15;
            c.parks = 2;
            c.border_growth_used = true;
            c.pending_reward = true;
        })
        .unit_with(0, UnitType::Warrior, p(1, 1), |u| {
            u.origin = Some(CityId(0));
            u.kills = 2;
            u.phase = Phase::Moved;
            u.last_dir = Some(Dir { dr: -1, dc: 1 });
        })
        .unit_with(1, UnitType::Rider, p(2, 3), |u| {
            u.origin = Some(CityId(1));
            u.veteran = true;
            u.max_health = 15;
            u.health = 9;
            u.promotion_available = true;
            u.phase = Phase::Attacked;
            u.last_dir = Some(Dir { dr: 1, dc: 0 });
        })
        .unit(0, UnitType::Archer, p(4, 0))
        .techs(0, &[Tech::Organization, Tech::Climbing])
        .techs(1, &[Tech::Hunting, Tech::Archery, Tech::Mining])
        .stars(0, 7)
        .stars(1, 11)
        .explore(0, &[p(0, 0), p(0, 1), p(1, 1), p(2, 2)])
        .explore(1, &[p(4, 4), p(3, 3)])
        .turn(4)
        .current(1)
        .config(Config {
            full_observability: false,
            turn_cap: 30,
            score_weights: ScoreWeights::default(),
        })
        .rng(Rng::new(99))
        .build();
    s.next_unit_id = 9;
    s
}

fn digest(s: &State) -> u64 {
    state_digest(s)
}

#[test]
fn digest_is_fnv1a_over_the_brief_byte_stream() {
    // The reference encoder's hash against FNV-1a 64-bit test vectors (Fowler/Noll/Vo reference
    // test suite).
    assert_eq!(fnv1a64(b""), 0xcbf29ce484222325);
    assert_eq!(fnv1a64(b"a"), 0xaf63dc4c8601ec8c);
    assert_eq!(fnv1a64(b"foobar"), 0x85944171f73967e8);
    let s = rich_state();
    assert_eq!(digest(&s), my_digest(&s));
    let empty = StateBuilder::new(3).build();
    assert_eq!(digest(&empty), my_digest(&empty));
    for seed in 0..5 {
        let s = new_game_from_seed(seed, &MapParams::default(), &Config::default());
        assert_eq!(digest(&s), my_digest(&s), "seed {seed}");
    }
}

/// Every non-terminal state of a few random playouts. Terminal states are only compared by the
/// property tests below because the `Outcome` encoding is an assumption (Q6).
#[test]
fn digest_matches_the_reference_along_playouts() {
    for seed in 0..4 {
        let (states, _) = play_independent(
            seed,
            &MapParams::default(),
            &cfg(4, seed % 2 == 1),
            400,
            seed + 1000,
        );
        for (i, s) in states.iter().enumerate() {
            if s.outcome.is_none() {
                assert_eq!(digest(s), my_digest(s), "seed {seed} state {i}");
            }
        }
    }
}

#[test]
fn equal_states_built_two_ways_have_equal_digests() {
    let a = StateBuilder::new(5)
        .stars(0, 3)
        .techs(1, &[Tech::Riding])
        .city(0, p(2, 2))
        .unit(1, UnitType::Rider, p(0, 0))
        .build();
    let b = StateBuilder::new(5)
        .unit(1, UnitType::Rider, p(0, 0))
        .city(0, p(2, 2))
        .techs(1, &[Tech::Riding])
        .stars(0, 3)
        .build();
    assert_eq!(a, b);
    assert_eq!(digest(&a), digest(&b));

    let g1 = new_game_from_seed(17, &MapParams::default(), &Config::default());
    let g2 = new_game_from_seed(17, &MapParams::default(), &Config::default());
    assert_eq!(digest(&g1), digest(&g2));
    assert_eq!(digest(&g1), digest(&g1.clone()));
    let r = rich_state();
    assert_eq!(digest(&r), digest(&r.clone()));
}

#[test]
fn digest_does_not_mutate_and_is_repeatable() {
    let s = rich_state();
    let before = s.clone();
    let d1 = digest(&s);
    let d2 = digest(&s);
    assert_eq!(d1, d2);
    assert_eq!(
        s, before,
        "the digest must not touch the state (rng included)"
    );
}

#[test]
fn digest_ignores_the_tables() {
    let s = rich_state();
    let mut fresh = s.clone();
    fresh.tables = Arc::new(Tables::load().unwrap());
    assert!(!Arc::ptr_eq(&s.tables, &fresh.tables));
    assert_eq!(digest(&s), digest(&fresh));

    // Different rules data (Warrior cost 2 -> 9) still gives the same digest.
    let units =
        include_str!("../../../data/units.json").replacen(r#""cost": 2,"#, r#""cost": 9,"#, 1);
    let modified = Tables::from_json(
        &units,
        include_str!("../../../data/techs.json"),
        include_str!("../../../data/site_actions.json"),
        include_str!("../../../data/rewards.json"),
    )
    .unwrap();
    assert_eq!(modified.unit(UnitType::Warrior).cost, Some(9));
    let mut other = s.clone();
    other.tables = Arc::new(modified);
    assert_eq!(digest(&s), digest(&other));
}

type Edit = (&'static str, fn(&mut State));

/// One single-field edit per field of `State` (except `tables`) and of every nested struct. The
/// exhaustive destructuring below stops compiling when a field is added, so the table must be
/// extended with it.
fn single_field_edits() -> Vec<Edit> {
    let s = rich_state();
    let State {
        size: _,
        turn: _,
        current: _,
        rng: _,
        config: _,
        tables: _,
        tiles: _,
        units: _,
        cities: _,
        players: _,
        next_unit_id: _,
        outcome: _,
    } = &s;
    let Config {
        full_observability: _,
        turn_cap: _,
        score_weights: _,
    } = &s.config;
    let ScoreWeights {
        unit_per_star: _,
        super_unit: _,
        territory_per_tile: _,
        exploration_per_tile: _,
        city_base: _,
        city_per_level: _,
        population_surplus: _,
        park: _,
        tech_per_tier: _,
    } = &s.config.score_weights;
    let TileState {
        terrain: _,
        resource: _,
        improvement: _,
        village: _,
        city: _,
        owner_city: _,
        unit: _,
    } = &s.tiles[0];
    let Unit {
        id: _,
        kind: _,
        owner: _,
        pos: _,
        health: _,
        max_health: _,
        origin: _,
        kills: _,
        veteran: _,
        promotion_available: _,
        phase: _,
        last_dir: _,
    } = &s.units[0];
    let City {
        id: _,
        owner: _,
        pos: _,
        level: _,
        population: _,
        is_capital: _,
        workshop: _,
        parks: _,
        wall: _,
        border_growth_used: _,
        pending_reward: _,
    } = &s.cities[0];
    let PlayerState {
        stars: _,
        techs: _,
        explored: _,
    } = &s.players[0];

    // Tile indices on the 5x5 rich board: (0,1) = 1 forest+animal, (0,4) = 4 village,
    // (1,1) = 6 capital + unit 0, (4,4) = 24 mountain+metal+mine.
    vec![
        ("size", |s| s.size = 6),
        ("turn", |s| s.turn += 1),
        ("current", |s| s.current = 0),
        ("rng", |s| {
            s.rng.next_u32();
        }),
        ("config.full_observability", |s| {
            s.config.full_observability = true
        }),
        ("config.turn_cap", |s| s.config.turn_cap = 31),
        ("weights.unit_per_star", |s| {
            s.config.score_weights.unit_per_star += 1
        }),
        ("weights.super_unit", |s| {
            s.config.score_weights.super_unit += 1
        }),
        ("weights.territory_per_tile", |s| {
            s.config.score_weights.territory_per_tile += 1
        }),
        ("weights.exploration_per_tile", |s| {
            s.config.score_weights.exploration_per_tile += 1
        }),
        ("weights.city_base", |s| {
            s.config.score_weights.city_base += 1
        }),
        ("weights.city_per_level", |s| {
            s.config.score_weights.city_per_level += 1
        }),
        ("weights.population_surplus", |s| {
            s.config.score_weights.population_surplus += 1
        }),
        ("weights.park", |s| s.config.score_weights.park += 1),
        ("weights.tech_per_tier", |s| {
            s.config.score_weights.tech_per_tier += 1
        }),
        ("tile.terrain", |s| s.tiles[1].terrain = Terrain::Field),
        ("tile.resource some->none", |s| s.tiles[1].resource = None),
        ("tile.resource none->some", |s| {
            s.tiles[2].resource = Some(Resource::Fruit)
        }),
        ("tile.resource kind", |s| {
            s.tiles[1].resource = Some(Resource::Crop)
        }),
        ("tile.improvement some->none", |s| {
            s.tiles[24].improvement = None
        }),
        ("tile.improvement kind", |s| {
            s.tiles[24].improvement = Some(Improvement::Farm)
        }),
        ("tile.improvement none->some", |s| {
            s.tiles[1].improvement = Some(Improvement::LumberHut)
        }),
        ("tile.village", |s| s.tiles[4].village = false),
        ("tile.city some->none", |s| s.tiles[6].city = None),
        ("tile.city none->some", |s| {
            s.tiles[0].city = Some(CityId(1))
        }),
        ("tile.owner_city some->none", |s| {
            s.tiles[0].owner_city = None
        }),
        ("tile.owner_city id", |s| {
            s.tiles[0].owner_city = Some(CityId(1))
        }),
        ("tile.unit some->none", |s| s.tiles[6].unit = None),
        ("tile.unit id", |s| s.tiles[6].unit = Some(UnitId(2))),
        ("tile.unit none->some", |s| {
            s.tiles[0].unit = Some(UnitId(0))
        }),
        ("unit.id", |s| s.units[0].id = UnitId(5)),
        ("unit.kind", |s| s.units[0].kind = UnitType::Defender),
        ("unit.owner", |s| s.units[0].owner = 1),
        ("unit.pos.row", |s| s.units[0].pos.row = 2),
        ("unit.pos.col", |s| s.units[0].pos.col = 2),
        ("unit.health", |s| s.units[0].health -= 1),
        ("unit.max_health", |s| s.units[0].max_health += 5),
        ("unit.origin some->none", |s| s.units[0].origin = None),
        ("unit.origin id", |s| s.units[0].origin = Some(CityId(1))),
        ("unit.origin none->some", |s| {
            s.units[2].origin = Some(CityId(0))
        }),
        ("unit.kills", |s| s.units[0].kills += 1),
        ("unit.veteran", |s| s.units[0].veteran = true),
        ("unit.promotion_available", |s| {
            s.units[0].promotion_available = true
        }),
        ("unit.phase", |s| s.units[0].phase = Phase::Done),
        ("unit.last_dir some->none", |s| s.units[0].last_dir = None),
        ("unit.last_dir.dr", |s| {
            s.units[0].last_dir = Some(Dir { dr: 0, dc: 1 })
        }),
        ("unit.last_dir.dc", |s| {
            s.units[0].last_dir = Some(Dir { dr: -1, dc: -1 })
        }),
        ("unit.last_dir none->some", |s| {
            s.units[2].last_dir = Some(Dir { dr: 1, dc: 1 })
        }),
        ("unit (last one).health", |s| s.units[2].health -= 1),
        ("city.id", |s| s.cities[0].id = CityId(7)),
        ("city.owner", |s| s.cities[0].owner = 1),
        ("city.pos", |s| s.cities[0].pos = p(0, 0)),
        ("city.level", |s| s.cities[0].level += 1),
        ("city.population", |s| s.cities[0].population += 1),
        ("city.is_capital", |s| s.cities[0].is_capital = false),
        ("city.workshop", |s| s.cities[0].workshop = false),
        ("city.parks", |s| s.cities[0].parks = 1),
        ("city.wall", |s| s.cities[0].wall = false),
        ("city.border_growth_used", |s| {
            s.cities[0].border_growth_used = true
        }),
        ("city.pending_reward", |s| s.cities[0].pending_reward = true),
        ("city (last one).parks", |s| s.cities[1].parks += 1),
        ("player0.stars", |s| s.players[0].stars += 1),
        ("player1.stars", |s| s.players[1].stars += 1),
        ("player0.techs", |s| s.players[0].techs.insert(Tech::Riding)),
        ("player1.techs", |s| s.players[1].techs = TechSet::new()),
        ("player0.explored", |s| s.players[0].explored[24] = true),
        ("player1.explored", |s| s.players[1].explored[0] = true),
        ("next_unit_id", |s| s.next_unit_id += 1),
        ("outcome none->draw", |s| s.outcome = Some(Outcome::Draw)),
        ("outcome none->win0", |s| s.outcome = Some(Outcome::Win(0))),
        ("outcome none->win1", |s| s.outcome = Some(Outcome::Win(1))),
    ]
}

#[test]
fn every_single_field_edit_changes_the_digest() {
    let base = rich_state();
    let d0 = digest(&base);
    for (name, edit) in single_field_edits() {
        let mut s = base.clone();
        edit(&mut s);
        assert_ne!(s, base, "{name}: the edit must change the state");
        assert_ne!(digest(&s), d0, "{name}: the digest ignores this field");
    }
}

#[test]
fn outcome_payloads_are_distinguished() {
    let mut a = rich_state();
    let mut b = a.clone();
    let mut c = a.clone();
    a.outcome = Some(Outcome::Win(0));
    b.outcome = Some(Outcome::Win(1));
    c.outcome = Some(Outcome::Draw);
    assert_ne!(digest(&a), digest(&b));
    assert_ne!(digest(&a), digest(&c));
    assert_ne!(digest(&b), digest(&c));
}

#[test]
fn rng_state_alone_changes_the_digest() {
    let a = StateBuilder::new(4).rng(Rng::new(0)).build();
    let b = StateBuilder::new(4).rng(Rng::new(1)).build();
    assert_ne!(digest(&a), digest(&b));
    // Same rng advanced by one output.
    let mut c = a.clone();
    c.rng.next_u32();
    assert_ne!(digest(&a), digest(&c));
    // Advanced by two: still different from both.
    let mut d = c.clone();
    d.rng.next_u32();
    assert_ne!(digest(&c), digest(&d));
    assert_ne!(digest(&a), digest(&d));
}

#[test]
fn unit_and_city_order_matter() {
    let s = rich_state();
    let mut t = s.clone();
    t.units.swap(0, 1);
    assert_ne!(digest(&s), digest(&t));
    let mut u = s.clone();
    u.cities.swap(0, 1);
    assert_ne!(digest(&s), digest(&u));
}

#[test]
fn consecutive_playout_states_have_different_digests() {
    for seed in 0..12 {
        let (states, actions) = play_independent(
            seed,
            &MapParams::default(),
            &cfg(3 + (seed as u32 % 3), seed % 3 == 0),
            2000,
            seed ^ 0xabcd,
        );
        assert!(states.last().unwrap().is_terminal(), "seed {seed}");
        for (i, w) in states.windows(2).enumerate() {
            assert_ne!(
                digest(&w[0]),
                digest(&w[1]),
                "seed {seed}: action {i} {:?} left the digest unchanged",
                actions[i]
            );
        }
    }
}
