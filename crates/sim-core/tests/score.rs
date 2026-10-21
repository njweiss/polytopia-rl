//! Milestone 6: `score` and `score_breakdown` (docs/MILESTONE_6 "Score", F6 to F8, "Tests" item 1;
//! spec "Score"). Every expected number is hand arithmetic from the brief's default weights:
//! unit 5 per star of cost (Warrior 2, Rider/Archer/Defender 3), Super Unit 50, territory 20 per
//! tile, exploration 5 per explored tile, city 100 + 50 * (level - 1) + 5 * surplus population
//! over `L(L+1)/2 - 1` of the current level, park 250, tech 100 * tier (Climbing, Hunting,
//! Organization, Riding tier 1; the rest tier 2).

mod common;

use common::*;
use sim_core::data::{Tech, UnitType};
use sim_core::map::MapParams;
use sim_core::score::{score, score_breakdown, ScoreBreakdown};
use sim_core::setup::new_game_from_seed;
use sim_core::state::{CityId, Config, Phase, PlayerId, ScoreWeights, State, Unit, UnitId};
use UnitType::*;

fn start(seed: u64) -> State {
    new_game_from_seed(seed, &MapParams::default(), &Config::default())
}

/// The spec's starting breakdown: Warrior 10, 9 tiles = 180, 25 tiles = 125, capital 100, no
/// parks, Organization 100.
const START: ScoreBreakdown = ScoreBreakdown {
    units: 10,
    territory: 180,
    exploration: 125,
    cities: 100,
    parks: 0,
    techs: 100,
};

/// Adds a unit (next id, Fresh, no origin) on the first row-major tile with no unit and no city.
fn add_unit(s: &mut State, owner: PlayerId, kind: UnitType) {
    let at = all_pos(s.size)
        .into_iter()
        .find(|&q| s.tile(q).unit.is_none() && s.tile(q).city.is_none())
        .unwrap();
    let health = s.tables.unit(kind).health;
    let id = UnitId(s.next_unit_id);
    s.next_unit_id += 1;
    s.units.push(Unit {
        id,
        kind,
        owner,
        pos: at,
        health,
        max_health: health,
        origin: None,
        kills: 0,
        veteran: false,
        promotion_available: false,
        phase: Phase::Fresh,
        last_dir: None,
    });
    tile_mut(s, at).unit = Some(id);
}

fn breakdown(s: &State, player: PlayerId) -> ScoreBreakdown {
    let b = score_breakdown(s, player);
    assert_eq!(score(s, player), b.total(), "score == breakdown total");
    b
}

/// `START` with `change` applied.
fn start_with(change: impl FnOnce(&mut ScoreBreakdown)) -> ScoreBreakdown {
    let mut b = ScoreBreakdown { ..START };
    change(&mut b);
    b
}

/// Applies `edit` to a fresh game and checks player 0's breakdown is `want`, while player 1 keeps
/// the starting 515.
fn check(seed: u64, edit: impl FnOnce(&mut State), want: ScoreBreakdown) {
    let mut s = start(seed);
    edit(&mut s);
    assert_eq!(breakdown(&s, 0), want, "seed {seed}");
    assert_eq!(breakdown(&s, 1), START, "seed {seed}: player 1 untouched");
}

#[test]
fn starting_score_is_515_for_both_players() {
    // The default weights are the brief's.
    assert_eq!(
        Config::default().score_weights,
        ScoreWeights {
            unit_per_star: 5,
            super_unit: 50,
            territory_per_tile: 20,
            exploration_per_tile: 5,
            city_base: 100,
            city_per_level: 50,
            population_surplus: 5,
            park: 250,
            tech_per_tier: 100,
        }
    );
    for seed in 0..30 {
        let s = start(seed);
        for pl in 0..2 {
            assert_eq!(breakdown(&s, pl), START, "seed {seed} player {pl}");
            assert_eq!(score(&s, pl), 515, "seed {seed} player {pl}");
        }
    }
}

/// F8: exploration counts the mask, so full observability gives 121 * 5 = 605 to each player.
#[test]
fn full_observability_scores_every_tile_explored() {
    let cfg = Config {
        full_observability: true,
        ..Config::default()
    };
    for seed in 0..10 {
        let s = new_game_from_seed(seed, &MapParams::default(), &cfg);
        for pl in 0..2 {
            let want = start_with(|b| b.exploration = 605);
            assert_eq!(breakdown(&s, pl), want, "seed {seed} player {pl}");
            assert_eq!(score(&s, pl), 995, "seed {seed} player {pl}");
        }
    }
}

#[test]
fn each_unit_scores_five_per_star_of_cost() {
    for (kind, points) in [(Warrior, 10), (Rider, 15), (Archer, 15), (Defender, 15)] {
        check(
            1,
            |s| add_unit(s, 0, kind),
            start_with(|b| b.units = 10 + points),
        );
    }
}

/// F6: the Super Unit scores a flat 50 (it has no cost).
#[test]
fn a_super_unit_scores_a_flat_50() {
    check(
        2,
        |s| add_unit(s, 0, SuperUnit),
        start_with(|b| b.units = 60),
    );
    // A dead unit no longer scores.
    check(
        2,
        |s| {
            let w = s.units.iter().position(|u| u.owner == 0).unwrap();
            let at = s.units.remove(w).pos;
            tile_mut(s, at).unit = None;
        },
        start_with(|b| b.units = 0),
    );
}

#[test]
fn one_more_territory_tile_scores_20() {
    check(
        3,
        |s| {
            let q = all_pos(s.size)
                .into_iter()
                .find(|&q| s.tile(q).owner_city.is_none())
                .unwrap();
            tile_mut(s, q).owner_city = Some(CityId(0));
        },
        start_with(|b| b.territory = 200),
    );
    // A tile of player 1's city scores for player 1, wherever it lies.
    let mut s = start(3);
    let q = all_pos(s.size)
        .into_iter()
        .find(|&q| s.tile(q).owner_city.is_none())
        .unwrap();
    tile_mut(&mut s, q).owner_city = Some(CityId(1));
    assert_eq!(breakdown(&s, 0), START);
    assert_eq!(breakdown(&s, 1), start_with(|b| b.territory = 200));
}

#[test]
fn one_more_explored_tile_scores_5() {
    check(
        4,
        |s| {
            let i = s.players[0].explored.iter().position(|&e| !e).unwrap();
            s.players[0].explored[i] = true;
        },
        start_with(|b| b.exploration = 130),
    );
}

/// The brief's example: level 3 with population 7 is 100 + 2 * 50 + 5 * (7 - 5) = 210.
#[test]
fn city_points_follow_level_and_surplus_population() {
    let set = |level: u32, population: u32| {
        move |s: &mut State| {
            let c = s.cities.iter_mut().find(|c| c.owner == 0).unwrap();
            c.level = level;
            c.population = population;
        }
    };
    check(5, set(3, 7), start_with(|b| b.cities = 210));
    // Exactly at the threshold: no surplus. Level 2 at 2, level 5 at 14 + 3.
    check(5, set(2, 2), start_with(|b| b.cities = 150));
    check(5, set(5, 17), start_with(|b| b.cities = 100 + 200 + 15));
    // Level 1 counts every population as surplus (threshold 0).
    check(5, set(1, 1), start_with(|b| b.cities = 105));
    // Below the threshold the surplus saturates at 0.
    check(5, set(3, 2), start_with(|b| b.cities = 200));
}

#[test]
fn each_city_scores_on_its_own() {
    // A second city for player 0 at level 2 with population 3 (surplus 1): 100 + 50 + 5.
    check(
        6,
        |s| {
            let at = all_pos(s.size)
                .into_iter()
                .find(|&q| s.tile(q).city.is_none() && s.tile(q).unit.is_none())
                .unwrap();
            let mut c = s.cities[0].clone();
            c.id = CityId(2);
            (c.owner, c.pos, c.level, c.population, c.is_capital) = (0, at, 2, 3, false);
            s.cities.push(c);
            tile_mut(s, at).city = Some(CityId(2));
        },
        start_with(|b| b.cities = 100 + 155),
    );
}

#[test]
fn each_park_scores_250() {
    let parks =
        |n: u32| move |s: &mut State| s.cities.iter_mut().find(|c| c.owner == 0).unwrap().parks = n;
    check(7, parks(1), start_with(|b| b.parks = 250));
    check(7, parks(3), start_with(|b| b.parks = 750));
}

#[test]
fn techs_score_100_per_tier_including_the_starting_one() {
    let add = |techs: &'static [Tech]| {
        move |s: &mut State| {
            for &t in techs {
                s.players[0].techs.insert(t);
            }
        }
    };
    check(8, add(&[Tech::Hunting]), start_with(|b| b.techs = 200));
    check(
        8,
        add(&[Tech::Hunting, Tech::Archery]),
        start_with(|b| b.techs = 400),
    );
    // Every tech: 4 tier 1 + 5 tier 2 = 1400.
    check(
        8,
        add(&[
            Tech::Climbing,
            Tech::Hunting,
            Tech::Riding,
            Tech::Archery,
            Tech::Forestry,
            Tech::Farming,
            Tech::Strategy,
            Tech::Mining,
        ]),
        start_with(|b| b.techs = 1400),
    );
    // Without the starting Organization nothing is scored for techs.
    check(
        8,
        |s| s.players[0].techs = sim_core::state::TechSet::new(),
        start_with(|b| b.techs = 0),
    );
}

/// Every weight is read from `config.score_weights`: distinct weights per term, on a game with
/// one extra Rider, a Super Unit, a level 3 city with population 7, two parks and Hunting.
#[test]
fn custom_weights_scale_each_term() {
    let mut s = start(9);
    add_unit(&mut s, 0, Rider);
    add_unit(&mut s, 0, SuperUnit);
    let c = s.cities.iter_mut().find(|c| c.owner == 0).unwrap();
    (c.level, c.population, c.parks) = (3, 7, 2);
    s.players[0].techs.insert(Tech::Hunting);
    s.config.score_weights = ScoreWeights {
        unit_per_star: 1,
        super_unit: 2,
        territory_per_tile: 3,
        exploration_per_tile: 7,
        city_base: 11,
        city_per_level: 13,
        population_surplus: 17,
        park: 19,
        tech_per_tier: 23,
    };
    let want = ScoreBreakdown {
        units: 2 + 3 + 2,             // Warrior 1*2, Rider 1*3, Super Unit 2
        territory: 9 * 3,             // 9 tiles
        exploration: 25 * 7,          // 25 tiles
        cities: 11 + 13 * 2 + 17 * 2, // level 3, surplus 7 - 5
        parks: 2 * 19,
        techs: 23 * 2, // Organization and Hunting, tier 1 each
    };
    assert_eq!(breakdown(&s, 0), want);
    assert_eq!(score(&s, 0), 7 + 27 + 175 + 71 + 38 + 46);

    // Doubling every default weight doubles every total.
    let mut d = start(9);
    add_unit(&mut d, 1, Defender);
    let w = ScoreWeights::default();
    d.config.score_weights = ScoreWeights {
        unit_per_star: 2 * w.unit_per_star,
        super_unit: 2 * w.super_unit,
        territory_per_tile: 2 * w.territory_per_tile,
        exploration_per_tile: 2 * w.exploration_per_tile,
        city_base: 2 * w.city_base,
        city_per_level: 2 * w.city_per_level,
        population_surplus: 2 * w.population_surplus,
        park: 2 * w.park,
        tech_per_tier: 2 * w.tech_per_tier,
    };
    assert_eq!(score(&d, 0), 1030);
    assert_eq!(score(&d, 1), 2 * (515 + 15));
}
