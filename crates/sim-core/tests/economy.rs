//! Milestone 4: income, research, training, site actions, population, levels and rewards
//! (docs/MILESTONE_4 "Rules"; the spec's tile-action notes overrule the brief's D4 table, see
//! docs/QUESTIONS.md). Milestone 5 replaced the reward menus (level 2: Workshop, Explorer; level
//! 5 and up: Park, SuperUnit) and excluded Super Units from capacity; the tests here follow.
//! Every expected number is worked by hand from the brief (Appendix tables,
//! `L(L+1)/2 - 1`, `4 + cities x tier`) in the comment next to it, never from the code under test.

mod common;

use common::*;
use sim_core::actions::{apply, legal_actions, Action, ActionError};
use sim_core::data::{Reward, SiteAction, Tech, UnitType, ALL_TECHS};
use sim_core::economy::{
    cumulative_population_for_level, income, research_cost, reward_menu, supported_units,
};
use sim_core::grid::Pos;
use sim_core::map::{MapParams, Resource, Terrain};
use sim_core::setup::new_game_from_seed;
use sim_core::state::{City, CityId, Config, Improvement, Phase, State, Unit, UnitId};
use sim_core::testing::StateBuilder;
use Reward::*;
use SiteAction::*;
use Tech::*;
use UnitType::*;

// ---------- helpers ----------

/// Applies an action that must be legal (and checks it is listed).
fn ok(s: &mut State, a: Action) {
    assert!(
        legal_actions(s).contains(&a),
        "{a:?} should be legal; legal = {:?}",
        legal_actions(s)
    );
    assert_eq!(apply(s, &a), Ok(()), "{a:?}");
}

/// Asserts an action is illegal: not listed, `apply` errors and changes nothing.
fn illegal(s: &mut State, a: Action) {
    assert!(!legal_actions(s).contains(&a), "{a:?} should not be legal");
    let before = s.clone();
    assert_eq!(apply(s, &a), Err(ActionError::Illegal), "{a:?}");
    assert_eq!(*s, before, "illegal {a:?} changed the state");
}

fn train(city: u32, kind: UnitType) -> Action {
    Action::Train {
        city: CityId(city),
        kind,
    }
}
fn site(at: Pos, action: SiteAction) -> Action {
    Action::Site { at, action }
}
fn research(tech: Tech) -> Action {
    Action::Research { tech }
}
fn choose(city: u32, reward: Reward) -> Action {
    Action::ChooseReward {
        city: CityId(city),
        reward,
    }
}

fn city(s: &State, id: u32) -> &City {
    s.cities.iter().find(|c| c.id == CityId(id)).unwrap()
}
fn stars(s: &State, player: u8) -> u32 {
    s.players[player as usize].stars
}

const EVERY_TECH: [Tech; 9] = ALL_TECHS;

// ---------- economy helpers ----------

#[test]
fn cumulative_population_thresholds() {
    // L(L+1)/2 - 1 for L = 1..8.
    let want = [0, 2, 5, 9, 14, 20, 27, 35];
    for (i, w) in want.iter().enumerate() {
        let level = i as u32 + 1;
        assert_eq!(cumulative_population_for_level(level), *w, "level {level}");
    }
}

#[test]
fn income_per_city_terms() {
    let one = |edit: fn(&mut City)| {
        let s = StateBuilder::new(5).city_with(0, p(2, 2), edit).build();
        income(&s, 0)
    };
    assert_eq!(one(|c| c.is_capital = true), 2, "level 1 capital: 1 + 1");
    assert_eq!(
        one(|c| {
            c.is_capital = true;
            c.workshop = true;
        }),
        3,
        "level 1 capital with workshop: 1 + 1 + 1"
    );
    assert_eq!(one(|c| c.level = 3), 3, "level 3 non-capital");
    assert_eq!(
        one(|c| {
            c.level = 3;
            c.parks = 2;
        }),
        5,
        "level 3 + 2 parks"
    );
    assert_eq!(one(|c| c.wall = true), 1, "a wall gives no income");
    assert_eq!(
        one(|c| c.border_growth_used = true),
        1,
        "border growth gives no income"
    );
}

#[test]
fn income_sums_over_the_players_cities_only() {
    let s = StateBuilder::new(9)
        // P0: 2 (level 1 capital) + 5 (level 3, 2 parks) + 3 (level 2, workshop, wall) = 10.
        .city_with(0, p(1, 1), |c| c.is_capital = true)
        .city_with(0, p(1, 7), |c| {
            c.level = 3;
            c.parks = 2;
        })
        .city_with(0, p(7, 1), |c| {
            c.level = 2;
            c.workshop = true;
            c.wall = true;
        })
        // P1: level 4 capital with workshop and 1 park = 4 + 1 + 1 + 1 = 7.
        .city_with(1, p(7, 7), |c| {
            c.level = 4;
            c.is_capital = true;
            c.workshop = true;
            c.parks = 1;
        })
        .build();
    assert_eq!(income(&s, 0), 10);
    assert_eq!(income(&s, 1), 7);
    let none = StateBuilder::new(3).build();
    assert_eq!(income(&none, 0), 0, "no cities, no income");
}

#[test]
fn research_cost_uses_tier_and_the_players_city_count() {
    // P0 owns 1 city, P1 owns 3.
    let s = StateBuilder::new(9)
        .city(0, p(1, 1))
        .city(1, p(1, 7))
        .city(1, p(7, 1))
        .city(1, p(7, 7))
        .build();
    // 4 + 1 x 1 = 5; 4 + 1 x 2 = 6.
    assert_eq!(research_cost(&s, 0, Climbing), 5);
    assert_eq!(research_cost(&s, 0, Archery), 6);
    // 4 + 3 x 1 = 7; 4 + 3 x 2 = 10.
    assert_eq!(research_cost(&s, 1, Riding), 7);
    assert_eq!(research_cost(&s, 1, Mining), 10);
}

#[test]
fn supported_units_count_living_units_with_that_origin() {
    let s = StateBuilder::new(5)
        .city(0, p(0, 0))
        .city(0, p(4, 4))
        .unit_with(0, Warrior, p(2, 0), |u| u.origin = Some(CityId(0)))
        .unit_with(0, Archer, p(2, 1), |u| u.origin = Some(CityId(0)))
        .unit_with(0, Warrior, p(2, 2), |u| u.origin = Some(CityId(1)))
        .unit(0, Rider, p(2, 3)) // orphan: origin None (D8)
        .unit_with(0, UnitType::SuperUnit, p(3, 0), |u| {
            u.origin = Some(CityId(0))
        }) // never counts (M5)
        .unit_with(0, UnitType::SuperUnit, p(3, 1), |u| {
            u.origin = Some(CityId(1))
        })
        .build();
    assert_eq!(supported_units(&s, CityId(0)), 2);
    assert_eq!(supported_units(&s, CityId(1)), 1);
}

#[test]
fn reward_menu_follows_the_level_and_drops_used_border_growth() {
    let menu = |level: u32, used: bool| {
        let s = StateBuilder::new(3)
            .city_with(0, p(1, 1), |c| {
                c.level = level;
                c.border_growth_used = used;
            })
            .build();
        reward_menu(&s, CityId(0))
    };
    assert_eq!(menu(2, false), [Workshop, Explorer]);
    assert_eq!(menu(3, false), [CityWall, Resources]);
    assert_eq!(menu(4, false), [PopulationGrowth, BorderGrowth]);
    assert_eq!(menu(4, true), [PopulationGrowth]);
    assert_eq!(menu(5, false), [Park, Reward::SuperUnit]);
    assert_eq!(menu(6, true), [Park, Reward::SuperUnit]);
    assert_eq!(menu(9, false), [Park, Reward::SuperUnit]);
}

// ---------- income timing ----------

#[test]
fn income_starts_on_each_players_second_turn() {
    // P0: level 1 capital, income 2. P1: level 2 capital with workshop, income 2 + 1 + 1 = 4.
    let mut s = StateBuilder::new(7)
        .city_with(0, p(1, 1), |c| c.is_capital = true)
        .city_with(1, p(5, 5), |c| {
            c.is_capital = true;
            c.level = 2;
            c.population = 2;
            c.workshop = true;
        })
        .stars(0, 5)
        .stars(1, 5)
        .build();
    let mut seen = vec![(s.current, s.turn, stars(&s, 0), stars(&s, 1))];
    for _ in 0..5 {
        ok(&mut s, Action::EndTurn);
        seen.push((s.current, s.turn, stars(&s, 0), stars(&s, 1)));
    }
    assert_eq!(
        seen,
        [
            (0, 0, 5, 5),  // start
            (1, 0, 5, 5),  // P1's first turn: no income (D1)
            (0, 1, 7, 5),  // P0's second turn: +2
            (1, 1, 7, 9),  // P1's second turn: +4
            (0, 2, 9, 9),  // +2
            (1, 2, 9, 13), // +4
        ]
    );
}

#[test]
fn income_from_new_game() {
    // Both capitals are level 1: income 2 each. Start 5 stars each.
    for seed in 0..5 {
        let mut s = new_game_from_seed(seed, &MapParams::default(), &Config::default());
        let mut seen = vec![(stars(&s, 0), stars(&s, 1))];
        for _ in 0..4 {
            ok(&mut s, Action::EndTurn);
            seen.push((stars(&s, 0), stars(&s, 1)));
        }
        assert_eq!(
            seen,
            [(5, 5), (5, 5), (7, 5), (7, 7), (9, 7)],
            "seed {seed}"
        );
    }
}

#[test]
fn income_uses_the_cities_at_turn_start() {
    // P1 to move in round 0; ending it starts P0's turn 1 and pays P0's current cities.
    let mut s = StateBuilder::new(7)
        .city_with(0, p(1, 1), |c| {
            c.level = 3;
            c.population = 5;
            c.workshop = true;
            c.is_capital = true;
        })
        .current(1)
        .build();
    ok(&mut s, Action::EndTurn);
    // 3 + 1 + 1 = 5.
    assert_eq!((s.current, s.turn, stars(&s, 0)), (0, 1, 5));
    assert_eq!(stars(&s, 1), 0, "only the incoming player is paid");
}

// ---------- research ----------

#[test]
fn research_spends_the_cost_and_adds_the_tech() {
    let mut s = StateBuilder::new(5)
        .city(0, p(2, 2))
        .techs(0, &[Organization])
        .stars(0, 6)
        .build();
    ok(&mut s, research(Climbing)); // 4 + 1 x 1 = 5
    assert_eq!(stars(&s, 0), 1);
    assert!(s.players[0].techs.contains(Climbing));
    assert!(s.players[0].techs.contains(Organization));
    assert!(!s.players[1].techs.contains(Climbing));
}

#[test]
fn research_needs_the_prerequisite() {
    let mut s = StateBuilder::new(5)
        .city(0, p(2, 2))
        .techs(0, &[Organization])
        .stars(0, 20)
        .build();
    illegal(&mut s, research(Archery)); // needs Hunting
    illegal(&mut s, research(Forestry)); // needs Hunting
    illegal(&mut s, research(Mining)); // needs Climbing
    ok(&mut s, research(Farming)); // needs Organization: 4 + 1 x 2 = 6
    assert_eq!(stars(&s, 0), 14);
    ok(&mut s, research(Hunting)); // 5
    ok(&mut s, research(Archery)); // 6
    assert_eq!(stars(&s, 0), 3);
}

#[test]
fn research_of_a_known_tech_is_illegal() {
    let mut s = StateBuilder::new(5)
        .city(0, p(2, 2))
        .techs(0, &[Organization, Hunting])
        .stars(0, 20)
        .build();
    illegal(&mut s, research(Organization));
    illegal(&mut s, research(Hunting));
}

#[test]
fn research_needs_enough_stars_exactly() {
    let mut s = StateBuilder::new(5).city(0, p(2, 2)).stars(0, 4).build();
    illegal(&mut s, research(Climbing)); // costs 5
    s.players[0].stars = 5;
    ok(&mut s, research(Climbing));
    assert_eq!(stars(&s, 0), 0);
}

#[test]
fn research_cost_grows_with_cities() {
    // 3 cities: Climbing costs 4 + 3 = 7, Mining 4 + 6 = 10.
    let build = |stars: u32| {
        StateBuilder::new(9)
            .city(0, p(1, 1))
            .city(0, p(1, 7))
            .city(0, p(7, 1))
            .city(1, p(7, 7))
            .techs(0, &[Climbing])
            .stars(0, stars)
            .build()
    };
    let mut s = build(9);
    illegal(&mut s, research(Mining));
    ok(&mut s, research(Riding));
    assert_eq!(stars(&s, 0), 2);
    let mut s = build(10);
    ok(&mut s, research(Mining));
    assert_eq!(stars(&s, 0), 0);
}

#[test]
fn research_actions_listed_in_tech_order() {
    // 1 city. Known: Organization. Tier 1 costs 5, tier 2 costs 6.
    let s = StateBuilder::new(5)
        .city(0, p(2, 2))
        .techs(0, &[Organization])
        .stars(0, 6)
        .build();
    let got: Vec<Action> = legal_actions(&s)
        .into_iter()
        .filter(|a| matches!(a, Action::Research { .. }))
        .collect();
    let want: Vec<Action> = [Climbing, Hunting, Riding, Farming, Strategy]
        .into_iter()
        .map(research)
        .collect();
    assert_eq!(got, want);
    let mut s = s;
    s.players[0].stars = 5;
    let got: Vec<Action> = legal_actions(&s)
        .into_iter()
        .filter(|a| matches!(a, Action::Research { .. }))
        .collect();
    let want: Vec<Action> = [Climbing, Hunting, Riding]
        .into_iter()
        .map(research)
        .collect();
    assert_eq!(got, want);
}

// ---------- train ----------

/// P0 city 0 at (2,2) (level 1), one P0 Warrior from it at (0,0); P1 city 1 at (4,4).
fn barracks(stars: u32, techs: &[Tech]) -> StateBuilder {
    StateBuilder::new(5)
        .city_with(0, p(2, 2), |c| c.is_capital = true)
        .city(1, p(4, 4))
        .unit_with(0, Warrior, p(0, 0), |u| u.origin = Some(CityId(0)))
        .techs(0, techs)
        .stars(0, stars)
        .explore_all()
}

#[test]
fn train_creates_a_done_unit_on_the_city_tile() {
    let mut s = barracks(10, &[]).build();
    s.next_unit_id = 7; // ids keep counting past dead units
    let explored = s.players.clone();
    ok(&mut s, train(0, Warrior));
    assert_eq!(stars(&s, 0), 8, "Warrior costs 2");
    assert_eq!(s.next_unit_id, 8);
    let u = s.units.last().unwrap();
    assert_eq!(
        *u,
        Unit {
            id: UnitId(7),
            kind: Warrior,
            owner: 0,
            pos: p(2, 2),
            health: 10,
            max_health: 10,
            origin: Some(CityId(0)),
            kills: 0,
            veteran: false,
            promotion_available: false,
            phase: Phase::Done,
            last_dir: None,
        }
    );
    assert_eq!(s.tile(p(2, 2)).unit, Some(UnitId(7)));
    assert_eq!(s.players[0].explored, explored[0].explored);
    assert_eq!(s.players[1], explored[1]);
    // Phase Done: the new unit has no actions this turn.
    assert!(legal_actions(&s).iter().all(|a| !matches!(a,
        Action::Move { unit, .. } | Action::Attack { unit, .. } | Action::Recover { unit }
        if *unit == UnitId(7))));
}

#[test]
fn trained_units_get_consecutive_ids_and_stay_sorted() {
    let mut s = StateBuilder::new(7)
        .city_with(0, p(1, 1), |c| c.level = 2)
        .city(0, p(5, 5))
        .stars(0, 10)
        .build();
    s.next_unit_id = 3;
    ok(&mut s, train(1, Warrior));
    ok(&mut s, train(0, Warrior));
    let ids: Vec<(UnitId, Pos)> = s.units.iter().map(|u| (u.id, u.pos)).collect();
    assert_eq!(ids, [(UnitId(3), p(5, 5)), (UnitId(4), p(1, 1))]);
    assert_eq!(s.next_unit_id, 5);
    assert_eq!(stars(&s, 0), 6);
}

#[test]
fn train_needs_an_empty_city_tile() {
    let mut s = barracks(10, &[])
        .unit(0, Warrior, p(2, 2)) // friendly occupant
        .build();
    illegal(&mut s, train(0, Warrior));
    let mut s = barracks(10, &[])
        .unit(1, Warrior, p(2, 2)) // enemy occupant
        .build();
    illegal(&mut s, train(0, Warrior));
}

#[test]
fn train_needs_the_unlocking_tech() {
    let mut s = barracks(10, &[Organization]).build();
    illegal(&mut s, train(0, Rider)); // Riding
    illegal(&mut s, train(0, Archer)); // Archery
    illegal(&mut s, train(0, Defender)); // Strategy
    let mut s = barracks(10, &[Riding]).build();
    ok(&mut s, train(0, Rider));
    assert_eq!(stars(&s, 0), 7, "Rider costs 3");
    let mut s = barracks(10, &[Archery]).build();
    ok(&mut s, train(0, Archer));
    let mut s = barracks(10, &[Strategy]).build();
    ok(&mut s, train(0, Defender));
    assert_eq!(s.units.last().unwrap().health, 15);
}

#[test]
fn super_unit_is_never_trainable() {
    let mut s = barracks(100, &EVERY_TECH).build();
    s.cities[0].level = 9;
    s.cities[0].population = 44; // cum(9) = 44, below cum(10) = 54
    illegal(&mut s, train(0, UnitType::SuperUnit));
}

#[test]
fn train_needs_enough_stars_exactly() {
    let mut s = barracks(1, &[]).build();
    illegal(&mut s, train(0, Warrior));
    let mut s = barracks(2, &[]).build();
    ok(&mut s, train(0, Warrior));
    assert_eq!(stars(&s, 0), 0);
    let mut s = barracks(2, &[Riding]).build();
    illegal(&mut s, train(0, Rider));
}

#[test]
fn train_only_in_own_cities() {
    let mut s = barracks(10, &[]).build();
    illegal(&mut s, train(1, Warrior)); // P1's city
    illegal(&mut s, train(5, Warrior)); // no such city
}

#[test]
fn level_1_city_supports_two_units() {
    // new_game's Warrior (origin = capital) + 1 trained; the third is illegal while both live.
    let mut s = new_game_from_seed(3, &MapParams::default(), &Config::default());
    let cap = s.cities[0].pos;
    let step = legal_actions(&s)
        .into_iter()
        .find(|a| matches!(a, Action::Move { unit, .. } if *unit == UnitId(0)))
        .expect("the starting Warrior can step off the capital");
    ok(&mut s, step);
    ok(&mut s, train(0, Warrior));
    assert_eq!(stars(&s, 0), 3);
    ok(&mut s, Action::EndTurn);
    ok(&mut s, Action::EndTurn);
    // P0 turn 1: 3 + income 2 = 5 stars. Move the trained Warrior off the capital.
    assert_eq!(stars(&s, 0), 5);
    let trained = s.tile(cap).unit.unwrap();
    let step = legal_actions(&s)
        .into_iter()
        .find(|a| matches!(a, Action::Move { unit, .. } if *unit == trained))
        .expect("the trained Warrior can step off the capital");
    ok(&mut s, step);
    assert_eq!(s.tile(cap).unit, None);
    illegal(&mut s, train(0, Warrior));
}

#[test]
fn capacity_is_level_plus_one_and_counts_only_this_citys_units() {
    // Level 1, two units from this city elsewhere: full.
    let full = |level: u32| {
        StateBuilder::new(5)
            .city_with(0, p(2, 2), |c| {
                c.level = level;
                c.population = [0, 0, 2, 5][level as usize];
            })
            .unit_with(0, Warrior, p(0, 0), |u| u.origin = Some(CityId(0)))
            .unit_with(0, Warrior, p(0, 4), |u| u.origin = Some(CityId(0)))
            .stars(0, 10)
            .build()
    };
    illegal(&mut full(1), train(0, Warrior));
    ok(&mut full(2), train(0, Warrior)); // level 2 supports 3
                                         // Orphans and other cities' units do not count.
    let mut s = StateBuilder::new(7)
        .city(0, p(1, 1))
        .city(0, p(5, 5))
        .unit(0, Warrior, p(3, 0))
        .unit(0, Warrior, p(3, 1))
        .unit_with(0, Warrior, p(3, 2), |u| u.origin = Some(CityId(1)))
        .unit_with(0, Warrior, p(3, 3), |u| u.origin = Some(CityId(1)))
        .unit_with(0, Warrior, p(3, 4), |u| u.origin = Some(CityId(0)))
        .stars(0, 10)
        .build();
    ok(&mut s, train(0, Warrior));
    illegal(&mut s, train(1, Warrior));
}

#[test]
fn a_dead_unit_frees_its_capacity() {
    // P0 Warrior (6 hp, from city 0) attacks a full-health P1 Warrior and dies in retaliation
    // (milestone 1 golden row 2). Then city 0 supports 1 unit again.
    let mut s = StateBuilder::new(5)
        .city(0, p(2, 2))
        .unit_with(0, Warrior, p(0, 0), |u| u.origin = Some(CityId(0)))
        .unit_with(0, Warrior, p(0, 3), |u| {
            u.origin = Some(CityId(0));
            u.health = 6;
        })
        .unit(1, Warrior, p(0, 4))
        .stars(0, 10)
        .explore_all()
        .build();
    illegal(&mut s, train(0, Warrior));
    ok(
        &mut s,
        Action::Attack {
            unit: UnitId(1),
            target: p(0, 4),
        },
    );
    assert!(s.unit(UnitId(1)).is_none());
    ok(&mut s, train(0, Warrior));
}

#[test]
fn train_actions_listed_per_city_in_unit_order() {
    let s = StateBuilder::new(7)
        .city(0, p(1, 1))
        .city(1, p(3, 3))
        .city(0, p(5, 5))
        .techs(0, &EVERY_TECH)
        .stars(0, 3)
        .build();
    let got: Vec<Action> = legal_actions(&s)
        .into_iter()
        .filter(|a| matches!(a, Action::Train { .. }))
        .collect();
    let mut want = Vec::new();
    for c in [0, 2] {
        for k in [Warrior, Rider, Archer, Defender] {
            want.push(train(c, k));
        }
    }
    assert_eq!(got, want);
}

// ---------- site actions ----------

/// 5x5, P0 city 0 at (2,2) (level 3, population 5: below cum(4) = 9, so +1 or +2 never levels
/// up), owning rows 1-3 x cols 1-3. P0 has every tech and 20 stars. `t` is (1,1).
fn farmland(setup: impl FnOnce(StateBuilder) -> StateBuilder) -> State {
    let b = StateBuilder::new(5)
        .city_with(0, p(2, 2), |c| {
            c.level = 3;
            c.population = 5;
        })
        .city(1, p(4, 4))
        .techs(0, &EVERY_TECH)
        .stars(0, 20)
        .explore_all();
    setup(b).build()
}

const T: Pos = Pos::new(1, 1);

/// Edits a `farmland` board.
type Setup = fn(StateBuilder) -> StateBuilder;

fn tile_summary(s: &State, q: Pos) -> (Terrain, Option<Resource>, Option<Improvement>) {
    let t = s.tile(q);
    (t.terrain, t.resource, t.improvement)
}

#[test]
fn harvest_animal_golden() {
    let mut s = farmland(|b| b.terrain(T, Terrain::Forest).resource(T, Resource::Animal));
    ok(&mut s, site(T, HarvestAnimal));
    assert_eq!(stars(&s, 0), 18);
    assert_eq!(tile_summary(&s, T), (Terrain::Forest, None, None));
    assert_eq!(city(&s, 0).population, 6);
}

#[test]
fn harvest_fruit_golden() {
    let mut s = farmland(|b| b.resource(T, Resource::Fruit));
    ok(&mut s, site(T, HarvestFruit));
    assert_eq!(stars(&s, 0), 18);
    assert_eq!(tile_summary(&s, T), (Terrain::Field, None, None));
    assert_eq!(city(&s, 0).population, 6);
}

#[test]
fn lumber_hut_golden_keeps_the_forest() {
    let mut s = farmland(|b| b.terrain(T, Terrain::Forest));
    ok(&mut s, site(T, LumberHut));
    assert_eq!(stars(&s, 0), 17);
    assert_eq!(
        tile_summary(&s, T),
        (Terrain::Forest, None, Some(Improvement::LumberHut)),
        "D3: terrain stays Forest"
    );
    assert_eq!(city(&s, 0).population, 6);
}

#[test]
fn lumber_hut_on_an_animal_keeps_the_animal() {
    // Spec: "Lumber hut can be placed on a tile with an animal" (overrules D4).
    let mut s = farmland(|b| b.terrain(T, Terrain::Forest).resource(T, Resource::Animal));
    ok(&mut s, site(T, LumberHut));
    assert_eq!(
        tile_summary(&s, T),
        (
            Terrain::Forest,
            Some(Resource::Animal),
            Some(Improvement::LumberHut)
        )
    );
    assert_eq!(city(&s, 0).population, 6);
    // The animal can still be hunted; the hut stays.
    ok(&mut s, site(T, HarvestAnimal));
    assert_eq!(
        tile_summary(&s, T),
        (Terrain::Forest, None, Some(Improvement::LumberHut))
    );
    assert_eq!(city(&s, 0).population, 7);
    assert_eq!(stars(&s, 0), 15); // 20 - 3 - 2
}

#[test]
fn clear_forest_golden() {
    let mut s = farmland(|b| b.terrain(T, Terrain::Forest));
    ok(&mut s, site(T, ClearForest));
    assert_eq!(stars(&s, 0), 21, "costs 0, gives +1");
    assert_eq!(tile_summary(&s, T), (Terrain::Field, None, None));
    assert_eq!(city(&s, 0).population, 5, "no population");
}

#[test]
fn clear_forest_forfeits_the_animal() {
    // Spec: clear forest on a tile with an animal is allowed and forfeits it (overrules D4).
    let mut s = farmland(|b| b.terrain(T, Terrain::Forest).resource(T, Resource::Animal));
    ok(&mut s, site(T, ClearForest));
    assert_eq!(tile_summary(&s, T), (Terrain::Field, None, None));
    assert_eq!(stars(&s, 0), 21);
    assert_eq!(city(&s, 0).population, 5);
}

#[test]
fn clear_forest_works_with_zero_stars() {
    let mut s = farmland(|b| b.terrain(T, Terrain::Forest).stars(0, 0));
    ok(&mut s, site(T, ClearForest));
    assert_eq!(stars(&s, 0), 1);
}

#[test]
fn farm_golden_keeps_the_crop() {
    // Spec: farm is an improvement and does not consume the crop (overrules the brief's table).
    let mut s = farmland(|b| b.resource(T, Resource::Crop));
    ok(&mut s, site(T, Farm));
    assert_eq!(stars(&s, 0), 15);
    assert_eq!(
        tile_summary(&s, T),
        (
            Terrain::Field,
            Some(Resource::Crop),
            Some(Improvement::Farm)
        )
    );
    assert_eq!(city(&s, 0).population, 7);
}

#[test]
fn mine_golden_keeps_the_metal() {
    // Spec: mine is an improvement and does not consume the metal (overrules the brief's table).
    let mut s = farmland(|b| b.terrain(T, Terrain::Mountain).resource(T, Resource::Metal));
    ok(&mut s, site(T, Mine));
    assert_eq!(stars(&s, 0), 15);
    assert_eq!(
        tile_summary(&s, T),
        (
            Terrain::Mountain,
            Some(Resource::Metal),
            Some(Improvement::Mine)
        )
    );
    assert_eq!(city(&s, 0).population, 7);
}

#[test]
fn site_actions_change_nothing_else() {
    let cases: [(Setup, SiteAction); 3] = [
        (|b| b.resource(T, Resource::Crop), Farm),
        (|b| b.terrain(T, Terrain::Forest), ClearForest),
        (|b| b.resource(T, Resource::Fruit), HarvestFruit),
    ];
    for (setup, a) in cases {
        let mut s = farmland(setup);
        let before = s.clone();
        ok(&mut s, site(T, a));
        assert_eq!(s, exp_after(&before, &site(T, a)), "{a:?}");
    }
}

#[test]
fn friendly_unit_on_the_tile_is_fine_enemy_blocks() {
    // D10.
    let mut s = farmland(|b| b.resource(T, Resource::Fruit).unit(0, Warrior, T));
    ok(&mut s, site(T, HarvestFruit));
    let mut s = farmland(|b| b.resource(T, Resource::Fruit).unit(1, Warrior, T));
    illegal(&mut s, site(T, HarvestFruit));
    // An enemy next to the tile does not block.
    let mut s = farmland(|b| b.resource(T, Resource::Fruit).unit(1, Warrior, p(0, 0)));
    ok(&mut s, site(T, HarvestFruit));
}

#[test]
fn site_actions_need_a_tile_owned_by_the_players_city() {
    // (0,0) is unowned (Chebyshev 2 from the city at (2,2)).
    let mut s = farmland(|b| b.resource(p(0, 0), Resource::Fruit));
    assert_eq!(s.tile(p(0, 0)).owner_city, None, "test bug");
    illegal(&mut s, site(p(0, 0), HarvestFruit));
    // (3,4) belongs to P1's city 1 at (4,4) (city 0 at (2,2) does not reach column 4).
    let mut s = farmland(|b| b.resource(p(3, 4), Resource::Fruit));
    assert_eq!(s.tile(p(3, 4)).owner_city, Some(CityId(1)), "test bug");
    illegal(&mut s, site(p(3, 4), HarvestFruit));
    // Off the map.
    illegal(&mut s, site(p(5, 0), HarvestFruit));
    illegal(&mut s, site(p(255, 255), ClearForest));
}

#[test]
fn each_site_action_needs_its_tech() {
    // (action, tile setup, tech) from the Appendix.
    let cases: [(SiteAction, Setup, Tech); 6] = [
        (
            HarvestAnimal,
            |b| b.terrain(T, Terrain::Forest).resource(T, Resource::Animal),
            Hunting,
        ),
        (
            HarvestFruit,
            |b| b.resource(T, Resource::Fruit),
            Organization,
        ),
        (LumberHut, |b| b.terrain(T, Terrain::Forest), Forestry),
        (ClearForest, |b| b.terrain(T, Terrain::Forest), Forestry),
        (Farm, |b| b.resource(T, Resource::Crop), Farming),
        (
            Mine,
            |b| b.terrain(T, Terrain::Mountain).resource(T, Resource::Metal),
            Mining,
        ),
    ];
    for (a, setup, tech) in cases {
        let others: Vec<Tech> = EVERY_TECH.into_iter().filter(|&t| t != tech).collect();
        let mut s = farmland(|b| setup(b).techs(0, &others));
        illegal(&mut s, site(T, a));
        let mut s = farmland(|b| setup(b).techs(0, &[tech]));
        ok(&mut s, site(T, a));
    }
}

#[test]
fn site_actions_need_their_tile_condition() {
    let none: [(SiteAction, Setup); 10] = [
        (HarvestAnimal, |b| b.terrain(T, Terrain::Forest)),
        (HarvestAnimal, |b| b.resource(T, Resource::Fruit)),
        (HarvestFruit, |b| b.resource(T, Resource::Crop)),
        (HarvestFruit, |b| b),
        (LumberHut, |b| b),
        (LumberHut, |b| b.terrain(T, Terrain::Mountain)),
        (ClearForest, |b| b.resource(T, Resource::Fruit)),
        (Farm, |b| b.resource(T, Resource::Fruit)),
        (Mine, |b| b.terrain(T, Terrain::Mountain)),
        (Mine, |b| b.resource(T, Resource::Crop)),
    ];
    for (a, setup) in none {
        let mut s = farmland(setup);
        illegal(&mut s, site(T, a));
    }
}

#[test]
fn an_improvement_blocks_a_second_one() {
    let mut s = farmland(|b| {
        b.resource(T, Resource::Crop)
            .improvement(T, Improvement::Farm)
    });
    illegal(&mut s, site(T, Farm));
    let mut s = farmland(|b| {
        b.terrain(T, Terrain::Mountain)
            .resource(T, Resource::Metal)
            .improvement(T, Improvement::Mine)
    });
    illegal(&mut s, site(T, Mine));
    let mut s = farmland(|b| {
        b.terrain(T, Terrain::Forest)
            .improvement(T, Improvement::LumberHut)
    });
    illegal(&mut s, site(T, LumberHut));
    illegal(&mut s, site(T, ClearForest));
    // Built through apply: the farm makes the second Farm illegal although the crop remains.
    let mut s = farmland(|b| b.resource(T, Resource::Crop));
    ok(&mut s, site(T, Farm));
    illegal(&mut s, site(T, Farm));
    // Hut first, then ClearForest is illegal.
    let mut s = farmland(|b| b.terrain(T, Terrain::Forest));
    ok(&mut s, site(T, LumberHut));
    illegal(&mut s, site(T, ClearForest));
}

#[test]
fn site_actions_need_enough_stars() {
    let mut s = farmland(|b| b.resource(T, Resource::Crop).stars(0, 4));
    illegal(&mut s, site(T, Farm));
    let mut s = farmland(|b| b.resource(T, Resource::Crop).stars(0, 5));
    ok(&mut s, site(T, Farm));
    assert_eq!(stars(&s, 0), 0);
    let mut s = farmland(|b| b.terrain(T, Terrain::Forest).stars(0, 2));
    illegal(&mut s, site(T, LumberHut));
    let mut s = farmland(|b| b.resource(T, Resource::Fruit).stars(0, 1));
    illegal(&mut s, site(T, HarvestFruit));
}

#[test]
fn population_goes_to_the_city_that_owns_the_tile() {
    // 7x7: city 0 at (1,1) owns rows 0-2 x cols 0-2; city 1 at (1,3) then owns cols 3-4 of
    // rows 0-2 (column 2 was claimed first). (1,2) is next to city 1 but owned by city 0; (2,3)
    // is owned by city 1.
    let mut s = StateBuilder::new(7)
        .city(0, p(1, 1))
        .city(0, p(1, 3))
        .resource(p(1, 2), Resource::Fruit)
        .resource(p(2, 3), Resource::Fruit)
        .techs(0, &[Organization])
        .stars(0, 10)
        .build();
    assert_eq!(s.tile(p(1, 2)).owner_city, Some(CityId(0)), "test bug");
    assert_eq!(s.tile(p(2, 3)).owner_city, Some(CityId(1)), "test bug");
    ok(&mut s, site(p(2, 3), HarvestFruit));
    assert_eq!((city(&s, 0).population, city(&s, 1).population), (0, 1));
    ok(&mut s, site(p(1, 2), HarvestFruit));
    assert_eq!((city(&s, 0).population, city(&s, 1).population), (1, 1));
}

#[test]
fn site_actions_listed_row_major_then_in_action_order() {
    // City 0 at (2,2) owns (1,1)..(3,3). P0 has Hunting, Forestry, Organization, 10 stars.
    // (1,1) forest + animal: HarvestAnimal, LumberHut, ClearForest.
    // (1,3) fruit: HarvestFruit. (3,1) forest: LumberHut, ClearForest. (0,0) unowned fruit: none.
    // (3,3) crop: no Farming, none.
    let s = StateBuilder::new(5)
        .city(0, p(2, 2))
        .terrain(p(1, 1), Terrain::Forest)
        .resource(p(1, 1), Resource::Animal)
        .resource(p(1, 3), Resource::Fruit)
        .terrain(p(3, 1), Terrain::Forest)
        .resource(p(0, 0), Resource::Fruit)
        .resource(p(3, 3), Resource::Crop)
        .techs(0, &[Hunting, Forestry, Organization])
        .stars(0, 10)
        .build();
    let got: Vec<Action> = legal_actions(&s)
        .into_iter()
        .filter(|a| matches!(a, Action::Site { .. }))
        .collect();
    assert_eq!(
        got,
        [
            site(p(1, 1), HarvestAnimal),
            site(p(1, 1), LumberHut),
            site(p(1, 1), ClearForest),
            site(p(1, 3), HarvestFruit),
            site(p(3, 1), LumberHut),
            site(p(3, 1), ClearForest),
        ]
    );
}

// ---------- levels and rewards ----------

#[test]
fn population_two_reaches_level_two_with_a_pending_workshop() {
    let mut s = farmland(|b| b.resource(T, Resource::Fruit));
    s.cities[0].level = 1;
    s.cities[0].population = 0;
    ok(&mut s, site(T, HarvestFruit));
    assert_eq!(
        (city(&s, 0).level, city(&s, 0).pending_reward),
        (1, false),
        "1 < 2"
    );
    tile_mut(&mut s, p(3, 1)).resource = Some(Resource::Fruit);
    ok(&mut s, site(p(3, 1), HarvestFruit));
    let c = city(&s, 0);
    assert_eq!((c.population, c.level, c.pending_reward), (2, 2, true));
    assert_eq!(
        legal_actions(&s),
        [choose(0, Workshop), choose(0, Explorer)]
    );
    ok(&mut s, choose(0, Workshop));
    let c = city(&s, 0);
    assert_eq!(
        (c.level, c.pending_reward, c.workshop, c.population),
        (2, false, true, 2)
    );
    assert!(legal_actions(&s).contains(&Action::EndTurn));
}

#[test]
fn level_thresholds_are_exact() {
    // (level, population) one short of cum(level + 1); +1 crosses it, a second city one further
    // short does not. cum: 2, 5, 9, 14, 20.
    for (level, pop) in [(1, 1), (2, 4), (3, 8), (4, 13), (5, 19)] {
        let mut s = StateBuilder::new(7)
            .city_with(0, p(1, 1), |c| {
                c.level = level;
                c.population = pop;
            })
            .city_with(0, p(5, 5), |c| {
                c.level = level;
                c.population = pop - 1;
            })
            .resource(p(0, 0), Resource::Fruit)
            .resource(p(6, 6), Resource::Fruit)
            .techs(0, &[Organization])
            .stars(0, 10)
            .build();
        ok(&mut s, site(p(6, 6), HarvestFruit));
        let c = city(&s, 1);
        assert_eq!(
            (c.level, c.pending_reward),
            (level, false),
            "level {level}, population {}",
            pop
        );
        ok(&mut s, site(p(0, 0), HarvestFruit));
        let c = city(&s, 0);
        assert_eq!(
            (c.level, c.pending_reward, c.population),
            (level + 1, true, pop + 1),
            "level {level} -> {}",
            level + 1
        );
    }
}

/// One P0 city 0 at (3,3) on a 7x7 board, pending at `level` with `pop`, 10 stars.
fn pending(level: u32, pop: u32) -> State {
    StateBuilder::new(7)
        .city_with(0, p(3, 3), |c| {
            c.level = level;
            c.population = pop;
            c.pending_reward = true;
        })
        .stars(0, 10)
        .explore_all()
        .build()
}

#[test]
fn reward_effects() {
    let check = |level: u32, pop: u32, reward: Reward, edit: fn(&mut City), stars_after: u32| {
        let mut s = pending(level, pop);
        let mut want = city(&s, 0).clone();
        want.pending_reward = false;
        edit(&mut want);
        ok(&mut s, choose(0, reward));
        assert_eq!(*city(&s, 0), want, "{reward:?}");
        assert_eq!(stars(&s, 0), stars_after, "{reward:?}");
    };
    check(2, 2, Workshop, |c| c.workshop = true, 10);
    check(3, 5, CityWall, |c| c.wall = true, 10);
    check(3, 5, Resources, |_| {}, 15); // +5 stars
    check(4, 9, PopulationGrowth, |c| c.population = 12, 10); // +3, 12 < cum(5) = 14
    check(5, 14, Park, |c| c.parks = 1, 10);
    check(
        6,
        20,
        Park,
        |c| c.parks = 1, // 6 x 7 / 2 - 1 = 20
        10,
    );
}

#[test]
fn parks_stack() {
    let mut s = StateBuilder::new(7)
        .city_with(0, p(3, 3), |c| {
            c.level = 6;
            c.population = 20;
            c.parks = 2;
            c.pending_reward = true;
        })
        .build();
    ok(&mut s, choose(0, Park));
    assert_eq!(city(&s, 0).parks, 3);
    assert_eq!(income(&s, 0), 6 + 3);
}

#[test]
fn choose_reward_needs_a_menu_entry_and_a_pending_city() {
    let mut s = pending(3, 5);
    illegal(&mut s, choose(0, Workshop));
    illegal(&mut s, choose(0, Park));
    illegal(&mut s, choose(0, PopulationGrowth));
    illegal(&mut s, choose(1, CityWall)); // no such city
    assert_eq!(
        legal_actions(&s),
        [choose(0, CityWall), choose(0, Resources)]
    );
    let mut s = pending(4, 9);
    illegal(&mut s, choose(0, Park));
    illegal(&mut s, choose(0, Resources));
    illegal(&mut s, choose(0, Explorer));
    illegal(&mut s, choose(0, Reward::SuperUnit));
    let mut s = pending(5, 14);
    illegal(&mut s, choose(0, Workshop));
    illegal(&mut s, choose(0, Explorer));
    assert_eq!(
        legal_actions(&s),
        [choose(0, Park), choose(0, Reward::SuperUnit)]
    );
    // Not pending: nothing to choose.
    let mut s = StateBuilder::new(7)
        .city_with(0, p(3, 3), |c| {
            c.level = 2;
            c.population = 2;
        })
        .build();
    illegal(&mut s, choose(0, Workshop));
}

#[test]
fn choose_reward_only_for_the_current_players_city() {
    // P1's city is pending; P0 to move cannot choose for it, and is not blocked by it.
    let mut s = StateBuilder::new(7)
        .city(0, p(1, 1))
        .city_with(1, p(5, 5), |c| {
            c.level = 2;
            c.population = 2;
            c.pending_reward = true;
        })
        .build();
    illegal(&mut s, choose(1, Workshop));
    assert!(legal_actions(&s).contains(&Action::EndTurn));
}

#[test]
fn border_growth_claims_the_expected_tiles_and_reveals_them() {
    // 7x7. City 0 (P0) at (1,1), level 4, pending. City 1 (P1) at (3,4) owns rows 2-4 x
    // cols 3-5. Village at (3,0). Radius 2 around (1,1) inside the map is rows 0-3 x cols 0-3.
    //
    //      0 1 2 3 4 5 6
    //   0  A A A + . . .      A = city 0's tiles (radius 1)
    //   1  A C A + . . .      B = city 1's tiles, D = city 1
    //   2  A A A B B B .      + = claimed by Border Growth
    //   3  v + + B D B .      v = village (skipped)
    //   4  . . . B B B .
    let mut s = StateBuilder::new(7)
        .city_with(0, p(1, 1), |c| {
            c.level = 4;
            c.population = 9;
            c.pending_reward = true;
        })
        .city(1, p(3, 4))
        .village(p(3, 0))
        .explore(0, &[p(0, 0), p(1, 1)])
        .explore(1, &[p(6, 6)])
        .build();
    let before = s.clone();
    ok(&mut s, choose(0, BorderGrowth));
    let claimed = [p(0, 3), p(1, 3), p(3, 1), p(3, 2)];
    for q in all_pos(7) {
        let want = if claimed.contains(&q) {
            Some(CityId(0))
        } else {
            before.tile(q).owner_city
        };
        assert_eq!(s.tile(q).owner_city, want, "owner of {q:?}");
    }
    assert_eq!(s.tile(p(3, 0)).owner_city, None, "village skipped");
    assert_eq!(
        s.tile(p(2, 3)).owner_city,
        Some(CityId(1)),
        "owned tile kept"
    );
    let mut want_explored = vec![p(0, 0), p(1, 1)];
    want_explored.extend(claimed);
    want_explored.sort();
    assert_eq!(explored_set(&s, 0), want_explored);
    assert_eq!(s.players[1], before.players[1]);
    let c = city(&s, 0);
    assert_eq!(
        (
            c.border_growth_used,
            c.pending_reward,
            c.level,
            c.population
        ),
        (true, false, 4, 9)
    );
}

#[test]
fn border_growth_is_legal_when_nothing_is_left_to_claim() {
    // 3x3 board: the city already owns every tile (D7).
    let mut s = StateBuilder::new(3)
        .city_with(0, p(1, 1), |c| {
            c.level = 4;
            c.population = 9;
            c.pending_reward = true;
        })
        .build();
    let tiles = s.tiles.clone();
    ok(&mut s, choose(0, BorderGrowth));
    assert_eq!(s.tiles, tiles);
    assert!(city(&s, 0).border_growth_used);
}

#[test]
fn border_growth_only_once_per_city() {
    let mut s = pending(4, 9);
    s.cities[0].border_growth_used = true;
    assert_eq!(legal_actions(&s), [choose(0, PopulationGrowth)]);
    illegal(&mut s, choose(0, BorderGrowth));
    // Through play: take it at level 4, then a later level-4 pending (synthetic) lacks it.
    let mut s = pending(4, 9);
    ok(&mut s, choose(0, BorderGrowth));
    s.cities[0].pending_reward = true;
    assert_eq!(reward_menu(&s, CityId(0)), [PopulationGrowth]);
    illegal(&mut s, choose(0, BorderGrowth));
}

#[test]
fn cascade_golden() {
    // Level 4, pending, population 12. PopulationGrowth: 15 >= cum(5) = 14 -> level 5, pending
    // again with the Park menu. Park: 15 < cum(6) = 20 -> done.
    let mut s = pending(4, 12);
    ok(&mut s, choose(0, PopulationGrowth));
    let c = city(&s, 0);
    assert_eq!((c.level, c.population, c.pending_reward), (5, 15, true));
    assert_eq!(
        legal_actions(&s),
        [choose(0, Park), choose(0, Reward::SuperUnit)]
    );
    ok(&mut s, choose(0, Park));
    let c = city(&s, 0);
    assert_eq!(
        (c.level, c.population, c.pending_reward, c.parks),
        (5, 15, false, 1)
    );
}

#[test]
fn only_one_level_per_check() {
    // Synthetic: level 4, pending, population 17; PopulationGrowth makes it 20 (>= cum(6) = 20). Each resolution raises the
    // level by exactly one: 4 -> 5 (pending), then 5 -> 6 (pending), then 20 < cum(7) = 27.
    let mut s = pending(4, 17);
    ok(&mut s, choose(0, PopulationGrowth)); // 20
    assert_eq!((city(&s, 0).level, city(&s, 0).pending_reward), (5, true));
    ok(&mut s, choose(0, Park));
    assert_eq!((city(&s, 0).level, city(&s, 0).pending_reward), (6, true));
    ok(&mut s, choose(0, Park));
    let c = city(&s, 0);
    assert_eq!((c.level, c.pending_reward, c.parks), (6, false, 2));
}

#[test]
fn a_farm_can_level_a_city_up() {
    // Level 1, population 0: Farm (+2) reaches cum(2) = 2.
    let mut s = farmland(|b| b.resource(T, Resource::Crop));
    s.cities[0].level = 1;
    s.cities[0].population = 0;
    ok(&mut s, site(T, Farm));
    let c = city(&s, 0);
    assert_eq!((c.level, c.population, c.pending_reward), (2, 2, true));
}

#[test]
fn a_pending_reward_blocks_every_other_action() {
    // P0: city 0 pending at level 2; a Fresh Warrior with a promotion available; stars, techs, a
    // fruit tile and a free city tile so Train, Site and Research would otherwise be legal.
    let mut s = StateBuilder::new(5)
        .city_with(0, p(2, 2), |c| {
            c.level = 2;
            c.population = 2;
            c.pending_reward = true;
        })
        .unit_with(0, Warrior, p(0, 0), |u| {
            u.kills = 3;
            u.promotion_available = true;
        })
        .unit(1, Warrior, p(0, 1))
        .resource(p(1, 1), Resource::Fruit)
        .techs(0, &[Organization])
        .stars(0, 20)
        .explore_all()
        .build();
    assert_eq!(
        legal_actions(&s),
        [choose(0, Workshop), choose(0, Explorer)]
    );
    for a in [
        Action::Move {
            unit: UnitId(0),
            to: p(1, 0),
        },
        Action::Attack {
            unit: UnitId(0),
            target: p(0, 1),
        },
        Action::Recover { unit: UnitId(0) },
        Action::Promote { unit: UnitId(0) },
        train(0, Warrior),
        site(p(1, 1), HarvestFruit),
        research(Climbing),
        Action::EndTurn,
    ] {
        illegal(&mut s, a);
    }
    ok(&mut s, choose(0, Workshop));
    // Unblocked: everything listed above is legal again.
    let legal = legal_actions(&s);
    assert!(legal.contains(&train(0, Warrior)));
    assert!(legal.contains(&site(p(1, 1), HarvestFruit)));
    assert!(legal.contains(&research(Climbing)));
    assert!(legal.contains(&Action::Promote { unit: UnitId(0) }));
    assert!(legal.contains(&Action::EndTurn));
}

#[test]
fn several_pending_cities_resolve_by_ascending_id() {
    // City 0 not pending; cities 1 (level 3) and 2 (level 2) pending; city 3 is P1's.
    let mut s = StateBuilder::new(9)
        .city(0, p(1, 1))
        .city_with(0, p(1, 7), |c| {
            c.level = 3;
            c.population = 5;
            c.pending_reward = true;
        })
        .city_with(0, p(7, 1), |c| {
            c.level = 2;
            c.population = 2;
            c.pending_reward = true;
        })
        .city(1, p(7, 7))
        .build();
    assert_eq!(
        legal_actions(&s),
        [choose(1, CityWall), choose(1, Resources)]
    );
    illegal(&mut s, choose(2, Workshop));
    ok(&mut s, choose(1, Resources));
    assert_eq!(stars(&s, 0), 5);
    assert_eq!(
        legal_actions(&s),
        [choose(2, Workshop), choose(2, Explorer)]
    );
    ok(&mut s, choose(2, Workshop));
    assert!(legal_actions(&s).contains(&Action::EndTurn));
}

#[test]
fn choose_reward_changes_nothing_else() {
    // Explorer and SuperUnit are replayed by `common::exp_after` (independent walk and push).
    let cases: [(u32, u32, Reward); 8] = [
        (2, 3, Workshop),
        (2, 4, Explorer),
        (5, 15, Reward::SuperUnit),
        (3, 6, CityWall),
        (3, 8, Resources),
        (4, 11, PopulationGrowth),
        (4, 9, BorderGrowth),
        (5, 18, Park),
    ];
    for (level, pop, reward) in cases {
        let mut s = pending(level, pop);
        let before = s.clone();
        ok(&mut s, choose(0, reward));
        assert_eq!(s, exp_after(&before, &choose(0, reward)), "{reward:?}");
    }
}

// ---------- legal_actions order ----------

#[test]
fn legal_actions_full_canonical_order() {
    // 5x5. City 0 (P0, capital) at (1,1) owns rows 0-2 x cols 0-2; city 1 (P0) at (3,3) owns the
    // rest of rows 2-4 x cols 2-4. Unit 0 (P0 Warrior from city 0, Done, promotion available)
    // at (0,0): only Promote. Unit 1 (P1) at (4,0).
    // P0: Organization + Hunting, 6 stars. Two cities: tier 1 costs 6, tier 2 costs 8.
    // Train: Warrior (2) in both cities (Rider, Archer, Defender locked).
    // Sites: (0,2) fruit; (2,0) forest + animal (no Forestry: hunt only); (4,4) fruit;
    // (2,2) crop (no Farming: none); (4,0) fruit under P1's unit, unowned anyway.
    let s = StateBuilder::new(5)
        .city_with(0, p(1, 1), |c| c.is_capital = true)
        .city(0, p(3, 3))
        .unit_with(0, Warrior, p(0, 0), |u| {
            u.origin = Some(CityId(0));
            u.phase = Phase::Done;
            u.kills = 3;
            u.promotion_available = true;
        })
        .unit(1, Warrior, p(4, 0))
        .resource(p(0, 2), Resource::Fruit)
        .terrain(p(2, 0), Terrain::Forest)
        .resource(p(2, 0), Resource::Animal)
        .resource(p(4, 4), Resource::Fruit)
        .resource(p(2, 2), Resource::Crop)
        .resource(p(4, 0), Resource::Fruit)
        .techs(0, &[Organization, Hunting])
        .stars(0, 6)
        .explore_all()
        .build();
    let want = vec![
        Action::Promote { unit: UnitId(0) },
        train(0, Warrior),
        train(1, Warrior),
        site(p(0, 2), HarvestFruit),
        site(p(2, 0), HarvestAnimal),
        site(p(4, 4), HarvestFruit),
        research(Climbing),
        research(Riding),
        Action::EndTurn,
    ];
    assert_eq!(bf_legal(&s), want, "test bug: brute force disagrees");
    assert_eq!(legal_actions(&s), want);
}

#[test]
fn economy_actions_only_for_the_current_player() {
    // P1 to move; P0's city, tiles and stars offer nothing to P1.
    let mut s = StateBuilder::new(5)
        .city(0, p(1, 1))
        .city(1, p(3, 3))
        .resource(p(0, 0), Resource::Fruit)
        .techs(0, &EVERY_TECH)
        .stars(0, 50)
        .current(1)
        .build();
    illegal(&mut s, train(0, Warrior));
    illegal(&mut s, site(p(0, 0), HarvestFruit));
    illegal(&mut s, research(Climbing)); // P1 has 0 stars
    assert_eq!(legal_actions(&s), [Action::EndTurn]);
}
