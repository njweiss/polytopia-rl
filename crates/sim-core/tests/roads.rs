//! Milestone 16 "Tests" items 1 to 3 and 7: `road_usable`, `BuildRoad` legality and apply under
//! v2, `BuildRoad` never legal under v1, and `StateBuilder::config` swapping the tables
//! (docs/MILESTONE_16 "Road tiles", "BuildRoad legality", "BuildRoad apply").
//!
//! Every expectation is replayed from the brief in this file (`exp_road_legal`, `exp_usable`);
//! nothing here calls `road_usable`, `legal_actions` or `apply` to compute an expectation.
//! Numbers from the brief: `road_cost` 3, `road_tech` Roads.

mod common;

use std::sync::Arc;

use common::{all_pos, idx, p, pick, random_board, tile_mut};
use sim_core::actions::{apply, legal_actions, Action, ActionError};
use sim_core::data::{SiteAction, Tables, Tech, UnitType};
use sim_core::grid::Pos;
use sim_core::map::{MapParams, Resource, Terrain};
use sim_core::rng::Rng;
use sim_core::roads::road_usable;
use sim_core::setup::new_game_from_seed;
use sim_core::state::{Config, Improvement, PlayerId, Ruleset, State};
use sim_core::testing::StateBuilder;

const ROAD_COST: u32 = 3;

fn v2_config() -> Config {
    Config {
        ruleset: Ruleset::V2,
        ..Config::default()
    }
}

/// The builder's board under v2 (`StateBuilder::config` also sets the v2 tables).
fn v2(b: StateBuilder) -> State {
    b.config(v2_config()).build()
}

/// 7x7, everything explored, player 0 to move with Riding and Roads and `stars` stars.
/// Player 0's capital at (1,1) owns rows 0-2 x cols 0-2; player 1's capital at (5,5) owns rows
/// 4-6 x cols 4-6; every other tile is neutral.
fn base(stars: u32) -> StateBuilder {
    StateBuilder::new(7)
        .city_with(0, p(1, 1), |c| c.is_capital = true)
        .city_with(1, p(5, 5), |c| c.is_capital = true)
        .techs(0, &[Tech::Riding, Tech::Roads])
        .stars(0, stars)
        .explore_all()
}

fn road(at: Pos) -> Action {
    Action::BuildRoad { at }
}

fn roads_in(legal: &[Action]) -> Vec<Pos> {
    legal
        .iter()
        .filter_map(|a| match *a {
            Action::BuildRoad { at } => Some(at),
            _ => None,
        })
        .collect()
}

fn assert_illegal(s: &State, a: Action) {
    let mut t = s.clone();
    assert_eq!(apply(&mut t, &a), Err(ActionError::Illegal), "{a:?}");
    assert_eq!(&t, s, "{a:?}: a refused action leaves the state unchanged");
}

/// `a` is listed by `legal_actions` and `apply` accepts it.
fn assert_legal(s: &State, a: Action) {
    assert!(legal_actions(s).contains(&a), "{a:?} not listed");
    let mut t = s.clone();
    assert_eq!(apply(&mut t, &a), Ok(()), "{a:?}");
}

fn assert_refused(s: &State, at: Pos) {
    assert!(!legal_actions(s).contains(&road(at)), "{at:?} listed");
    assert_illegal(s, road(at));
}

fn owner_of_tile(s: &State, q: Pos) -> Option<PlayerId> {
    s.tile(q)
        .owner_city
        .map(|c| s.cities.iter().find(|x| x.id == c).unwrap().owner)
}

/// The brief's "Usable": a road or a city, and not in the other player's territory.
fn exp_usable(s: &State, player: PlayerId, q: Pos) -> bool {
    let t = s.tile(q);
    (t.road || t.city.is_some()) && owner_of_tile(s, q).is_none_or(|o| o == player)
}

/// The brief's BuildRoad legality, pending-reward block and outcome included.
fn exp_road_legal(s: &State, q: Pos) -> bool {
    let me = s.current;
    if s.config.ruleset != Ruleset::V2
        || s.outcome.is_some()
        || s.cities.iter().any(|c| c.owner == me && c.pending_reward)
    {
        return false;
    }
    let pl = &s.players[me as usize];
    if !pl.techs.contains(Tech::Roads) || pl.stars < ROAD_COST {
        return false;
    }
    if q.row >= s.size || q.col >= s.size {
        return false;
    }
    let t = s.tile(q);
    if t.road || t.city.is_some() || t.terrain == Terrain::Mountain {
        return false;
    }
    let occupant = s.units.iter().find(|u| u.pos == q);
    match owner_of_tile(s, q) {
        Some(o) if o == me => occupant.is_none_or(|u| u.owner == me),
        Some(_) => false,
        None => s.players[me as usize].explored[idx(s, q)] && occupant.is_none(),
    }
}

// ---------------------------------------------------------------- road_usable

#[test]
fn road_usable_follows_roads_cities_and_territory() {
    let mut s = v2(base(0));
    for q in [p(0, 0), p(3, 3), p(6, 6)] {
        tile_mut(&mut s, q).road = true;
    }
    // (tile, usable for player 0, usable for player 1)
    let cases = [
        (p(0, 0), true, false),  // road in player 0's territory
        (p(3, 3), true, true),   // neutral road
        (p(6, 6), false, true),  // road in player 1's territory
        (p(1, 1), true, false),  // player 0's city (R2)
        (p(5, 5), false, true),  // player 1's city
        (p(3, 4), false, false), // neutral, no road
        (p(0, 1), false, false), // own territory, no road
        (p(6, 5), false, false), // enemy territory, no road
    ];
    for (q, u0, u1) in cases {
        assert_eq!(exp_usable(&s, 0, q), u0, "oracle self-check {q:?}");
        assert_eq!(road_usable(&s, 0, q), u0, "player 0 at {q:?}");
        assert_eq!(road_usable(&s, 1, q), u1, "player 1 at {q:?}");
    }
}

#[test]
fn road_usable_matches_the_rule_on_random_boards() {
    for seed in 0..60u64 {
        let mut s = random_board(seed);
        s.config.ruleset = Ruleset::V2;
        s.tables = Arc::new(Tables::load_for(Ruleset::V2).unwrap());
        let mut rng = Rng::new(seed ^ 0x16);
        for q in all_pos(s.size) {
            if s.tile(q).terrain != Terrain::Mountain && rng.gen_range(3) == 0 {
                tile_mut(&mut s, q).road = true;
            }
        }
        for q in all_pos(s.size) {
            for player in 0..2u8 {
                assert_eq!(
                    road_usable(&s, player, q),
                    exp_usable(&s, player, q),
                    "seed {seed}, player {player}, {q:?}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------- legality

#[test]
fn build_road_needs_the_roads_tech() {
    let without = v2(base(10).techs(0, &[Tech::Riding]));
    assert!(roads_in(&legal_actions(&without)).is_empty());
    assert_illegal(&without, road(p(3, 3)));
    assert_illegal(&without, road(p(0, 0)));

    let with = v2(base(10));
    assert_legal(&with, road(p(3, 3)));
    assert_legal(&with, road(p(0, 0)));
}

#[test]
fn build_road_needs_three_stars() {
    let poor = v2(base(ROAD_COST - 1));
    assert!(roads_in(&legal_actions(&poor)).is_empty());
    assert_illegal(&poor, road(p(3, 3)));

    let mut exact = v2(base(ROAD_COST));
    assert_legal(&exact, road(p(3, 3)));
    apply(&mut exact, &road(p(3, 3))).unwrap();
    assert_eq!(exact.players[0].stars, 0);
}

#[test]
fn no_road_on_a_mountain() {
    let s = v2(base(10)
        .terrain(p(0, 0), Terrain::Mountain)
        .terrain(p(3, 3), Terrain::Mountain)
        .techs(0, &[Tech::Riding, Tech::Roads, Tech::Climbing]));
    assert_refused(&s, p(0, 0));
    assert_refused(&s, p(3, 3));
    assert_legal(&s, road(p(0, 1)));
}

#[test]
fn no_road_on_a_city_tile() {
    // The own capital (empty tile, own territory) and an own non-capital city in neutral land.
    let s = v2(base(10).city(0, p(3, 1)));
    assert_eq!(s.tile(p(1, 1)).unit, None);
    assert_refused(&s, p(1, 1));
    assert_refused(&s, p(3, 1));
    assert_refused(&s, p(5, 5));
}

#[test]
fn no_second_road_on_a_tile() {
    let mut s = v2(base(10));
    tile_mut(&mut s, p(3, 3)).road = true;
    tile_mut(&mut s, p(0, 0)).road = true;
    assert_refused(&s, p(3, 3));
    assert_refused(&s, p(0, 0));

    // And after building one, the same tile is refused.
    let mut t = v2(base(10));
    apply(&mut t, &road(p(3, 2))).unwrap();
    assert_refused(&t, p(3, 2));
}

#[test]
fn own_territory_allows_an_own_unit_but_not_an_enemy_unit() {
    let s = v2(base(10)
        .unit(0, UnitType::Warrior, p(0, 0))
        .unit(1, UnitType::Warrior, p(0, 2))
        .hide(0, &[p(2, 2)]));
    assert_legal(&s, road(p(0, 0)));
    assert_refused(&s, p(0, 2));
    // The exploration condition is the neutral rule's only: own territory follows the site rule.
    assert_legal(&s, road(p(2, 2)));
}

#[test]
fn neutral_tiles_need_exploration_and_no_unit_at_all() {
    let s = v2(base(10)
        .hide(0, &[p(3, 0)])
        .unit(0, UnitType::Warrior, p(3, 3))
        .unit(1, UnitType::Warrior, p(3, 5)));
    assert_legal(&s, road(p(3, 2))); // explored, empty
    assert_refused(&s, p(3, 0)); // unexplored
    assert_refused(&s, p(3, 3)); // own unit
    assert_refused(&s, p(3, 5)); // enemy unit
}

#[test]
fn a_neutral_village_tile_is_allowed() {
    let s = v2(base(10).village(p(3, 3)));
    assert_eq!(s.tile(p(3, 3)).owner_city, None);
    assert_legal(&s, road(p(3, 3)));
    // A village with a unit on it is not.
    let t = v2(base(10)
        .village(p(3, 3))
        .unit(0, UnitType::Warrior, p(3, 3)));
    assert_refused(&t, p(3, 3));
}

#[test]
fn enemy_territory_is_never_allowed() {
    let s = v2(base(10).explore_all());
    for q in [p(4, 4), p(6, 6), p(4, 6)] {
        assert_eq!(owner_of_tile(&s, q), Some(1));
        assert_refused(&s, q);
    }
    // Not even for an own unit standing there.
    let t = v2(base(10).unit(0, UnitType::Warrior, p(4, 4)));
    assert_refused(&t, p(4, 4));
}

#[test]
fn build_road_is_blocked_while_a_reward_is_pending() {
    let s = v2(StateBuilder::new(7)
        .city_with(0, p(1, 1), |c| {
            c.is_capital = true;
            c.level = 2;
            c.population = 2;
            c.pending_reward = true;
        })
        .city_with(1, p(5, 5), |c| c.is_capital = true)
        .techs(0, &[Tech::Riding, Tech::Roads])
        .stars(0, 10)
        .explore_all());
    let legal = legal_actions(&s);
    assert!(legal
        .iter()
        .all(|a| matches!(a, Action::ChooseReward { .. })));
    assert_illegal(&s, road(p(3, 3)));
    assert_illegal(&s, road(p(0, 0)));
    // Once the reward is chosen, the same roads are legal.
    let mut t = s.clone();
    apply(
        &mut t,
        &Action::ChooseReward {
            city: sim_core::state::CityId(0),
            reward: sim_core::data::Reward::Workshop,
        },
    )
    .unwrap();
    assert_legal(&t, road(p(3, 3)));
    assert_legal(&t, road(p(0, 0)));
}

#[test]
fn build_road_comes_after_site_actions_and_before_research_in_row_major_order() {
    let s = v2(base(20)
        .terrain(p(0, 1), Terrain::Forest)
        .resource(p(2, 2), Resource::Fruit)
        .techs(
            0,
            &[Tech::Riding, Tech::Roads, Tech::Organization, Tech::Hunting],
        )
        .unit(0, UnitType::Warrior, p(2, 0)));
    let legal = legal_actions(&s);
    let pos = |f: &dyn Fn(&Action) -> bool| -> Vec<usize> {
        legal
            .iter()
            .enumerate()
            .filter(|(_, a)| f(a))
            .map(|(i, _)| i)
            .collect()
    };
    let sites = pos(&|a| matches!(a, Action::Site { .. }));
    let roads = pos(&|a| matches!(a, Action::BuildRoad { .. }));
    let research = pos(&|a| matches!(a, Action::Research { .. }));
    let trains = pos(&|a| matches!(a, Action::Train { .. }));
    let units = pos(&|a| matches!(a, Action::Move { .. } | Action::Recover { .. }));
    assert!(!sites.is_empty() && !research.is_empty() && !trains.is_empty() && !units.is_empty());
    assert!(!roads.is_empty());
    // One contiguous block between the site actions and research.
    assert_eq!(roads.last().unwrap() - roads[0] + 1, roads.len());
    assert!(sites.last().unwrap() < &roads[0]);
    assert!(roads.last().unwrap() < &research[0]);
    assert!(trains.last().unwrap() < &roads[0] && units.last().unwrap() < &roads[0]);
    assert_eq!(legal.last(), Some(&Action::EndTurn));
    // Exactly the rule's tiles, row-major.
    let expected: Vec<Pos> = all_pos(7)
        .into_iter()
        .filter(|&q| exp_road_legal(&s, q))
        .collect();
    assert_eq!(roads_in(&legal), expected);
    // Spot checks: own forest and the own unit's tile are in; the city and enemy land are not.
    assert!(expected.contains(&p(0, 1)) && expected.contains(&p(2, 0)));
    assert!(!expected.contains(&p(1, 1)) && !expected.contains(&p(4, 4)));
}

#[test]
fn build_road_list_matches_the_rule_on_random_v2_boards() {
    let mut listed = 0;
    for seed in 0..120u64 {
        let mut s = random_board(seed);
        s.config.ruleset = Ruleset::V2;
        s.tables = Arc::new(Tables::load_for(Ruleset::V2).unwrap());
        let mut rng = Rng::new(seed ^ 0x16_0001);
        let me = s.current as usize;
        if rng.gen_range(4) != 0 {
            s.players[me].techs.insert(Tech::Roads);
        }
        s.players[me].stars = [0, 2, 3, 4, 9][rng.gen_range(5) as usize];
        for q in all_pos(s.size) {
            if s.tile(q).terrain != Terrain::Mountain
                && s.tile(q).city.is_none()
                && rng.gen_range(5) == 0
            {
                tile_mut(&mut s, q).road = true;
            }
        }
        let expected: Vec<Pos> = all_pos(s.size)
            .into_iter()
            .filter(|&q| exp_road_legal(&s, q))
            .collect();
        let legal = legal_actions(&s);
        assert_eq!(roads_in(&legal), expected, "seed {seed}");
        listed += expected.len();
        // `apply` agrees tile by tile (an out-of-bounds tile too).
        for q in all_pos(s.size + 1) {
            let mut t = s.clone();
            let ok = apply(&mut t, &road(q)).is_ok();
            assert_eq!(ok, expected.contains(&q), "seed {seed}: apply at {q:?}");
            if !ok {
                assert_eq!(t, s, "seed {seed}: refused at {q:?}");
            }
        }
    }
    assert!(listed > 200, "only {listed} legal roads over all seeds");
}

// ---------------------------------------------------------------- apply

#[test]
fn apply_costs_three_stars_and_sets_only_the_road() {
    let s = v2(base(10)
        .terrain(p(0, 1), Terrain::Forest)
        .resource(p(0, 1), Resource::Animal)
        .improvement(p(0, 1), Improvement::LumberHut)
        .resource(p(2, 2), Resource::Metal)
        .improvement(p(2, 2), Improvement::Mine)
        .village(p(3, 3))
        .unit(0, UnitType::Warrior, p(2, 2)));
    for q in [p(0, 1), p(2, 2), p(3, 3), p(3, 0)] {
        let mut after = s.clone();
        apply(&mut after, &road(q)).unwrap();
        let mut expected = s.clone();
        expected.players[0].stars -= ROAD_COST;
        tile_mut(&mut expected, q).road = true;
        assert_eq!(after, expected, "{q:?}");
        assert_eq!(after.tile(q).terrain, s.tile(q).terrain);
        assert_eq!(after.tile(q).resource, s.tile(q).resource);
        assert_eq!(after.tile(q).improvement, s.tile(q).improvement);
        assert_eq!(after.tile(q).village, s.tile(q).village);
    }
}

// ---------------------------------------------------------------- v1

#[test]
fn under_v1_build_road_is_never_legal() {
    // The v2 board's every legal road tile, under v1 with Roads in the tech set anyway.
    let v2_state = v2(base(10).village(p(3, 3)));
    let legal_v2 = roads_in(&legal_actions(&v2_state));
    assert!(!legal_v2.is_empty());
    let v1 = base(10).village(p(3, 3)).build();
    assert_eq!(v1.config.ruleset, Ruleset::V1);
    assert!(roads_in(&legal_actions(&v1)).is_empty());
    for q in all_pos(7) {
        assert_illegal(&v1, road(q));
    }
}

#[test]
fn random_v1_playouts_never_see_a_road() {
    for seed in 0..10u64 {
        let cfg = Config {
            turn_cap: 12,
            ..Config::default()
        };
        let mut s = new_game_from_seed(seed, &MapParams::default(), &cfg);
        // Roads in the tech set and stars to spare: still nothing under v1.
        for pl in &mut s.players {
            pl.techs.insert(Tech::Roads);
            pl.stars += 20;
        }
        let mut rng = Rng::new(seed ^ 0x16_0002);
        let mut steps = 0;
        while !s.is_terminal() && steps < 2000 {
            let legal = legal_actions(&s);
            assert!(roads_in(&legal).is_empty(), "seed {seed}");
            if steps % 29 == 0 {
                let q = Pos::new(rng.gen_range(11) as u8, rng.gen_range(11) as u8);
                assert_illegal(&s, road(q));
            }
            let a = pick(&mut rng, &legal);
            apply(&mut s, &a).unwrap();
            assert!(s.tiles.iter().all(|t| !t.road), "seed {seed}");
            assert!(
                s.cities.iter().all(|c| c.connected_to.is_none()),
                "seed {seed}"
            );
            steps += 1;
        }
    }
}

// ---------------------------------------------------------------- StateBuilder

#[test]
fn state_builder_config_sets_the_tables_of_the_ruleset() {
    let s = StateBuilder::new(5).config(v2_config()).build();
    assert_eq!(s.tables.ruleset(), Ruleset::V2);
    // A v2 tech's cost and the v2 scalars are readable.
    assert_eq!(s.tables.tech(Tech::Roads).tier, 2);
    assert_eq!(s.tables.v2_rules().road_cost, 3);
    // And back to v1.
    let t = StateBuilder::new(5)
        .config(v2_config())
        .config(Config::default())
        .build();
    assert_eq!(t.tables.ruleset(), Ruleset::V1);
    assert_eq!(*t.tables, Tables::load().unwrap());
    // The default builder keeps the v1 tables.
    assert_eq!(StateBuilder::new(5).build().tables.ruleset(), Ruleset::V1);
    // A v2 site action's row is there too.
    assert_eq!(s.tables.site_action(SiteAction::BurnForest).cost, 3);
}
