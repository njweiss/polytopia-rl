//! Milestone 17 tests (items 1-6 and 8): v1 identity for M17 actions, `BurnForest`, economy
//! buildings (`Sawmill`, `Windmill`, `Forge`, `Market`), `Disband`, `Destroy`, signed population &
//! score surplus floor, canonical `legal_actions` order, and `StateBuilder::building`.

mod common;

use common::{p, pick, tile_mut};
use sim_core::actions::{apply, legal_actions, Action, ActionError};
use sim_core::buildings::{building_level, market_income};
use sim_core::data::{Building, Reward, SiteAction, Tech, UnitType, ALL_BUILDINGS, ALL_TECHS};
use sim_core::economy::income;
use sim_core::grid::Pos;
use sim_core::map::{MapParams, Resource, Terrain};
use sim_core::rng::Rng;
use sim_core::score::{score, score_breakdown};
use sim_core::setup::new_game_from_seed;
use sim_core::state::{City, CityId, Config, Improvement, Outcome, Phase, Ruleset, State, UnitId};
use sim_core::testing::StateBuilder;

fn v2_config() -> Config {
    Config {
        ruleset: Ruleset::V2,
        ..Config::default()
    }
}

fn v2(b: StateBuilder) -> State {
    b.config(v2_config()).build()
}

fn city(s: &State, id: u32) -> &City {
    s.city(CityId(id))
}

fn assert_illegal(s: &State, a: Action) {
    assert!(!legal_actions(s).contains(&a), "{a:?} should not be legal");
    let mut t = s.clone();
    assert_eq!(apply(&mut t, &a), Err(ActionError::Illegal), "{a:?}");
    assert_eq!(t, *s, "{a:?}: illegal action mutated state");
}

fn ok(s: &mut State, a: Action) {
    assert!(
        legal_actions(s).contains(&a),
        "{a:?} should be legal; legal = {:?}",
        legal_actions(s)
    );
    assert_eq!(apply(s, &a), Ok(()), "{a:?}");
}

fn build(at: Pos, building: Building) -> Action {
    Action::BuildBuilding { at, building }
}

fn disband(unit: u32) -> Action {
    Action::Disband { unit: UnitId(unit) }
}

fn destroy(at: Pos) -> Action {
    Action::Destroy { at }
}

fn burn(at: Pos) -> Action {
    Action::Site {
        at,
        action: SiteAction::BurnForest,
    }
}

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

// ============================================================================
// 1. v1 identity
// ============================================================================

#[test]
fn under_v1_m17_actions_are_never_legal_or_applicable() {
    // Rich v1 board with forests, fields next to huts/farms/mines/buildings, units, and stars.
    let mut s = StateBuilder::new(5)
        .city_with(0, p(2, 2), |c| c.is_capital = true)
        .city_with(1, p(4, 4), |c| c.is_capital = true)
        .terrain(p(1, 1), Terrain::Forest)
        .improvement(p(1, 1), Improvement::LumberHut)
        .terrain(p(1, 2), Terrain::Forest)
        .resource(p(2, 1), Resource::Crop)
        .improvement(p(2, 1), Improvement::Farm)
        .unit(0, UnitType::Warrior, p(2, 2))
        .techs(0, &ALL_TECHS)
        .stars(0, 50)
        .explore_all()
        .build();
    // Even if v2 techs are synthetically inserted into a V1 state, M17 actions stay illegal.
    for t in [
        Tech::Construction,
        Tech::FreeSpirit,
        Tech::Chivalry,
        Tech::Mathematics,
        Tech::Smithery,
        Tech::Trade,
    ] {
        s.players[0].techs.insert(t);
    }
    assert_eq!(s.config.ruleset, Ruleset::V1);
    let legal = legal_actions(&s);
    assert!(legal.iter().all(|a| !matches!(
        a,
        Action::Site {
            action: SiteAction::BurnForest,
            ..
        } | Action::BuildBuilding { .. }
            | Action::Disband { .. }
            | Action::Destroy { .. }
    )));

    assert_illegal(&s, burn(p(1, 2)));
    for b in ALL_BUILDINGS {
        assert_illegal(&s, build(p(1, 3), b));
        assert_illegal(&s, build(p(2, 3), b));
    }
    assert_illegal(&s, disband(0));
    assert_illegal(&s, destroy(p(1, 1)));
    assert_illegal(&s, destroy(p(2, 1)));
}

#[test]
fn random_v1_playouts_never_emit_or_accept_m17_actions() {
    for seed in 0..8u64 {
        let mut s = new_game_from_seed(seed, &MapParams::default(), &Config::default());
        for pl in &mut s.players {
            for t in ALL_V2_TECHS {
                pl.techs.insert(t);
            }
            pl.stars += 30;
        }
        let mut rng = Rng::new(seed ^ 0x17_0001);
        let mut steps = 0;
        while !s.is_terminal() && steps < 1500 {
            let legal = legal_actions(&s);
            for a in &legal {
                assert!(
                    !matches!(
                        a,
                        Action::Site {
                            action: SiteAction::BurnForest,
                            ..
                        } | Action::BuildBuilding { .. }
                            | Action::Disband { .. }
                            | Action::Destroy { .. }
                    ),
                    "seed {seed}: {a:?} legal under V1"
                );
            }
            if steps % 31 == 0 {
                assert_illegal(&s, burn(p(0, 0)));
                assert_illegal(&s, build(p(0, 0), Building::Sawmill));
                assert_illegal(&s, destroy(p(0, 0)));
                if let Some(u) = s.units.iter().find(|u| u.owner == s.current) {
                    assert_illegal(&s, disband(u.id.0));
                }
            }
            let a = pick(&mut rng, &legal);
            apply(&mut s, &a).unwrap();
            assert!(s
                .tiles
                .iter()
                .all(|t| t.building.is_none() && t.building_level == 0));
            steps += 1;
        }
    }
}

// ============================================================================
// 2. BurnForest
// ============================================================================

#[test]
fn burn_forest_legality_and_apply() {
    let base = || {
        StateBuilder::new(5)
            .city_with(0, p(2, 2), |c| c.is_capital = true)
            .city_with(1, p(4, 4), |c| c.is_capital = true)
            .terrain(p(1, 1), Terrain::Forest)
            .resource(p(1, 1), Resource::Animal)
            .terrain(p(1, 2), Terrain::Forest)
            .terrain(p(1, 3), Terrain::Forest)
            .improvement(p(1, 3), Improvement::LumberHut)
            .terrain(p(2, 1), Terrain::Mountain)
            .terrain(p(3, 4), Terrain::Forest) // P1's territory
            .terrain(p(0, 4), Terrain::Forest) // neutral
            .techs(0, &[Tech::Organization, Tech::Farming, Tech::Construction])
            .stars(0, 10)
            .explore_all()
    };

    // Missing Construction -> illegal.
    let no_tech = v2(base().techs(0, &[Tech::Organization, Tech::Farming]));
    assert_illegal(&no_tech, burn(p(1, 1)));

    // 2 stars -> illegal; 3 stars -> legal.
    let poor = v2(base().stars(0, 2));
    assert_illegal(&poor, burn(p(1, 1)));

    // Terrain / improvement / territory / enemy unit checks.
    let s = v2(base()
        .unit(0, UnitType::Warrior, p(1, 2))
        .unit(1, UnitType::Warrior, p(2, 3))
        .terrain(p(2, 3), Terrain::Forest));
    assert_illegal(&s, burn(p(1, 3))); // has LumberHut
    assert_illegal(&s, burn(p(2, 1))); // Mountain
    assert_illegal(&s, burn(p(2, 0))); // Field
    assert_illegal(&s, burn(p(3, 4))); // enemy territory
    assert_illegal(&s, burn(p(0, 4))); // neutral
    assert_illegal(&s, burn(p(2, 3))); // enemy unit
    assert_illegal(&s, burn(p(5, 0))); // out of bounds

    // Apply on forest with animal + road, and forest with friendly unit:
    let mut s = v2(base().stars(0, 6).unit(0, UnitType::Warrior, p(1, 2)));
    tile_mut(&mut s, p(1, 1)).road = true;

    ok(&mut s, burn(p(1, 1)));
    assert_eq!(s.players[0].stars, 3);
    assert_eq!(city(&s, 0).population, 0, "BurnForest gives 0 population");
    let t1 = s.tile(p(1, 1));
    assert_eq!(t1.terrain, Terrain::Field);
    assert_eq!(t1.resource, Some(Resource::Crop));
    assert!(t1.road, "road stays intact");

    // Friendly unit does not block.
    ok(&mut s, burn(p(1, 2)));
    assert_eq!(s.players[0].stars, 0);
    assert_eq!(s.tile(p(1, 2)).terrain, Terrain::Field);
    assert_eq!(s.tile(p(1, 2)).resource, Some(Resource::Crop));

    // Cannot burn again (now Field with Crop).
    s.players[0].stars = 10;
    assert_illegal(&s, burn(p(1, 1)));
    // Can build a Farm on the burned forest's new Crop!
    ok(
        &mut s,
        Action::Site {
            at: p(1, 1),
            action: SiteAction::Farm,
        },
    );
    assert_eq!(city(&s, 0).population, 2);
}

#[test]
fn burn_forest_blocked_by_pending_reward() {
    let mut s = v2(StateBuilder::new(5)
        .city_with(0, p(2, 2), |c| {
            c.is_capital = true;
            c.level = 2;
            c.population = 2;
            c.pending_reward = true;
        })
        .terrain(p(1, 1), Terrain::Forest)
        .techs(0, &[Tech::Construction])
        .stars(0, 5)
        .explore_all());
    assert_illegal(&s, burn(p(1, 1)));
    ok(
        &mut s,
        Action::ChooseReward {
            city: CityId(0),
            reward: Reward::Workshop,
        },
    );
    ok(&mut s, burn(p(1, 1)));
}

// ============================================================================
// 3. Economy buildings (Sawmill, Windmill, Forge, Market)
// ============================================================================

#[test]
fn building_placement_legality_tech_cost_tile_and_adjacency() {
    let setup = || {
        StateBuilder::new(7)
            .city_with(0, p(1, 1), |c| {
                c.is_capital = true;
                c.level = 3;
                c.population = 5;
            })
            .city_with(1, p(5, 5), |c| c.is_capital = true)
            .terrain(p(0, 0), Terrain::Forest)
            .improvement(p(0, 0), Improvement::LumberHut)
            .resource(p(0, 1), Resource::Crop)
            .improvement(p(0, 1), Improvement::Farm)
            .terrain(p(0, 2), Terrain::Mountain)
            .resource(p(0, 2), Resource::Metal)
            .improvement(p(0, 2), Improvement::Mine)
            .resource(p(1, 0), Resource::Fruit) // field with fruit next to hut/farm
            .resource(p(2, 0), Resource::Crop) // field with crop next to hut/farm
            .terrain(p(2, 1), Terrain::Forest) // forest next to hut/farm/mine
            .terrain(p(2, 2), Terrain::Mountain) // mountain next to mine
            // (1,2) is plain empty field adjacent to Farm(0,1) and Mine(0,2)
            // (1,0) has Fruit, (2,0) has Crop
            // (4,4) is enemy territory, (3,3) is neutral
            .techs(0, &ALL_V2_TECHS)
            .stars(0, 20)
            .explore_all()
    };

    // Tech check for each building:
    for (b, tech) in [
        (Building::Sawmill, Tech::Mathematics),
        (Building::Windmill, Tech::Construction),
        (Building::Forge, Tech::Smithery),
    ] {
        let others: Vec<Tech> = ALL_V2_TECHS.into_iter().filter(|&t| t != tech).collect();
        let s = v2(setup().techs(0, &others));
        assert_illegal(&s, build(p(1, 2), b));
        assert_illegal(&s, build(p(0, 0), b));
    }

    // Cost check: 4 stars illegal, 5 stars legal.
    // Wait: (1,2) is adjacent to Farm(0,1) and Mine(0,2), NOT LumberHut(0,0) (dist 2 in col)!
    // (1,1) is city, (0,1) is Farm. Put a Sawmill candidate on (1,0)? No, (1,0) has Fruit.
    // Let's check Windmill at (1,2):
    let s4 = v2(setup().stars(0, 4));
    assert_illegal(&s4, build(p(1, 2), Building::Windmill));
    let mut s5 = v2(setup().stars(0, 5));
    ok(&mut s5, build(p(1, 2), Building::Windmill));
    assert_eq!(s5.players[0].stars, 0);

    // Tile condition checks on Windmill / Sawmill / Forge:
    let s = v2(setup());
    // (1,2) is adjacent to Farm(0,1) and Mine(0,2), NOT LumberHut(0,0) -> Sawmill illegal, Windmill/Forge legal
    assert_illegal(&s, build(p(1, 2), Building::Sawmill));
    assert!(legal_actions(&s).contains(&build(p(1, 2), Building::Windmill)));
    assert!(legal_actions(&s).contains(&build(p(1, 2), Building::Forge)));
    // P1: Fruit or Crop on field blocks building even when adjacent to Farm/LumberHut!
    assert_illegal(&s, build(p(1, 0), Building::Sawmill)); // Fruit
    assert_illegal(&s, build(p(1, 0), Building::Windmill)); // Fruit
    assert_illegal(&s, build(p(2, 0), Building::Windmill)); // Crop
                                                            // Terrain: Forest, Mountain, City, Existing improvement block building:
    assert_illegal(&s, build(p(2, 1), Building::Windmill)); // Forest
    assert_illegal(&s, build(p(2, 2), Building::Forge)); // Mountain
    assert_illegal(&s, build(p(1, 1), Building::Windmill)); // City tile
    assert_illegal(&s, build(p(0, 1), Building::Windmill)); // Existing Farm
                                                            // Neutral / enemy territory / out of bounds:
    assert_illegal(&s, build(p(3, 3), Building::Windmill));
    assert_illegal(&s, build(p(4, 4), Building::Windmill));
    assert_illegal(&s, build(p(7, 0), Building::Windmill));

    // Enemy unit blocks, friendly unit allows:
    let s_enemy = v2(setup().unit(1, UnitType::Warrior, p(1, 2)));
    assert_illegal(&s_enemy, build(p(1, 2), Building::Windmill));
    let mut s_friend = v2(setup().unit(0, UnitType::Warrior, p(1, 2)));
    ok(&mut s_friend, build(p(1, 2), Building::Windmill));

    // Road on empty field does NOT block BuildBuilding and stays intact:
    let mut s_road = v2(setup());
    tile_mut(&mut s_road, p(1, 2)).road = true;
    ok(&mut s_road, build(p(1, 2), Building::Forge));
    assert!(s_road.tile(p(1, 2)).road);
    assert_eq!(s_road.tile(p(1, 2)).building, Some(Building::Forge));

    // Pending reward blocks BuildBuilding:
    let s_pend = v2(StateBuilder::new(5)
        .city_with(0, p(1, 1), |c| {
            c.level = 2;
            c.population = 2;
            c.pending_reward = true;
        })
        .resource(p(0, 1), Resource::Crop)
        .improvement(p(0, 1), Improvement::Farm)
        .techs(0, &ALL_V2_TECHS)
        .stars(0, 20)
        .explore_all());
    assert_illegal(&s_pend, build(p(0, 0), Building::Windmill));
}

#[test]
fn one_per_city_enforced_per_city_not_per_player() {
    // City 0 at (1,1), City 1 at (1,4). Both have a Farm and two empty fields.
    let mut s = v2(StateBuilder::new(7)
        .city_with(0, p(1, 1), |c| {
            c.is_capital = true;
            c.level = 5;
            c.population = 14;
        })
        .city_with(0, p(1, 4), |c| {
            c.level = 5;
            c.population = 14;
        })
        .resource(p(0, 1), Resource::Crop)
        .improvement(p(0, 1), Improvement::Farm)
        .resource(p(0, 4), Resource::Crop)
        .improvement(p(0, 4), Improvement::Farm)
        .techs(0, &ALL_V2_TECHS)
        .stars(0, 40)
        .explore_all());
    ok(&mut s, build(p(0, 0), Building::Windmill));
    // Cannot build another Windmill in city 0 (e.g., (0,2) or (1,0)), nor replace (0,0).
    assert_illegal(&s, build(p(0, 2), Building::Windmill));
    assert_illegal(&s, build(p(1, 0), Building::Windmill));
    assert_illegal(&s, build(p(0, 0), Building::Forge));
    // CAN build a Windmill in city 1!
    ok(&mut s, build(p(0, 3), Building::Windmill));
    assert_illegal(&s, build(p(0, 5), Building::Windmill));

    // Same for Market (one per city): build Market next to Windmill at (0,0) on (1,0) in city 0.
    ok(&mut s, build(p(1, 0), Building::Market));
    assert_illegal(&s, build(p(0, 2), Building::Market)); // Wait: (0,2) is adjacent to Windmill? No, (0,2) is adjacent to (0,3) in city 1!
                                                          // (2,0) is adjacent to Market? No, (1,0) is Market, (0,0) is Windmill; (2,0) isn't adjacent to (0,0).
                                                          // Check (0,2): owned by city 0, adjacent to city 1's Windmill at (0,3) -> blocked by one_per_city on city 0!
    assert_eq!(s.tile(p(0, 2)).owner_city, Some(CityId(0)));
    assert_illegal(&s, build(p(0, 2), Building::Market));
    // (1,3) is owned by city 1, adjacent to Windmill(0,3) -> Market allowed in city 1!
    ok(&mut s, build(p(1, 3), Building::Market));
}

#[test]
fn friendly_cross_city_counts_enemy_and_neutral_do_not() {
    // 7x7: City 0 (P0) at (1,1) owns (0..=2, 0..=2).
    // City 1 (P0) at (1,4) owns (0..=2, 3..=5).
    // City 2 (P1) at (4,1) owns (3..=5, 0..=2).
    // Neutral improvement at (3,3) (outside all 3 cities).
    // Target field (2,2) in City 0 has 8 neighbors:
    //   (1,3) owned by P0 City 1: Farm (friendly cross-city -> COUNTS!)
    //   (3,1) owned by P1 City 2: Farm (enemy -> does NOT count!)
    //   (3,2) owned by P1 City 2: Farm (enemy -> does NOT count!)
    //   (3,3) neutral: Farm (neutral -> does NOT count!)
    let make = |with_friendly: bool| {
        let mut b = StateBuilder::new(7)
            .city_with(0, p(1, 1), |c| {
                c.is_capital = true;
                c.level = 5;
                c.population = 14;
            })
            .city_with(0, p(1, 4), |c| {
                c.level = 5;
                c.population = 14;
            })
            .city_with(1, p(4, 1), |c| c.is_capital = true)
            .resource(p(3, 1), Resource::Crop)
            .improvement(p(3, 1), Improvement::Farm)
            .resource(p(3, 2), Resource::Crop)
            .improvement(p(3, 2), Improvement::Farm)
            .resource(p(3, 3), Resource::Crop)
            .improvement(p(3, 3), Improvement::Farm)
            .techs(0, &ALL_V2_TECHS)
            .stars(0, 30)
            .explore_all();
        if with_friendly {
            b = b
                .resource(p(1, 3), Resource::Crop)
                .improvement(p(1, 3), Improvement::Farm);
        }
        b
    };

    // First without friendly cross-city Farm at (1,3): only enemy & neutral farms around (2,2)!
    let s_no_friendly = v2(make(false));
    assert_eq!(s_no_friendly.tile(p(2, 2)).owner_city, Some(CityId(0)));
    assert_illegal(&s_no_friendly, build(p(2, 2), Building::Windmill));

    // Add friendly cross-city Farm at (1,3) in City 1:
    let mut s = v2(make(true));
    ok(&mut s, build(p(2, 2), Building::Windmill));
    assert_eq!(building_level(&s, p(2, 2)), 1);
    assert_eq!(s.tile(p(2, 2)).building_level, 1);
    assert_eq!(
        city(&s, 0).population,
        15,
        "City 0 got +1 from cross-city Farm"
    );
    assert_eq!(
        city(&s, 1).population,
        14,
        "City 1 did not gain from City 0's Windmill"
    );
}

#[test]
fn building_levels_population_per_level_and_forge_plus_two() {
    // City 0 at (2,2), level 6, pop 20 (cum(7)=27, plenty of headroom).
    // Put 3 Mines around (1,1): (0,0), (0,1), (0,2).
    // Put 2 LumberHuts around (3,3): (2,3), (3,2).
    let mut s = v2(StateBuilder::new(5)
        .city_with(0, p(2, 2), |c| {
            c.is_capital = true;
            c.level = 6;
            c.population = 20;
        })
        .terrain(p(0, 0), Terrain::Mountain)
        .resource(p(0, 0), Resource::Metal)
        .improvement(p(0, 0), Improvement::Mine)
        .terrain(p(0, 1), Terrain::Mountain)
        .resource(p(0, 1), Resource::Metal)
        .improvement(p(0, 1), Improvement::Mine)
        .terrain(p(0, 2), Terrain::Mountain)
        .resource(p(0, 2), Resource::Metal)
        .improvement(p(0, 2), Improvement::Mine)
        .terrain(p(2, 3), Terrain::Forest)
        .improvement(p(2, 3), Improvement::LumberHut)
        .terrain(p(3, 2), Terrain::Forest)
        .improvement(p(3, 2), Improvement::LumberHut)
        .techs(0, &ALL_V2_TECHS)
        .stars(0, 30)
        .explore_all());
    for q in [p(0, 0), p(0, 1), p(0, 2)] {
        tile_mut(&mut s, q).owner_city = Some(CityId(0));
    }

    ok(&mut s, build(p(1, 1), Building::Forge));
    assert_eq!(building_level(&s, p(1, 1)), 3);
    assert_eq!(s.tile(p(1, 1)).building_level, 3);
    // Forge gives +2 per level -> +6!
    assert_eq!(city(&s, 0).population, 26);

    ok(&mut s, build(p(3, 3), Building::Sawmill));
    assert_eq!(building_level(&s, p(3, 3)), 2);
    assert_eq!(s.tile(p(3, 3)).building_level, 2);
    // Sawmill gives +1 per level -> +2 -> 28 -> crosses cum(7)=27!
    assert_eq!(city(&s, 0).population, 28);
    assert_eq!(city(&s, 0).level, 7);
    assert!(city(&s, 0).pending_reward);
}

#[test]
fn adding_adjacent_improvement_raises_level_and_cascades_level_up() {
    // City 0 at (1,1) level 1 pop 1 (cum(2)=2, cum(3)=5).
    // Existing Mine at (0,0), existing Forge at (1,0) (level 1, pop already set to 1 by builder).
    // Metal at (2,0) next to Forge(1,0).
    // Building Mine at (2,0): Mine gives +2 (pop -> 3, level 2 pending) AND Forge syncs 1 -> 2
    // giving +2 more (pop -> 5 == cum(3)!). Resolving Workshop immediately cascades to level 3!
    let mut s = v2(StateBuilder::new(5)
        .city_with(0, p(1, 1), |c| {
            c.is_capital = true;
            c.level = 1;
            c.population = 1;
        })
        .terrain(p(0, 0), Terrain::Mountain)
        .resource(p(0, 0), Resource::Metal)
        .improvement(p(0, 0), Improvement::Mine)
        .building(p(1, 0), Building::Forge)
        .terrain(p(2, 0), Terrain::Mountain)
        .resource(p(2, 0), Resource::Metal)
        .techs(0, &ALL_V2_TECHS)
        .stars(0, 20)
        .explore_all());
    assert_eq!(s.tile(p(1, 0)).building_level, 1);

    ok(
        &mut s,
        Action::Site {
            at: p(2, 0),
            action: SiteAction::Mine,
        },
    );
    assert_eq!(building_level(&s, p(1, 0)), 2);
    assert_eq!(s.tile(p(1, 0)).building_level, 2);
    assert_eq!(city(&s, 0).population, 5);
    assert_eq!(city(&s, 0).level, 2);
    assert!(city(&s, 0).pending_reward);

    ok(
        &mut s,
        Action::ChooseReward {
            city: CityId(0),
            reward: Reward::Workshop,
        },
    );
    assert_eq!(city(&s, 0).level, 3);
    assert!(city(&s, 0).pending_reward);
}

#[test]
fn adding_improvement_levels_up_a_different_friendly_city_via_sync() {
    // City 0 at (1,1) owns (1,2) (Forest).
    // City 1 at (1,4) owns (1,3), which has a Sawmill (currently level 1 from hut at (0,3)).
    // City 1 is at level 1, pop 1 (1 short of level 2).
    // Building a LumberHut on (1,2) (City 0) raises City 0 to pop 1 AND raises City 1's Sawmill
    // from level 1 -> 2, giving City 1 +1 -> pop 2 -> level 2 pending!
    let mut s = v2(StateBuilder::new(7)
        .city_with(0, p(1, 1), |c| c.is_capital = true)
        .city_with(0, p(1, 4), |c| c.population = 1)
        .terrain(p(0, 3), Terrain::Forest)
        .improvement(p(0, 3), Improvement::LumberHut)
        .building(p(1, 3), Building::Sawmill)
        .terrain(p(1, 2), Terrain::Forest)
        .techs(0, &ALL_V2_TECHS)
        .stars(0, 20)
        .explore_all());
    assert_eq!(s.tile(p(1, 3)).building_level, 1);

    ok(
        &mut s,
        Action::Site {
            at: p(1, 2),
            action: SiteAction::LumberHut,
        },
    );
    assert_eq!(city(&s, 0).population, 1);
    assert_eq!(city(&s, 0).level, 1);
    assert_eq!(s.tile(p(1, 3)).building_level, 2);
    assert_eq!(city(&s, 1).population, 2);
    assert_eq!(city(&s, 1).level, 2);
    assert!(city(&s, 1).pending_reward);
}

#[test]
fn market_income_sums_neighbor_levels_caps_at_eight_and_updates_dynamically() {
    // City 0 at (2,2), level 20, pop 209 (cum(21)=230, 21 pop headroom).
    // Market at (2,1) (Trade).
    // Sawmill at (1,1) next to 1 LumberHut at (0,0) (and Forest+no hut at (0,1), (0,2)).
    // Forge at (3,1) next to 4 Mines at (4,0), (4,1), (4,2), (3,0) -> level 4.
    // Windmill at (2,0) next to 1 Farm at (1,0) -> level 1.
    // Note: all 3 economy buildings are 8-adjacent to Market at (2,1).
    let mut s = v2(StateBuilder::new(6)
        .city_with(0, p(2, 2), |c| {
            c.is_capital = true;
            c.level = 20;
            c.population = 209;
        })
        // Extra friendly city at (4,4) owning row 3..=5 x col 3..=5? No, keep (3,0),(4,0..=2) in City 0?
        // Wait: City 0 at (2,2) owns (1..=3, 1..=3). (0,0), (4,0..=2), (1,0), (2,0), (3,0) are outside radius 1!
        // Let's use a 2nd P0 city or expand City 0, or place buildings cleanly inside (1..=3, 1..=3)!
        // Let's put City 0 at (2,2) with extra tiles claimed via city_with + tile_mut in a helper:
        .terrain(p(0, 0), Terrain::Forest)
        .improvement(p(0, 0), Improvement::LumberHut)
        .terrain(p(0, 1), Terrain::Forest)
        .terrain(p(0, 2), Terrain::Forest)
        .resource(p(1, 0), Resource::Crop)
        .improvement(p(1, 0), Improvement::Farm)
        .terrain(p(4, 0), Terrain::Mountain)
        .resource(p(4, 0), Resource::Metal)
        .improvement(p(4, 0), Improvement::Mine)
        .terrain(p(4, 1), Terrain::Mountain)
        .resource(p(4, 1), Resource::Metal)
        .improvement(p(4, 1), Improvement::Mine)
        .terrain(p(4, 2), Terrain::Mountain)
        .resource(p(4, 2), Resource::Metal)
        .improvement(p(4, 2), Improvement::Mine)
        .terrain(p(3, 0), Terrain::Mountain)
        .resource(p(3, 0), Resource::Metal)
        .improvement(p(3, 0), Improvement::Mine)
        .terrain(p(4, 3), Terrain::Mountain)
        .resource(p(4, 3), Resource::Metal)
        .techs(0, &ALL_V2_TECHS)
        .stars(0, 50)
        .explore_all());
    // Give City 0 ownership of the outer ring tiles we need for improvements/buildings:
    for q in [
        p(0, 0),
        p(0, 1),
        p(0, 2),
        p(1, 0),
        p(2, 0),
        p(3, 0),
        p(4, 0),
        p(4, 1),
        p(4, 2),
        p(4, 3),
    ] {
        tile_mut(&mut s, q).owner_city = Some(CityId(0));
    }

    // Build Sawmill at (1,1), Windmill at (2,0), Forge at (3,1):
    ok(&mut s, build(p(1, 1), Building::Sawmill)); // level 1
    ok(&mut s, build(p(2, 0), Building::Windmill)); // level 1
    ok(&mut s, build(p(3, 1), Building::Forge)); // level 4
    assert_eq!(building_level(&s, p(1, 1)), 1);
    assert_eq!(building_level(&s, p(2, 0)), 1);
    assert_eq!(building_level(&s, p(3, 1)), 4);
    // Non-Market buildings have 0 market_income:
    assert_eq!(market_income(&s, p(1, 1)), 0);
    assert_eq!(market_income(&s, p(3, 1)), 0);
    // Empty tile has 0 building_level and 0 market_income:
    assert_eq!(building_level(&s, p(2, 1)), 0);
    assert_eq!(market_income(&s, p(2, 1)), 0);

    // Build Market at (2,1): sum of neighbor levels = 1 + 1 + 4 = 6 (not count 3!).
    ok(&mut s, build(p(2, 1), Building::Market));
    assert_eq!(building_level(&s, p(2, 1)), 6);
    assert_eq!(s.tile(p(2, 1)).building_level, 6);
    assert_eq!(market_income(&s, p(2, 1)), 6);
    // City 0: level 20 + capital 1 + market 6 = 27.
    assert_eq!(income(&s, 0), 27);

    // Add 2 LumberHuts next to Sawmill(1,1) -> Sawmill level 3 -> Market level 8, income 8.
    ok(
        &mut s,
        Action::Site {
            at: p(0, 1),
            action: SiteAction::LumberHut,
        },
    );
    ok(
        &mut s,
        Action::Site {
            at: p(0, 2),
            action: SiteAction::LumberHut,
        },
    );
    assert_eq!(building_level(&s, p(1, 1)), 3);
    assert_eq!(building_level(&s, p(2, 1)), 8);
    assert_eq!(market_income(&s, p(2, 1)), 8);

    // Add a 5th Mine at (4,3)? Wait: (4,3) is distance 2 from (3,1). Put a Mine on (2,3)? No, keep test simple:
    // put a 2nd Farm at (3,0)? No, let's check cap > 8 by placing a Forest+LumberHut at (1,2) next to Sawmill(1,1):
    tile_mut(&mut s, p(1, 2)).terrain = Terrain::Forest;
    ok(
        &mut s,
        Action::Site {
            at: p(1, 2),
            action: SiteAction::LumberHut,
        },
    );
    // Sawmill is now level 4 -> Market building_level = 4 + 1 + 4 = 9, but market_income is capped at 8!
    assert_eq!(building_level(&s, p(1, 1)), 4);
    assert_eq!(building_level(&s, p(2, 1)), 9);
    assert_eq!(s.tile(p(2, 1)).building_level, 9);
    assert_eq!(market_income(&s, p(2, 1)), 8, "capped at max_output=8");
    // Turn-start stars reflect Market income:
    assert_eq!(city(&s, 0).population, 209 + 1 + 1 + 8 + 2 + 2 + 2);
    assert!(!city(&s, 0).pending_reward);
    let expected_income = 20 + 1 + 8; // level 20 + capital 1 + market 8 = 29
    assert_eq!(income(&s, 0), expected_income);
    let stars_before = s.players[0].stars;
    ok(&mut s, Action::EndTurn); // P1 turn 0
    ok(&mut s, Action::EndTurn); // P0 turn 1 -> +income
    assert_eq!(s.players[0].stars, stars_before + expected_income);
}

#[test]
fn market_can_be_placed_next_to_level_zero_building() {
    // P2: Sawmill exists with 0 adjacent huts (level 0); Market can be placed next to it.
    let mut s = v2(StateBuilder::new(5)
        .city_with(0, p(2, 2), |c| c.is_capital = true)
        .building(p(1, 1), Building::Sawmill)
        .techs(0, &ALL_V2_TECHS)
        .stars(0, 10)
        .explore_all());
    assert_eq!(building_level(&s, p(1, 1)), 0);
    ok(&mut s, build(p(1, 2), Building::Market));
    assert_eq!(building_level(&s, p(1, 2)), 0);
    assert_eq!(s.tile(p(1, 2)).building_level, 0);
    assert_eq!(market_income(&s, p(1, 2)), 0);

    // Building a LumberHut at (0,0) lifts Sawmill to 1 and Market to 1!
    tile_mut(&mut s, p(0, 0)).terrain = Terrain::Forest;
    tile_mut(&mut s, p(0, 0)).owner_city = Some(CityId(0));
    ok(
        &mut s,
        Action::Site {
            at: p(0, 0),
            action: SiteAction::LumberHut,
        },
    );
    assert_eq!(building_level(&s, p(1, 1)), 1);
    assert_eq!(building_level(&s, p(1, 2)), 1);
    assert_eq!(s.tile(p(1, 2)).building_level, 1);
    assert_eq!(market_income(&s, p(1, 2)), 1);
}

#[test]
fn capture_transfers_adjacency_and_skipped_on_capital_capture() {
    // 7x7:
    // City 0 (P0 capital) at (1,1), level 3 pop 5, Sawmill at (1,2) (next to City 2's LumberHut at (1,3)).
    // City 1 (P1 capital) at (5,5), level 3 pop 5.
    // City 2 (P0) at (1,4), level 3 pop 5, LumberHut at (1,3), Sawmill at (2,4) (next to (1,3)).
    // P1 Warrior on City 2 at (1,4) ready to capture.
    let mut s = v2(StateBuilder::new(7)
        .city_with(0, p(1, 1), |c| {
            c.is_capital = true;
            c.level = 3;
            c.population = 5;
        })
        .city_with(1, p(5, 5), |c| {
            c.is_capital = true;
            c.level = 3;
            c.population = 5;
        })
        .city_with(0, p(1, 4), |c| {
            c.level = 3;
            c.population = 5;
        })
        .terrain(p(1, 3), Terrain::Forest)
        .improvement(p(1, 3), Improvement::LumberHut)
        .building(p(1, 2), Building::Sawmill)
        .building(p(2, 4), Building::Sawmill)
        .unit(1, UnitType::Warrior, p(1, 4))
        .current(1)
        .explore_all());
    assert_eq!(s.tile(p(1, 2)).building_level, 1);
    assert_eq!(s.tile(p(2, 4)).building_level, 1);

    // P1 captures City 2:
    // - City 2 (owner P1) has Sawmill(2,4) still next to City 2's LumberHut(1,3) -> level stays 1.
    // - City 0 (owner P0) Sawmill(1,2) loses adjacency to LumberHut(1,3) -> level 1 -> 0, City 0 pop 5 -> 4!
    ok(&mut s, Action::Capture { unit: UnitId(0) });
    assert_eq!(s.tile(p(1, 2)).building_level, 0);
    assert_eq!(building_level(&s, p(1, 2)), 0);
    assert_eq!(city(&s, 0).population, 4);
    assert_eq!(city(&s, 0).level, 3, "level never drops");
    assert_eq!(s.tile(p(2, 4)).building_level, 1);
    assert_eq!(city(&s, 2).population, 5);

    // Q48: Capital capture ending the game skips sync_buildings!
    // P0 Warrior on P1 capital (1,4)? Let's set up a fresh capital capture board:
    // City 0 (P0 capital) at (1,1), Sawmill at (1,2) next to (1,3) (owned by P1 capital City 1 at (1,4), LumberHut at (1,3)).
    // P0 Warrior stands on (1,4). Before capture, Sawmill(1,2) is level 0.
    // If sync_buildings ran after capital capture, Sawmill(1,2) would jump to 1 and City 0 pop would jump!
    let mut cap_game = v2(StateBuilder::new(7)
        .city_with(0, p(1, 1), |c| {
            c.is_capital = true;
            c.level = 3;
            c.population = 5;
        })
        .city_with(1, p(1, 4), |c| {
            c.is_capital = true;
            c.level = 3;
            c.population = 5;
        })
        .terrain(p(1, 3), Terrain::Forest)
        .improvement(p(1, 3), Improvement::LumberHut)
        .building(p(1, 2), Building::Sawmill)
        .unit(0, UnitType::Warrior, p(1, 4))
        .explore_all());
    assert_eq!(cap_game.tile(p(1, 2)).building_level, 0);
    ok(&mut cap_game, Action::Capture { unit: UnitId(0) });
    assert_eq!(cap_game.outcome, Some(Outcome::Win(0)));
    assert_eq!(
        cap_game.tile(p(1, 2)).building_level,
        0,
        "Q48: sync_buildings skipped"
    );
    assert_eq!(
        city(&cap_game, 0).population,
        5,
        "Q48: population unchanged"
    );
}

// ============================================================================
// 4. Disband
// ============================================================================

#[test]
fn disband_legality_across_phases_and_pending_reward() {
    let base = || {
        StateBuilder::new(5)
            .city_with(0, p(2, 2), |c| c.is_capital = true)
            .unit(0, UnitType::Warrior, p(0, 0))
            .unit(1, UnitType::Warrior, p(4, 4))
            .techs(0, &[Tech::Riding, Tech::FreeSpirit])
            .explore_all()
    };

    // Without FreeSpirit -> illegal.
    let no_tech = v2(base().techs(0, &[Tech::Riding]));
    assert_illegal(&no_tech, disband(0));

    // Enemy unit or non-existent unit -> illegal.
    let s = v2(base());
    assert_illegal(&s, disband(1));
    assert_illegal(&s, disband(99));

    // Blocked by pending reward:
    let s_pend = v2(base().city_with(0, p(0, 2), |c| {
        c.level = 2;
        c.population = 2;
        c.pending_reward = true;
    }));
    assert_illegal(&s_pend, disband(0));

    // Legal in EVERY phase (Fresh, Moved, Attacked, Done) and even with 0 stars:
    for phase in [Phase::Fresh, Phase::Moved, Phase::Attacked, Phase::Done] {
        let mut s = v2(base().unit_with(0, UnitType::Rider, p(1, 1), |u| u.phase = phase));
        ok(&mut s, disband(2));
        assert!(s.unit(UnitId(2)).is_none());
        assert_eq!(s.tile(p(1, 1)).unit, None);
        assert_eq!(s.players[0].stars, 1);
    }
}

#[test]
fn disband_refunds_table_values_frees_capacity_and_is_not_a_kill() {
    let cases = [
        (UnitType::Warrior, 1u32),
        (UnitType::Rider, 1),
        (UnitType::Archer, 1),
        (UnitType::Defender, 1),
        (UnitType::Swordsman, 2),
        (UnitType::Catapult, 4),
        (UnitType::Knight, 4),
        (UnitType::SuperUnit, 5),
    ];
    for (kind, expected_refund) in cases {
        let mut s = StateBuilder::new(5)
            .config(v2_config())
            .city_with(0, p(2, 2), |c| c.is_capital = true)
            .unit_with(0, kind, p(0, 0), |u| u.origin = Some(CityId(0)))
            .unit_with(0, UnitType::Warrior, p(0, 1), |u| {
                u.origin = Some(CityId(0))
            })
            .techs(0, &[Tech::Riding, Tech::FreeSpirit])
            .stars(0, 0)
            .explore_all()
            .build();
        ok(&mut s, disband(0));
        assert_eq!(s.players[0].stars, expected_refund, "{kind:?}");
        assert!(s.unit(UnitId(0)).is_none());
        assert_eq!(s.tile(p(0, 0)).unit, None);
        // Not a kill: remaining unit's kills stay 0.
        assert_eq!(s.unit(UnitId(1)).unwrap().kills, 0);
        // Capacity freed: if non-SuperUnit, city 0 dropped from 2 to 1 and can train a Warrior.
        if kind != UnitType::SuperUnit {
            s.players[0].stars = 5;
            ok(
                &mut s,
                Action::Train {
                    city: CityId(0),
                    kind: UnitType::Warrior,
                },
            );
        }
    }
}

// ============================================================================
// 5. Destroy
// ============================================================================

#[test]
fn destroy_legality_and_apply_on_improvements_buildings_and_roads() {
    let setup = || {
        let mut s = v2(StateBuilder::new(7)
            .city_with(0, p(1, 1), |c| {
                c.is_capital = true;
                c.level = 5;
                c.population = 14;
            })
            .city_with(1, p(5, 5), |c| c.is_capital = true)
            .terrain(p(0, 0), Terrain::Forest)
            .resource(p(0, 0), Resource::Animal)
            .improvement(p(0, 0), Improvement::LumberHut)
            .resource(p(0, 1), Resource::Crop)
            .improvement(p(0, 1), Improvement::Farm)
            .terrain(p(0, 2), Terrain::Mountain)
            .resource(p(0, 2), Resource::Metal)
            .improvement(p(0, 2), Improvement::Mine)
            .building(p(1, 0), Building::Sawmill)
            .building(p(2, 1), Building::Market)
            .resource(p(2, 2), Resource::Fruit)
            .terrain(p(5, 4), Terrain::Forest)
            .improvement(p(5, 4), Improvement::LumberHut) // enemy
            .techs(0, &[Tech::Riding, Tech::FreeSpirit, Tech::Chivalry])
            .stars(0, 0)
            .explore_all());
        tile_mut(&mut s, p(0, 0)).road = true;
        tile_mut(&mut s, p(1, 0)).road = true;
        tile_mut(&mut s, p(2, 0)).road = true; // road-only tile
        s
    };

    // Missing Chivalry -> illegal.
    let no_tech = v2(StateBuilder::new(5)
        .city(0, p(1, 1))
        .resource(p(0, 1), Resource::Crop)
        .improvement(p(0, 1), Improvement::Farm)
        .techs(0, &[Tech::Riding, Tech::FreeSpirit])
        .explore_all());
    assert_illegal(&no_tech, destroy(p(0, 1)));

    // Illegal targets:
    let s = setup();
    assert_illegal(&s, destroy(p(2, 0))); // road-only tile
    assert_illegal(&s, destroy(p(1, 2))); // empty field
    assert_illegal(&s, destroy(p(2, 2))); // resource-only (Fruit)
    assert_illegal(&s, destroy(p(1, 1))); // city-only tile
    assert_illegal(&s, destroy(p(5, 4))); // enemy improvement
    assert_illegal(&s, destroy(p(3, 3))); // neutral empty
    assert_illegal(&s, destroy(p(7, 0))); // out of bounds

    // Enemy unit blocks, friendly unit allows:
    let s_enemy = v2(StateBuilder::new(5)
        .city(0, p(1, 1))
        .resource(p(0, 1), Resource::Crop)
        .improvement(p(0, 1), Improvement::Farm)
        .unit(1, UnitType::Warrior, p(0, 1))
        .techs(0, &[Tech::Chivalry])
        .explore_all());
    assert_illegal(&s_enemy, destroy(p(0, 1)));

    let mut s_friend = v2(StateBuilder::new(5)
        .city_with(0, p(1, 1), |c| {
            c.level = 3;
            c.population = 5;
        })
        .resource(p(0, 1), Resource::Crop)
        .improvement(p(0, 1), Improvement::Farm)
        .unit(0, UnitType::Warrior, p(0, 1))
        .techs(0, &[Tech::Chivalry])
        .explore_all());
    ok(&mut s_friend, destroy(p(0, 1)));
    assert_eq!(city(&s_friend, 0).population, 3);
    assert_eq!(s_friend.tile(p(0, 1)).improvement, None);
    assert_eq!(
        s_friend.tile(p(0, 1)).resource,
        Some(Resource::Crop),
        "P3: crop stays"
    );

    // Full cascading destroy test on `setup()`:
    let mut s = setup();
    assert_eq!(s.tile(p(1, 0)).building_level, 1);
    assert_eq!(s.tile(p(2, 1)).building_level, 1);
    assert_eq!(market_income(&s, p(2, 1)), 1);

    // 1. Destroy LumberHut at (0,0) (pop -1) -> cascades Sawmill(1,0) 1->0 (pop -1 more) and Market(2,1) 1->0!
    ok(&mut s, destroy(p(0, 0)));
    assert_eq!(s.players[0].stars, 0, "no cost, no refund");
    assert_eq!(s.tile(p(0, 0)).improvement, None);
    assert_eq!(s.tile(p(0, 0)).terrain, Terrain::Forest, "terrain stays");
    assert_eq!(
        s.tile(p(0, 0)).resource,
        Some(Resource::Animal),
        "P3: animal stays"
    );
    assert!(s.tile(p(0, 0)).road, "road stays");
    assert_eq!(s.tile(p(1, 0)).building_level, 0);
    assert_eq!(s.tile(p(2, 1)).building_level, 0);
    assert_eq!(market_income(&s, p(2, 1)), 0);
    assert_eq!(city(&s, 0).population, 12);

    // 2. Destroy Mine at (0,2) -> pop -2, metal and mountain stay:
    ok(&mut s, destroy(p(0, 2)));
    assert_eq!(s.tile(p(0, 2)).improvement, None);
    assert_eq!(s.tile(p(0, 2)).terrain, Terrain::Mountain);
    assert_eq!(s.tile(p(0, 2)).resource, Some(Resource::Metal));
    assert_eq!(city(&s, 0).population, 10);

    // 3. Destroy Sawmill at (1,0) (currently level 0 -> pop -0) -> road stays, building gone:
    ok(&mut s, destroy(p(1, 0)));
    assert_eq!(s.tile(p(1, 0)).building, None);
    assert_eq!(s.tile(p(1, 0)).building_level, 0);
    assert!(s.tile(p(1, 0)).road, "road stays");
    assert_eq!(city(&s, 0).population, 10);

    // 4. Rebuild Sawmill? First rebuild LumberHut on (0,0) -> pop +1, then Sawmill on (1,0) (one_per_city reset)!
    s.players[0].stars = 20;
    s.players[0].techs.insert(Tech::Forestry);
    s.players[0].techs.insert(Tech::Mathematics);
    ok(
        &mut s,
        Action::Site {
            at: p(0, 0),
            action: SiteAction::LumberHut,
        },
    );
    ok(&mut s, build(p(1, 0), Building::Sawmill));
    assert_eq!(s.tile(p(1, 0)).building_level, 1);
    assert_eq!(s.tile(p(2, 1)).building_level, 1);
    assert_eq!(city(&s, 0).population, 12);

    // 5. Destroy Sawmill at level 1 -> subtracts 1 * pop_per_level = 1 and updates Market 1 -> 0:
    ok(&mut s, destroy(p(1, 0)));
    assert_eq!(s.tile(p(1, 0)).building, None);
    assert_eq!(s.tile(p(2, 1)).building_level, 0);
    assert_eq!(city(&s, 0).population, 11);
}

#[test]
fn destroy_blocked_by_pending_reward() {
    let mut s = v2(StateBuilder::new(5)
        .city_with(0, p(2, 2), |c| {
            c.is_capital = true;
            c.level = 2;
            c.population = 2;
            c.pending_reward = true;
        })
        .resource(p(1, 1), Resource::Crop)
        .improvement(p(1, 1), Improvement::Farm)
        .techs(0, &[Tech::Chivalry])
        .explore_all());
    assert_illegal(&s, destroy(p(1, 1)));
    ok(
        &mut s,
        Action::ChooseReward {
            city: CityId(0),
            reward: Reward::Workshop,
        },
    );
    ok(&mut s, destroy(p(1, 1)));
}

// ============================================================================
// 6. Signed population & score floor
// ============================================================================

#[test]
fn signed_population_goes_negative_without_lowering_level_and_regrows_from_negative() {
    // Level 2 city at pop 0 (e.g. had connection/building losses), connected to capital, with Mine & Forge.
    // Start at level 2, pop 2, with Mine(0,0), Forge(1,0), Market? No, let's start at level 2 pop 0
    // with Mine(0,0) and Forge(1,0) (level 1 -> 2 pop), plus another Mine(2,0) (Forge level 2).
    // Destroying Mine(0,0) (-2) + Mine(2,0) (-2) + Forge(1,0) dropping 2->0 (-4) pushes pop 0 -> -8!
    let mut s = v2(StateBuilder::new(5)
        .city_with(0, p(1, 1), |c| {
            c.is_capital = true;
            c.level = 2;
            c.population = 0;
        })
        .terrain(p(0, 0), Terrain::Mountain)
        .resource(p(0, 0), Resource::Metal)
        .improvement(p(0, 0), Improvement::Mine)
        .terrain(p(2, 0), Terrain::Mountain)
        .resource(p(2, 0), Resource::Metal)
        .improvement(p(2, 0), Improvement::Mine)
        .building(p(1, 0), Building::Forge)
        .resource(p(0, 2), Resource::Fruit)
        .resource(p(1, 2), Resource::Fruit)
        .resource(p(2, 2), Resource::Fruit)
        .techs(0, &ALL_V2_TECHS)
        .stars(0, 50)
        .explore_all());
    assert_eq!(s.tile(p(1, 0)).building_level, 2);

    ok(&mut s, destroy(p(0, 0))); // -2 (Mine) and -2 (Forge 2->1) => -4
    assert_eq!(city(&s, 0).population, -4);
    assert_eq!(city(&s, 0).level, 2, "level never drops");
    assert!(!city(&s, 0).pending_reward);

    ok(&mut s, destroy(p(2, 0))); // -2 (Mine) and -2 (Forge 1->0) => -8
    assert_eq!(city(&s, 0).population, -8);
    assert_eq!(city(&s, 0).level, 2);

    // Score surplus floors at 0 (no panic, no u32 wrap!):
    // Capital (level 2): city_base 100 + city_per_level 50 + surplus 0 = 150.
    assert_eq!(score_breakdown(&s, 0).cities, 150);

    // Regrowing from -8: level 3 needs cum(3) = 5.
    // Harvesting fruits (+1) and rebuilding Mines (+2) + Forge (+4):
    ok(
        &mut s,
        Action::Site {
            at: p(0, 2),
            action: SiteAction::HarvestFruit,
        },
    );
    assert_eq!(city(&s, 0).population, -7);
    assert_eq!(city(&s, 0).level, 2);
    assert!(!city(&s, 0).pending_reward);
    assert_eq!(score_breakdown(&s, 0).cities, 150);

    // Also verify score_breakdown for positive below-threshold (level 3, pop 2 -> 200) and level 1 negative (pop -3 -> 100):
    let mut t = s.clone();
    t.cities[0].level = 3;
    t.cities[0].population = 2;
    assert_eq!(score_breakdown(&t, 0).cities, 200);
    t.cities[0].level = 1;
    t.cities[0].population = -3;
    assert_eq!(score_breakdown(&t, 0).cities, 100);
    assert!(score(&t, 0) < 10_000, "no u32 underflow wrap");
}

#[test]
fn roads_lose_handles_negative_population_without_panicking() {
    // City 2 connected to capital City 0; destroy buildings on City 0 pushing City 0 negative,
    // then break the road connection via village capture -> roads::lose subtracts 1 from negative pop!
    let mut s = v2(StateBuilder::new(9)
        .city_with(0, p(1, 1), |c| {
            c.is_capital = true;
            c.level = 2;
            c.population = 0;
        })
        .city_with(1, p(7, 7), |c| c.is_capital = true)
        .city(0, p(1, 5))
        .village(p(1, 3))
        .terrain(p(0, 0), Terrain::Mountain)
        .resource(p(0, 0), Resource::Metal)
        .improvement(p(0, 0), Improvement::Mine)
        .unit(1, UnitType::Warrior, p(2, 3))
        .techs(0, &ALL_V2_TECHS)
        .stars(0, 20)
        .explore_all());
    for q in [p(1, 2), p(1, 4)] {
        tile_mut(&mut s, q).road = true;
    }
    // Building the neutral road at (1,3) triggers update_connections: City 2 connects to City 0 (+1 each -> City 0 pop = 1).
    ok(&mut s, Action::BuildRoad { at: p(1, 3) });
    assert_eq!(city(&s, 0).population, 1);
    assert_eq!(city(&s, 2).connected_to, Some(CityId(0)));

    // Destroy Mine at (0,0): City 0 pop goes 1 -> -1.
    ok(&mut s, destroy(p(0, 0)));
    assert_eq!(city(&s, 0).population, -1);

    // P1 moves onto village (1,3), then captures it on the next turn: breaks connection -> roads::lose subtracts 1 from -1 -> -2!
    ok(&mut s, Action::EndTurn);
    ok(
        &mut s,
        Action::Move {
            unit: UnitId(0),
            to: p(1, 3),
        },
    );
    ok(&mut s, Action::EndTurn);
    ok(&mut s, Action::EndTurn);
    ok(&mut s, Action::Capture { unit: UnitId(0) });
    assert_eq!(city(&s, 0).population, -2);
    assert_eq!(city(&s, 0).level, 2);
    assert_eq!(score_breakdown(&s, 0).cities, 250);
}

// ============================================================================
// 8. Canonical legal_actions order
// ============================================================================

#[test]
fn canonical_legal_actions_order_with_all_m17_actions() {
    // Board with:
    // - Unit 0 (Warrior, P0, Fresh, promotion_available) at (2,2) (city 0)
    // - City 0 (P0 capital) at (2,2) -> Train blocked by unit 0, OR leave city 0 empty and put unit 0 at (1,2)!
    //   Let's put unit 0 at (1,2) so City 0 at (2,2) can Train too!
    // - (1,1): Forest -> LumberHut, ClearForest, BurnForest (and BuildRoad)
    // - (2,1): Farm (already built) -> Destroy at (2,1), BuildRoad at (2,1)
    // - (3,1): Empty Field adjacent to Farm(2,1) -> BuildBuilding(Windmill), BuildRoad at (3,1)
    let s = v2(StateBuilder::new(5)
        .city_with(0, p(2, 2), |c| {
            c.is_capital = true;
            c.level = 3;
            c.population = 5;
        })
        .terrain(p(1, 1), Terrain::Forest)
        .resource(p(2, 1), Resource::Crop)
        .improvement(p(2, 1), Improvement::Farm)
        .unit_with(0, UnitType::Warrior, p(1, 2), |u| {
            u.kills = 3;
            u.promotion_available = true;
        })
        .techs(0, &ALL_V2_TECHS)
        .stars(0, 20)
        .explore_all());

    let legal = legal_actions(&s);
    let idx_of = |pred: &dyn Fn(&Action) -> bool| -> Vec<usize> {
        legal
            .iter()
            .enumerate()
            .filter(|(_, a)| pred(a))
            .map(|(i, _)| i)
            .collect()
    };

    let moves = idx_of(&|a| matches!(a, Action::Move { .. }));
    let recovers = idx_of(&|a| matches!(a, Action::Recover { .. }));
    let promotes = idx_of(&|a| matches!(a, Action::Promote { .. }));
    let disbands = idx_of(&|a| matches!(a, Action::Disband { .. }));
    let trains = idx_of(&|a| matches!(a, Action::Train { .. }));
    let sites = idx_of(&|a| matches!(a, Action::Site { .. }));
    let roads = idx_of(&|a| matches!(a, Action::BuildRoad { .. }));
    let buildings = idx_of(&|a| matches!(a, Action::BuildBuilding { .. }));
    let destroys = idx_of(&|a| matches!(a, Action::Destroy { .. }));
    let end_turn = idx_of(&|a| matches!(a, Action::EndTurn));

    assert!(!moves.is_empty() && !recovers.is_empty() && !promotes.is_empty());
    assert_eq!(disbands.len(), 1);
    assert!(!trains.is_empty() && !sites.is_empty() && !roads.is_empty());
    assert!(!buildings.is_empty() && !destroys.is_empty());
    assert_eq!(end_turn, vec![legal.len() - 1]);

    // Order: Move .. Recover .. Promote .. Disband -> Train -> Site -> BuildRoad -> BuildBuilding -> Destroy -> EndTurn.
    assert!(moves.last() < recovers.first());
    assert!(recovers.last() < promotes.first());
    assert!(promotes.last() < disbands.first());
    assert!(disbands.last() < trains.first());
    assert!(trains.last() < sites.first());
    assert!(sites.last() < roads.first());
    assert!(roads.last() < buildings.first());
    assert!(buildings.last() < destroys.first());
    assert!(destroys.last() < end_turn.first());

    // On (1,1), BurnForest comes after LumberHut and ClearForest:
    let site_actions_at_11: Vec<SiteAction> = legal
        .iter()
        .filter_map(|a| match *a {
            Action::Site { at, action } if at == p(1, 1) => Some(action),
            _ => None,
        })
        .collect();
    assert_eq!(
        site_actions_at_11,
        vec![
            SiteAction::LumberHut,
            SiteAction::ClearForest,
            SiteAction::BurnForest
        ]
    );
}

#[test]
fn building_level_returns_zero_on_unowned_or_empty_tiles() {
    let mut s = v2(StateBuilder::new(5)
        .city(0, p(2, 2))
        .terrain(p(0, 0), Terrain::Forest)
        .improvement(p(0, 0), Improvement::LumberHut)
        .building(p(0, 1), Building::Sawmill));
    // (0,1) is unowned (distance 2 from (2,2)), so building_level is 0 even though (0,0) has a LumberHut.
    assert_eq!(s.tile(p(0, 1)).owner_city, None);
    assert_eq!(building_level(&s, p(0, 1)), 0);
    assert_eq!(s.tile(p(0, 1)).building_level, 0);
    // (1,1) has no building.
    assert_eq!(building_level(&s, p(1, 1)), 0);
    // Synthetically un-building (0,1) leaves 0.
    tile_mut(&mut s, p(0, 1)).building = None;
    assert_eq!(building_level(&s, p(0, 1)), 0);
}
