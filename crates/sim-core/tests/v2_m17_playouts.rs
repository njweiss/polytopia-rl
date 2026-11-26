//! Milestone 17 tests (items 7 and 8): M16 carry-overs (Q49 scripted test, `road_step_cost`
//! load-time validation, gated single-BFS `update_connections` including `BorderGrowth`), plus
//! randomized v2 playouts exercising all M17 actions with invariants and replay determinism.

mod common;

use common::{all_pos, exp_cum, p, tile_mut};
use sim_core::actions::{apply, legal_actions, Action};
use sim_core::buildings::{building_level, market_income};
use sim_core::data::{Building, Reward, SiteAction, Tables, Tech, UnitType, ALL_BUILDINGS};
use sim_core::map::{MapParams, Terrain};
use sim_core::rng::Rng;
use sim_core::roads::is_connected;
use sim_core::score::score_breakdown;
use sim_core::setup::new_game_from_seed;
use sim_core::state::{City, CityId, Config, Ruleset, State, UnitId};
use sim_core::testing::StateBuilder;

const V2_UNITS: &str = include_str!("../../../data/v2/units.json");
const V2_TECHS: &str = include_str!("../../../data/v2/techs.json");
const V2_SITES: &str = include_str!("../../../data/v2/site_actions.json");
const V2_BUILDINGS: &str = include_str!("../../../data/v2/buildings.json");
const V2_MONUMENTS: &str = include_str!("../../../data/v2/monuments.json");
const V2_RULES: &str = include_str!("../../../data/v2/rules.json");

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

fn ok(s: &mut State, a: Action) {
    assert!(
        legal_actions(s).contains(&a),
        "{a:?} should be legal; legal = {:?}",
        legal_actions(s)
    );
    assert_eq!(apply(s, &a), Ok(()), "{a:?}");
}

// ============================================================================
// 7a. Q49 scripted test
// ============================================================================

/// Q49: Player 0 builds a neutral road that connects Player 1's non-capital city (pop 1 -> 2,
/// level 2, pending_reward = true).
/// - Player 0 is NOT blocked by Player 1's pending reward; Player 0 keeps acting and ends turn.
/// - When Player 1's turn starts, Player 1's only legal action is ChooseReward for that city.
/// - If Player 0 captures that pending city on Player 0's turn, it keeps pending_reward = true
///   and Player 0 must choose the reward immediately.
#[test]
fn q49_opponent_pending_reward_from_neutral_road_and_capture_of_pending_city() {
    // 9x9:
    // City 0 (P0 capital) at (1,1).
    // City 1 (P1 capital) at (5,1), level 3 pop 5.
    // City 2 (P1 non-capital) at (5,5), level 1 pop 1 (1 short of level 2).
    // Roads already on (5,2) and (5,4); (5,3) is neutral.
    // P0 has a Fresh Warrior on (5,5) (City 2) and another Warrior on (1,1) to verify P0 isn't blocked.
    let mut base = v2(StateBuilder::new(9)
        .city_with(0, p(1, 1), |c| {
            c.is_capital = true;
            c.level = 3;
            c.population = 5;
        })
        .city_with(1, p(5, 1), |c| {
            c.is_capital = true;
            c.level = 3;
            c.population = 5;
        })
        .city_with(1, p(5, 5), |c| {
            c.level = 1;
            c.population = 1;
        })
        .unit(0, UnitType::Warrior, p(5, 5))
        .unit(0, UnitType::Warrior, p(1, 1))
        .techs(0, &[Tech::Riding, Tech::Roads])
        .stars(0, 10)
        .explore_all());
    tile_mut(&mut base, p(5, 2)).road = true;
    tile_mut(&mut base, p(5, 4)).road = true;

    // P0 builds the neutral road at (5,3): City 2 connects to City 1 -> City 2 becomes level 2, pending!
    ok(&mut base, Action::BuildRoad { at: p(5, 3) });
    assert_eq!(city(&base, 2).connected_to, Some(CityId(1)));
    assert_eq!(city(&base, 2).population, 2);
    assert_eq!(city(&base, 2).level, 2);
    assert!(city(&base, 2).pending_reward);
    assert_eq!(city(&base, 1).population, 6);

    // Branch A: P0 does NOT capture City 2, just moves unit 1 and ends turn.
    let mut branch_a = base.clone();
    let legal_p0 = legal_actions(&branch_a);
    assert!(!legal_p0
        .iter()
        .any(|a| matches!(a, Action::ChooseReward { .. })));
    assert!(legal_p0.contains(&Action::EndTurn));
    ok(
        &mut branch_a,
        Action::Move {
            unit: UnitId(1),
            to: p(1, 2),
        },
    );
    ok(&mut branch_a, Action::EndTurn);
    assert_eq!(branch_a.current, 1);
    assert_eq!(
        legal_actions(&branch_a),
        vec![
            Action::ChooseReward {
                city: CityId(2),
                reward: Reward::Workshop
            },
            Action::ChooseReward {
                city: CityId(2),
                reward: Reward::Explorer
            },
        ]
    );
    ok(
        &mut branch_a,
        Action::ChooseReward {
            city: CityId(2),
            reward: Reward::Workshop,
        },
    );
    assert!(!city(&branch_a, 2).pending_reward);
    assert!(city(&branch_a, 2).workshop);
    assert!(legal_actions(&branch_a).contains(&Action::EndTurn));

    // Branch B: P0 captures City 2 while pending_reward is true!
    let mut branch_b = base;
    ok(&mut branch_b, Action::Capture { unit: UnitId(0) });
    assert_eq!(city(&branch_b, 2).owner, 0);
    assert!(
        city(&branch_b, 2).pending_reward,
        "captured city keeps its pending reward"
    );
    assert_eq!(
        legal_actions(&branch_b),
        vec![
            Action::ChooseReward {
                city: CityId(2),
                reward: Reward::Workshop
            },
            Action::ChooseReward {
                city: CityId(2),
                reward: Reward::Explorer
            },
        ],
        "capturer must resolve pending reward immediately"
    );
    ok(
        &mut branch_b,
        Action::ChooseReward {
            city: CityId(2),
            reward: Reward::Workshop,
        },
    );
    assert!(!city(&branch_b, 2).pending_reward);
    assert!(city(&branch_b, 2).workshop);
    assert!(legal_actions(&branch_b).contains(&Action::EndTurn));
}

// ============================================================================
// 7b. Tables::from_json_v2 road_step_cost validation
// ============================================================================

#[test]
fn tables_v2_rejects_invalid_road_step_cost() {
    let with_cost = |val: &str| {
        let rules = V2_RULES.replacen(
            "\"road_step_cost\": 0.5,",
            &format!("\"road_step_cost\": {val},"),
            1,
        );
        Tables::from_json_v2(
            V2_UNITS,
            V2_TECHS,
            V2_SITES,
            V2_BUILDINGS,
            V2_MONUMENTS,
            &rules,
        )
    };
    for bad in ["0.0", "-0.5", "-1.0", "0.3", "0.25", "1.2", "1e99"] {
        assert!(
            with_cost(bad).is_err(),
            "should reject road_step_cost={bad}"
        );
    }
    for good in ["0.5", "1.0", "1.5"] {
        assert!(
            with_cost(good).is_ok(),
            "should accept road_step_cost={good}"
        );
    }
}

// ============================================================================
// 7c. Gated update_connections (including BorderGrowth)
// ============================================================================

/// BorderGrowth claims a neutral road tile (3,2) that Player 1's city 2 relied on to connect to
/// Player 1's capital: update_connections MUST run on BorderGrowth and break Player 1's connection.
#[test]
fn border_growth_claiming_neutral_road_updates_connections() {
    // 9x9:
    // City 0 (P0 capital) at (1,1), level 4 pop 9 pending BorderGrowth. Owns (0..=2, 0..=2).
    // City 1 (P1 capital) at (5,1), level 3 pop 5. Owns (4..=6, 0..=2).
    // City 2 (P1) at (3,5), level 3 pop 5. Owns (2..=4, 4..=6).
    // Roads on (4,1) -> (3,1) -> (3,2) (neutral, distance 2 from City 0!) -> (3,3) (neutral) -> (3,4).
    //Wait: (3,1) is also distance 2 from City 0 at (1,1), so BorderGrowth on City 0 claims (3,1), (3,2) in P0 territory!
    let mut s = v2(StateBuilder::new(9)
        .city_with(0, p(1, 1), |c| {
            c.is_capital = true;
            c.level = 4;
            c.population = 9;
        })
        .city_with(1, p(5, 1), |c| {
            c.is_capital = true;
            c.level = 3;
            c.population = 5;
        })
        .city_with(1, p(3, 5), |c| {
            c.level = 3;
            c.population = 5;
        })
        .techs(0, &[Tech::Riding, Tech::Roads])
        .stars(0, 10)
        .explore_all());
    for q in [p(4, 1), p(3, 1), p(3, 2), p(3, 4)] {
        tile_mut(&mut s, q).road = true;
    }
    // Building neutral road (3,3) triggers update_connections: City 2 connects to City 1!
    ok(&mut s, Action::BuildRoad { at: p(3, 3) });
    assert_eq!(city(&s, 2).connected_to, Some(CityId(1)));
    assert_eq!(city(&s, 1).population, 6);
    assert_eq!(city(&s, 2).population, 6);

    // Now set City 0 pending_reward = true to pick BorderGrowth:
    s.cities[0].pending_reward = true;
    ok(
        &mut s,
        Action::ChooseReward {
            city: CityId(0),
            reward: Reward::BorderGrowth,
        },
    );
    // (3,2) and (3,1) were claimed by City 0 (P0) -> P1 can no longer use those roads!
    assert_eq!(s.tile(p(3, 2)).owner_city, Some(CityId(0)));
    assert!(!is_connected(&s, CityId(2)));
    assert_eq!(city(&s, 2).connected_to, None);
    assert_eq!(city(&s, 1).population, 5);
    assert_eq!(city(&s, 2).population, 5);
}

// ============================================================================
// 8. Random v2 playouts exercising all M17 actions, invariants & replay
// ============================================================================

fn pick_m17(legal: &[Action], rng: &mut Rng) -> Action {
    let m17: Vec<Action> = legal
        .iter()
        .copied()
        .filter(|a| {
            matches!(
                a,
                Action::Site {
                    action: SiteAction::BurnForest
                        | SiteAction::LumberHut
                        | SiteAction::Farm
                        | SiteAction::Mine,
                    ..
                } | Action::BuildBuilding { .. }
                    | Action::Destroy { .. }
                    | Action::Disband { .. }
                    | Action::BuildRoad { .. }
                    | Action::ChooseReward { .. }
                    | Action::Capture { .. }
            )
        })
        .collect();
    if !m17.is_empty() && rng.gen_range(2) == 0 {
        return m17[rng.gen_range(m17.len() as u32) as usize];
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

fn check_v2_invariants(s: &State, ctx: &str) {
    // 1. O(1) city lookup matches linear search, and tile building_level matches building_level().
    for (i, c) in s.cities.iter().enumerate() {
        assert_eq!(c.id, CityId(i as u32), "{ctx}: dense city id");
        assert_eq!(s.city(c.id), c, "{ctx}: O(1) city()");
        assert!(c.level >= 1, "{ctx}: level >= 1");
        if !c.pending_reward {
            assert!(
                c.population < exp_cum(c.level + 1) as i32,
                "{ctx}: missed level up: {c:?}"
            );
        }
    }

    // 2. Non-terminal invariants: building_level synced and connections match is_connected().
    if s.outcome.is_none() {
        for q in all_pos(s.size) {
            let t = s.tile(q);
            assert_eq!(
                t.building_level as u32,
                building_level(s, q),
                "{ctx}: building_level mismatch at {q:?}"
            );
            if let Some(b) = t.building {
                assert!(t.owner_city.is_some(), "{ctx}: unowned building at {q:?}");
                assert_eq!(t.terrain, Terrain::Field, "{ctx}: building on non-field");
                assert!(t.improvement.is_none() && t.city.is_none());
                if b == Building::Market {
                    assert!(market_income(s, q) <= 8, "{ctx}: market cap");
                }
            }
        }
        for c in &s.cities {
            let conn = is_connected(s, c.id);
            assert_eq!(
                c.connected_to.is_some(),
                conn,
                "{ctx}: connected_to mismatch for {:?} (conn={conn})",
                c.id
            );
        }
    }

    // 3. Score surplus never wraps or panics.
    for pl in 0..2u8 {
        let b = score_breakdown(s, pl);
        assert!(b.total() < 1_000_000, "{ctx}: suspicious score {b:?}");
    }
}

#[test]
fn random_v2_playouts_exercise_all_m17_actions_and_replay_identically() {
    let mut burn_count = 0usize;
    let mut building_counts = [0usize; 4];
    let mut disband_count = 0usize;
    let mut destroy_count = 0usize;
    let mut border_growth_count = 0usize;

    for seed in 0..24u64 {
        let cfg = Config {
            turn_cap: 16,
            ..v2_config()
        };
        let mut s = new_game_from_seed(seed, &MapParams::default(), &cfg);
        for pl in &mut s.players {
            for t in ALL_V2_TECHS {
                pl.techs.insert(t);
            }
            pl.stars += 60;
        }
        // Boost one city to level 3 so BorderGrowth shows up in playouts too.
        for c in &mut s.cities {
            c.level = 3;
            c.population = 8;
        }
        let start = s.clone();
        let mut rng = Rng::new(seed ^ 0x17_beef);
        let mut log = Vec::new();
        let mut states = Vec::new();
        while !s.is_terminal() && log.len() < 1500 {
            let legal = legal_actions(&s);
            let a = pick_m17(&legal, &mut rng);
            apply(&mut s, &a).unwrap();
            let ctx = format!("seed {seed} step {}", log.len());
            check_v2_invariants(&s, &ctx);
            match a {
                Action::Site {
                    action: SiteAction::BurnForest,
                    ..
                } => burn_count += 1,
                Action::BuildBuilding { building, .. } => {
                    building_counts[ALL_BUILDINGS.iter().position(|&b| b == building).unwrap()] +=
                        1;
                }
                Action::Disband { .. } => disband_count += 1,
                Action::Destroy { .. } => destroy_count += 1,
                Action::ChooseReward {
                    reward: Reward::BorderGrowth,
                    ..
                } => border_growth_count += 1,
                _ => {}
            }
            log.push(a);
            states.push(s.clone());
        }

        // Replay determinism check:
        let mut r = start;
        for (i, a) in log.iter().enumerate() {
            apply(&mut r, a).unwrap();
            assert_eq!(r, states[i], "seed {seed}: replay mismatch at {i}");
        }
    }

    assert!(burn_count > 10, "only {burn_count} BurnForests");
    for (i, &cnt) in building_counts.iter().enumerate() {
        assert!(cnt > 5, "only {cnt} {:?} built", ALL_BUILDINGS[i]);
    }
    assert!(disband_count > 10, "only {disband_count} Disbands");
    assert!(destroy_count > 10, "only {destroy_count} Destroys");
    assert!(
        border_growth_count > 2,
        "only {border_growth_count} BorderGrowths"
    );
}
