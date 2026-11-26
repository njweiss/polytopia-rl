//! Milestone 18 tests (items 2 and 3): corner lighthouses (placement at game start, discovery via
//! all fog-reveal paths, tile exclusivity, Q48/Q49 interactions) and the v2 Explorer step 3b
//! lighthouse bonus (docs/MILESTONE_18 "Lighthouses", "Explorer lighthouse bonus").

mod common;

use common::{all_pos, cheb, p, tile_mut};
use sim_core::actions::{apply, legal_actions, Action, ActionError};
use sim_core::data::{Reward, Tech, UnitType, ALL_BUILDINGS, ALL_MONUMENTS};
use sim_core::explorer::explorer_walk;
use sim_core::grid::Pos;
use sim_core::map::{MapParams, Terrain};
use sim_core::monuments::{corner_index, corners};
use sim_core::rng::Rng;
use sim_core::setup::new_game_from_seed;
use sim_core::state::{CityId, Config, Improvement, MonumentSet, Outcome, Ruleset, State, UnitId};
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

fn ok(s: &mut State, a: Action) {
    assert!(
        legal_actions(s).contains(&a),
        "{a:?} should be legal; legal = {:?}",
        legal_actions(s)
    );
    assert_eq!(apply(s, &a), Ok(()), "{a:?}");
}

fn assert_illegal(s: &State, a: Action) {
    assert!(!legal_actions(s).contains(&a), "{a:?} should not be legal");
    let mut t = s.clone();
    assert_eq!(apply(&mut t, &a), Err(ActionError::Illegal), "{a:?}");
    assert_eq!(t, *s, "{a:?}: illegal action mutated state");
}

// ============================================================================
// 1. corners / corner_index helpers
// ============================================================================

#[test]
fn corners_and_corner_index_cover_the_four_corners_in_row_major_order() {
    for size in [6u8, 7, 11, 20] {
        let last = size - 1;
        let expected = [p(0, 0), p(0, last), p(last, 0), p(last, last)];
        assert_eq!(corners(size), expected, "size {size}");
        for q in all_pos(size) {
            let want = expected.iter().position(|&c| c == q);
            assert_eq!(corner_index(size, q), want, "size {size}, pos {q:?}");
        }
    }
}

// ============================================================================
// 2. Lighthouse placement at game start
// ============================================================================

#[test]
fn v2_new_game_places_four_fogged_field_lighthouses_even_at_capital_distance_two() {
    let mut saw_dist2_capital = 0usize;
    let mut saw_cleared_non_field = 0usize;
    let mut saw_cleared_resource = 0usize;

    for seed in 0..80u64 {
        let v1 = new_game_from_seed(seed, &MapParams::default(), &Config::default());
        let v2 = new_game_from_seed(seed, &MapParams::default(), &v2_config());
        let c4 = [p(0, 0), p(0, 10), p(10, 0), p(10, 10)];

        for &q in &c4 {
            // V1 never has lighthouses.
            assert!(
                !v1.tile(q).lighthouse,
                "seed {seed}: v1 lighthouse at {q:?}"
            );
            // Track vacuity: V1 actually generated forests/mountains/resources on corners.
            if v1.tile(q).terrain != Terrain::Field {
                saw_cleared_non_field += 1;
            }
            if v1.tile(q).resource.is_some() {
                saw_cleared_resource += 1;
            }
            if v2.cities.iter().any(|c| cheb(c.pos, q) == 2) {
                saw_dist2_capital += 1;
            }

            // V2 corner invariants.
            let t = v2.tile(q);
            assert!(t.lighthouse, "seed {seed}: missing lighthouse at {q:?}");
            assert_eq!(t.terrain, Terrain::Field, "seed {seed}: terrain at {q:?}");
            assert_eq!(t.resource, None, "seed {seed}: resource at {q:?}");
            assert!(!t.village, "seed {seed}: village at {q:?}");
            assert_eq!(t.city, None, "seed {seed}: city at {q:?}");
            assert!(
                !v2.is_explored(0, q) && !v2.is_explored(1, q),
                "seed {seed}: corner {q:?} must start fogged for both players"
            );
        }

        // Non-corner tiles never have lighthouses and stay identical to V1.
        for q in all_pos(v2.size) {
            if !c4.contains(&q) {
                assert!(
                    !v2.tile(q).lighthouse,
                    "seed {seed}: extra lighthouse {q:?}"
                );
                assert_eq!(v2.tile(q), v1.tile(q), "seed {seed}: non-corner {q:?}");
                for pl in 0..2 {
                    assert_eq!(
                        v2.is_explored(pl, q),
                        v1.is_explored(pl, q),
                        "seed {seed}: explored mismatch at {q:?}"
                    );
                }
            }
        }
        assert_eq!(v2.players[0].lighthouses_discovered, 0);
        assert_eq!(v2.players[1].lighthouses_discovered, 0);
        assert_eq!(v2.cities[0].population, 0);
        assert_eq!(v2.cities[1].population, 0);
    }

    // Vacuity guards: the seed range hit distance-2 capitals and overwritten corner features.
    assert!(
        saw_dist2_capital >= 5,
        "only {saw_dist2_capital} distance-2 capital corners"
    );
    assert!(
        saw_cleared_non_field >= 20,
        "only {saw_cleared_non_field} non-field corners cleared"
    );
    assert!(
        saw_cleared_resource >= 1,
        "only {saw_cleared_resource} corner resources cleared"
    );
}

#[test]
fn v2_full_observability_starts_all_explored_with_zero_lighthouses_discovered() {
    let cfg = Config {
        full_observability: true,
        ruleset: Ruleset::V2,
        ..Config::default()
    };
    for seed in 0..10u64 {
        let s = new_game_from_seed(seed, &MapParams::default(), &cfg);
        let c4 = [p(0, 0), p(0, 10), p(10, 0), p(10, 10)];
        for &q in &c4 {
            assert!(s.tile(q).lighthouse);
            assert_eq!(s.tile(q).terrain, Terrain::Field);
            assert_eq!(s.tile(q).resource, None);
        }
        for pl in 0..2u8 {
            assert!(
                s.players[pl as usize].explored.iter().all(|&e| e),
                "seed {seed}: full_observability must explore every tile"
            );
            assert_eq!(s.players[pl as usize].lighthouses_discovered, 0);
        }
        assert_eq!(s.cities[0].population, 0);
        assert_eq!(s.cities[1].population, 0);
    }
}

// ============================================================================
// 3. Lighthouse discovery paths (M2), once-per-player, both players, Q48/Q49
// ============================================================================

#[test]
fn lighthouse_discovery_by_move_grants_once_per_player_and_both_players_can_claim() {
    // 7x7: P0 capital at (3,3) (level 3 pop 5), P1 capital at (5,5) (level 3 pop 5).
    // Corner (0,0) has a lighthouse, initially hidden from both players.
    // P0 Warrior at (2,2) moves to (1,1) -> reveals (0,0).
    // P1 Warrior at (2,1) moves to (1,0) -> reveals (0,0) for P1.
    let mut s = v2(StateBuilder::new(7)
        .lighthouse(p(0, 0))
        .city_with(0, p(3, 3), |c| {
            c.is_capital = true;
            c.level = 3;
            c.population = 5;
        })
        .city_with(1, p(5, 5), |c| {
            c.is_capital = true;
            c.level = 3;
            c.population = 5;
        })
        .unit(0, UnitType::Warrior, p(2, 2))
        .unit(1, UnitType::Warrior, p(2, 1))
        .explore_all()
        .hide(0, &[p(0, 0)])
        .hide(1, &[p(0, 0)]));

    // P0 discovers (0,0) by moving to (1,1).
    ok(
        &mut s,
        Action::Move {
            unit: UnitId(0),
            to: p(1, 1),
        },
    );
    assert!(s.is_explored(0, p(0, 0)));
    assert!(!s.is_explored(1, p(0, 0)));
    assert_eq!(s.players[0].lighthouses_discovered, 1 << 0);
    assert_eq!(s.players[1].lighthouses_discovered, 0);
    assert_eq!(s.city(CityId(0)).population, 6);
    assert_eq!(s.city(CityId(1)).population, 5);

    // End turn -> P1 moves to (1,0) and discovers the SAME corner (0,0).
    ok(&mut s, Action::EndTurn);
    ok(
        &mut s,
        Action::Move {
            unit: UnitId(1),
            to: p(1, 0),
        },
    );
    assert!(s.is_explored(1, p(0, 0)));
    assert_eq!(s.players[1].lighthouses_discovered, 1 << 0);
    assert_eq!(s.city(CityId(1)).population, 6);

    // End turn -> P0 steps directly onto (0,0) (revisit): NO second +1 population.
    ok(&mut s, Action::EndTurn);
    ok(
        &mut s,
        Action::Move {
            unit: UnitId(0),
            to: p(0, 0),
        },
    );
    assert_eq!(s.players[0].lighthouses_discovered, 1 << 0);
    assert_eq!(s.city(CityId(0)).population, 6);
}

#[test]
fn lighthouse_discovery_by_kill_move_grants_capital_population_and_levels_up() {
    // 7x7: P0 capital at (5,5) at level 1 pop 1 (1 short of level 2!).
    // P0 Warrior at (2,2) kills a 1-HP P1 Warrior at (1,1) -> kill-moves onto (1,1) -> reveals (0,0).
    let mut s = v2(StateBuilder::new(7)
        .lighthouse(p(0, 0))
        .city_with(0, p(5, 5), |c| {
            c.is_capital = true;
            c.level = 1;
            c.population = 1;
        })
        .city_with(1, p(5, 1), |c| c.is_capital = true)
        .unit(0, UnitType::Warrior, p(2, 2))
        .unit_with(1, UnitType::Warrior, p(1, 1), |u| u.health = 1)
        .explore_all()
        .hide(0, &[p(0, 0)]));

    ok(
        &mut s,
        Action::Attack {
            unit: UnitId(0),
            target: p(1, 1),
        },
    );
    assert_eq!(s.unit(UnitId(0)).unwrap().pos, p(1, 1));
    assert!(s.is_explored(0, p(0, 0)));
    assert_eq!(s.players[0].lighthouses_discovered, 1 << 0);
    assert_eq!(s.city(CityId(0)).population, 2);
    assert_eq!(s.city(CityId(0)).level, 2);
    assert!(s.city(CityId(0)).pending_reward);
}

#[test]
fn lighthouse_discovery_by_border_growth_and_city_capture() {
    // 7x7: P0 capital at (2,2), level 4 pop 9 pending BorderGrowth. Corner (0,0) is hidden and unowned.
    // BorderGrowth claims (0,0) and reveals it -> discovers corner 0, capital pop 9 -> 10.
    let mut s = v2(StateBuilder::new(7)
        .lighthouse(p(0, 0))
        .city_with(0, p(2, 2), |c| {
            c.is_capital = true;
            c.level = 4;
            c.population = 9;
            c.pending_reward = true;
        })
        .city_with(1, p(5, 5), |c| {
            c.is_capital = true;
            c.level = 3;
            c.population = 5;
        })
        .explore_all()
        .hide(0, &[p(0, 0)])
        .hide(1, &[p(0, 0)]));

    ok(
        &mut s,
        Action::ChooseReward {
            city: CityId(0),
            reward: Reward::BorderGrowth,
        },
    );
    assert_eq!(s.tile(p(0, 0)).owner_city, Some(CityId(0)));
    assert!(s.is_explored(0, p(0, 0)));
    assert_eq!(s.players[0].lighthouses_discovered, 1 << 0);
    assert_eq!(s.city(CityId(0)).population, 10);

    // Non-terminal city capture revealing corner (0,6) via captured city's territory:
    // P1 owns a non-capital city at (2,4) whose territory includes (0,6) (from earlier BorderGrowth).
    // P0 captures city at (2,4) with (0,6) hidden -> reveals (0,6) -> P0 discovers corner 1!
    let mut s2 = v2(StateBuilder::new(7)
        .lighthouse(p(0, 6))
        .city_with(0, p(5, 1), |c| {
            c.is_capital = true;
            c.level = 3;
            c.population = 5;
        })
        .city_with(1, p(5, 5), |c| c.is_capital = true)
        .city(1, p(2, 4))
        .unit(0, UnitType::Warrior, p(2, 4))
        .explore_all()
        .hide(0, &[p(0, 6)]));
    tile_mut(&mut s2, p(0, 6)).owner_city = Some(CityId(2));

    ok(&mut s2, Action::Capture { unit: UnitId(0) });
    assert!(s2.is_explored(0, p(0, 6)));
    assert_eq!(s2.players[0].lighthouses_discovered, 1 << 1);
    assert_eq!(s2.city(CityId(0)).population, 6);
}

#[test]
fn lighthouse_discovery_by_forced_super_unit_push_triggers_q49_opponent_reward() {
    // 7x7: P0 owns City 0 (capital at (2,2), level 5, pop 14, pending SuperUnit).
    // P1 owns City 1 (capital at (5,5), level 1, pop 1 -> 1 short of level 2!).
    // A P1 Warrior stands on (2,2) with last_dir = SE (dr=1, dc=1), so enemy push is opposite: NW -> (1,1)!
    // Pushing P1's Warrior onto (1,1) reveals (0,0) (lighthouse) for P1 -> P1 capital gains +1 pop ->
    // level 2, pending_reward = true (Q49: P0 is NOT blocked; P1 must choose on P1's turn).
    let mut s = v2(StateBuilder::new(7)
        .lighthouse(p(0, 0))
        .city_with(0, p(2, 2), |c| {
            c.is_capital = true;
            c.level = 5;
            c.population = 14;
            c.pending_reward = true;
        })
        .city_with(1, p(5, 5), |c| {
            c.is_capital = true;
            c.level = 1;
            c.population = 1;
        })
        .unit_with(1, UnitType::Warrior, p(2, 2), |u| {
            u.last_dir = Some(sim_core::state::Dir { dr: 1, dc: 1 });
        })
        .explore_all()
        .hide(1, &[p(0, 0)]));

    ok(
        &mut s,
        Action::ChooseReward {
            city: CityId(0),
            reward: Reward::SuperUnit,
        },
    );
    assert_eq!(s.unit(UnitId(0)).unwrap().pos, p(1, 1));
    assert!(s.is_explored(1, p(0, 0)));
    assert_eq!(s.players[1].lighthouses_discovered, 1 << 0);
    assert_eq!(s.city(CityId(1)).population, 2);
    assert_eq!(s.city(CityId(1)).level, 2);
    assert!(s.city(CityId(1)).pending_reward);
    // Q49: P0 is not blocked by P1's pending reward.
    assert!(legal_actions(&s).contains(&Action::EndTurn));
    ok(&mut s, Action::EndTurn);
    assert_eq!(s.current, 1);
    assert_eq!(
        legal_actions(&s),
        vec![
            Action::ChooseReward {
                city: CityId(1),
                reward: Reward::Workshop
            },
            Action::ChooseReward {
                city: CityId(1),
                reward: Reward::Explorer
            },
        ]
    );
}

#[test]
fn q48_terminal_capital_capture_skips_lighthouse_discovery() {
    // 7x7: P1's capital at (2,2) owns corner (0,0) (lighthouse, hidden from P0).
    // P0 Warrior on (2,2) captures P1's capital -> game ends immediately -> Q48 skips discovery!
    let mut s = v2(StateBuilder::new(7)
        .lighthouse(p(0, 0))
        .city_with(0, p(5, 5), |c| {
            c.is_capital = true;
            c.level = 3;
            c.population = 5;
        })
        .city_with(1, p(2, 2), |c| c.is_capital = true)
        .unit(0, UnitType::Warrior, p(2, 2))
        .explore_all()
        .hide(0, &[p(0, 0)]));
    tile_mut(&mut s, p(0, 0)).owner_city = Some(CityId(1));

    ok(&mut s, Action::Capture { unit: UnitId(0) });
    assert_eq!(s.outcome, Some(Outcome::Win(0)));
    assert_eq!(s.players[0].lighthouses_discovered, 0);
    assert_eq!(s.city(CityId(0)).population, 5);
}

#[test]
fn lighthouse_blocks_road_building_and_monument_but_allows_units() {
    // 7x7: P0 capital at (1,1) owns (0,0) which has a lighthouse.
    // Also place a LumberHut at (0,1) so Sawmill would otherwise be legal at (0,0),
    // and unlock all monuments for P0.
    // Neutral explored lighthouse at (0,6) also blocks BuildRoad.
    let mut s = v2(StateBuilder::new(7)
        .lighthouse(p(0, 0))
        .lighthouse(p(0, 6))
        .city_with(0, p(1, 1), |c| c.is_capital = true)
        .city_with(1, p(5, 5), |c| c.is_capital = true)
        .terrain(p(0, 1), Terrain::Forest)
        .improvement(p(0, 1), Improvement::LumberHut)
        .unit(0, UnitType::Warrior, p(1, 0))
        .techs(
            0,
            &[
                Tech::Riding,
                Tech::Roads,
                Tech::Forestry,
                Tech::Mathematics,
                Tech::Farming,
                Tech::Construction,
                Tech::Climbing,
                Tech::Mining,
                Tech::Smithery,
                Tech::Trade,
            ],
        )
        .stars(0, 30)
        .player_with(0, |pl| {
            pl.monuments_unlocked = MonumentSet::from_slice(&ALL_MONUMENTS);
        })
        .explore_all());

    assert_eq!(s.tile(p(0, 0)).owner_city, Some(CityId(0)));
    assert_eq!(s.tile(p(0, 6)).owner_city, None);

    // Owned lighthouse rejects BuildRoad, BuildBuilding, BuildMonument.
    assert_illegal(&s, Action::BuildRoad { at: p(0, 0) });
    for b in ALL_BUILDINGS {
        assert_illegal(
            &s,
            Action::BuildBuilding {
                at: p(0, 0),
                building: b,
            },
        );
    }
    for m in ALL_MONUMENTS {
        assert_illegal(
            &s,
            Action::BuildMonument {
                at: p(0, 0),
                monument: m,
            },
        );
    }

    // Neutral explored lighthouse rejects BuildRoad.
    assert_illegal(&s, Action::BuildRoad { at: p(0, 6) });

    // Contrast: adjacent plain owned field (1,0) allows BuildRoad and BuildMonument (after moving off/on),
    // and moving the Warrior onto (0,0) IS legal!
    ok(
        &mut s,
        Action::Move {
            unit: UnitId(0),
            to: p(0, 0),
        },
    );
    assert_eq!(s.unit(UnitId(0)).unwrap().pos, p(0, 0));
}

// ============================================================================
// 4. Explorer lighthouse bonus (step 3b)
// ============================================================================

/// Helper: 20x20 board, all Mountains except `open` tiles (Field). No Climbing.
fn walled(size: u8, open: &[Pos]) -> StateBuilder {
    let mut b = StateBuilder::new(size);
    for q in all_pos(size) {
        if !open.contains(&q) {
            b = b.terrain(q, Terrain::Mountain);
        }
    }
    b
}

/// Geometry: capital at (1,2), corner lighthouse at (0,0), open tiles (0,0), (1,1), (1,2),
/// and east corridor (2,3)..=(2,18). Pre-explore (1,0), (2,0), (3,2).
/// Start reveal around (1,2) reveals rows 0..=2 x cols 1..=3.
/// Move 1 distance-1 candidates:
/// - (1,1): 1 fog tile {(0,0)} hiding a lighthouse -> V2 score = 1.0 + 3.5 = 4.5 (V1 score = 1.0).
/// - (2,3): 4 plain fog tiles {(1,4), (2,4), (3,3), (3,4)} -> score = 4.0.
///
/// Under V2: deterministically picks (1,1) (0 RNG draws!), discovers (0,0) (+1 pop), then walks
/// (1,2) -> (2,3) -> .. -> (2,15), revealing (2,16) but NOT (2,17).
///
/// Under V1 (or once (0,0) is already explored): picks (2,3) and marches east to (2,17), revealing
/// (2,18) and NEVER visiting (1,1).
fn explorer_fork_board(ruleset: Ruleset, seed: u64) -> State {
    let mut open = vec![p(0, 0), p(1, 1), p(1, 2)];
    open.extend((3..=18).map(|c| p(2, c)));
    let mut b = walled(20, &open)
        .lighthouse(p(0, 0))
        .city_with(0, p(1, 2), |c| {
            c.is_capital = true;
            c.level = 2;
            c.population = 2;
            c.pending_reward = true;
        })
        .city_with(1, p(18, 18), |c| c.is_capital = true)
        .explore(0, &[p(1, 0), p(2, 0), p(3, 2)])
        .rng(Rng::new(seed));
    if ruleset == Ruleset::V2 {
        b = b.config(v2_config());
    }
    b.build()
}

#[test]
fn v2_explorer_lighthouse_bonus_beats_four_plain_fog_tiles_after_cap() {
    for seed in 0..8u64 {
        let before = explorer_fork_board(Ruleset::V2, seed);
        let mut s = before.clone();
        ok(
            &mut s,
            Action::ChooseReward {
                city: CityId(0),
                reward: Reward::Explorer,
            },
        );
        // Deterministic: no tie, so RNG is untouched!
        assert_eq!(
            s.rng, before.rng,
            "seed {seed}: 4.5 vs 4.0 must not tie (catches capping after bonus)"
        );
        assert!(s.is_explored(0, p(0, 0)), "seed {seed}: lighthouse found");
        assert_eq!(s.players[0].lighthouses_discovered, 1 << 0);
        assert_eq!(s.city(CityId(0)).population, 3, "capital gained +1 pop");
        assert!(s.is_explored(0, p(2, 16)));
        assert!(!s.is_explored(0, p(2, 17)));
    }
}

#[test]
fn v2_explorer_stops_giving_bonus_once_lighthouse_is_already_explored() {
    // Same fork, except (0,0) is ALREADY explored and (1,0) is left hidden so (1,1) is still a
    // fog candidate (1 plain fog tile, adjacent to an ALREADY-EXPLORED lighthouse).
    // (1,1) scores 1.0 < (2,3) at 4.0 -> Explorer deterministically marches east to (2,17).
    let mut open = vec![p(0, 0), p(1, 1), p(1, 2)];
    open.extend((3..=18).map(|c| p(2, c)));
    let before = walled(20, &open)
        .lighthouse(p(0, 0))
        .city_with(0, p(1, 2), |c| {
            c.is_capital = true;
            c.level = 2;
            c.population = 2;
            c.pending_reward = true;
        })
        .city_with(1, p(18, 18), |c| c.is_capital = true)
        .explore(0, &[p(0, 0), p(2, 0), p(3, 2)])
        .rng(Rng::new(42))
        .config(v2_config())
        .build();
    let mut s = before.clone();
    ok(
        &mut s,
        Action::ChooseReward {
            city: CityId(0),
            reward: Reward::Explorer,
        },
    );
    assert_eq!(s.rng, before.rng, "4.0 vs 1.0 must not tie");
    assert!(
        !s.is_explored(0, p(1, 0)),
        "Explorer must not head to already-explored lighthouse"
    );
    assert!(s.is_explored(0, p(2, 18)), "Explorer marched east 15 steps");
    assert_eq!(s.players[0].lighthouses_discovered, 0);
    assert_eq!(s.city(CityId(0)).population, 2);
}

#[test]
fn v1_explorer_ignores_lighthouse_bonus_on_same_geometry() {
    let before = explorer_fork_board(Ruleset::V1, 0);
    let mut s = before.clone();
    explorer_walk(&mut s, CityId(0));
    assert_eq!(s.rng, before.rng);
    assert!(
        !s.is_explored(0, p(0, 0)),
        "under V1, (2,3) with 4 fog beats (1,1) with 1 fog"
    );
    assert!(s.is_explored(0, p(2, 18)));
    assert_eq!(s.players[0].lighthouses_discovered, 0);
    assert_eq!(s.city(CityId(0)).population, 2);
}

#[test]
fn v2_explorer_nearest_distance_still_beats_farther_lighthouse() {
    // Capital at (1,3). Open: (0,0) lighthouse, (1,1), (1,2), (1,3), and (2,4)..=(2,18).
    // Pre-explore (0,1), (1,1), (2,1), (1,0), (2,0) so (1,2) has 0 fog.
    // Distance 1 candidate: (2,4) (plain fog). Distance 2 candidate: (1,1) (unexplored lighthouse (0,0)).
    // Step 2 nearest-distance wins -> Explorer marches east and never reveals (0,0).
    let mut open = vec![p(0, 0), p(1, 1), p(1, 2), p(1, 3)];
    open.extend((4..=18).map(|c| p(2, c)));
    let before = walled(20, &open)
        .lighthouse(p(0, 0))
        .city_with(0, p(1, 3), |c| {
            c.is_capital = true;
            c.level = 2;
            c.population = 2;
            c.pending_reward = true;
        })
        .city_with(1, p(18, 18), |c| c.is_capital = true)
        .explore(0, &[p(0, 1), p(1, 1), p(2, 1), p(1, 0), p(2, 0)])
        .rng(Rng::new(9))
        .config(v2_config())
        .build();
    let mut s = before.clone();
    ok(
        &mut s,
        Action::ChooseReward {
            city: CityId(0),
            reward: Reward::Explorer,
        },
    );
    assert_eq!(s.rng, before.rng);
    assert!(!s.is_explored(0, p(0, 0)));
    assert!(s.is_explored(0, p(2, 19)));
    assert_eq!(s.players[0].lighthouses_discovered, 0);
}

#[test]
fn v2_explorer_discovering_lighthouse_can_level_up_capital() {
    // Capital at (1,2) starts at level 2, pop 4 (1 short of level 3 = 5), pending Explorer!
    // Explorer discovers (0,0) -> capital pop becomes 5 -> after Explorer finishes, level_up_check
    // promotes capital to level 3 with pending_reward = true!
    let mut s = explorer_fork_board(Ruleset::V2, 0);
    s.cities[0].population = 4;
    ok(
        &mut s,
        Action::ChooseReward {
            city: CityId(0),
            reward: Reward::Explorer,
        },
    );
    assert_eq!(s.players[0].lighthouses_discovered, 1 << 0);
    assert_eq!(s.city(CityId(0)).population, 5);
    assert_eq!(s.city(CityId(0)).level, 3);
    assert!(s.city(CityId(0)).pending_reward);
}
