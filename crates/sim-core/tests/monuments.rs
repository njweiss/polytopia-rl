//! Milestone 18 tests (items 1, 4, 5, 6): v1 identity, `MonumentSet`, monument unlocks
//! (`ParkOfFortune`, `GateOfPower`, `EyeOfGod`), `BuildMonument` legality/order/apply, `Destroy`
//! on monuments, `ScoreBreakdown.monuments` & capture transfer, and `disband_refund_divisor == 0`
//! load-time validation (docs/MILESTONE_18).

mod common;

use common::{p, pick, tile_mut};
use sim_core::actions::{apply, legal_actions, Action, ActionError};
use sim_core::data::{
    Building, Monument, Reward, SiteAction, Tables, Tech, UnitType, ALL_BUILDINGS, ALL_MONUMENTS,
    ALL_TECHS,
};
use sim_core::grid::Pos;
use sim_core::map::{MapParams, Resource, Terrain};
use sim_core::rng::Rng;
use sim_core::score::{score, score_breakdown};
use sim_core::setup::new_game_from_seed;
use sim_core::state::{CityId, Config, Improvement, MonumentSet, Ruleset, State, UnitId};
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

fn build_mon(at: Pos, monument: Monument) -> Action {
    Action::BuildMonument { at, monument }
}

// ============================================================================
// 1. MonumentSet & ALL_MONUMENTS + v1 identity
// ============================================================================

#[test]
fn monument_set_operations_and_all_monuments_order() {
    assert_eq!(
        ALL_MONUMENTS,
        [
            Monument::ParkOfFortune,
            Monument::GateOfPower,
            Monument::EyeOfGod
        ]
    );
    let empty = MonumentSet::new();
    assert_eq!(empty.bits(), 0);
    for m in ALL_MONUMENTS {
        assert!(!empty.contains(m));
    }
    let mut s = MonumentSet::new();
    s.insert(Monument::ParkOfFortune);
    assert_eq!(s.bits(), 0b001);
    assert!(s.contains(Monument::ParkOfFortune));
    assert!(!s.contains(Monument::GateOfPower));
    assert!(!s.contains(Monument::EyeOfGod));

    s.insert(Monument::EyeOfGod);
    assert_eq!(s.bits(), 0b101);
    assert_eq!(
        s,
        MonumentSet::from_slice(&[Monument::EyeOfGod, Monument::ParkOfFortune])
    );

    let full = MonumentSet::from_slice(&ALL_MONUMENTS);
    assert_eq!(full.bits(), 0b111);
    for m in ALL_MONUMENTS {
        assert!(full.contains(m));
    }
}

#[test]
fn v1_identity_no_lighthouses_monuments_kills_or_monument_score() {
    // Scripted V1 state with synthetically unlocked monuments and a monument on a tile:
    // BuildMonument is still illegal and ScoreBreakdown.monuments is 0 under V1.
    let s = StateBuilder::new(5)
        .city_with(0, p(2, 2), |c| {
            c.is_capital = true;
            c.level = 5;
            c.population = 14;
        })
        .city_with(1, p(4, 4), |c| c.is_capital = true)
        .monument(p(1, 1), Monument::ParkOfFortune)
        .unit(0, UnitType::Warrior, p(2, 3))
        .unit_with(1, UnitType::Warrior, p(3, 3), |u| u.health = 1)
        .techs(0, &ALL_TECHS)
        .stars(0, 30)
        .player_with(0, |pl| {
            pl.monuments_unlocked = MonumentSet::from_slice(&ALL_MONUMENTS);
        })
        .explore_all()
        .build();

    assert_eq!(s.config.ruleset, Ruleset::V1);
    assert_eq!(score_breakdown(&s, 0).monuments, 0);
    assert!(legal_actions(&s)
        .iter()
        .all(|a| !matches!(a, Action::BuildMonument { .. })));
    for m in ALL_MONUMENTS {
        assert_illegal(&s, build_mon(p(1, 2), m));
    }

    // Attack kill under V1 does NOT increment PlayerState.kills.
    let mut after_kill = s.clone();
    ok(
        &mut after_kill,
        Action::Attack {
            unit: UnitId(0),
            target: p(3, 3),
        },
    );
    assert_eq!(after_kill.players[0].kills, 0);

    // Random V1 playouts never produce lighthouses, monuments, kills, or monument score.
    for seed in 0..6u64 {
        let mut g = new_game_from_seed(seed, &MapParams::default(), &Config::default());
        let mut rng = Rng::new(seed ^ 0x18_0001);
        let mut steps = 0;
        while !g.is_terminal() && steps < 800 {
            let legal = legal_actions(&g);
            assert!(legal
                .iter()
                .all(|a| !matches!(a, Action::BuildMonument { .. })));
            if steps % 40 == 0 {
                assert_illegal(&g, build_mon(p(0, 0), Monument::ParkOfFortune));
            }
            let a = pick(&mut rng, &legal);
            apply(&mut g, &a).unwrap();
            assert!(g
                .tiles
                .iter()
                .all(|t| !t.lighthouse && t.monument.is_none()));
            for pl in 0..2u8 {
                let p_st = &g.players[pl as usize];
                assert_eq!(p_st.kills, 0);
                assert_eq!(p_st.lighthouses_discovered, 0);
                assert_eq!(p_st.monuments_unlocked, MonumentSet::new());
                assert_eq!(p_st.monuments_placed, MonumentSet::new());
                assert_eq!(score_breakdown(&g, pl).monuments, 0);
            }
            steps += 1;
        }
    }
}

// ============================================================================
// 2. Monument unlocks
// ============================================================================

#[test]
fn park_of_fortune_unlocks_at_level_5_by_growth_cascade_and_monument() {
    // (a) Growth via Site action: level 4 pop 13 + 1 (HarvestFruit) -> level 5 -> unlocks ParkOfFortune.
    let mut s = v2(StateBuilder::new(5)
        .city_with(0, p(2, 2), |c| {
            c.is_capital = true;
            c.level = 4;
            c.population = 13;
        })
        .city_with(1, p(4, 4), |c| c.is_capital = true)
        .resource(p(1, 1), Resource::Fruit)
        .techs(0, &[Tech::Organization])
        .stars(0, 5)
        .explore_all());
    assert!(!s.players[0]
        .monuments_unlocked
        .contains(Monument::ParkOfFortune));
    ok(
        &mut s,
        Action::Site {
            at: p(1, 1),
            action: SiteAction::HarvestFruit,
        },
    );
    assert_eq!(s.city(CityId(0)).level, 5);
    assert!(s.players[0]
        .monuments_unlocked
        .contains(Monument::ParkOfFortune));

    // (b) Cascade via PopulationGrowth (+3): level 4 pending, pop 11 + 3 = 14 -> cascades to level 5!
    let mut s2 = v2(StateBuilder::new(5)
        .city_with(0, p(2, 2), |c| {
            c.is_capital = true;
            c.level = 4;
            c.population = 11;
            c.pending_reward = true;
        })
        .city_with(1, p(4, 4), |c| c.is_capital = true)
        .explore_all());
    ok(
        &mut s2,
        Action::ChooseReward {
            city: CityId(0),
            reward: Reward::PopulationGrowth,
        },
    );
    assert_eq!(s2.city(CityId(0)).level, 5);
    assert!(s2.players[0]
        .monuments_unlocked
        .contains(Monument::ParkOfFortune));

    // (c) Placing GateOfPower (+3 pop) on a level-4 city at pop 11 -> level 5 -> unlocks ParkOfFortune immediately!
    let mut s3 = v2(StateBuilder::new(5)
        .city_with(0, p(2, 2), |c| {
            c.is_capital = true;
            c.level = 4;
            c.population = 11;
        })
        .city_with(1, p(4, 4), |c| c.is_capital = true)
        .player_with(0, |pl| pl.monuments_unlocked.insert(Monument::GateOfPower))
        .explore_all());
    ok(&mut s3, build_mon(p(1, 1), Monument::GateOfPower));
    assert_eq!(s3.city(CityId(0)).level, 5);
    assert!(s3.players[0]
        .monuments_unlocked
        .contains(Monument::ParkOfFortune));
}

#[test]
fn park_of_fortune_unlocks_on_capturing_enemy_level_5_city_and_stays_unlocked_if_lost() {
    // 7x7: P1 owns non-capital City 2 at (3,3) at level 5 pop 14.
    // P0 Warrior stands on (3,3) and captures it -> P0 unlocks ParkOfFortune!
    // Then P1 captures City 2 back -> P0's ParkOfFortune remains unlocked permanently!
    let mut s = v2(StateBuilder::new(7)
        .city_with(0, p(1, 1), |c| c.is_capital = true)
        .city_with(1, p(5, 5), |c| c.is_capital = true)
        .city_with(1, p(3, 3), |c| {
            c.level = 5;
            c.population = 14;
        })
        .unit(0, UnitType::Warrior, p(3, 3))
        .unit(1, UnitType::Warrior, p(3, 4))
        .explore_all());

    assert!(!s.players[0]
        .monuments_unlocked
        .contains(Monument::ParkOfFortune));
    ok(&mut s, Action::Capture { unit: UnitId(0) });
    assert_eq!(s.city(CityId(2)).owner, 0);
    assert!(
        s.players[0]
            .monuments_unlocked
            .contains(Monument::ParkOfFortune),
        "capturing an enemy level-5 city must unlock ParkOfFortune"
    );

    // P0's warrior moves off (3,3) next turn; let's just move P1's warrior in after killing/moving.
    // Simpler: end turn, P1 attacks and kills P0's warrior (set HP to 1 first), next turn captures back.
    s.units[0].health = 1;
    ok(&mut s, Action::EndTurn);
    ok(
        &mut s,
        Action::Attack {
            unit: UnitId(1),
            target: p(3, 3),
        },
    );
    assert_eq!(s.unit(UnitId(1)).unwrap().pos, p(3, 3));
    ok(&mut s, Action::EndTurn);
    ok(&mut s, Action::EndTurn);
    ok(&mut s, Action::Capture { unit: UnitId(1) });
    assert_eq!(s.city(CityId(2)).owner, 1);
    assert!(
        s.players[0]
            .monuments_unlocked
            .contains(Monument::ParkOfFortune),
        "unlock must remain permanently even after losing the level-5 city"
    );
}

#[test]
fn gate_of_power_kill_tracking_attack_retaliation_super_unit_and_exclusions() {
    // (a) Attack kill by normal unit at 9 -> 10 unlocks GateOfPower (not at 9).
    let mut s = v2(StateBuilder::new(5)
        .city_with(0, p(1, 1), |c| c.is_capital = true)
        .city_with(1, p(4, 4), |c| c.is_capital = true)
        .unit(0, UnitType::Warrior, p(2, 2))
        .unit_with(1, UnitType::Warrior, p(2, 3), |u| u.health = 1)
        .unit_with(1, UnitType::Warrior, p(3, 2), |u| u.health = 1)
        .player_with(0, |pl| pl.kills = 8)
        .explore_all());
    // 9th kill: still locked.
    ok(
        &mut s,
        Action::Attack {
            unit: UnitId(0),
            target: p(2, 3),
        },
    );
    assert_eq!(s.players[0].kills, 9);
    assert!(!s.players[0]
        .monuments_unlocked
        .contains(Monument::GateOfPower));
    ok(&mut s, Action::EndTurn);
    ok(&mut s, Action::EndTurn);
    // 10th kill: unlocked!
    ok(
        &mut s,
        Action::Attack {
            unit: UnitId(0),
            target: p(3, 2),
        },
    );
    assert_eq!(s.players[0].kills, 10);
    assert!(s.players[0]
        .monuments_unlocked
        .contains(Monument::GateOfPower));

    // (b) Retaliation kill: P0's 1-HP Warrior attacks P1's full-HP Warrior and dies ->
    // P1's kills goes 9 -> 10 and unlocks GateOfPower for P1!
    let mut s_ret = v2(StateBuilder::new(5)
        .city_with(0, p(1, 1), |c| c.is_capital = true)
        .city_with(1, p(4, 4), |c| c.is_capital = true)
        .unit_with(0, UnitType::Warrior, p(2, 2), |u| u.health = 1)
        .unit(1, UnitType::Warrior, p(2, 3))
        .player_with(1, |pl| pl.kills = 9)
        .explore_all());
    ok(
        &mut s_ret,
        Action::Attack {
            unit: UnitId(0),
            target: p(2, 3),
        },
    );
    assert!(
        s_ret.unit(UnitId(0)).is_none(),
        "attacker died to retaliation"
    );
    assert_eq!(s_ret.players[0].kills, 0);
    assert_eq!(s_ret.players[1].kills, 10);
    assert!(
        s_ret.players[1]
            .monuments_unlocked
            .contains(Monument::GateOfPower),
        "retaliation kill must increment defender owner's kills and unlock GateOfPower"
    );

    // (c) Super Unit kill increments PlayerState.kills (9 -> 10 unlocks GateOfPower).
    let mut s_su = v2(StateBuilder::new(5)
        .city_with(0, p(1, 1), |c| c.is_capital = true)
        .city_with(1, p(4, 4), |c| c.is_capital = true)
        .unit(0, UnitType::SuperUnit, p(2, 2))
        .unit(1, UnitType::Warrior, p(2, 3))
        .player_with(0, |pl| pl.kills = 9)
        .explore_all());
    ok(
        &mut s_su,
        Action::Attack {
            unit: UnitId(0),
            target: p(2, 3),
        },
    );
    assert!(s_su.unit(UnitId(1)).is_none());
    assert_eq!(s_su.players[0].kills, 10);
    assert!(s_su.players[0]
        .monuments_unlocked
        .contains(Monument::GateOfPower));

    // (d) Disband does NOT increment either player's kills.
    let mut s_dis = v2(StateBuilder::new(5)
        .city_with(0, p(1, 1), |c| c.is_capital = true)
        .city_with(1, p(4, 4), |c| c.is_capital = true)
        .unit(0, UnitType::Warrior, p(2, 2))
        .techs(0, &[Tech::Riding, Tech::FreeSpirit])
        .player_with(0, |pl| pl.kills = 9)
        .player_with(1, |pl| pl.kills = 9)
        .explore_all());
    ok(&mut s_dis, Action::Disband { unit: UnitId(0) });
    assert_eq!(s_dis.players[0].kills, 9);
    assert_eq!(s_dis.players[1].kills, 9);
    assert!(!s_dis.players[0]
        .monuments_unlocked
        .contains(Monument::GateOfPower));
    assert!(!s_dis.players[1]
        .monuments_unlocked
        .contains(Monument::GateOfPower));

    // (e) Forced Super Unit push removal (all 8 neighbors occupied) does NOT increment kills.
    let mut b_push = StateBuilder::new(5)
        .city_with(0, p(2, 2), |c| {
            c.is_capital = true;
            c.level = 5;
            c.population = 14;
            c.pending_reward = true;
        })
        .city_with(1, p(4, 4), |c| c.is_capital = true)
        .unit(1, UnitType::Warrior, p(2, 2))
        .player_with(0, |pl| pl.kills = 9)
        .player_with(1, |pl| pl.kills = 9)
        .explore_all();
    for dr in -1i32..=1 {
        for dc in -1i32..=1 {
            if (dr, dc) != (0, 0) {
                b_push = b_push.unit(0, UnitType::Warrior, p((2 + dr) as u8, (2 + dc) as u8));
            }
        }
    }
    let mut s_push = v2(b_push);
    ok(
        &mut s_push,
        Action::ChooseReward {
            city: CityId(0),
            reward: Reward::SuperUnit,
        },
    );
    assert!(s_push.unit(UnitId(0)).is_none(), "pushed enemy was removed");
    assert_eq!(s_push.players[0].kills, 9);
    assert_eq!(s_push.players[1].kills, 9);
    assert!(!s_push.players[0]
        .monuments_unlocked
        .contains(Monument::GateOfPower));
}

#[test]
fn eye_of_god_unlocks_only_when_all_four_distinct_lighthouses_are_discovered() {
    // 7x7 with lighthouses on all 4 corners, all hidden from P0.
    // P0 already discovered 2 corners (bits 0 and 1); two Warriors are ready to reveal corners 2 and 3.
    let mut s = v2(StateBuilder::new(7)
        .lighthouse(p(0, 0))
        .lighthouse(p(0, 6))
        .lighthouse(p(6, 0))
        .lighthouse(p(6, 6))
        .city_with(0, p(3, 3), |c| {
            c.is_capital = true;
            c.level = 4;
            c.population = 9;
        })
        .city_with(1, p(3, 5), |c| c.is_capital = true)
        .unit(0, UnitType::Warrior, p(4, 1))
        .unit(0, UnitType::Warrior, p(4, 5))
        .unit(1, UnitType::Warrior, p(4, 6))
        .player_with(0, |pl| pl.lighthouses_discovered = 0b0011)
        .explore_all()
        .hide(0, &[p(6, 0), p(6, 6)])
        .hide(1, &[p(6, 6)]));

    // P0 discovers 3rd corner (6,0): NOT yet unlocked.
    ok(
        &mut s,
        Action::Move {
            unit: UnitId(0),
            to: p(5, 1),
        },
    );
    assert_eq!(s.players[0].lighthouses_discovered, 0b0111);
    assert!(!s.players[0].monuments_unlocked.contains(Monument::EyeOfGod));

    // P1 discovers corner (6,6): does NOT count for P0!
    ok(&mut s, Action::EndTurn);
    ok(
        &mut s,
        Action::Move {
            unit: UnitId(2),
            to: p(5, 6),
        },
    );
    assert_eq!(s.players[1].lighthouses_discovered, 1 << 3);
    assert!(!s.players[0].monuments_unlocked.contains(Monument::EyeOfGod));

    // P0 revisits corner (6,0): still 3, NOT unlocked.
    ok(&mut s, Action::EndTurn);
    ok(
        &mut s,
        Action::Move {
            unit: UnitId(0),
            to: p(6, 0),
        },
    );
    assert_eq!(s.players[0].lighthouses_discovered, 0b0111);
    assert!(!s.players[0].monuments_unlocked.contains(Monument::EyeOfGod));

    // P0 discovers 4th corner (6,6): EyeOfGod unlocks!
    ok(
        &mut s,
        Action::Move {
            unit: UnitId(1),
            to: p(5, 5),
        },
    );
    assert_eq!(s.players[0].lighthouses_discovered, 0b1111);
    assert!(s.players[0].monuments_unlocked.contains(Monument::EyeOfGod));
    assert!(!s.players[1].monuments_unlocked.contains(Monument::EyeOfGod));
}

// ============================================================================
// 3. BuildMonument legality, canonical order, apply, Destroy, score & capture
// ============================================================================

#[test]
fn build_monument_legality_checks_every_condition_and_is_free() {
    // 7x7: P0 capital at (2,2) owns (1..=3, 1..=3). P0 has 0 stars!
    // Unlock ParkOfFortune & GateOfPower (EyeOfGod locked); mark GateOfPower already placed.
    // Tiles in P0 territory:
    // - (1,1): plain Field -> LEGAL (even with 0 stars!)
    // - (1,2): plain Field with own Warrior -> LEGAL
    // - (1,3): plain Field with Road -> LEGAL
    // - (2,1): plain Field with enemy Warrior -> ILLEGAL
    // - (2,2): city tile -> ILLEGAL
    // - (2,3): Field with Fruit resource -> ILLEGAL (M3)
    // - (3,1): Forest -> ILLEGAL
    // - (3,2): Mountain -> ILLEGAL
    // - (3,3): Field with Sawmill building -> ILLEGAL
    // Also test a tile with improvement (Farm on Crop) and a tile with an existing monument,
    // plus unowned tile (0,4), enemy tile (5,4), and out-of-bounds (7,7).
    let mut s = v2(StateBuilder::new(7)
        .city_with(0, p(2, 2), |c| c.is_capital = true)
        .city_with(1, p(5, 5), |c| c.is_capital = true)
        .unit(0, UnitType::Warrior, p(1, 2))
        .unit(1, UnitType::Warrior, p(2, 1))
        .resource(p(2, 3), Resource::Fruit)
        .terrain(p(3, 1), Terrain::Forest)
        .terrain(p(3, 2), Terrain::Mountain)
        .terrain(p(3, 0), Terrain::Forest)
        .building(p(3, 3), Building::Sawmill)
        .stars(0, 0)
        .player_with(0, |pl| {
            pl.monuments_unlocked =
                MonumentSet::from_slice(&[Monument::ParkOfFortune, Monument::GateOfPower]);
            pl.monuments_placed = MonumentSet::from_slice(&[Monument::GateOfPower]);
        })
        .explore_all());
    tile_mut(&mut s, p(1, 3)).road = true;

    let legal = legal_actions(&s);
    // Only ParkOfFortune on (1,1), (1,2), (1,3) should be legal.
    let mons: Vec<(Pos, Monument)> = legal
        .iter()
        .filter_map(|a| match *a {
            Action::BuildMonument { at, monument } => Some((at, monument)),
            _ => None,
        })
        .collect();
    assert_eq!(
        mons,
        vec![
            (p(1, 1), Monument::ParkOfFortune),
            (p(1, 2), Monument::ParkOfFortune),
            (p(1, 3), Monument::ParkOfFortune),
        ]
    );

    // Locked or already placed monuments are illegal even on (1,1).
    assert_illegal(&s, build_mon(p(1, 1), Monument::EyeOfGod));
    assert_illegal(&s, build_mon(p(1, 1), Monument::GateOfPower));

    // Every blocked tile is illegal for ParkOfFortune.
    for bad in [
        p(2, 1),
        p(2, 2),
        p(2, 3),
        p(3, 1),
        p(3, 2),
        p(3, 3),
        p(0, 4),
        p(5, 4),
        p(7, 7),
    ] {
        assert_illegal(&s, build_mon(bad, Monument::ParkOfFortune));
    }

    // Improvement or existing monument on (1,1) also blocks BuildMonument.
    let mut with_imp = s.clone();
    tile_mut(&mut with_imp, p(1, 1)).improvement = Some(Improvement::Farm);
    assert_illegal(&with_imp, build_mon(p(1, 1), Monument::ParkOfFortune));

    let mut with_mon = s.clone();
    tile_mut(&mut with_mon, p(1, 1)).monument = Some(Monument::GateOfPower);
    assert_illegal(&with_mon, build_mon(p(1, 1), Monument::ParkOfFortune));

    // Pending reward blocks BuildMonument.
    let mut pending = s.clone();
    pending.cities[0].level = 2;
    pending.cities[0].pending_reward = true;
    assert_illegal(&pending, build_mon(p(1, 1), Monument::ParkOfFortune));

    // Actually apply on (1,3) (road + 0 stars): succeeds and costs 0 stars!
    ok(&mut s, build_mon(p(1, 3), Monument::ParkOfFortune));
    assert_eq!(s.players[0].stars, 0);
    assert_eq!(s.tile(p(1, 3)).monument, Some(Monument::ParkOfFortune));
    assert!(s.tile(p(1, 3)).road, "existing road stays intact");
}

#[test]
fn legal_actions_canonical_order_places_build_monument_between_build_building_and_destroy() {
    // 5x5: P0 capital at (2,2), level 3 pop 5.
    // (1,1): Forest + LumberHut -> enables Sawmill on (1,2) & (2,1), and Destroy on (1,1).
    // (1,2), (2,1): plain Fields -> enable BuildBuilding AND BuildMonument.
    // Unlock EyeOfGod and ParkOfFortune (check ALL_MONUMENTS order: ParkOfFortune before EyeOfGod!).
    let s = v2(StateBuilder::new(5)
        .city_with(0, p(2, 2), |c| {
            c.is_capital = true;
            c.level = 3;
            c.population = 5;
        })
        .terrain(p(1, 1), Terrain::Forest)
        .improvement(p(1, 1), Improvement::LumberHut)
        .techs(0, &ALL_V2_TECHS)
        .stars(0, 20)
        .player_with(0, |pl| {
            pl.monuments_unlocked =
                MonumentSet::from_slice(&[Monument::EyeOfGod, Monument::ParkOfFortune]);
        })
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
    let buildings = idx_of(&|a| matches!(a, Action::BuildBuilding { .. }));
    let monuments = idx_of(&|a| matches!(a, Action::BuildMonument { .. }));
    let destroys = idx_of(&|a| matches!(a, Action::Destroy { .. }));

    assert!(!buildings.is_empty());
    assert!(!monuments.is_empty());
    assert!(!destroys.is_empty());
    assert!(
        buildings.last() < monuments.first(),
        "BuildMonument must come after BuildBuilding"
    );
    assert!(
        monuments.last() < destroys.first(),
        "BuildMonument must come before Destroy"
    );

    // Within BuildMonument: row-major, then ALL_MONUMENTS order (ParkOfFortune before EyeOfGod).
    let mon_list: Vec<(Pos, Monument)> = legal
        .iter()
        .filter_map(|a| match *a {
            Action::BuildMonument { at, monument } => Some((at, monument)),
            _ => None,
        })
        .collect();
    assert_eq!(mon_list[0], (p(1, 2), Monument::ParkOfFortune));
    assert_eq!(mon_list[1], (p(1, 2), Monument::EyeOfGod));
    assert_eq!(mon_list[2], (p(1, 3), Monument::ParkOfFortune));
    assert_eq!(mon_list[3], (p(1, 3), Monument::EyeOfGod));
}

#[test]
fn build_monument_apply_score_road_building_exclusivity_destroy_and_no_rebuild() {
    // 5x5: P0 capital at (2,2), level 1 pop 0 -> placing monument (+3 pop) levels up to level 2!
    // Place LumberHut at (1,1) so Sawmill would be legal on (1,2) before monument, illegal after.
    let mut s = v2(StateBuilder::new(5)
        .city_with(0, p(2, 2), |c| c.is_capital = true)
        .city_with(1, p(4, 4), |c| c.is_capital = true)
        .terrain(p(1, 1), Terrain::Forest)
        .improvement(p(1, 1), Improvement::LumberHut)
        .techs(0, &ALL_V2_TECHS)
        .stars(0, 20)
        .player_with(0, |pl| {
            pl.monuments_unlocked = MonumentSet::from_slice(&[Monument::GateOfPower]);
        })
        .explore_all());

    let before_bd = score_breakdown(&s, 0);
    assert_eq!(before_bd.monuments, 0);
    assert_eq!(score(&s, 0), before_bd.total());

    // Place GateOfPower at (1,2): +3 pop (0 -> 3 -> level 2 with surplus 1, pending reward!), +400 score!
    ok(&mut s, build_mon(p(1, 2), Monument::GateOfPower));
    assert_eq!(s.tile(p(1, 2)).monument, Some(Monument::GateOfPower));
    assert!(s.players[0]
        .monuments_placed
        .contains(Monument::GateOfPower));
    assert_eq!(s.city(CityId(0)).population, 3);
    assert_eq!(s.city(CityId(0)).level, 2);
    assert!(s.city(CityId(0)).pending_reward);

    let after_bd = score_breakdown(&s, 0);
    assert_eq!(after_bd.monuments, 400);
    assert_eq!(score(&s, 0), after_bd.total());
    // Score delta: +400 monument + 50 level-up + 5 surplus pop (3 - 2 = 1) = +455.
    assert_eq!(after_bd.total() - before_bd.total(), 455);

    // Resolve level-2 reward so we can test BuildRoad, BuildBuilding, and Destroy.
    ok(
        &mut s,
        Action::ChooseReward {
            city: CityId(0),
            reward: Reward::Workshop,
        },
    );

    // BuildBuilding is ILLEGAL on the monument tile (1,2) even though LumberHut is adjacent!
    for b in ALL_BUILDINGS {
        assert_illegal(
            &s,
            Action::BuildBuilding {
                at: p(1, 2),
                building: b,
            },
        );
    }

    // BuildRoad IS legal on a monument tile (M3: road may share a tile with a monument).
    ok(&mut s, Action::BuildRoad { at: p(1, 2) });
    assert!(s.tile(p(1, 2)).road);
    assert_eq!(s.tile(p(1, 2)).monument, Some(Monument::GateOfPower));

    // Destroy on (1,2): removes monument, -3 pop (3 -> 0; drop below 0 tested next), -400 score,
    // keeps road, level stays 2, cannot rebuild GateOfPower!
    s.cities[0].population = 1; // so 1 - 3 = -2 tests negative population too!
    ok(&mut s, Action::Destroy { at: p(1, 2) });
    assert_eq!(s.tile(p(1, 2)).monument, None);
    assert!(s.tile(p(1, 2)).road, "Destroy must leave road intact");
    assert_eq!(s.city(CityId(0)).population, -2);
    assert_eq!(s.city(CityId(0)).level, 2, "level never goes down");
    assert_eq!(score_breakdown(&s, 0).monuments, 0);
    assert!(s.players[0]
        .monuments_placed
        .contains(Monument::GateOfPower));
    assert_illegal(&s, build_mon(p(1, 2), Monument::GateOfPower));
    assert_illegal(&s, build_mon(p(1, 3), Monument::GateOfPower));
}

#[test]
fn capturing_city_with_monument_transfers_400_score_to_new_owner() {
    // 9x9: P0 owns non-capital City 2 at (4,4) (full 9-tile territory) with two monuments.
    // P1 captures City 2 -> 800 monument score transfers from P0 to P1!
    let mut s = v2(StateBuilder::new(9)
        .city_with(0, p(1, 1), |c| c.is_capital = true)
        .city_with(1, p(7, 7), |c| c.is_capital = true)
        .city(0, p(4, 4))
        .monument(p(3, 3), Monument::ParkOfFortune)
        .monument(p(3, 4), Monument::GateOfPower)
        .unit(1, UnitType::Warrior, p(4, 4))
        .current(1)
        .explore_all());

    assert_eq!(s.tile(p(3, 3)).owner_city, Some(CityId(2)));
    assert_eq!(s.tile(p(3, 4)).owner_city, Some(CityId(2)));
    assert_eq!(score_breakdown(&s, 0).monuments, 800);
    assert_eq!(score_breakdown(&s, 1).monuments, 0);
    let p0_before = score(&s, 0);
    let p1_before = score(&s, 1);

    ok(&mut s, Action::Capture { unit: UnitId(0) });
    assert_eq!(s.city(CityId(2)).owner, 1);
    assert_eq!(score_breakdown(&s, 0).monuments, 0);
    assert_eq!(score_breakdown(&s, 1).monuments, 800);
    assert_eq!(score(&s, 0), score_breakdown(&s, 0).total());
    assert_eq!(score(&s, 1), score_breakdown(&s, 1).total());
    // City 2 (100) + 9 tiles (180) + 2 monuments (800) = 1080 transferred!
    assert_eq!(p0_before - score(&s, 0), 1080);
    assert_eq!(score(&s, 1) - p1_before, 1080);
}

// ============================================================================
// 4. M17 review nit: disband_refund_divisor == 0 load-time validation
// ============================================================================

#[test]
fn tables_v2_rejects_zero_disband_refund_divisor() {
    let with_div = |val: &str| {
        let rules = V2_RULES.replacen(
            "\"disband_refund_divisor\": 2,",
            &format!("\"disband_refund_divisor\": {val},"),
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
    assert!(
        with_div("0").is_err(),
        "disband_refund_divisor == 0 must be rejected"
    );
    assert!(with_div("1").is_ok());
    assert!(with_div("2").is_ok());
}
