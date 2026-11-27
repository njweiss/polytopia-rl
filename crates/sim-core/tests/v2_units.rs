//! Milestone 19 tests: v2 unit training (`Swordsman`, `Catapult`, `Knight`), combat formulas,
//! movement, `Stiff`, and `Persist` (docs/MILESTONE_19 "Rules", "Tests" items 2-3).
//!
//! Combat expectations are hand-computed directly from the spec's `f32` half-up formulas
//! (docs/SPEC.md "Combat", "Game v2 rules" -> "Units").

mod common;

use std::sync::Arc;

use common::{all_pos, idx, p};
use sim_core::actions::{apply, legal_actions, Action, ActionError};
use sim_core::data::{Reward, Tables, Tech, UnitType, ALL_TECHS};
use sim_core::map::Terrain;
use sim_core::movement::{defense_bonus, reachable_tiles};
use sim_core::score::score_breakdown;
use sim_core::state::{CityId, Config, Dir, Phase, Ruleset, State, UnitId};
use sim_core::testing::StateBuilder;

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

fn v2_open(n: u8) -> StateBuilder {
    StateBuilder::new(n).config(v2_config()).explore_all()
}

fn assert_illegal(s: &State, a: Action) {
    let mut copy = s.clone();
    assert_eq!(apply(&mut copy, &a), Err(ActionError::Illegal), "{a:?}");
    assert_eq!(&copy, s, "{a:?}: refused action must not mutate state");
}

// ---------------------------------------------------------------- Training v2 units

#[test]
fn v2_units_require_their_tech_and_star_cost() {
    let cases = [
        (UnitType::Swordsman, Tech::Smithery, 5u32),
        (UnitType::Catapult, Tech::Mathematics, 8u32),
        (UnitType::Knight, Tech::Chivalry, 8u32),
    ];
    for (kind, tech, cost) in cases {
        let train = Action::Train {
            city: CityId(0),
            kind,
        };
        // Every tech EXCEPT the required one: illegal even with 99 stars.
        let others: Vec<Tech> = ALL_V2_TECHS
            .iter()
            .copied()
            .filter(|&t| t != tech)
            .collect();
        let no_tech = v2_open(5)
            .city(0, p(2, 2))
            .techs(0, &others)
            .stars(0, 99)
            .build();
        assert!(
            !legal_actions(&no_tech).contains(&train),
            "{kind:?} no tech"
        );
        assert_illegal(&no_tech, train);

        // With tech, one star short: illegal.
        let short = v2_open(5)
            .city(0, p(2, 2))
            .techs(0, &[tech])
            .stars(0, cost - 1)
            .build();
        assert!(!legal_actions(&short).contains(&train), "{kind:?} short");
        assert_illegal(&short, train);

        // Exact stars + tech: legal.
        let mut exact = v2_open(5)
            .city(0, p(2, 2))
            .techs(0, &[tech])
            .stars(0, cost)
            .build();
        assert!(legal_actions(&exact).contains(&train), "{kind:?} exact");
        apply(&mut exact, &train).unwrap();
        assert_eq!(exact.players[0].stars, 0, "{kind:?} spent");
        assert_eq!(exact.units.len(), 1);
        let u = &exact.units[0];
        assert_eq!(u.kind, kind);
        assert_eq!(u.pos, p(2, 2));
        assert_eq!(u.origin, Some(CityId(0)));
        assert_eq!(u.phase, Phase::Done);
    }
}

#[test]
fn v2_train_canonical_order_across_cities_and_unit_types() {
    let s = v2_open(7)
        .city_with(0, p(1, 1), |c| c.level = 3)
        .city_with(0, p(5, 5), |c| c.level = 3)
        .techs(0, &ALL_V2_TECHS)
        .stars(0, 99)
        .build();
    let trains: Vec<Action> = legal_actions(&s)
        .into_iter()
        .filter(|a| matches!(a, Action::Train { .. }))
        .collect();
    let expected_kinds = [
        UnitType::Warrior,
        UnitType::Rider,
        UnitType::Archer,
        UnitType::Defender,
        UnitType::Swordsman,
        UnitType::Catapult,
        UnitType::Knight,
    ];
    let mut expected = Vec::new();
    for cid in [CityId(0), CityId(1)] {
        for kind in expected_kinds {
            expected.push(Action::Train { city: cid, kind });
        }
    }
    assert_eq!(trains, expected);
}

#[test]
fn v2_trained_units_stats_score_and_disband_refund() {
    let cases = [
        (UnitType::Swordsman, 5u32, 15u32, 25u32, 2u32),
        (UnitType::Catapult, 8, 10, 40, 4),
        (UnitType::Knight, 8, 10, 40, 4),
    ];
    for (kind, cost, hp, unit_score, refund) in cases {
        let mut s = v2_open(5)
            .city(0, p(2, 2))
            .techs(0, &ALL_V2_TECHS)
            .stars(0, 20)
            .build();
        let score_before = score_breakdown(&s, 0);
        apply(
            &mut s,
            &Action::Train {
                city: CityId(0),
                kind,
            },
        )
        .unwrap();
        assert_eq!(s.players[0].stars, 20 - cost, "{kind:?}");
        let u = &s.units[0];
        assert_eq!(u.health, hp, "{kind:?} hp");
        assert_eq!(u.max_health, hp, "{kind:?} max_hp");
        assert_eq!(u.phase, Phase::Done, "{kind:?} phase");
        assert_eq!(u.origin, Some(CityId(0)), "{kind:?} origin");

        let score_after = score_breakdown(&s, 0);
        assert_eq!(
            score_after.units - score_before.units,
            unit_score,
            "{kind:?} score"
        );
        assert_eq!(
            score_after.total() - score_before.total(),
            unit_score,
            "{kind:?} total score"
        );

        let uid = u.id;
        apply(&mut s, &Action::Disband { unit: uid }).unwrap();
        assert_eq!(s.players[0].stars, 20 - cost + refund, "{kind:?} refund");
        assert!(s.units.is_empty());
        assert_eq!(score_breakdown(&s, 0).units, 0);
    }
}

#[test]
fn v2_train_respects_occupied_city_and_unit_cap() {
    // Occupied city tile blocks training all v2 units.
    let occupied = v2_open(5)
        .city(0, p(2, 2))
        .unit(0, UnitType::Warrior, p(2, 2))
        .techs(0, &ALL_V2_TECHS)
        .stars(0, 99)
        .build();
    for kind in [UnitType::Swordsman, UnitType::Catapult, UnitType::Knight] {
        let a = Action::Train {
            city: CityId(0),
            kind,
        };
        assert!(!legal_actions(&occupied).contains(&a));
        assert_illegal(&occupied, a);
    }

    // Level 1 city cap is level + 1 = 2 supported units.
    let capped = v2_open(5)
        .city(0, p(2, 2))
        .unit_with(0, UnitType::Warrior, p(0, 0), |u| {
            u.origin = Some(CityId(0))
        })
        .unit_with(0, UnitType::Warrior, p(0, 1), |u| {
            u.origin = Some(CityId(0))
        })
        .techs(0, &ALL_V2_TECHS)
        .stars(0, 99)
        .build();
    for kind in [UnitType::Swordsman, UnitType::Catapult, UnitType::Knight] {
        let a = Action::Train {
            city: CityId(0),
            kind,
        };
        assert!(!legal_actions(&capped).contains(&a));
        assert_illegal(&capped, a);
    }
}

#[test]
fn v1_never_allows_training_v2_units() {
    let s = StateBuilder::new(5)
        .explore_all()
        .city(0, p(2, 2))
        .techs(0, &ALL_TECHS)
        .stars(0, 99)
        .build();
    assert_eq!(s.config.ruleset, Ruleset::V1);
    let legal = legal_actions(&s);
    for kind in [UnitType::Swordsman, UnitType::Catapult, UnitType::Knight] {
        let a = Action::Train {
            city: CityId(0),
            kind,
        };
        assert!(!legal.contains(&a));
        assert_illegal(&s, a);
    }
}

// ---------------------------------------------------------------- Swordsman

#[test]
fn swordsman_combat_dash_and_fortify_match_hand_computed_formulas() {
    // 1. Swordsman 15/15 vs Warrior 10/10 (bonus 1.0):
    //    af = 3.0, df = 2.0, td = 5.0
    //    to_def = round(3/5 * 3 * 4.5) = round(8.1) = 8 -> Warrior 2/10
    //    to_atk = round(2/5 * 2 * 4.5) = round(3.6) = 4 -> Swordsman 11/15
    //    Also exercises Dash: move (2,1) -> (2,2), then attack (2,3).
    let mut s = v2_open(5)
        .unit(0, UnitType::Swordsman, p(2, 1))
        .unit(1, UnitType::Warrior, p(2, 3))
        .build();
    apply(
        &mut s,
        &Action::Move {
            unit: UnitId(0),
            to: p(2, 2),
        },
    )
    .unwrap();
    assert_eq!(s.unit(UnitId(0)).unwrap().phase, Phase::Moved);
    assert!(legal_actions(&s).contains(&Action::Attack {
        unit: UnitId(0),
        target: p(2, 3),
    }));
    apply(
        &mut s,
        &Action::Attack {
            unit: UnitId(0),
            target: p(2, 3),
        },
    )
    .unwrap();
    assert_eq!(s.unit(UnitId(0)).unwrap().health, 11);
    assert_eq!(s.unit(UnitId(0)).unwrap().phase, Phase::Attacked);
    assert_eq!(s.unit(UnitId(1)).unwrap().health, 2);

    // 2. Swordsman 15/15 Fortify in own city (1.5x):
    //    Warrior 10/10 attacks: af = 2.0, df = 3.0 * 1.5 = 4.5, td = 6.5
    //    to_def = round(2/6.5 * 2 * 4.5) = round(2.7692) = 3 -> Swordsman 12/15
    //    to_atk = round(4.5/6.5 * 3 * 4.5) = round(9.3462) = 9 -> Warrior 1/10
    let mut s_city = v2_open(5)
        .city(1, p(2, 2))
        .unit(0, UnitType::Warrior, p(2, 1))
        .unit(1, UnitType::Swordsman, p(2, 2))
        .build();
    assert_eq!(defense_bonus(&s_city, UnitId(1)), 1.5);
    apply(
        &mut s_city,
        &Action::Attack {
            unit: UnitId(0),
            target: p(2, 2),
        },
    )
    .unwrap();
    assert_eq!(s_city.unit(UnitId(1)).unwrap().health, 12);
    assert_eq!(s_city.unit(UnitId(0)).unwrap().health, 1);

    // 3. Swordsman 15/15 Fortify in own walled city (4.0x):
    //    Warrior 10/10 attacks: af = 2.0, df = 3.0 * 4.0 = 12.0, td = 14.0
    //    to_def = round(2/14 * 2 * 4.5) = round(1.2857) = 1 -> Swordsman 14/15
    //    to_atk = round(12/14 * 3 * 4.5) = round(11.5714) = 12 -> Warrior dies (0/10)
    let mut s_wall = v2_open(5)
        .city_with(1, p(2, 2), |c| c.wall = true)
        .unit(0, UnitType::Warrior, p(2, 1))
        .unit(1, UnitType::Swordsman, p(2, 2))
        .build();
    assert_eq!(defense_bonus(&s_wall, UnitId(1)), 4.0);
    apply(
        &mut s_wall,
        &Action::Attack {
            unit: UnitId(0),
            target: p(2, 2),
        },
    )
    .unwrap();
    assert!(
        s_wall.unit(UnitId(0)).is_none(),
        "Warrior dies to wall retaliation"
    );
    let sw = s_wall.unit(UnitId(1)).unwrap();
    assert_eq!(sw.health, 14);
    assert_eq!(sw.kills, 1);
    assert_eq!(s_wall.players[1].kills, 1);
}

// ---------------------------------------------------------------- Catapult & Stiff

#[test]
fn catapult_range_3_ranged_kill_no_kill_move_and_last_dir() {
    let s = v2_open(7)
        .unit(0, UnitType::Catapult, p(3, 1))
        .unit(1, UnitType::Warrior, p(3, 2)) // dist 1
        .unit(1, UnitType::Warrior, p(3, 3)) // dist 2
        .unit(1, UnitType::Warrior, p(3, 4)) // dist 3
        .unit(1, UnitType::Warrior, p(3, 5)) // dist 4 (out of range)
        .build();
    let legal = legal_actions(&s);
    assert!(legal.contains(&Action::Attack {
        unit: UnitId(0),
        target: p(3, 2),
    }));
    assert!(legal.contains(&Action::Attack {
        unit: UnitId(0),
        target: p(3, 3),
    }));
    assert!(legal.contains(&Action::Attack {
        unit: UnitId(0),
        target: p(3, 4),
    }));
    assert!(!legal.contains(&Action::Attack {
        unit: UnitId(0),
        target: p(3, 5),
    }));
    assert_illegal(
        &s,
        Action::Attack {
            unit: UnitId(0),
            target: p(3, 5),
        },
    );

    // Attack Warrior at distance 3: af = 4.0, df = 2.0, td = 6.0 -> to_def = 12 (kill!).
    // Ranged unit must NOT kill-move, updates last_dir, and enters Phase::Attacked.
    let mut s_kill = s.clone();
    apply(
        &mut s_kill,
        &Action::Attack {
            unit: UnitId(0),
            target: p(3, 4),
        },
    )
    .unwrap();
    assert!(s_kill.unit(UnitId(3)).is_none(), "target Warrior died");
    let cat = s_kill.unit(UnitId(0)).unwrap();
    assert_eq!(cat.pos, p(3, 1), "ranged Catapult never kill-moves");
    assert_eq!(cat.phase, Phase::Attacked);
    assert_eq!(cat.last_dir, Some(Dir { dr: 0, dc: 1 }));
    assert_eq!(cat.kills, 1);
    assert_eq!(s_kill.tile(p(3, 4)).unit, None);
}

#[test]
fn catapult_has_no_dash_cannot_attack_after_moving() {
    let mut s = v2_open(7)
        .unit(0, UnitType::Catapult, p(3, 1))
        .unit(1, UnitType::Warrior, p(3, 4))
        .build();
    apply(
        &mut s,
        &Action::Move {
            unit: UnitId(0),
            to: p(3, 2),
        },
    )
    .unwrap();
    assert_eq!(s.unit(UnitId(0)).unwrap().phase, Phase::Moved);
    let atk = Action::Attack {
        unit: UnitId(0),
        target: p(3, 4),
    };
    assert!(!legal_actions(&s).contains(&atk));
    assert_illegal(&s, atk);
}

#[test]
fn catapult_defense_zero_takes_full_attack_times_4_5_even_in_walled_city() {
    // Catapult has defense 0 and no Fortify -> defense_force == 0.0 ->
    // to_def = round_half_up(attacker.attack * 4.5) even when attacker has 1 HP and Catapult
    // stands in its own walled city.
    // Warrior (attack 2.0, 1/10 HP): round(2.0 * 4.5) = 9 -> Catapult 1/10; no retaliation.
    let mut s = v2_open(5)
        .city_with(1, p(2, 2), |c| c.wall = true)
        .unit_with(0, UnitType::Warrior, p(2, 1), |u| u.health = 1)
        .unit(1, UnitType::Catapult, p(2, 2))
        .build();
    assert_eq!(defense_bonus(&s, UnitId(1)), 1.0, "Catapult has no Fortify");
    apply(
        &mut s,
        &Action::Attack {
            unit: UnitId(0),
            target: p(2, 2),
        },
    )
    .unwrap();
    assert_eq!(s.unit(UnitId(1)).unwrap().health, 1);
    assert_eq!(s.unit(UnitId(0)).unwrap().health, 1);

    // Defender (attack 1.0, 1/15 HP): round_half_up(1.0 * 4.5) = 5 -> Catapult 5/10.
    let mut s2 = v2_open(5)
        .city_with(1, p(2, 2), |c| c.wall = true)
        .unit_with(0, UnitType::Defender, p(2, 1), |u| u.health = 1)
        .unit(1, UnitType::Catapult, p(2, 2))
        .build();
    apply(
        &mut s2,
        &Action::Attack {
            unit: UnitId(0),
            target: p(2, 2),
        },
    )
    .unwrap();
    assert_eq!(s2.unit(UnitId(1)).unwrap().health, 5);
    assert_eq!(s2.unit(UnitId(0)).unwrap().health, 1);
}

#[test]
fn stiff_suppresses_retaliation_even_with_positive_defense() {
    // Custom v2 table: give Catapult defense 3 and health 15 while keeping `skills: ["stiff"]`,
    // so `Stiff` itself is tested independently of `defense == 0`.
    let custom_units = r#"[
      { "unit": "swordsman", "cost": 5, "health": 15, "attack": 3,   "defense": 3, "movement": 1, "range": 1, "skills": ["dash", "fortify"],            "unlocked_by": "smithery" },
      { "unit": "catapult",  "cost": 8, "health": 15, "attack": 4,   "defense": 3, "movement": 1, "range": 3, "skills": ["stiff"],                      "unlocked_by": "mathematics" },
      { "unit": "knight",    "cost": 8, "health": 10, "attack": 3.5, "defense": 1, "movement": 3, "range": 1, "skills": ["dash", "fortify", "persist"], "unlocked_by": "chivalry" }
    ]"#;
    let tables = Tables::from_json_v2(
        custom_units,
        include_str!("../../../data/v2/techs.json"),
        include_str!("../../../data/v2/site_actions.json"),
        include_str!("../../../data/v2/buildings.json"),
        include_str!("../../../data/v2/monuments.json"),
        include_str!("../../../data/v2/rules.json"),
    )
    .expect("custom v2 tables load");

    // Warrior 10/10 attacks Stiff Catapult (def 3, 15/15 HP, range 3) at distance 1.
    // af = 2.0, df = 3.0, td = 5.0 -> to_def = round(3.6) = 4 (Catapult survives at 11/15).
    // Without Stiff, retaliation would deal round(3/5 * 3 * 4.5) = round(8.1) = 8 to Warrior.
    // With Stiff, Warrior takes 0 retaliation and stays at 10/10 HP.
    let mut s = v2_open(5)
        .unit(0, UnitType::Warrior, p(2, 1))
        .unit_with(1, UnitType::Catapult, p(2, 2), |u| {
            u.health = 15;
            u.max_health = 15;
        })
        .build();
    s.tables = Arc::new(tables);
    apply(
        &mut s,
        &Action::Attack {
            unit: UnitId(0),
            target: p(2, 2),
        },
    )
    .unwrap();
    assert_eq!(s.unit(UnitId(1)).unwrap().health, 11);
    assert_eq!(
        s.unit(UnitId(0)).unwrap().health,
        10,
        "Stiff defender with defense > 0 must never retaliate"
    );
}

#[test]
fn catapult_takes_retaliation_when_attacking_in_defender_range_but_not_at_range_2_or_3() {
    // Catapult 10/10 vs Defender 15/15 (range 1, bonus 1.0):
    // af = 4.0, df = 3.0, td = 7.0
    // to_def = round(4/7 * 4 * 4.5) = round(10.2857) = 10 -> Defender 5/15
    // to_atk = round(3/7 * 3 * 4.5) = round(5.7857) = 6 (only at dist 1)
    let mut s_adj = v2_open(7)
        .unit(0, UnitType::Catapult, p(3, 1))
        .unit(1, UnitType::Defender, p(3, 2))
        .build();
    apply(
        &mut s_adj,
        &Action::Attack {
            unit: UnitId(0),
            target: p(3, 2),
        },
    )
    .unwrap();
    assert_eq!(s_adj.unit(UnitId(1)).unwrap().health, 5);
    assert_eq!(s_adj.unit(UnitId(0)).unwrap().health, 4);

    let mut s_far = v2_open(7)
        .unit(0, UnitType::Catapult, p(3, 1))
        .unit(1, UnitType::Defender, p(3, 4))
        .build();
    apply(
        &mut s_far,
        &Action::Attack {
            unit: UnitId(0),
            target: p(3, 4),
        },
    )
    .unwrap();
    assert_eq!(s_far.unit(UnitId(1)).unwrap().health, 5);
    assert_eq!(s_far.unit(UnitId(0)).unwrap().health, 10);
}

// ---------------------------------------------------------------- Knight: combat, movement, Fortify

#[test]
fn knight_3_5_attack_combat_math_and_fortify() {
    // 1. Knight 10/10 (atk 3.5) vs Warrior 10/10 (bonus 1.0):
    //    af = 3.5, df = 2.0, td = 5.5 -> to_def = round(3.5/5.5 * 3.5 * 4.5) = round(10.0227) = 10.
    //    (If attack were truncated to 3.0, to_def would be 8 and Warrior would survive!)
    let mut s_w = v2_open(5)
        .unit(0, UnitType::Knight, p(2, 1))
        .unit(1, UnitType::Warrior, p(2, 2))
        .build();
    apply(
        &mut s_w,
        &Action::Attack {
            unit: UnitId(0),
            target: p(2, 2),
        },
    )
    .unwrap();
    assert!(
        s_w.unit(UnitId(1)).is_none(),
        "Knight 3.5 atk one-shots Warrior 10/10"
    );
    let k = s_w.unit(UnitId(0)).unwrap();
    assert_eq!(k.health, 10);
    assert_eq!(k.pos, p(2, 2));
    assert_eq!(k.phase, Phase::Moved);

    // 2. Knight 10/10 vs Defender 15/15 (bonus 1.0):
    //    af = 3.5, df = 3.0, td = 6.5
    //    to_def = round(55.125 / 6.5) = round(8.4808) = 8 -> Defender 7/15
    //    to_atk = round(40.5 / 6.5) = round(6.2308) = 6 -> Knight 4/10
    let mut s_d = v2_open(5)
        .unit(0, UnitType::Knight, p(2, 1))
        .unit(1, UnitType::Defender, p(2, 2))
        .build();
    apply(
        &mut s_d,
        &Action::Attack {
            unit: UnitId(0),
            target: p(2, 2),
        },
    )
    .unwrap();
    assert_eq!(s_d.unit(UnitId(1)).unwrap().health, 7);
    assert_eq!(s_d.unit(UnitId(0)).unwrap().health, 4);
    assert_eq!(s_d.unit(UnitId(0)).unwrap().phase, Phase::Attacked);

    // 3. Knight 10/10 vs Swordsman 15/15 (bonus 1.0): same defense 3 -> Swordsman 7/15, Knight 4/10.
    let mut s_sw = v2_open(5)
        .unit(0, UnitType::Knight, p(2, 1))
        .unit(1, UnitType::Swordsman, p(2, 2))
        .build();
    apply(
        &mut s_sw,
        &Action::Attack {
            unit: UnitId(0),
            target: p(2, 2),
        },
    )
    .unwrap();
    assert_eq!(s_sw.unit(UnitId(1)).unwrap().health, 7);
    assert_eq!(s_sw.unit(UnitId(0)).unwrap().health, 4);

    // 4. Knight Fortify in own city (1.5x) and walled city (4.0x):
    //    City 1.5x: Warrior 10/10 vs Knight 10/10 -> af = 2, df = 1.5, td = 3.5
    //    to_def = round(18/3.5) = 5 -> Knight 5/10; to_atk = round(6.75/3.5) = 2 -> Warrior 8/10.
    let mut s_kc = v2_open(5)
        .city(1, p(2, 2))
        .unit(0, UnitType::Warrior, p(2, 1))
        .unit(1, UnitType::Knight, p(2, 2))
        .build();
    assert_eq!(defense_bonus(&s_kc, UnitId(1)), 1.5);
    apply(
        &mut s_kc,
        &Action::Attack {
            unit: UnitId(0),
            target: p(2, 2),
        },
    )
    .unwrap();
    assert_eq!(s_kc.unit(UnitId(1)).unwrap().health, 5);
    assert_eq!(s_kc.unit(UnitId(0)).unwrap().health, 8);

    //    Wall 4.0x: af = 2, df = 4, td = 6 -> to_def = 3 (Knight 7/10), to_atk = 3 (Warrior 7/10).
    let mut s_kw = v2_open(5)
        .city_with(1, p(2, 2), |c| c.wall = true)
        .unit(0, UnitType::Warrior, p(2, 1))
        .unit(1, UnitType::Knight, p(2, 2))
        .build();
    assert_eq!(defense_bonus(&s_kw, UnitId(1)), 4.0);
    apply(
        &mut s_kw,
        &Action::Attack {
            unit: UnitId(0),
            target: p(2, 2),
        },
    )
    .unwrap();
    assert_eq!(s_kw.unit(UnitId(1)).unwrap().health, 7);
    assert_eq!(s_kw.unit(UnitId(0)).unwrap().health, 7);
}

#[test]
fn knight_movement_3_plain_roads_6_zoc_and_forest_stop() {
    // Plain 9x9: Knight at (4,4) reaches every tile within Chebyshev distance 3 except (4,4).
    let plain = v2_open(9).unit(0, UnitType::Knight, p(4, 4)).build();
    let expected_plain: Vec<_> = all_pos(9)
        .into_iter()
        .filter(|&q| q != p(4, 4) && q.row.abs_diff(4).max(q.col.abs_diff(4)) <= 3)
        .collect();
    assert_eq!(expected_plain.len(), 48);
    assert_eq!(reachable_tiles(&plain, UnitId(0)), expected_plain);

    // Road line along row 4 from col 1 to col 8: Knight at (4,1) has 6 half-points, so reaches
    // up to (4,7) (6 road steps) and NOT (4,8) (7 road steps).
    let mut road = v2_open(9).unit(0, UnitType::Knight, p(4, 1)).build();
    for c in 1..=8 {
        let i = idx(&road, p(4, c));
        road.tiles[i].road = true;
    }
    let reach_road = reachable_tiles(&road, UnitId(0));
    assert!(reach_road.contains(&p(4, 7)), "6 road steps reachable");
    assert!(!reach_road.contains(&p(4, 8)), "7 road steps unreachable");

    // Forest without road stops movement on arrival.
    let mut forest_block = v2_open(9).unit(0, UnitType::Knight, p(4, 1)).build();
    for r in 0..9 {
        let i = idx(&forest_block, p(r, 2));
        forest_block.tiles[i].terrain = Terrain::Forest;
    }
    let reach_f = reachable_tiles(&forest_block, UnitId(0));
    assert!(reach_f.contains(&p(4, 2)));
    assert!(
        !reach_f.contains(&p(4, 3)),
        "forest stops Knight without road"
    );

    // Enemy ZoC stops Knight even along a road.
    let mut zoc_road = v2_open(9)
        .unit(0, UnitType::Knight, p(4, 1))
        .unit(1, UnitType::Warrior, p(3, 3))
        .build();
    for c in 1..=7 {
        let i = idx(&zoc_road, p(4, c));
        zoc_road.tiles[i].road = true;
    }
    // Every tile in col 2..=4 on row 4 is adjacent to enemy at (3,3).
    // Knight can reach (4,2) (adjacent to (3,3)), then stops — cannot continue to (4,7).
    let reach_zoc = reachable_tiles(&zoc_road, UnitId(0));
    assert!(reach_zoc.contains(&p(4, 2)));
    assert!(!reach_zoc.contains(&p(4, 7)), "ZoC stops road run");
}

// ---------------------------------------------------------------- Knight Persist

#[test]
fn knight_persist_chains_five_kills_after_dash_with_no_cap() {
    // Knight moves (3,0) -> (3,1) (Dash), then kills 5 full-HP Warriors in a line.
    let mut s = v2_open(9)
        .unit(0, UnitType::Knight, p(3, 0))
        .unit(1, UnitType::Warrior, p(3, 2))
        .unit(1, UnitType::Warrior, p(3, 3))
        .unit(1, UnitType::Warrior, p(3, 4))
        .unit(1, UnitType::Warrior, p(3, 5))
        .unit(1, UnitType::Warrior, p(3, 6))
        .build();
    apply(
        &mut s,
        &Action::Move {
            unit: UnitId(0),
            to: p(3, 1),
        },
    )
    .unwrap();
    assert_eq!(s.unit(UnitId(0)).unwrap().phase, Phase::Moved);

    for (k, col) in (2..=6u8).enumerate() {
        let target = p(3, col);
        assert!(
            legal_actions(&s).contains(&Action::Attack {
                unit: UnitId(0),
                target,
            }),
            "chain kill #{k} at {target:?} must be legal"
        );
        apply(
            &mut s,
            &Action::Attack {
                unit: UnitId(0),
                target,
            },
        )
        .unwrap();
        let u = s.unit(UnitId(0)).unwrap();
        assert_eq!(u.pos, target);
        assert_eq!(
            u.phase,
            Phase::Moved,
            "Persist keeps Phase::Moved after kill #{k}"
        );
        assert_eq!(u.kills, (k + 1) as u32);
        assert_eq!(u.last_dir, Some(Dir { dr: 0, dc: 1 }));
    }
    assert_eq!(s.players[0].kills, 5);
    assert!(s.unit(UnitId(0)).unwrap().promotion_available);
}

#[test]
fn knight_persist_mid_chain_promote_heals_and_continues_chain() {
    // Injured Knight (6/10 HP) kills 3 weak Warriors (4/10 HP each) -> eligible to Promote.
    // At 6/10 HP it could NOT one-shot a 10/10 Warrior (would deal 8), but promoting mid-chain
    // heals to 15/15 while preserving Phase::Moved so it one-shots the 4th and 5th 10/10 Warriors!
    let mut s = v2_open(9)
        .unit_with(0, UnitType::Knight, p(3, 0), |u| u.health = 6)
        .unit_with(1, UnitType::Warrior, p(3, 1), |u| u.health = 4)
        .unit_with(1, UnitType::Warrior, p(3, 2), |u| u.health = 4)
        .unit_with(1, UnitType::Warrior, p(3, 3), |u| u.health = 4)
        .unit(1, UnitType::Warrior, p(3, 4))
        .unit(1, UnitType::Warrior, p(3, 5))
        .build();

    for col in 1..=3u8 {
        apply(
            &mut s,
            &Action::Attack {
                unit: UnitId(0),
                target: p(3, col),
            },
        )
        .unwrap();
    }
    let k = s.unit(UnitId(0)).unwrap();
    assert_eq!(k.kills, 3);
    assert!(k.promotion_available);
    assert_eq!(k.phase, Phase::Moved);
    assert!(legal_actions(&s).contains(&Action::Promote { unit: UnitId(0) }));

    apply(&mut s, &Action::Promote { unit: UnitId(0) }).unwrap();
    let k = s.unit(UnitId(0)).unwrap();
    assert!(k.veteran);
    assert_eq!(k.health, 15);
    assert_eq!(k.max_health, 15);
    assert_eq!(
        k.phase,
        Phase::Moved,
        "Promote mid-chain must preserve Phase::Moved"
    );

    // Continue chaining with full veteran attack force (15/15 -> ratio 1.0 -> deals 10).
    apply(
        &mut s,
        &Action::Attack {
            unit: UnitId(0),
            target: p(3, 4),
        },
    )
    .unwrap();
    assert_eq!(s.unit(UnitId(0)).unwrap().pos, p(3, 4));
    assert_eq!(s.unit(UnitId(0)).unwrap().phase, Phase::Moved);
    apply(
        &mut s,
        &Action::Attack {
            unit: UnitId(0),
            target: p(3, 5),
        },
    )
    .unwrap();
    assert_eq!(s.unit(UnitId(0)).unwrap().pos, p(3, 5));
    assert_eq!(s.unit(UnitId(0)).unwrap().kills, 5);
}

#[test]
fn knight_persist_non_lethal_attack_ends_chain() {
    let mut s = v2_open(7)
        .unit(0, UnitType::Knight, p(3, 1))
        .unit(1, UnitType::Warrior, p(3, 2))
        .unit(1, UnitType::Defender, p(3, 3))
        .unit(1, UnitType::Warrior, p(2, 2))
        .build();
    // Kill 1: Warrior at (3,2).
    apply(
        &mut s,
        &Action::Attack {
            unit: UnitId(0),
            target: p(3, 2),
        },
    )
    .unwrap();
    assert_eq!(s.unit(UnitId(0)).unwrap().phase, Phase::Moved);

    // Non-lethal attack on Defender at (3,3): Defender 7/15, Knight 4/10, Phase::Attacked.
    apply(
        &mut s,
        &Action::Attack {
            unit: UnitId(0),
            target: p(3, 3),
        },
    )
    .unwrap();
    let k = s.unit(UnitId(0)).unwrap();
    assert_eq!(k.phase, Phase::Attacked);
    assert_eq!(k.health, 4);

    // No further Attack, Move, Capture, or Recover legal for the Knight.
    let legal = legal_actions(&s);
    assert!(
        !legal.iter().any(|a| matches!(
            a,
            Action::Attack { unit, .. }
                | Action::Move { unit, .. }
                | Action::Capture { unit }
                | Action::Recover { unit }
                if *unit == UnitId(0)
        )),
        "expected no follow-up actions for Knight after non-lethal hit: {legal:?}"
    );
    assert_illegal(
        &s,
        Action::Attack {
            unit: UnitId(0),
            target: p(2, 2),
        },
    );
}

#[test]
fn knight_persist_forbids_move_capture_and_recover_after_kill() {
    let mut s = v2_open(7)
        .village(p(2, 2))
        .city(1, p(2, 3))
        .techs(0, &[Tech::FreeSpirit])
        .unit(0, UnitType::Knight, p(2, 1))
        .unit_with(1, UnitType::Warrior, p(2, 2), |u| u.health = 4)
        .unit_with(1, UnitType::Warrior, p(2, 3), |u| u.health = 4)
        .build();

    // Kill enemy on village -> Knight kill-moves onto village in Phase::Moved.
    apply(
        &mut s,
        &Action::Attack {
            unit: UnitId(0),
            target: p(2, 2),
        },
    )
    .unwrap();
    assert_eq!(s.unit(UnitId(0)).unwrap().pos, p(2, 2));
    assert_eq!(s.unit(UnitId(0)).unwrap().phase, Phase::Moved);

    let legal = legal_actions(&s);
    assert!(legal.contains(&Action::Attack {
        unit: UnitId(0),
        target: p(2, 3),
    }));
    assert!(
        legal.contains(&Action::Disband { unit: UnitId(0) }),
        "Disband stays legal mid-chain"
    );
    assert!(
        !legal.iter().any(|a| matches!(a, Action::Move { .. })),
        "Move must be illegal after Persist kill"
    );
    assert!(!legal.contains(&Action::Capture { unit: UnitId(0) }));
    assert!(!legal.contains(&Action::Recover { unit: UnitId(0) }));
    assert_illegal(&s, Action::Capture { unit: UnitId(0) });
    assert_illegal(&s, Action::Recover { unit: UnitId(0) });
    assert_illegal(
        &s,
        Action::Move {
            unit: UnitId(0),
            to: p(2, 1),
        },
    );

    // Chain kill onto enemy city -> still cannot Capture, Move, or Recover.
    apply(
        &mut s,
        &Action::Attack {
            unit: UnitId(0),
            target: p(2, 3),
        },
    )
    .unwrap();
    assert_eq!(s.unit(UnitId(0)).unwrap().pos, p(2, 3));
    assert_eq!(s.unit(UnitId(0)).unwrap().phase, Phase::Moved);
    assert_illegal(&s, Action::Capture { unit: UnitId(0) });
    assert_illegal(&s, Action::Recover { unit: UnitId(0) });
    assert_illegal(
        &s,
        Action::Move {
            unit: UnitId(0),
            to: p(2, 2),
        },
    );
}

#[test]
fn knight_persist_kill_on_unclimbable_mountain_stays_in_place_and_continues() {
    // Owner lacks Climbing: kill-move onto mountain (2,2) is skipped, but Persist still sets
    // Phase::Moved so the Knight can attack another adjacent enemy at (2,0).
    let mut s = v2_open(5)
        .terrain(p(2, 2), Terrain::Mountain)
        .unit(0, UnitType::Knight, p(2, 1))
        .unit_with(1, UnitType::Warrior, p(2, 2), |u| u.health = 4)
        .unit_with(1, UnitType::Warrior, p(2, 0), |u| u.health = 4)
        .build();

    apply(
        &mut s,
        &Action::Attack {
            unit: UnitId(0),
            target: p(2, 2),
        },
    )
    .unwrap();
    assert!(s.unit(UnitId(1)).is_none(), "mountain defender died");
    let k = s.unit(UnitId(0)).unwrap();
    assert_eq!(
        k.pos,
        p(2, 1),
        "no kill-move onto mountain without Climbing"
    );
    assert_eq!(
        k.phase,
        Phase::Moved,
        "Persist still grants follow-up attack"
    );

    assert!(legal_actions(&s).contains(&Action::Attack {
        unit: UnitId(0),
        target: p(2, 0),
    }));
    apply(
        &mut s,
        &Action::Attack {
            unit: UnitId(0),
            target: p(2, 0),
        },
    )
    .unwrap();
    assert!(s.unit(UnitId(2)).is_none());
    assert_eq!(s.unit(UnitId(0)).unwrap().pos, p(2, 0));
    assert_eq!(s.unit(UnitId(0)).unwrap().kills, 2);
}

#[test]
fn knight_persist_resumes_after_mid_chain_lighthouse_pending_reward() {
    // Corner (0,0) is a hidden lighthouse. P0's capital at (5,5) is level 1 with pop 1 (needs 2).
    // Knight kills enemy 1 at (1,1), kill-moves to (1,1), reveals corner (0,0) -> capital hits
    // pop 2 -> level 2 pending_reward! ChooseReward resolves first, then Knight keeps Phase::Moved
    // and kills enemy 2 at (1,2).
    let mut s = v2_open(7)
        .lighthouse(p(0, 0))
        .hide(0, &[p(0, 0)])
        .city_with(0, p(5, 5), |c| {
            c.is_capital = true;
            c.level = 1;
            c.population = 1;
        })
        .unit(0, UnitType::Knight, p(2, 2))
        .unit_with(1, UnitType::Warrior, p(1, 1), |u| u.health = 4)
        .unit_with(1, UnitType::Warrior, p(1, 2), |u| u.health = 4)
        .build();

    apply(
        &mut s,
        &Action::Attack {
            unit: UnitId(0),
            target: p(1, 1),
        },
    )
    .unwrap();
    assert!(s.is_explored(0, p(0, 0)));
    assert_eq!(s.cities[0].level, 2);
    assert!(s.cities[0].pending_reward);
    assert_eq!(s.unit(UnitId(0)).unwrap().phase, Phase::Moved);

    // Forced reward blocks everything else until resolved.
    let legal = legal_actions(&s);
    assert_eq!(
        legal,
        vec![
            Action::ChooseReward {
                city: CityId(0),
                reward: Reward::Workshop,
            },
            Action::ChooseReward {
                city: CityId(0),
                reward: Reward::Explorer,
            },
        ]
    );

    apply(
        &mut s,
        &Action::ChooseReward {
            city: CityId(0),
            reward: Reward::Workshop,
        },
    )
    .unwrap();
    assert!(!s.cities[0].pending_reward);
    assert_eq!(s.unit(UnitId(0)).unwrap().phase, Phase::Moved);
    assert!(legal_actions(&s).contains(&Action::Attack {
        unit: UnitId(0),
        target: p(1, 2),
    }));

    apply(
        &mut s,
        &Action::Attack {
            unit: UnitId(0),
            target: p(1, 2),
        },
    )
    .unwrap();
    assert!(s.unit(UnitId(2)).is_none());
    assert_eq!(s.unit(UnitId(0)).unwrap().pos, p(1, 2));
    assert_eq!(s.unit(UnitId(0)).unwrap().kills, 2);
}

#[test]
fn knight_retaliation_kill_does_not_grant_persist_or_change_phase() {
    // Attacker Warrior 2/10 attacks defending Knight 10/10 (in Phase::Done).
    // af = 0.4, df = 1.0, td = 1.4 -> to_def = round(0.4/1.4 * 2 * 4.5) = round(2.5714) = 3
    // (Knight 7/10), to_atk = round(1.0/1.4 * 1 * 4.5) = round(3.2143) = 3 -> Warrior dies!
    // Defending Knight stays in place, keeps Phase::Done, gets kill credit, no Persist.
    let mut s = v2_open(5)
        .unit_with(0, UnitType::Warrior, p(2, 1), |u| u.health = 2)
        .unit_with(1, UnitType::Knight, p(2, 2), |u| u.phase = Phase::Done)
        .build();
    apply(
        &mut s,
        &Action::Attack {
            unit: UnitId(0),
            target: p(2, 2),
        },
    )
    .unwrap();
    assert!(
        s.unit(UnitId(0)).is_none(),
        "attacking Warrior died to retaliation"
    );
    let k = s.unit(UnitId(1)).unwrap();
    assert_eq!(k.pos, p(2, 2));
    assert_eq!(k.health, 7);
    assert_eq!(k.kills, 1);
    assert_eq!(
        k.phase,
        Phase::Done,
        "retaliation kill must not alter defender phase"
    );
    assert_eq!(s.players[1].kills, 1);
}
