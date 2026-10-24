//! Milestone 7: serde derives on sim-core's public data types (docs/MILESTONE_7 "Changes to
//! sim-core", "Tests" item 1). The JSON strings are written by hand from serde's default
//! representation (externally tagged enums, newtype structs as their inner value, the existing
//! snake_case renames on the data enums, struct fields in declaration order).

use sim_core::actions::Action;
use sim_core::data::{Reward, SiteAction, Tech, UnitType};
use sim_core::grid::Pos;
use sim_core::map::MapParams;
use sim_core::state::{CityId, Config, Outcome, ScoreWeights, UnitId};

fn to_json<T: serde::Serialize>(v: &T) -> String {
    serde_json::to_string(v).expect("serialize")
}

fn round_trip<T>(v: &T) -> T
where
    T: serde::Serialize + serde::de::DeserializeOwned,
{
    serde_json::from_str(&to_json(v)).expect("deserialize")
}

#[test]
fn brief_json_strings_are_exact() {
    assert_eq!(to_json(&Action::EndTurn), r#""EndTurn""#);
    assert_eq!(
        to_json(&Action::Move {
            unit: UnitId(3),
            to: Pos { row: 2, col: 3 }
        }),
        r#"{"Move":{"unit":3,"to":{"row":2,"col":3}}}"#
    );
    assert_eq!(
        to_json(&Action::Train {
            city: CityId(0),
            kind: UnitType::Warrior
        }),
        r#"{"Train":{"city":0,"kind":"warrior"}}"#
    );
    assert_eq!(to_json(&Outcome::Win(1)), r#"{"Win":1}"#);
    assert_eq!(to_json(&Outcome::Draw), r#""Draw""#);
}

#[test]
fn other_action_variants_follow_the_default_representation() {
    let cases: Vec<(Action, &str)> = vec![
        (
            Action::Attack {
                unit: UnitId(12),
                target: Pos::new(0, 10),
            },
            r#"{"Attack":{"unit":12,"target":{"row":0,"col":10}}}"#,
        ),
        (
            Action::Capture { unit: UnitId(1) },
            r#"{"Capture":{"unit":1}}"#,
        ),
        (
            Action::Recover { unit: UnitId(2) },
            r#"{"Recover":{"unit":2}}"#,
        ),
        (
            Action::Promote { unit: UnitId(4) },
            r#"{"Promote":{"unit":4}}"#,
        ),
        (
            Action::Site {
                at: Pos::new(0, 4),
                action: SiteAction::LumberHut,
            },
            r#"{"Site":{"at":{"row":0,"col":4},"action":"lumber_hut"}}"#,
        ),
        (
            Action::Research {
                tech: Tech::Organization,
            },
            r#"{"Research":{"tech":"organization"}}"#,
        ),
        (
            Action::ChooseReward {
                city: CityId(1),
                reward: Reward::SuperUnit,
            },
            r#"{"ChooseReward":{"city":1,"reward":"super_unit"}}"#,
        ),
        (
            Action::Train {
                city: CityId(2),
                kind: UnitType::SuperUnit,
            },
            r#"{"Train":{"city":2,"kind":"super_unit"}}"#,
        ),
    ];
    for (a, json) in cases {
        assert_eq!(to_json(&a), json, "{a:?}");
    }
}

#[test]
fn newtypes_and_data_enums_are_plain_values() {
    assert_eq!(to_json(&UnitId(7)), "7");
    assert_eq!(to_json(&CityId(0)), "0");
    assert_eq!(to_json(&Pos::new(10, 1)), r#"{"row":10,"col":1}"#);
    assert_eq!(to_json(&Tech::Climbing), r#""climbing""#);
    assert_eq!(to_json(&Tech::Mining), r#""mining""#);
    assert_eq!(to_json(&SiteAction::HarvestFruit), r#""harvest_fruit""#);
    assert_eq!(to_json(&SiteAction::ClearForest), r#""clear_forest""#);
    assert_eq!(to_json(&Reward::PopulationGrowth), r#""population_growth""#);
    assert_eq!(to_json(&Reward::BorderGrowth), r#""border_growth""#);
    assert_eq!(to_json(&UnitType::Defender), r#""defender""#);
    assert_eq!(to_json(&Outcome::Win(0)), r#"{"Win":0}"#);
}

#[test]
fn default_config_json_is_exact() {
    assert_eq!(
        to_json(&Config::default()),
        concat!(
            r#"{"full_observability":false,"turn_cap":30,"score_weights":{"#,
            r#""unit_per_star":5,"super_unit":50,"territory_per_tile":20,"#,
            r#""exploration_per_tile":5,"city_base":100,"city_per_level":50,"#,
            r#""population_surplus":5,"park":250,"tech_per_tier":100}}"#
        )
    );
}

#[test]
fn every_action_variant_round_trips() {
    let actions = vec![
        Action::Move {
            unit: UnitId(0),
            to: Pos::new(0, 0),
        },
        Action::Move {
            unit: UnitId(u32::MAX),
            to: Pos::new(255, 254),
        },
        Action::Attack {
            unit: UnitId(5),
            target: Pos::new(3, 9),
        },
        Action::Capture { unit: UnitId(6) },
        Action::Recover { unit: UnitId(7) },
        Action::Promote { unit: UnitId(8) },
        Action::Train {
            city: CityId(3),
            kind: UnitType::Archer,
        },
        Action::Train {
            city: CityId(0),
            kind: UnitType::Rider,
        },
        Action::Site {
            at: Pos::new(10, 10),
            action: SiteAction::Mine,
        },
        Action::Site {
            at: Pos::new(1, 2),
            action: SiteAction::HarvestAnimal,
        },
        Action::Site {
            at: Pos::new(2, 1),
            action: SiteAction::Farm,
        },
        Action::Research {
            tech: Tech::Hunting,
        },
        Action::Research {
            tech: Tech::Strategy,
        },
        Action::ChooseReward {
            city: CityId(4),
            reward: Reward::Workshop,
        },
        Action::ChooseReward {
            city: CityId(4),
            reward: Reward::Explorer,
        },
        Action::ChooseReward {
            city: CityId(1),
            reward: Reward::CityWall,
        },
        Action::ChooseReward {
            city: CityId(1),
            reward: Reward::Resources,
        },
        Action::ChooseReward {
            city: CityId(1),
            reward: Reward::Park,
        },
        Action::EndTurn,
    ];
    for a in &actions {
        assert_eq!(round_trip(a), *a, "{a:?}");
    }
    // As a list (how the replay log stores them).
    let back: Vec<Action> = round_trip(&actions);
    assert_eq!(back, actions);
}

#[test]
fn config_round_trips_with_every_weight() {
    let c = Config {
        full_observability: true,
        turn_cap: 37,
        score_weights: ScoreWeights {
            unit_per_star: 1,
            super_unit: 2,
            territory_per_tile: 3,
            exploration_per_tile: 4,
            city_base: 5,
            city_per_level: 6,
            population_surplus: 7,
            park: 8,
            tech_per_tier: 9,
        },
    };
    assert_eq!(round_trip(&c), c);
    assert_eq!(round_trip(&Config::default()), Config::default());
    assert_eq!(round_trip(&c.score_weights), c.score_weights);
}

#[test]
fn map_params_floats_round_trip_bit_for_bit() {
    let check = |m: &MapParams| {
        let back = round_trip(m);
        assert_eq!(back, *m);
        let bits = |m: &MapParams| {
            [
                m.forest_prob,
                m.mountain_prob,
                m.fruit_prob,
                m.crop_prob,
                m.animal_prob,
                m.metal_prob,
                m.outer_ring_factor,
            ]
            .map(f32::to_bits)
        };
        assert_eq!(bits(&back), bits(m));
        assert_eq!(back.size, m.size);
        assert_eq!(back.min_capital_fruit, m.min_capital_fruit);
    };
    let d = MapParams::default();
    assert_eq!(d.outer_ring_factor, 1.0f32 / 3.0);
    check(&d);
    check(&MapParams {
        size: 7,
        forest_prob: 0.1,
        mountain_prob: 2.0 / 3.0,
        fruit_prob: 1.0e-7,
        crop_prob: 0.999_999_9,
        animal_prob: 0.0,
        metal_prob: 1.0,
        outer_ring_factor: 1.0 / 7.0,
        min_capital_fruit: 0,
    });
}

#[test]
fn outcomes_round_trip() {
    for o in [Outcome::Win(0), Outcome::Win(1), Outcome::Draw] {
        assert_eq!(round_trip(&o), o);
        assert_eq!(round_trip(&Some(o)), Some(o));
    }
    assert_eq!(round_trip(&None::<Outcome>), None);
}
