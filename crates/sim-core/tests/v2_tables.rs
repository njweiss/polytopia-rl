//! Milestone 15 "Tests" item 2: the v2 data tables (docs/MILESTONE_15 "Data files"). Every expected
//! number is copied by hand from the brief (itself the approved v2 rules, docs/V2_RULES_NOTES.md);
//! malformed data is hand-written JSON fed through `Tables::from_json_v2`.

use std::panic::{catch_unwind, AssertUnwindSafe};

use sim_core::combat::{compute_damage, Combatant};
use sim_core::data::{
    site_actions, techs, units, Building, LevelFrom, Monument, MonumentTask, SiteAction, Skill,
    Tables, Tech, UnitType, ALL_REWARDS, ALL_SITE_ACTIONS, ALL_TECHS, ALL_UNITS,
};
use sim_core::state::Ruleset;

const V2_UNITS: &str = include_str!("../../../data/v2/units.json");
const V2_TECHS: &str = include_str!("../../../data/v2/techs.json");
const V2_SITES: &str = include_str!("../../../data/v2/site_actions.json");
const V2_BUILDINGS: &str = include_str!("../../../data/v2/buildings.json");
const V2_MONUMENTS: &str = include_str!("../../../data/v2/monuments.json");
const V2_RULES: &str = include_str!("../../../data/v2/rules.json");

const V1_UNITS: &str = include_str!("../../../data/units.json");
const V1_TECHS: &str = include_str!("../../../data/techs.json");
const V1_SITES: &str = include_str!("../../../data/site_actions.json");
const V1_REWARDS: &str = include_str!("../../../data/rewards.json");

fn v2() -> Tables {
    Tables::load_for(Ruleset::V2).expect("the embedded v2 tables load")
}

/// `from_json_v2` with the embedded v2 files, except the ones the closure replaces.
fn v2_with(edit: impl FnOnce(&mut [String; 6])) -> Result<Tables, String> {
    let mut f = [
        V2_UNITS,
        V2_TECHS,
        V2_SITES,
        V2_BUILDINGS,
        V2_MONUMENTS,
        V2_RULES,
    ]
    .map(str::to_string);
    edit(&mut f);
    Tables::from_json_v2(&f[0], &f[1], &f[2], &f[3], &f[4], &f[5])
}

/// Replaces exactly one occurrence of `from` (the edit must hit the file).
fn swap(s: &str, from: &str, to: &str) -> String {
    assert_eq!(s.matches(from).count(), 1, "{from:?} must occur once");
    s.replacen(from, to, 1)
}

/// Inserts `row` as the first element of a JSON list.
fn prepend(list: &str, row: &str) -> String {
    let rest = list.strip_prefix('[').expect("a JSON list");
    format!("[\n  {row},{rest}")
}

#[test]
fn enum_variants_are_appended_in_the_brief_order() {
    use UnitType::*;
    assert_eq!(
        [Warrior, Rider, Archer, Defender, SuperUnit, Swordsman, Catapult, Knight].map(|u| u as u8),
        [0, 1, 2, 3, 4, 5, 6, 7]
    );
    assert_eq!(
        [
            Skill::Dash,
            Skill::Escape,
            Skill::Fortify,
            Skill::Static,
            Skill::Stiff,
            Skill::Persist
        ]
        .map(|s| s as u8),
        [0, 1, 2, 3, 4, 5]
    );
    let v2_techs = [
        Tech::Roads,
        Tech::FreeSpirit,
        Tech::Trade,
        Tech::Chivalry,
        Tech::Mathematics,
        Tech::Construction,
        Tech::Smithery,
    ];
    assert_eq!(v2_techs.map(|t| t as u16), [9, 10, 11, 12, 13, 14, 15]);
    assert_eq!(Tech::Mining as u16, 8);
    assert_eq!(SiteAction::BurnForest as u8, 6);
    assert_eq!(
        [
            Building::Sawmill,
            Building::Windmill,
            Building::Forge,
            Building::Market
        ]
        .map(|b| b as u8),
        [0, 1, 2, 3]
    );
    assert_eq!(
        [
            Monument::ParkOfFortune,
            Monument::GateOfPower,
            Monument::EyeOfGod
        ]
        .map(|m| m as u8),
        [0, 1, 2]
    );
}

#[test]
fn the_v1_lists_are_unchanged_and_the_ruleset_lists_extend_them() {
    use UnitType::*;
    assert_eq!(ALL_UNITS, [Warrior, Rider, Archer, Defender, SuperUnit]);
    assert_eq!(ALL_TECHS.len(), 9);
    assert_eq!(ALL_TECHS[8], Tech::Mining);
    assert_eq!(ALL_SITE_ACTIONS.len(), 6);
    assert_eq!(ALL_SITE_ACTIONS[5], SiteAction::Mine);
    assert_eq!(ALL_REWARDS.len(), 8);

    assert_eq!(units(Ruleset::V1), &ALL_UNITS[..]);
    assert_eq!(techs(Ruleset::V1), &ALL_TECHS[..]);
    assert_eq!(site_actions(Ruleset::V1), &ALL_SITE_ACTIONS[..]);
    assert_eq!(
        units(Ruleset::V2),
        &[Warrior, Rider, Archer, Defender, SuperUnit, Swordsman, Catapult, Knight][..]
    );
    let t = techs(Ruleset::V2);
    assert_eq!(t.len(), 16);
    assert_eq!(&t[..9], &ALL_TECHS[..]);
    assert_eq!(
        &t[9..],
        &[
            Tech::Roads,
            Tech::FreeSpirit,
            Tech::Trade,
            Tech::Chivalry,
            Tech::Mathematics,
            Tech::Construction,
            Tech::Smithery
        ][..]
    );
    let s = site_actions(Ruleset::V2);
    assert_eq!(s.len(), 7);
    assert_eq!(&s[..6], &ALL_SITE_ACTIONS[..]);
    assert_eq!(s[6], SiteAction::BurnForest);
}

#[test]
fn load_for_v1_is_load_and_load_for_v2_is_v2() {
    let v1 = Tables::load_for(Ruleset::V1).unwrap();
    assert_eq!(v1, Tables::load().unwrap());
    assert_eq!(v1.ruleset(), Ruleset::V1);
    assert_eq!(Tables::load().unwrap().ruleset(), Ruleset::V1);
    let t = v2();
    assert_eq!(t.ruleset(), Ruleset::V2);
    assert_eq!(t, v2(), "deterministic");
    assert_ne!(t, v1);
    // The embedded files through the test entry point give the same tables.
    assert_eq!(v2_with(|_| {}).unwrap(), t);
}

/// (unit, cost, health, attack, defense, movement, range, skills, unlocked_by)
type UnitRow = (UnitType, u32, u32, f32, u32, u32, u32, Vec<Skill>, Tech);

#[test]
fn v2_unit_rows_match_the_brief() {
    use Skill::*;
    let t = v2();
    // (unit, cost, health, attack, defense, movement, range, skills, unlocked_by)
    let rows: [UnitRow; 3] = [
        (
            UnitType::Swordsman,
            5,
            15,
            3.0,
            3,
            1,
            1,
            vec![Dash, Fortify],
            Tech::Smithery,
        ),
        (
            UnitType::Catapult,
            8,
            10,
            4.0,
            0,
            1,
            3,
            vec![Stiff],
            Tech::Mathematics,
        ),
        (
            UnitType::Knight,
            8,
            10,
            3.5,
            1,
            3,
            1,
            vec![Dash, Fortify, Persist],
            Tech::Chivalry,
        ),
    ];
    for (u, cost, health, attack, defense, movement, range, skills, tech) in rows {
        let s = t.unit(u);
        assert_eq!(s.cost, Some(cost), "{u:?} cost");
        assert_eq!(s.health, health, "{u:?} health");
        assert_eq!(s.attack.to_bits(), attack.to_bits(), "{u:?} attack");
        assert_eq!(s.defense, defense, "{u:?} defense");
        assert_eq!(s.movement, movement, "{u:?} movement");
        assert_eq!(s.range, range, "{u:?} range");
        assert_eq!(s.skills, skills, "{u:?} skills");
        assert_eq!(s.unlocked_by, Some(tech), "{u:?} unlocked_by");
    }
    // The v1 rows are the v1 tables' rows.
    let v1 = Tables::load().unwrap();
    for u in ALL_UNITS {
        assert_eq!(t.unit(u), v1.unit(u), "{u:?}");
    }
}

#[test]
fn the_knights_attack_is_not_truncated_in_combat() {
    // Knight 3.5 at full health against a full Warrior (defense 2), bonus 1.0, spec formulas:
    // attackForce 3.5, defenseForce 2, total 5.5; attackResult 3.5/5.5*3.5*4.5 = 10.02 -> 10,
    // defenseResult 2/5.5*2*4.5 = 3.27 -> 3. (A truncated attack 3 would give 8 and 4.)
    let t = v2();
    let c = |u: UnitType| {
        let s = t.unit(u);
        Combatant {
            attack: s.attack,
            defense: s.defense,
            health: s.health,
            max_health: s.health,
        }
    };
    let d = compute_damage(&c(UnitType::Knight), &c(UnitType::Warrior), 1.0);
    assert_eq!((d.to_defender, d.to_attacker), (10, 3));
}

#[test]
fn v2_tech_rows_match_the_brief() {
    let t = v2();
    let rows = [
        (Tech::Roads, 2, Tech::Riding),
        (Tech::FreeSpirit, 2, Tech::Riding),
        (Tech::Trade, 3, Tech::Roads),
        (Tech::Chivalry, 3, Tech::FreeSpirit),
        (Tech::Mathematics, 3, Tech::Forestry),
        (Tech::Construction, 3, Tech::Farming),
        (Tech::Smithery, 3, Tech::Mining),
    ];
    for (tech, tier, req) in rows {
        assert_eq!(t.tech(tech).tier, tier, "{tech:?} tier");
        assert_eq!(t.tech(tech).requires, Some(req), "{tech:?} requires");
        // Spec cost formula, 1 and 3 cities.
        assert_eq!(t.tech_cost(tech, 1), 4 + tier);
        assert_eq!(t.tech_cost(tech, 3), 4 + 3 * tier);
    }
    let v1 = Tables::load().unwrap();
    for k in ALL_TECHS {
        assert_eq!(t.tech(k), v1.tech(k), "{k:?}");
    }
}

#[test]
fn v2_site_action_row_matches_the_brief() {
    let t = v2();
    let b = t.site_action(SiteAction::BurnForest);
    assert_eq!(
        (b.cost, b.tech, b.population, b.stars),
        (3, Tech::Construction, 0, 0)
    );
    let v1 = Tables::load().unwrap();
    for a in ALL_SITE_ACTIONS {
        assert_eq!(t.site_action(a), v1.site_action(a), "{a:?}");
    }
}

#[test]
fn building_rows_match_the_brief() {
    let t = v2();
    // (building, cost, tech, adjacent, level_from, population_per_level, stars_per_level,
    //  max_output, one_per_city)
    let rows = [
        (
            Building::Sawmill,
            5,
            Tech::Mathematics,
            vec!["lumber_hut"],
            LevelFrom::Count,
            1,
            0,
            None,
        ),
        (
            Building::Windmill,
            5,
            Tech::Construction,
            vec!["farm"],
            LevelFrom::Count,
            1,
            0,
            None,
        ),
        (
            Building::Forge,
            5,
            Tech::Smithery,
            vec!["mine"],
            LevelFrom::Count,
            2,
            0,
            None,
        ),
        (
            Building::Market,
            5,
            Tech::Trade,
            vec!["sawmill", "windmill", "forge"],
            LevelFrom::Levels,
            0,
            1,
            Some(8),
        ),
    ];
    for (b, cost, tech, adjacent, level_from, pop, stars, max_output) in rows {
        let i = t.building(b);
        assert_eq!(i.cost, cost, "{b:?}");
        assert_eq!(i.tech, tech, "{b:?}");
        assert_eq!(i.adjacent, adjacent, "{b:?}");
        assert_eq!(i.level_from, level_from, "{b:?}");
        assert_eq!(i.population_per_level, pop, "{b:?}");
        assert_eq!(i.stars_per_level, stars, "{b:?}");
        assert_eq!(i.max_output, max_output, "{b:?}");
        assert!(i.one_per_city, "{b:?}");
    }
}

#[test]
fn monument_rows_and_v2_rules_match_the_brief() {
    let t = v2();
    let m = |x| {
        let i = t.monument(x);
        (i.task, i.threshold)
    };
    assert_eq!(m(Monument::ParkOfFortune), (MonumentTask::CityLevel, 5));
    assert_eq!(m(Monument::GateOfPower), (MonumentTask::Kills, 10));
    assert_eq!(m(Monument::EyeOfGod), (MonumentTask::Lighthouses, 4));

    let r = t.v2_rules();
    assert_eq!(r.road_cost, 3);
    assert_eq!(r.road_tech, Tech::Roads);
    assert_eq!(r.road_step_cost.to_bits(), 0.5f32.to_bits());
    assert_eq!(r.connection_population, 1);
    assert_eq!(r.disband_tech, Tech::FreeSpirit);
    assert_eq!(r.destroy_tech, Tech::Chivalry);
    assert_eq!(r.disband_refund_divisor, 2);
    assert_eq!(r.super_unit_value, 10);
    assert_eq!(r.lighthouse_population, 1);
    assert_eq!(r.explorer_lighthouse_bonus.to_bits(), 3.5f32.to_bits());
    assert_eq!(r.monument_population, 3);
    assert_eq!(r.monument_score, 400);
    // v1 scalars are unchanged under v2.
    assert_eq!(t.rules(), Tables::load().unwrap().rules());
}

#[test]
fn v2_accessors_panic_under_v1() {
    let v1 = Tables::load().unwrap();
    let quiet = |f: &dyn Fn()| catch_unwind(AssertUnwindSafe(f)).is_err();
    // The v2 tables answer them ...
    let t = v2();
    assert_eq!(t.building(Building::Forge).population_per_level, 2);
    assert_eq!(t.monument(Monument::EyeOfGod).threshold, 4);
    assert_eq!(t.v2_rules().monument_score, 400);
    // ... the v1 tables refuse.
    assert!(quiet(&|| {
        v1.building(Building::Sawmill);
    }));
    assert!(quiet(&|| {
        v1.monument(Monument::GateOfPower);
    }));
    assert!(quiet(&|| {
        v1.v2_rules();
    }));
}

#[test]
fn malformed_v2_json_is_an_error() {
    for i in 0..6 {
        assert!(
            v2_with(|f| f[i] = "not json".into()).is_err(),
            "file {i} not json"
        );
    }
    // A list where an object is expected and vice versa.
    assert!(v2_with(|f| f[5] = "[]".into()).is_err());
    assert!(v2_with(|f| f[3] = "{}".into()).is_err());
}

#[test]
fn a_missing_v2_row_is_an_error() {
    let knight = V2_UNITS.lines().find(|l| l.contains("\"knight\"")).unwrap();
    assert!(v2_with(|f| f[0] = swap(V2_UNITS, &format!(",\n{knight}"), "")).is_err());
    let smithery = V2_TECHS
        .lines()
        .find(|l| l.contains("\"smithery\""))
        .unwrap();
    assert!(v2_with(|f| f[1] = swap(V2_TECHS, &format!(",\n{smithery}"), "")).is_err());
    assert!(v2_with(|f| f[2] = "[]".into()).is_err());
    let market = V2_BUILDINGS
        .lines()
        .find(|l| l.contains("\"market\""))
        .unwrap();
    assert!(v2_with(|f| f[3] = swap(V2_BUILDINGS, &format!(",\n{market}"), "")).is_err());
    let eye = V2_MONUMENTS
        .lines()
        .find(|l| l.contains("\"eye_of_god\""))
        .unwrap();
    assert!(v2_with(|f| f[4] = swap(V2_MONUMENTS, &format!(",\n{eye}"), "")).is_err());
    // A missing scalar.
    assert!(v2_with(|f| f[5] = swap(V2_RULES, "\"monument_score\": 400", "\"x_\": 0")).is_err());
    assert!(v2_with(|f| f[5] = swap(V2_RULES, ",\n  \"monument_score\": 400", "")).is_err());
}

#[test]
fn a_duplicated_v2_row_is_an_error() {
    let row = |file: &str, key: &str| {
        file.lines()
            .find(|l| l.contains(key))
            .unwrap()
            .trim()
            .trim_end_matches(',')
            .to_string()
    };
    assert!(v2_with(|f| f[0] = prepend(V2_UNITS, &row(V2_UNITS, "\"knight\""))).is_err());
    assert!(v2_with(|f| f[1] = prepend(V2_TECHS, &row(V2_TECHS, "\"roads\""))).is_err());
    assert!(v2_with(|f| f[2] = prepend(V2_SITES, &row(V2_SITES, "\"burn_forest\""))).is_err());
    assert!(v2_with(|f| f[3] = prepend(V2_BUILDINGS, &row(V2_BUILDINGS, "\"sawmill\""))).is_err());
    assert!(
        v2_with(|f| f[4] = prepend(V2_MONUMENTS, &row(V2_MONUMENTS, "\"gate_of_power\""))).is_err()
    );
    // A v1 row repeated in a v2 file duplicates the v1 table's row.
    let warrior = row(V1_UNITS, "\"warrior\"");
    assert!(v2_with(|f| f[0] = prepend(V2_UNITS, &warrior)).is_err());
    let climbing = row(V1_TECHS, "\"climbing\"");
    assert!(v2_with(|f| f[1] = prepend(V2_TECHS, &climbing)).is_err());
    let farm = row(V1_SITES, "\"farm\"");
    assert!(v2_with(|f| f[2] = prepend(V2_SITES, &farm)).is_err());
}

#[test]
fn an_unknown_v2_row_or_field_is_an_error() {
    let musketeer = r#"{ "unit": "musketeer", "cost": 5, "health": 10, "attack": 3, "defense": 1, "movement": 1, "range": 1, "skills": [], "unlocked_by": null }"#;
    assert!(v2_with(|f| f[0] = prepend(V2_UNITS, musketeer)).is_err());
    let sailing = r#"{ "tech": "sailing", "tier": 1, "requires": null }"#;
    assert!(v2_with(|f| f[1] = prepend(V2_TECHS, sailing)).is_err());
    let fish = r#"{ "action": "fish", "cost": 2, "tech": "hunting", "population": 1, "stars": 0 }"#;
    assert!(v2_with(|f| f[2] = prepend(V2_SITES, fish)).is_err());
    let temple = r#"{ "building": "temple", "cost": 5, "tech": "trade", "adjacent": [], "level_from": "count", "population_per_level": 1, "stars_per_level": 0, "max_output": null, "one_per_city": true }"#;
    assert!(v2_with(|f| f[3] = prepend(V2_BUILDINGS, temple)).is_err());
    let altar = r#"{ "monument": "altar_of_peace", "task": "kills", "threshold": 1 }"#;
    assert!(v2_with(|f| f[4] = prepend(V2_MONUMENTS, altar)).is_err());
    // Unknown fields (deny_unknown_fields on every row and file).
    assert!(v2_with(|f| f[3] = swap(
        V2_BUILDINGS,
        "\"building\": \"forge\",",
        "\"building\": \"forge\", \"color\": 1,"
    ))
    .is_err());
    assert!(v2_with(|f| f[4] = swap(
        V2_MONUMENTS,
        "\"monument\": \"eye_of_god\",",
        "\"monument\": \"eye_of_god\", \"x\": 1,"
    ))
    .is_err());
    assert!(v2_with(|f| f[5] = swap(
        V2_RULES,
        "\"road_cost\": 3,",
        "\"road_cost\": 3, \"bridge_cost\": 5,"
    ))
    .is_err());
    // Unknown enum values inside a row.
    assert!(v2_with(|f| f[3] = swap(
        V2_BUILDINGS,
        "\"level_from\": \"levels\"",
        "\"level_from\": \"sum\""
    ))
    .is_err());
    assert!(
        v2_with(|f| f[4] = swap(V2_MONUMENTS, "\"task\": \"kills\"", "\"task\": \"wins\""))
            .is_err()
    );
}

#[test]
fn a_prerequisite_of_equal_or_higher_tier_is_an_error() {
    // Roads (tier 2) requiring Free Spirit (tier 2): equal tier.
    let equal = swap(
        V2_TECHS,
        r#""tech": "roads",        "tier": 2, "requires": "riding""#,
        r#""tech": "roads",        "tier": 2, "requires": "free_spirit""#,
    );
    assert!(v2_with(|f| f[1] = equal).is_err());
    // Roads (tier 2) requiring Trade (tier 3): higher tier (and a cycle).
    let higher = swap(
        V2_TECHS,
        r#""tech": "roads",        "tier": 2, "requires": "riding""#,
        r#""tech": "roads",        "tier": 2, "requires": "trade""#,
    );
    assert!(v2_with(|f| f[1] = higher).is_err());
    // A tech requiring itself.
    let itself = swap(
        V2_TECHS,
        r#""requires": "mining""#,
        r#""requires": "smithery""#,
    );
    assert!(v2_with(|f| f[1] = itself).is_err());
    // Smithery at tier 2 requiring Mining (tier 2).
    let lowered = swap(
        V2_TECHS,
        r#""tech": "smithery",     "tier": 3"#,
        r#""tech": "smithery",     "tier": 2"#,
    );
    assert!(v2_with(|f| f[1] = lowered).is_err());
    // Any lower tier is fine: Chivalry (3) requiring Riding (1), Trade (3) requiring Mining (2).
    let ok = swap(
        V2_TECHS,
        r#""requires": "free_spirit""#,
        r#""requires": "riding" "#,
    );
    assert!(v2_with(|f| f[1] = ok).is_ok());
    let ok = swap(
        V2_TECHS,
        r#""requires": "roads""#,
        r#""requires": "mining""#,
    );
    assert!(v2_with(|f| f[1] = ok).is_ok());
}

#[test]
fn an_unknown_adjacent_name_is_an_error() {
    let bad = swap(V2_BUILDINGS, r#"["lumber_hut"]"#, r#"["lumberhut"]"#);
    assert!(v2_with(|f| f[3] = bad).is_err());
    let bad = swap(
        V2_BUILDINGS,
        r#"["sawmill", "windmill", "forge"]"#,
        r#"["sawmill", "temple", "forge"]"#,
    );
    assert!(v2_with(|f| f[3] = bad).is_err());
    let bad = swap(V2_BUILDINGS, r#"["mine"]"#, r#"["metal"]"#);
    assert!(v2_with(|f| f[3] = bad).is_err());
    // Other known improvement or building names are accepted.
    let ok = swap(V2_BUILDINGS, r#"["mine"]"#, r#"["mine", "farm", "market"]"#);
    assert!(v2_with(|f| f[3] = ok).is_ok());
}

#[test]
fn v1_files_with_a_v2_row_are_an_error_not_a_panic() {
    // Under v1 a v2 variant has no slot: `from_json` must reject it instead of indexing past the
    // v1 lists.
    let knight = r#"{ "unit": "knight", "cost": 8, "health": 10, "attack": 3.5, "defense": 1, "movement": 3, "range": 1, "skills": ["dash"], "unlocked_by": "chivalry" }"#;
    let units = prepend(V1_UNITS, knight);
    let r = catch_unwind(|| Tables::from_json(&units, V1_TECHS, V1_SITES, V1_REWARDS));
    assert!(matches!(r, Ok(Err(_))), "knight in v1 units.json");
    let roads = r#"{ "tech": "roads", "tier": 2, "requires": "riding" }"#;
    let techs = prepend(V1_TECHS, roads);
    let r = catch_unwind(|| Tables::from_json(V1_UNITS, &techs, V1_SITES, V1_REWARDS));
    assert!(matches!(r, Ok(Err(_))), "roads in v1 techs.json");
    let burn = r#"{ "action": "burn_forest", "cost": 3, "tech": "construction", "population": 0, "stars": 0 }"#;
    let sites = prepend(V1_SITES, burn);
    let r = catch_unwind(|| Tables::from_json(V1_UNITS, V1_TECHS, &sites, V1_REWARDS));
    assert!(
        matches!(r, Ok(Err(_))),
        "burn_forest in v1 site_actions.json"
    );
    // And the v1 files themselves still load.
    assert!(Tables::from_json(V1_UNITS, V1_TECHS, V1_SITES, V1_REWARDS).is_ok());
}
