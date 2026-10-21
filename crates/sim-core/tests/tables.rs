//! Data-table tests (spec "Data tables"). Values are from the owner's wiki extract.

use sim_core::data::{Skill, Tables, Tech, UnitType, ALL_TECHS, ALL_UNITS};
use Skill::*;
use Tech::*;
use UnitType::*;

fn tables() -> Tables {
    Tables::load().expect("embedded tables must load")
}

struct Row {
    unit: UnitType,
    cost: Option<u32>,
    health: u32,
    attack: u32,
    defense: u32,
    movement: u32,
    range: u32,
    skills: Vec<Skill>,
    unlocked_by: Option<Tech>,
}

fn row(
    unit: UnitType,
    cost: Option<u32>,
    stats: [u32; 5],
    skills: Vec<Skill>,
    unlocked_by: Option<Tech>,
) -> Row {
    let [health, attack, defense, movement, range] = stats;
    Row {
        unit,
        cost,
        health,
        attack,
        defense,
        movement,
        range,
        skills,
        unlocked_by,
    }
}

#[test]
fn unit_table_matches_spec() {
    let t = tables();
    // stats = [health, attack, defense, movement, range]
    let rows = [
        row(
            Warrior,
            Some(2),
            [10, 2, 2, 1, 1],
            vec![Dash, Fortify],
            None,
        ),
        row(
            Rider,
            Some(3),
            [10, 2, 1, 2, 1],
            vec![Dash, Fortify, Escape],
            Some(Riding),
        ),
        row(
            Archer,
            Some(3),
            [10, 2, 1, 1, 2],
            vec![Dash, Fortify],
            Some(Archery),
        ),
        row(
            Defender,
            Some(3),
            [15, 1, 3, 1, 1],
            vec![Fortify],
            Some(Strategy),
        ),
        row(SuperUnit, None, [40, 5, 4, 1, 1], vec![Static], None),
    ];
    for r in rows {
        let u = r.unit;
        let s = t.unit(u);
        assert_eq!(s.cost, r.cost, "{u:?} cost");
        assert_eq!(s.health, r.health, "{u:?} health");
        assert_eq!(s.attack, r.attack, "{u:?} attack");
        assert_eq!(s.defense, r.defense, "{u:?} defense");
        assert_eq!(s.movement, r.movement, "{u:?} movement");
        assert_eq!(s.range, r.range, "{u:?} range");
        assert_eq!(s.skills.len(), r.skills.len(), "{u:?} skill count");
        for sk in r.skills {
            assert!(s.skills.contains(&sk), "{u:?} missing {sk:?}");
        }
        assert_eq!(s.unlocked_by, r.unlocked_by, "{u:?} unlocked_by");
    }
}

#[test]
fn super_unit_has_no_dash_escape_or_fortify() {
    let s = tables().unit(SuperUnit).clone();
    assert!(!s.skills.contains(&Dash));
    assert!(!s.skills.contains(&Escape));
    assert!(!s.skills.contains(&Fortify));
}

#[test]
fn tech_tree_matches_spec() {
    let t = tables();
    let rows = [
        (Climbing, 1, None),
        (Hunting, 1, None),
        (Organization, 1, None),
        (Riding, 1, None),
        (Archery, 2, Some(Hunting)),
        (Forestry, 2, Some(Hunting)),
        (Farming, 2, Some(Organization)),
        (Strategy, 2, Some(Organization)),
        (Mining, 2, Some(Climbing)),
    ];
    for (tech, tier, req) in rows {
        let info = t.tech(tech);
        assert_eq!(info.tier, tier, "{tech:?} tier");
        assert_eq!(info.requires, req, "{tech:?} requires");
    }
}

#[test]
fn tech_cost_is_4_plus_cities_times_tier() {
    let t = tables();
    assert_eq!(t.tech_cost(Climbing, 1), 5);
    assert_eq!(t.tech_cost(Archery, 1), 6);
    assert_eq!(t.tech_cost(Archery, 3), 10);
    assert_eq!(t.tech_cost(Riding, 4), 8);
    assert_eq!(t.tech_cost(Mining, 5), 14);
}

#[test]
fn every_enum_value_is_in_the_tables() {
    let t = tables();
    for u in ALL_UNITS {
        let _ = t.unit(u);
    }
    for k in ALL_TECHS {
        let _ = t.tech(k);
    }
}

#[test]
fn prerequisites_are_lower_tier_and_acyclic() {
    let t = tables();
    for k in ALL_TECHS {
        if let Some(r) = t.tech(k).requires {
            assert!(t.tech(r).tier < t.tech(k).tier, "{k:?} requires {r:?}");
        }
    }
}

#[test]
fn malformed_json_is_an_error_not_a_panic() {
    assert!(Tables::from_json("not json", "[]").is_err());
    assert!(Tables::from_json("[]", "[]").is_err()); // missing every unit and tech
}

#[test]
fn duplicate_or_missing_entries_are_rejected() {
    let units = include_str!("../../../data/units.json");
    let techs = include_str!("../../../data/techs.json");
    assert!(Tables::from_json(units, techs).is_ok());
    // Drop the last tech (mining): incomplete.
    let cut = techs
        .rfind(",\n  { \"tech\": \"mining\"")
        .expect("fixture shape");
    let incomplete = format!("{}\n]", &techs[..cut]);
    assert!(Tables::from_json(units, &incomplete).is_err());
    // Duplicate a tech.
    let dup = techs.replacen("\"mining\"", "\"hunting\"", 1);
    assert!(Tables::from_json(units, &dup).is_err());
}
