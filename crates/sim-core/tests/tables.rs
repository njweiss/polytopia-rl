//! Data-table tests (spec "Data tables"). Values are from the owner's wiki extract; the site
//! action table (milestone 4) and the reward table (milestone 5) are the briefs' Appendices.

use sim_core::data::{
    Reward, SiteAction, Skill, Tables, Tech, UnitType, ALL_SITE_ACTIONS, ALL_TECHS, ALL_UNITS,
};
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
        assert_eq!(s.attack, r.attack as f32, "{u:?} attack");
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

const UNITS: &str = include_str!("../../../data/units.json");
const TECHS: &str = include_str!("../../../data/techs.json");
const SITES: &str = include_str!("../../../data/site_actions.json");
const REWARDS: &str = include_str!("../../../data/rewards.json");

#[test]
fn malformed_json_is_an_error_not_a_panic() {
    assert!(Tables::from_json("not json", "[]", SITES, REWARDS).is_err());
    assert!(Tables::from_json("[]", "[]", SITES, REWARDS).is_err()); // missing every unit and tech
}

#[test]
fn duplicate_or_missing_entries_are_rejected() {
    assert!(Tables::from_json(UNITS, TECHS, SITES, REWARDS).is_ok());
    // Drop the last tech (mining): incomplete.
    let cut = TECHS
        .rfind(",\n  { \"tech\": \"mining\"")
        .expect("fixture shape");
    let incomplete = format!("{}\n]", &TECHS[..cut]);
    assert!(Tables::from_json(UNITS, &incomplete, SITES, REWARDS).is_err());
    // Duplicate a tech.
    let dup = TECHS.replacen("\"mining\"", "\"hunting\"", 1);
    assert!(Tables::from_json(UNITS, &dup, SITES, REWARDS).is_err());
}

// ---------- milestone 4: site actions and rewards ----------

#[test]
fn site_action_table_matches_the_appendix() {
    let t = tables();
    // (action, cost, tech, population, stars) from docs/MILESTONE_4 Appendix.
    let rows = [
        (SiteAction::HarvestAnimal, 2, Hunting, 1, 0),
        (SiteAction::HarvestFruit, 2, Organization, 1, 0),
        (SiteAction::LumberHut, 3, Forestry, 1, 0),
        (SiteAction::ClearForest, 0, Forestry, 0, 1),
        (SiteAction::Farm, 5, Farming, 2, 0),
        (SiteAction::Mine, 5, Mining, 2, 0),
    ];
    for (a, cost, tech, population, stars) in rows {
        let info = t.site_action(a);
        assert_eq!(info.cost, cost, "{a:?} cost");
        assert_eq!(info.tech, tech, "{a:?} tech");
        assert_eq!(info.population, population, "{a:?} population");
        assert_eq!(info.stars, stars, "{a:?} stars");
    }
}

#[test]
fn all_site_actions_order() {
    assert_eq!(
        ALL_SITE_ACTIONS,
        [
            SiteAction::HarvestAnimal,
            SiteAction::HarvestFruit,
            SiteAction::LumberHut,
            SiteAction::ClearForest,
            SiteAction::Farm,
            SiteAction::Mine,
        ]
    );
}

#[test]
fn reward_menus_cover_levels_2_to_4_and_an_open_ended_5_plus() {
    let t = tables();
    assert_eq!(t.reward_menu(2), [Reward::Workshop, Reward::Explorer]);
    assert_eq!(t.reward_menu(3), [Reward::CityWall, Reward::Resources]);
    assert_eq!(
        t.reward_menu(4),
        [Reward::PopulationGrowth, Reward::BorderGrowth]
    );
    for level in [5, 6, 7, 10, 50, 1000] {
        assert_eq!(
            t.reward_menu(level),
            [Reward::Park, Reward::SuperUnit],
            "level {level}"
        );
    }
    assert_eq!(t.resources_reward_stars(), 5);
    assert_eq!(t.population_growth_amount(), 3);
}

#[test]
fn malformed_site_or_reward_json_is_an_error() {
    assert!(Tables::from_json(UNITS, TECHS, "not json", REWARDS).is_err());
    assert!(Tables::from_json(UNITS, TECHS, SITES, "not json").is_err());
    assert!(Tables::from_json(UNITS, TECHS, "[]", REWARDS).is_err()); // every action missing
                                                                      // Unknown names.
    let bad = SITES.replacen("\"harvest_fruit\"", "\"harvest_fish\"", 1);
    assert!(Tables::from_json(UNITS, TECHS, &bad, REWARDS).is_err());
    let bad = REWARDS.replacen("\"park\"", "\"monument\"", 1);
    assert!(Tables::from_json(UNITS, TECHS, SITES, &bad).is_err());
    // Milestone 5 names: "explorer" and "super_unit" (snake_case, like the others).
    for (name, wrong) in [
        ("explorer", "\"Explorer\""),
        ("super_unit", "\"superunit\""),
    ] {
        let quoted = format!("\"{name}\"");
        assert!(REWARDS.contains(&quoted), "rewards.json lacks {quoted}");
        let bad = REWARDS.replacen(&quoted, wrong, 1);
        assert!(
            Tables::from_json(UNITS, TECHS, SITES, &bad).is_err(),
            "{wrong}"
        );
    }
    // A missing scalar.
    let cut = REWARDS
        .find(",\n  \"population_growth_amount\"")
        .expect("fixture shape");
    let bad = format!("{}\n}}", &REWARDS[..cut]);
    assert!(Tables::from_json(UNITS, TECHS, SITES, &bad).is_err());
}

#[test]
fn duplicate_or_missing_site_actions_are_rejected() {
    // Drop the last action (mine): incomplete.
    let cut = SITES
        .rfind(",\n  { \"action\": \"mine\"")
        .expect("fixture shape");
    let incomplete = format!("{}\n]", &SITES[..cut]);
    assert!(Tables::from_json(UNITS, TECHS, &incomplete, REWARDS).is_err());
    // Duplicate an action while all six stay present (a second farm row).
    let dup = SITES.replacen(
        "\n]",
        ",\n  { \"action\": \"farm\", \"cost\": 1, \"tech\": \"farming\", \"population\": 1, \"stars\": 0 }\n]",
        1,
    );
    assert_ne!(dup, SITES, "fixture shape");
    assert!(Tables::from_json(UNITS, TECHS, &dup, REWARDS).is_err());
}

#[test]
fn reward_menus_must_cover_every_level_from_2_up_exactly_once() {
    // A gap: no menu for level 3.
    let gap = REWARDS.replacen(
        "{ \"min_level\": 3, \"max_level\": 3,    \"options\": [\"city_wall\", \"resources\"] },\n    ",
        "",
        1,
    );
    assert_ne!(gap, REWARDS, "fixture shape");
    assert!(Tables::from_json(UNITS, TECHS, SITES, &gap).is_err());
    // An overlap: level 3 covered twice.
    let overlap = REWARDS.replacen(
        "\"min_level\": 2, \"max_level\": 2",
        "\"min_level\": 2, \"max_level\": 3",
        1,
    );
    assert_ne!(overlap, REWARDS, "fixture shape");
    assert!(Tables::from_json(UNITS, TECHS, SITES, &overlap).is_err());
    // No open-ended last menu: levels 6 and up would have none.
    let closed = REWARDS.replacen("\"max_level\": null", "\"max_level\": 5", 1);
    assert_ne!(closed, REWARDS, "fixture shape");
    assert!(Tables::from_json(UNITS, TECHS, SITES, &closed).is_err());
}
