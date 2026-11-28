//! Milestone 20 Phase A: `PlayerView` v2 visibility (`VisibleTile` v2 fields, viewer-only v2
//! counters/monuments) and `PlayerView::attack_preview` (docs/MILESTONE_20 "PlayerView v2 &
//! attack_preview").

use std::sync::Arc;

use sim_core::actions::{apply, Action};
use sim_core::combat::{compute_damage, resolve_attack, Combatant};
use sim_core::data::{Building, Monument, Skill, Tables, Tech, UnitType};
use sim_core::grid::Pos;
use sim_core::map::Terrain;
use sim_core::state::{
    CityId, Config, Improvement, MonumentSet, Outcome, Phase, Ruleset, State, UnitId,
};
use sim_core::testing::StateBuilder;
use sim_core::view::{AttackPreview, PlayerView};

fn p(row: u8, col: u8) -> Pos {
    Pos::new(row, col)
}

fn v2_cfg() -> Config {
    Config {
        ruleset: Ruleset::V2,
        ..Config::default()
    }
}

fn set_road(s: &mut State, q: Pos) {
    let i = q.row as usize * s.size as usize + q.col as usize;
    s.tiles[i].road = true;
}

/// Independent check of `attack_preview` against both `apply_attack` and the spec combat formula.
fn assert_preview_matches_apply(
    s: &State,
    attacker: UnitId,
    target: Pos,
    expected_bonus: f32,
    ctx: &str,
) -> AttackPreview {
    let viewer = s.current;
    let view = PlayerView::new(s, viewer);
    let preview = view
        .attack_preview(attacker, target)
        .unwrap_or_else(|| panic!("{ctx}: expected Some(AttackPreview), got None"));
    assert_eq!(
        preview.defense_bonus, expected_bonus,
        "{ctx}: defense_bonus"
    );

    let a_before = s.unit(attacker).expect("attacker alive");
    let d_id = s.tile(target).unit.expect("defender on target");
    let d_before = s.unit(d_id).expect("defender alive");

    // Cross-check against the spec's combat formula directly.
    let sa = s.tables.unit(a_before.kind);
    let sd = s.tables.unit(d_before.kind);
    let ca = Combatant {
        attack: sa.attack,
        defense: sa.defense,
        health: a_before.health,
        max_health: a_before.max_health,
    };
    let cd = Combatant {
        attack: sd.attack,
        defense: sd.defense,
        health: d_before.health,
        max_health: d_before.max_health,
    };
    let dist = (a_before.pos.row as i32 - d_before.pos.row as i32)
        .unsigned_abs()
        .max((a_before.pos.col as i32 - d_before.pos.col as i32).unsigned_abs());
    let can_retal = dist <= sd.range && !sd.skills.contains(&Skill::Stiff);
    let expected = resolve_attack(&ca, &cd, expected_bonus, can_retal);
    assert_eq!(
        preview.dealt,
        d_before.health - expected.defender_health,
        "{ctx}: dealt vs formula"
    );
    assert_eq!(
        preview.retaliation,
        a_before.health - expected.attacker_health,
        "{ctx}: retaliation vs formula"
    );
    assert_eq!(
        preview.defender_dies,
        expected.defender_health == 0,
        "{ctx}: defender_dies vs formula"
    );
    assert_eq!(
        preview.attacker_dies,
        expected.attacker_health == 0,
        "{ctx}: attacker_dies vs formula"
    );

    // Cross-check bit-for-bit against `sim_core::actions::apply`.
    let mut after = s.clone();
    apply(
        &mut after,
        &Action::Attack {
            unit: attacker,
            target,
        },
    )
    .unwrap_or_else(|e| panic!("{ctx}: apply_attack failed: {e:?}"));
    let a_after = after.unit(attacker).map_or(0, |u| u.health);
    let d_after = after.unit(d_id).map_or(0, |u| u.health);
    assert_eq!(
        preview.dealt,
        d_before.health - d_after,
        "{ctx}: dealt vs apply"
    );
    assert_eq!(
        preview.retaliation,
        a_before.health - a_after,
        "{ctx}: retaliation vs apply"
    );
    assert_eq!(preview.defender_dies, d_after == 0, "{ctx}: def dies");
    assert_eq!(preview.attacker_dies, a_after == 0, "{ctx}: atk dies");
    preview
}

#[test]
fn visible_tile_exposes_v2_fields_only_when_explored() {
    let mut s = StateBuilder::new(7)
        .config(v2_cfg())
        .city_with(0, p(2, 2), |c| c.is_capital = true)
        .city(0, p(5, 1))
        .city_with(1, p(5, 5), |c| c.is_capital = true)
        // Explored Sawmill at (1,2) adjacent to two lumber huts -> level 2.
        .terrain(p(1, 1), Terrain::Forest)
        .improvement(p(1, 1), Improvement::LumberHut)
        .terrain(p(1, 3), Terrain::Forest)
        .improvement(p(1, 3), Improvement::LumberHut)
        .building(p(1, 2), Building::Sawmill)
        // Explored Windmill at (2,3) adjacent to one farm -> level 1.
        .improvement(p(3, 3), Improvement::Farm)
        .building(p(2, 3), Building::Windmill)
        // Explored Forge at (3,1) adjacent to one mine at (3,2) -> level 1.
        .terrain(p(3, 2), Terrain::Mountain)
        .improvement(p(3, 2), Improvement::Mine)
        .building(p(3, 1), Building::Forge)
        // Explored Market at (2,1) adjacent to Sawmill (L2) and Forge (L1) -> level 3.
        .building(p(2, 1), Building::Market)
        // Explored corner lighthouse at (0,0) and monument at (4,1).
        .lighthouse(p(0, 0))
        .monument(p(4, 1), Monument::ParkOfFortune)
        // Unexplored v2 features in rows 4..7 x cols 4..7 and corner (0,6).
        .lighthouse(p(0, 6))
        .terrain(p(4, 5), Terrain::Forest)
        .improvement(p(4, 5), Improvement::LumberHut)
        .building(p(4, 6), Building::Sawmill)
        .monument(p(5, 6), Monument::EyeOfGod)
        .explore_all()
        .hide(0, &[p(0, 6), p(4, 6), p(5, 6), p(6, 5)])
        .build();
    // Sanity-check building levels before testing PlayerView.
    assert_eq!(s.tile(p(1, 2)).building_level, 2);
    assert_eq!(s.tile(p(2, 3)).building_level, 1);
    assert_eq!(s.tile(p(3, 1)).building_level, 1);
    assert_eq!(s.tile(p(2, 1)).building_level, 3);
    assert_eq!(s.tile(p(4, 6)).building_level, 1);
    // Roads on explored plain tile (0,2), explored building tile (1,2), and unexplored tile (6,5).
    set_road(&mut s, p(0, 2));
    set_road(&mut s, p(1, 2));
    set_road(&mut s, p(6, 5));

    let v = PlayerView::new(&s, 0);

    // Explored road-only tile.
    let t_road = v.tile(p(0, 2)).expect("explored");
    assert!(t_road.road);
    assert_eq!(t_road.building, None);
    assert_eq!(t_road.building_level, 0);
    assert!(!t_road.lighthouse);
    assert_eq!(t_road.monument, None);

    // Explored road + Sawmill (level 2).
    let t_saw = v.tile(p(1, 2)).expect("explored");
    assert!(t_saw.road);
    assert_eq!(t_saw.building, Some(Building::Sawmill));
    assert_eq!(t_saw.building_level, 2);
    assert!(!t_saw.lighthouse);
    assert_eq!(t_saw.monument, None);

    // Explored Windmill (level 1), Forge (level 1), Market (level 2).
    let t_wind = v.tile(p(2, 3)).expect("explored");
    assert_eq!(t_wind.building, Some(Building::Windmill));
    assert_eq!(t_wind.building_level, 1);

    let t_forge = v.tile(p(3, 1)).expect("explored");
    assert_eq!(t_forge.building, Some(Building::Forge));
    assert_eq!(t_forge.building_level, 1);

    let t_market = v.tile(p(2, 1)).expect("explored");
    assert_eq!(t_market.building, Some(Building::Market));
    assert_eq!(t_market.building_level, 3);

    // Explored corner lighthouse.
    let t_lh = v.tile(p(0, 0)).expect("explored");
    assert!(t_lh.lighthouse);
    assert!(!t_lh.road);
    assert_eq!(t_lh.building, None);
    assert_eq!(t_lh.monument, None);

    // Explored monument.
    let t_mon = v.tile(p(4, 1)).expect("explored");
    assert_eq!(t_mon.monument, Some(Monument::ParkOfFortune));
    assert_eq!(t_mon.building, None);
    assert_eq!(t_mon.building_level, 0);

    // Unexplored tiles return None even when holding lighthouse/building/monument/road.
    for q in [p(0, 6), p(4, 6), p(5, 6), p(6, 5)] {
        assert_eq!(v.tile(q), None, "unexplored {q:?} must be None");
    }

    // Player 1 explored everything and sees the tiles hidden from player 0.
    let v1 = PlayerView::new(&s, 1);
    assert!(v1.tile(p(0, 6)).unwrap().lighthouse);
    assert_eq!(v1.tile(p(4, 6)).unwrap().building, Some(Building::Sawmill));
    assert_eq!(v1.tile(p(4, 6)).unwrap().building_level, 1);
    assert_eq!(v1.tile(p(5, 6)).unwrap().monument, Some(Monument::EyeOfGod));
    assert!(v1.tile(p(6, 5)).unwrap().road);
}

#[test]
fn viewer_v2_counters_and_monuments_are_own_only() {
    let s = StateBuilder::new(6)
        .config(v2_cfg())
        .player_with(0, |pl| {
            pl.kills = 4;
            pl.lighthouses_discovered = 0b0101;
            pl.monuments_unlocked = MonumentSet::from_slice(&[Monument::ParkOfFortune]);
            pl.monuments_placed = MonumentSet::from_slice(&[Monument::ParkOfFortune]);
        })
        .player_with(1, |pl| {
            pl.kills = 12;
            pl.lighthouses_discovered = 0b1111;
            pl.monuments_unlocked = MonumentSet::from_slice(&[
                Monument::ParkOfFortune,
                Monument::GateOfPower,
                Monument::EyeOfGod,
            ]);
            pl.monuments_placed = MonumentSet::from_slice(&[Monument::GateOfPower]);
        })
        .build();

    let v0 = PlayerView::new(&s, 0);
    assert_eq!(v0.kills(), 4);
    assert_eq!(v0.lighthouses_discovered(), 0b0101);
    assert_eq!(
        v0.monuments_unlocked(),
        MonumentSet::from_slice(&[Monument::ParkOfFortune])
    );
    assert_eq!(
        v0.monuments_placed(),
        MonumentSet::from_slice(&[Monument::ParkOfFortune])
    );

    let v1 = PlayerView::new(&s, 1);
    assert_eq!(v1.kills(), 12);
    assert_eq!(v1.lighthouses_discovered(), 0b1111);
    assert_eq!(
        v1.monuments_unlocked(),
        MonumentSet::from_slice(&[
            Monument::ParkOfFortune,
            Monument::GateOfPower,
            Monument::EyeOfGod
        ])
    );
    assert_eq!(
        v1.monuments_placed(),
        MonumentSet::from_slice(&[Monument::GateOfPower])
    );

    // Mutating player 1's v2 state never affects player 0's view.
    let mut twin = s.clone();
    twin.players[1].kills = 99;
    twin.players[1].lighthouses_discovered = 0;
    twin.players[1].monuments_unlocked = MonumentSet::new();
    twin.players[1].monuments_placed = MonumentSet::new();
    let v0_twin = PlayerView::new(&twin, 0);
    assert_eq!(v0_twin.kills(), v0.kills());
    assert_eq!(
        v0_twin.lighthouses_discovered(),
        v0.lighthouses_discovered()
    );
    assert_eq!(v0_twin.monuments_unlocked(), v0.monuments_unlocked());
    assert_eq!(v0_twin.monuments_placed(), v0.monuments_placed());
}

#[test]
fn attack_preview_matches_apply_across_plain_hidden_tech_and_city_bonuses() {
    // 1. Plain field Warrior vs Warrior (exact .5 tie -> dealt=5, retal=5).
    let plain = StateBuilder::new(6)
        .config(v2_cfg())
        .explore_all()
        .unit(0, UnitType::Warrior, p(2, 2))
        .unit(1, UnitType::Warrior, p(2, 3))
        .build();
    let pr = assert_preview_matches_apply(&plain, UnitId(0), p(2, 3), 1.0, "plain");
    assert_eq!(
        pr,
        AttackPreview {
            dealt: 5,
            retaliation: 5,
            defender_dies: false,
            attacker_dies: false,
            defense_bonus: 1.0,
        }
    );

    // 2. Hidden enemy Archery on forest (1.0 without, 1.5 with); Swordsman vs Warrior so rounded
    // dealt strictly drops (8 -> 7) and retaliation strictly rises (4 -> 5) (Q56 A).
    let forest_no_arch = StateBuilder::new(6)
        .config(v2_cfg())
        .explore_all()
        .terrain(p(2, 3), Terrain::Forest)
        .unit(0, UnitType::Swordsman, p(2, 2))
        .unit(1, UnitType::Warrior, p(2, 3))
        .build();
    let p_no_arch =
        assert_preview_matches_apply(&forest_no_arch, UnitId(0), p(2, 3), 1.0, "forest no arch");
    let forest_arch = StateBuilder::new(6)
        .config(v2_cfg())
        .explore_all()
        .terrain(p(2, 3), Terrain::Forest)
        .techs(1, &[Tech::Archery])
        .unit(0, UnitType::Swordsman, p(2, 2))
        .unit(1, UnitType::Warrior, p(2, 3))
        .build();
    let p_arch = assert_preview_matches_apply(&forest_arch, UnitId(0), p(2, 3), 1.5, "forest arch");
    assert!(p_arch.dealt < p_no_arch.dealt);
    assert!(p_arch.retaliation > p_no_arch.retaliation);

    // 3. Hidden enemy Climbing on mountain (1.0 without, 1.5 with).
    let mtn_no_climb = StateBuilder::new(6)
        .config(v2_cfg())
        .explore_all()
        .terrain(p(2, 3), Terrain::Mountain)
        .unit(0, UnitType::Warrior, p(2, 2))
        .unit(1, UnitType::Warrior, p(2, 3))
        .build();
    assert_preview_matches_apply(&mtn_no_climb, UnitId(0), p(2, 3), 1.0, "mtn no climb");
    let mtn_climb = StateBuilder::new(6)
        .config(v2_cfg())
        .explore_all()
        .terrain(p(2, 3), Terrain::Mountain)
        .techs(1, &[Tech::Climbing])
        .unit(0, UnitType::Warrior, p(2, 2))
        .unit(1, UnitType::Warrior, p(2, 3))
        .build();
    assert_preview_matches_apply(&mtn_climb, UnitId(0), p(2, 3), 1.5, "mtn climb");

    // 4. Own city (1.5x with Fortify) vs walled city (4.0x with Fortify) vs SuperUnit without Fortify (1.0x).
    let city_fortify = StateBuilder::new(6)
        .config(v2_cfg())
        .explore_all()
        .city(1, p(2, 3))
        .unit(0, UnitType::Warrior, p(2, 2))
        .unit(1, UnitType::Warrior, p(2, 3))
        .build();
    assert_preview_matches_apply(&city_fortify, UnitId(0), p(2, 3), 1.5, "city fortify");

    let wall_fortify = StateBuilder::new(6)
        .config(v2_cfg())
        .explore_all()
        .city_with(1, p(2, 3), |c| c.wall = true)
        .unit(0, UnitType::Defender, p(2, 2))
        .unit(1, UnitType::Archer, p(2, 3))
        .build();
    let pr_wall =
        assert_preview_matches_apply(&wall_fortify, UnitId(0), p(2, 3), 4.0, "wall fortify");
    // Spec golden case 5: Defender 15 vs Archer 10 with 4.0 bonus -> Atk 11 (retal 4), Def 9 (dealt 1).
    assert_eq!(pr_wall.dealt, 1);
    assert_eq!(pr_wall.retaliation, 4);

    let wall_no_fortify = StateBuilder::new(6)
        .config(v2_cfg())
        .explore_all()
        .city_with(1, p(2, 3), |c| c.wall = true)
        .unit(0, UnitType::Warrior, p(2, 2))
        .unit(1, UnitType::SuperUnit, p(2, 3))
        .build();
    assert_preview_matches_apply(
        &wall_no_fortify,
        UnitId(0),
        p(2, 3),
        1.0,
        "wall superunit no fortify",
    );
}

#[test]
fn attack_preview_v2_units_stiff_range_lethal_and_suicidal() {
    // 5. Knight 3.5 attack vs Defender.
    let knight_vs_def = StateBuilder::new(6)
        .config(v2_cfg())
        .explore_all()
        .unit(0, UnitType::Knight, p(2, 2))
        .unit(1, UnitType::Defender, p(2, 3))
        .build();
    assert_preview_matches_apply(&knight_vs_def, UnitId(0), p(2, 3), 1.0, "knight vs def");

    // 6a. Catapult (defense == 0, Stiff) survives Defender's attack -> retaliation == 0.
    let def_vs_cat = StateBuilder::new(6)
        .config(v2_cfg())
        .explore_all()
        .unit(0, UnitType::Defender, p(2, 2))
        .unit(1, UnitType::Catapult, p(2, 3))
        .build();
    let pr_cat = assert_preview_matches_apply(&def_vs_cat, UnitId(0), p(2, 3), 1.0, "def vs cat");
    assert!(!pr_cat.defender_dies, "catapult must survive to test retal");
    assert_eq!(pr_cat.retaliation, 0);

    // 6b. Custom-table Stiff with defense > 0 (kills mutants that skip Skill::Stiff check!).
    let units_v2 = include_str!("../../../data/v2/units.json").replace(
        r#""defense": 0, "movement": 1, "range": 3, "skills": ["stiff"]"#,
        r#""defense": 3, "movement": 1, "range": 3, "skills": ["stiff"]"#,
    );
    let custom_tables = Tables::from_json_v2(
        &units_v2,
        include_str!("../../../data/v2/techs.json"),
        include_str!("../../../data/v2/site_actions.json"),
        include_str!("../../../data/v2/buildings.json"),
        include_str!("../../../data/v2/monuments.json"),
        include_str!("../../../data/v2/rules.json"),
    )
    .expect("custom v2 tables");
    assert_eq!(custom_tables.unit(UnitType::Catapult).defense, 3);
    let mut stiff_def = StateBuilder::new(6)
        .config(v2_cfg())
        .explore_all()
        .unit(0, UnitType::Warrior, p(2, 2))
        .unit(1, UnitType::Catapult, p(2, 3))
        .build();
    stiff_def.tables = Arc::new(custom_tables);
    // Sanity check: raw retaliation damage would be > 0 if Stiff weren't enforced.
    let raw = compute_damage(
        &Combatant {
            attack: 2.0,
            defense: 2,
            health: 10,
            max_health: 10,
        },
        &Combatant {
            attack: 4.0,
            defense: 3,
            health: 10,
            max_health: 10,
        },
        1.0,
    );
    assert!(raw.to_attacker > 0, "test premise: raw retal > 0");
    let pr_stiff =
        assert_preview_matches_apply(&stiff_def, UnitId(0), p(2, 3), 1.0, "custom stiff");
    assert!(!pr_stiff.defender_dies);
    assert_eq!(pr_stiff.retaliation, 0, "Stiff unit must never retaliate");

    // 7. Out-of-range defender: Archer at dist 2 vs Warrior (range 1), Catapult at dist 3 vs Archer (range 2).
    let archer_vs_warr = StateBuilder::new(6)
        .config(v2_cfg())
        .explore_all()
        .unit(0, UnitType::Archer, p(1, 1))
        .unit(1, UnitType::Warrior, p(1, 3))
        .build();
    let pr_rng =
        assert_preview_matches_apply(&archer_vs_warr, UnitId(0), p(1, 3), 1.0, "archer dist 2");
    assert!(!pr_rng.defender_dies);
    assert_eq!(pr_rng.retaliation, 0);

    let cat_vs_archer = StateBuilder::new(6)
        .config(v2_cfg())
        .explore_all()
        .unit(0, UnitType::Catapult, p(1, 1))
        .unit(1, UnitType::Archer, p(1, 4))
        .build();
    let pr_cat3 =
        assert_preview_matches_apply(&cat_vs_archer, UnitId(0), p(1, 4), 1.0, "cat dist 3");
    assert_eq!(pr_cat3.retaliation, 0);

    // 8. Lethal attack: dealt is capped at defender's current HP, defender_dies == true, retal == 0.
    let lethal = StateBuilder::new(6)
        .config(v2_cfg())
        .explore_all()
        .unit(0, UnitType::Swordsman, p(2, 2))
        .unit_with(1, UnitType::Warrior, p(2, 3), |u| u.health = 3)
        .build();
    let pr_lethal = assert_preview_matches_apply(&lethal, UnitId(0), p(2, 3), 1.0, "lethal");
    assert_eq!(pr_lethal.dealt, 3);
    assert_eq!(pr_lethal.retaliation, 0);
    assert!(pr_lethal.defender_dies);
    assert!(!pr_lethal.attacker_dies);

    // 9. Suicidal attack (spec golden case 2): Warrior 6/10 vs Warrior 10/10 -> atk dies (retal=6), def at 7 (dealt=3).
    let suicidal = StateBuilder::new(6)
        .config(v2_cfg())
        .explore_all()
        .unit_with(0, UnitType::Warrior, p(2, 2), |u| u.health = 6)
        .unit(1, UnitType::Warrior, p(2, 3))
        .build();
    let pr_suicide = assert_preview_matches_apply(&suicidal, UnitId(0), p(2, 3), 1.0, "suicidal");
    assert_eq!(pr_suicide.dealt, 3);
    assert_eq!(pr_suicide.retaliation, 6);
    assert!(!pr_suicide.defender_dies);
    assert!(pr_suicide.attacker_dies);
}

#[test]
fn attack_preview_returns_none_when_invalid() {
    let base = StateBuilder::new(6)
        .config(v2_cfg())
        .explore_all()
        .hide(0, &[p(2, 3)])
        .unit(0, UnitType::Warrior, p(2, 2)) // UnitId(0)
        .unit(0, UnitType::Warrior, p(2, 1)) // UnitId(1), friendly neighbor
        .unit(1, UnitType::Warrior, p(2, 3)) // UnitId(2), unexplored adjacent
        .unit(1, UnitType::Warrior, p(2, 4)) // UnitId(3), explored out of range (dist 2)
        .unit(1, UnitType::Warrior, p(3, 2)) // UnitId(4), explored adjacent (valid baseline)
        .build();

    let v0 = PlayerView::new(&base, 0);
    // Sanity check: valid target returns Some.
    assert!(v0.attack_preview(UnitId(0), p(3, 2)).is_some());
    // Unexplored target.
    assert_eq!(v0.attack_preview(UnitId(0), p(2, 3)), None);
    // Out-of-range target.
    assert_eq!(v0.attack_preview(UnitId(0), p(2, 4)), None);
    // Empty target.
    assert_eq!(v0.attack_preview(UnitId(0), p(1, 2)), None);
    // Friendly target.
    assert_eq!(v0.attack_preview(UnitId(0), p(2, 1)), None);
    // Self-tile.
    assert_eq!(v0.attack_preview(UnitId(0), p(2, 2)), None);
    // Out-of-bounds target.
    assert_eq!(v0.attack_preview(UnitId(0), p(6, 0)), None);
    assert_eq!(v0.attack_preview(UnitId(0), p(0, 6)), None);
    // Enemy unit id or nonexistent unit id.
    assert_eq!(v0.attack_preview(UnitId(4), p(2, 2)), None);
    assert_eq!(v0.attack_preview(UnitId(999), p(3, 2)), None);

    // Wrong turn (player 1's view on player 0's turn).
    let v1 = PlayerView::new(&base, 1);
    assert_eq!(v1.attack_preview(UnitId(4), p(2, 2)), None);

    // Terminal state.
    let mut over = base.clone();
    over.outcome = Some(Outcome::Win(0));
    assert_eq!(
        PlayerView::new(&over, 0).attack_preview(UnitId(0), p(3, 2)),
        None
    );

    // Note: attack_preview does not forbid a Done unit if all 5 spec preconditions hold.
    let mut done_unit = base.clone();
    done_unit.units[0].phase = Phase::Done;
    done_unit.cities.push(sim_core::state::City {
        id: CityId(0),
        owner: 0,
        pos: p(0, 0),
        level: 1,
        population: 0,
        is_capital: true,
        workshop: false,
        parks: 0,
        wall: false,
        border_growth_used: false,
        pending_reward: false,
        connected_to: None,
    });
    assert!(PlayerView::new(&done_unit, 0)
        .attack_preview(UnitId(0), p(3, 2))
        .is_some());
}
