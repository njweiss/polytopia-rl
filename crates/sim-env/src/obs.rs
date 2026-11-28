//! Observation encoding (docs/MILESTONE_9 "Observation"). Reads only a `PlayerView`.

use std::fmt::Debug;

use sim_core::actions::Action;
use sim_core::data::{
    techs, units, SiteAction, ALL_BUILDINGS, ALL_MONUMENTS, ALL_TECHS, ALL_UNITS,
};
use sim_core::grid::{all_tiles, Pos};
use sim_core::map::{ALL_RESOURCES, ALL_TERRAINS};
use sim_core::state::{Ruleset, ALL_IMPROVEMENTS, ALL_PHASES};
use sim_core::view::PlayerView;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObsSpec {
    pub height: usize,
    pub width: usize,
    pub channels: usize,
    pub globals: usize,
    pub channel_names: Vec<String>,
    pub global_names: Vec<String>,
}

/// The `snake_case` data name of a fieldless enum variant (`SuperUnit` -> `super_unit`).
pub fn snake_name<T: Debug>(x: T) -> String {
    let mut out = String::new();
    for (i, ch) in format!("{x:?}").chars().enumerate() {
        if ch.is_ascii_uppercase() {
            if i > 0 {
                out.push('_');
            }
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push(ch);
        }
    }
    out
}

fn group<T: Debug + Copy>(name: &str, variants: &[T]) -> Vec<String> {
    variants
        .iter()
        .map(|&v| format!("{name}.{}", snake_name(v)))
        .collect()
}

fn names(xs: &[&str]) -> Vec<String> {
    xs.iter().map(|s| s.to_string()).collect()
}

// Board channel offsets, in the brief's table order.
const EXPLORED: usize = 0;
const TERRAIN: usize = EXPLORED + 1;
const RESOURCE: usize = TERRAIN + ALL_TERRAINS.len();
const IMPROVEMENT: usize = RESOURCE + ALL_RESOURCES.len();
const VILLAGE: usize = IMPROVEMENT + ALL_IMPROVEMENTS.len();
const TERRITORY_MINE: usize = VILLAGE + 1;
const TERRITORY_ENEMY: usize = TERRITORY_MINE + 1;
const CITY_MINE: usize = TERRITORY_ENEMY + 1;
const CITY_ENEMY: usize = CITY_MINE + 1;
const CITY_LEVEL: usize = CITY_ENEMY + 1;
const CITY_CAPITAL: usize = CITY_LEVEL + 1;
const MY_UNIT: usize = CITY_CAPITAL + 1;
const ENEMY_UNIT: usize = MY_UNIT + ALL_UNITS.len();
const HEALTH: usize = ENEMY_UNIT + ALL_UNITS.len();
const PHASE: usize = HEALTH + 1;
const VETERAN: usize = PHASE + ALL_PHASES.len();
const CAN_PROMOTE: usize = VETERAN + 1;
const CAN_HARVEST: usize = CAN_PROMOTE + 1;
const CAN_BUILD: usize = CAN_HARVEST + 1;
pub const CHANNELS: usize = CAN_BUILD + 1;
const V2_EXTRA_UNITS: usize = 3;
const MY_UNIT_V2: usize = CHANNELS;
const ENEMY_UNIT_V2: usize = MY_UNIT_V2 + V2_EXTRA_UNITS;
const ROAD: usize = ENEMY_UNIT_V2 + V2_EXTRA_UNITS;
const CITY_CONNECTED: usize = ROAD + 1;
const BUILDING: usize = CITY_CONNECTED + 1;
const BUILDING_LEVEL: usize = BUILDING + ALL_BUILDINGS.len();
const LIGHTHOUSE: usize = BUILDING_LEVEL + 1;
const MONUMENT: usize = LIGHTHOUSE + 1;
const CAN_BUILD_ROAD: usize = MONUMENT + ALL_MONUMENTS.len();
const CAN_BUILD_BUILDING: usize = CAN_BUILD_ROAD + 1;
const CAN_BUILD_MONUMENT: usize = CAN_BUILD_BUILDING + 1;
const CAN_DESTROY: usize = CAN_BUILD_MONUMENT + 1;
const PREVIEW_TARGET_CAN_KILL: usize = CAN_DESTROY + 1;
const PREVIEW_TARGET_MAX_DMG: usize = PREVIEW_TARGET_CAN_KILL + 1;
const PREVIEW_TARGET_MIN_RETAL: usize = PREVIEW_TARGET_MAX_DMG + 1;
const PREVIEW_TARGET_DEF_BONUS: usize = PREVIEW_TARGET_MIN_RETAL + 1;
const PREVIEW_ATTACKER_CAN_KILL: usize = PREVIEW_TARGET_DEF_BONUS + 1;
const PREVIEW_ATTACKER_MAX_DMG: usize = PREVIEW_ATTACKER_CAN_KILL + 1;
pub const V2_CHANNELS: usize = PREVIEW_ATTACKER_MAX_DMG + 1;

const GLOBAL_SCALARS: [&str; 9] = [
    "stars",
    "income",
    "turn",
    "pending_reward",
    "my_score",
    "opp_score",
    "score_diff",
    "my_cities",
    "my_units",
];
pub const GLOBALS: usize = GLOBAL_SCALARS.len() + ALL_TECHS.len();
pub const V2_GLOBALS: usize = 33;

pub fn obs_spec(size: u8) -> ObsSpec {
    let mut channel_names = names(&["explored"]);
    channel_names.extend(group("terrain", &ALL_TERRAINS));
    channel_names.extend(group("resource", &ALL_RESOURCES));
    channel_names.extend(group("improvement", &ALL_IMPROVEMENTS));
    channel_names.extend(names(&[
        "village",
        "territory.mine",
        "territory.enemy",
        "city.mine",
        "city.enemy",
        "city.level",
        "city.capital",
    ]));
    channel_names.extend(group("my_unit", &ALL_UNITS));
    channel_names.extend(group("enemy_unit", &ALL_UNITS));
    channel_names.push("health".into());
    channel_names.extend(group("phase", &ALL_PHASES));
    channel_names.extend(names(&[
        "veteran",
        "can_promote",
        "can_harvest",
        "can_build",
    ]));
    debug_assert_eq!(channel_names.len(), CHANNELS);

    let mut global_names = names(&GLOBAL_SCALARS);
    global_names.extend(group("tech", &ALL_TECHS));
    ObsSpec {
        height: size as usize,
        width: size as usize,
        channels: CHANNELS,
        globals: GLOBALS,
        channel_names,
        global_names,
    }
}

pub fn obs_spec_for(size: u8, ruleset: Ruleset) -> ObsSpec {
    match ruleset {
        Ruleset::V1 => obs_spec(size),
        Ruleset::V2 => {
            let mut spec = obs_spec(size);
            let extra_units = &units(Ruleset::V2)[ALL_UNITS.len()..];
            let extra_techs = &techs(Ruleset::V2)[ALL_TECHS.len()..];
            spec.channel_names.extend(group("my_unit", extra_units));
            spec.channel_names.extend(group("enemy_unit", extra_units));
            spec.channel_names
                .extend(names(&["road", "city.connected"]));
            spec.channel_names.extend(group("building", &ALL_BUILDINGS));
            spec.channel_names
                .extend(names(&["building.level", "lighthouse"]));
            spec.channel_names.extend(group("monument", &ALL_MONUMENTS));
            spec.channel_names.extend(names(&[
                "can_build_road",
                "can_build_building",
                "can_build_monument",
                "can_destroy",
                "preview.target_can_kill",
                "preview.target_max_dmg",
                "preview.target_min_retal",
                "preview.target_def_bonus",
                "preview.attacker_can_kill",
                "preview.attacker_max_dmg",
            ]));
            debug_assert_eq!(spec.channel_names.len(), V2_CHANNELS);

            spec.global_names.extend(group("tech", extra_techs));
            spec.global_names
                .extend(names(&["my_kills", "my_lighthouses"]));
            spec.global_names
                .extend(group("monument_unlocked", &ALL_MONUMENTS));
            spec.global_names
                .extend(group("monument_placed", &ALL_MONUMENTS));
            debug_assert_eq!(spec.global_names.len(), V2_GLOBALS);

            spec.channels = V2_CHANNELS;
            spec.globals = V2_GLOBALS;
            spec
        }
    }
}

fn flag(b: bool) -> f32 {
    if b {
        1.0
    } else {
        0.0
    }
}

/// Channel-major `(C, H, W)`, length `channels * height * width`; overwrites all of `out`.
pub fn encode_board(view: &PlayerView, out: &mut [f32]) {
    encode_board_with_legal(view, &view.legal_actions(), out);
}

/// `encode_board` with `legal == view.legal_actions()` already in hand (`Env` caches it).
pub fn encode_board_with_legal(view: &PlayerView, legal: &[Action], out: &mut [f32]) {
    let n = view.size() as usize;
    let v2 = view.ruleset() == Ruleset::V2;
    let channels = if v2 { V2_CHANNELS } else { CHANNELS };
    assert_eq!(out.len(), channels * n * n, "encode_board: buffer length");
    out.fill(0.0);
    let me = view.player();
    let idx = |c: usize, p: Pos| (c * n + p.row as usize) * n + p.col as usize;
    for p in all_tiles(view.size()) {
        let Some(t) = view.tile(p) else {
            continue;
        };
        out[idx(EXPLORED, p)] = 1.0;
        out[idx(TERRAIN + t.terrain as usize, p)] = 1.0;
        if let Some(r) = t.resource {
            out[idx(RESOURCE + r as usize, p)] = 1.0;
        }
        if let Some(i) = t.improvement {
            out[idx(IMPROVEMENT + i as usize, p)] = 1.0;
        }
        out[idx(VILLAGE, p)] = flag(t.village);
        if let Some(owner) = t.territory {
            let c = if owner == me {
                TERRITORY_MINE
            } else {
                TERRITORY_ENEMY
            };
            out[idx(c, p)] = 1.0;
        }
        if let Some(c) = t.city {
            let ch = if c.owner == me { CITY_MINE } else { CITY_ENEMY };
            out[idx(ch, p)] = 1.0;
            out[idx(CITY_LEVEL, p)] = c.level.min(10) as f32 / 10.0;
            out[idx(CITY_CAPITAL, p)] = flag(c.is_capital);
        }
        if let Some(u) = t.unit {
            let kind_idx = u.kind as usize;
            if u.owner == me {
                if kind_idx < ALL_UNITS.len() {
                    out[idx(MY_UNIT + kind_idx, p)] = 1.0;
                } else if v2 {
                    out[idx(MY_UNIT_V2 + (kind_idx - ALL_UNITS.len()), p)] = 1.0;
                }
                out[idx(PHASE + u.phase as usize, p)] = 1.0;
            } else if kind_idx < ALL_UNITS.len() {
                out[idx(ENEMY_UNIT + kind_idx, p)] = 1.0;
            } else if v2 {
                out[idx(ENEMY_UNIT_V2 + (kind_idx - ALL_UNITS.len()), p)] = 1.0;
            }
            out[idx(HEALTH, p)] = u.health as f32 / u.max_health as f32;
            out[idx(VETERAN, p)] = flag(u.veteran);
        }
        if v2 {
            out[idx(ROAD, p)] = flag(t.road);
            if let Some(b) = t.building {
                out[idx(BUILDING + b as usize, p)] = 1.0;
                out[idx(BUILDING_LEVEL, p)] = t.building_level.min(8) as f32 / 8.0;
            }
            out[idx(LIGHTHOUSE, p)] = flag(t.lighthouse);
            if let Some(m) = t.monument {
                out[idx(MONUMENT + m as usize, p)] = 1.0;
            }
        }
    }
    if v2 {
        for c in view.my_cities() {
            if c.connected_to.is_some() && view.is_explored(c.pos) {
                out[idx(CITY_CONNECTED, c.pos)] = 1.0;
            }
        }
    }
    let units = view.my_units();
    let mut target_previewed = if v2 { vec![false; n * n] } else { Vec::new() };
    for &a in legal {
        match a {
            Action::Promote { unit } => {
                let u = units.iter().find(|u| u.id == unit).expect("own unit");
                out[idx(CAN_PROMOTE, u.pos)] = 1.0;
            }
            Action::Site { at, action } => {
                let c = match action {
                    SiteAction::HarvestAnimal | SiteAction::HarvestFruit => CAN_HARVEST,
                    _ => CAN_BUILD,
                };
                out[idx(c, at)] = 1.0;
            }
            Action::BuildRoad { at } if v2 => {
                out[idx(CAN_BUILD_ROAD, at)] = 1.0;
            }
            Action::BuildBuilding { at, .. } if v2 => {
                out[idx(CAN_BUILD_BUILDING, at)] = 1.0;
            }
            Action::BuildMonument { at, .. } if v2 => {
                out[idx(CAN_BUILD_MONUMENT, at)] = 1.0;
            }
            Action::Destroy { at } if v2 => {
                out[idx(CAN_DESTROY, at)] = 1.0;
            }
            Action::Attack { unit, target } if v2 => {
                let Some(prev) = view.attack_preview(unit, target) else {
                    continue;
                };
                let Some(d) = view.tile(target).and_then(|t| t.unit) else {
                    continue;
                };
                let u = units.iter().find(|u| u.id == unit).expect("own unit");
                let dmg_frac = prev.dealt as f32 / d.health as f32;
                let retal_frac = prev.retaliation as f32 / u.health as f32;
                let t_tile = target.row as usize * n + target.col as usize;
                if prev.defender_dies {
                    out[idx(PREVIEW_TARGET_CAN_KILL, target)] = 1.0;
                    out[idx(PREVIEW_ATTACKER_CAN_KILL, u.pos)] = 1.0;
                }
                let t_dmg = &mut out[idx(PREVIEW_TARGET_MAX_DMG, target)];
                *t_dmg = t_dmg.max(dmg_frac);
                let t_retal = &mut out[idx(PREVIEW_TARGET_MIN_RETAL, target)];
                if !target_previewed[t_tile] {
                    target_previewed[t_tile] = true;
                    *t_retal = retal_frac;
                } else {
                    *t_retal = t_retal.min(retal_frac);
                }
                out[idx(PREVIEW_TARGET_DEF_BONUS, target)] = prev.defense_bonus / 4.0;
                let a_dmg = &mut out[idx(PREVIEW_ATTACKER_MAX_DMG, u.pos)];
                *a_dmg = a_dmg.max(dmg_frac);
            }
            _ => {}
        }
    }
}

/// Length `globals`; overwrites all of `out`.
pub fn encode_globals(view: &PlayerView, out: &mut [f32]) {
    let v2 = view.ruleset() == Ruleset::V2;
    let expected = if v2 { V2_GLOBALS } else { GLOBALS };
    assert_eq!(out.len(), expected, "encode_globals: buffer length");
    let me = view.player();
    let (my, opp) = (view.score(me), view.score(1 - me));
    let tech_set = view.techs();
    let head = [
        view.stars().min(50) as f32 / 50.0,
        view.income().min(20) as f32 / 20.0,
        view.turn() as f32 / view.turn_cap() as f32,
        flag(view.pending_reward_city().is_some()),
        my.min(2000) as f32 / 2000.0,
        opp.min(2000) as f32 / 2000.0,
        ((my as f32 - opp as f32) / 1000.0).clamp(-1.0, 1.0),
        view.my_cities().len().min(8) as f32 / 8.0,
        view.my_units().len().min(16) as f32 / 16.0,
    ];
    out[..head.len()].copy_from_slice(&head);
    for (i, &t) in ALL_TECHS.iter().enumerate() {
        out[head.len() + i] = flag(tech_set.contains(t));
    }
    if v2 {
        let extra_techs = &techs(Ruleset::V2)[ALL_TECHS.len()..];
        let mut k = GLOBALS;
        for &t in extra_techs {
            out[k] = flag(tech_set.contains(t));
            k += 1;
        }
        out[k] = view.kills().min(10) as f32 / 10.0;
        k += 1;
        out[k] = view.lighthouses_discovered().count_ones() as f32 / 4.0;
        k += 1;
        let unlocked = view.monuments_unlocked();
        for &m in &ALL_MONUMENTS {
            out[k] = flag(unlocked.contains(m));
            k += 1;
        }
        let placed = view.monuments_placed();
        for &m in &ALL_MONUMENTS {
            out[k] = flag(placed.contains(m));
            k += 1;
        }
        debug_assert_eq!(k, V2_GLOBALS);
    }
}
