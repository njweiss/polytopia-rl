//! Observation encoding (docs/MILESTONE_9 "Observation"). Reads only a `PlayerView`.

use std::fmt::Debug;

use sim_core::actions::Action;
use sim_core::data::{SiteAction, ALL_TECHS, ALL_UNITS};
use sim_core::grid::{all_tiles, Pos};
use sim_core::map::{ALL_RESOURCES, ALL_TERRAINS};
use sim_core::state::{ALL_IMPROVEMENTS, ALL_PHASES};
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
    assert_eq!(out.len(), CHANNELS * n * n, "encode_board: buffer length");
    out.fill(0.0);
    let me = view.player();
    let mut set = |c: usize, p: Pos, x: f32| out[(c * n + p.row as usize) * n + p.col as usize] = x;
    for p in all_tiles(view.size()) {
        let Some(t) = view.tile(p) else {
            continue;
        };
        set(EXPLORED, p, 1.0);
        set(TERRAIN + t.terrain as usize, p, 1.0);
        if let Some(r) = t.resource {
            set(RESOURCE + r as usize, p, 1.0);
        }
        if let Some(i) = t.improvement {
            set(IMPROVEMENT + i as usize, p, 1.0);
        }
        set(VILLAGE, p, flag(t.village));
        if let Some(owner) = t.territory {
            set(
                if owner == me {
                    TERRITORY_MINE
                } else {
                    TERRITORY_ENEMY
                },
                p,
                1.0,
            );
        }
        if let Some(c) = t.city {
            set(if c.owner == me { CITY_MINE } else { CITY_ENEMY }, p, 1.0);
            set(CITY_LEVEL, p, c.level.min(10) as f32 / 10.0);
            set(CITY_CAPITAL, p, flag(c.is_capital));
        }
        if let Some(u) = t.unit {
            if u.owner == me {
                set(MY_UNIT + u.kind as usize, p, 1.0);
                set(PHASE + u.phase as usize, p, 1.0);
            } else {
                set(ENEMY_UNIT + u.kind as usize, p, 1.0);
            }
            set(HEALTH, p, u.health as f32 / u.max_health as f32);
            set(VETERAN, p, flag(u.veteran));
        }
    }
    let units = view.my_units();
    for &a in legal {
        match a {
            Action::Promote { unit } => {
                let u = units.iter().find(|u| u.id == unit).expect("own unit");
                set(CAN_PROMOTE, u.pos, 1.0);
            }
            Action::Site { at, action } => {
                let c = match action {
                    SiteAction::HarvestAnimal | SiteAction::HarvestFruit => CAN_HARVEST,
                    _ => CAN_BUILD,
                };
                set(c, at, 1.0);
            }
            _ => {}
        }
    }
}

/// Length `globals`; overwrites all of `out`.
pub fn encode_globals(view: &PlayerView, out: &mut [f32]) {
    assert_eq!(out.len(), GLOBALS, "encode_globals: buffer length");
    let me = view.player();
    let (my, opp) = (view.score(me), view.score(1 - me));
    let techs = view.techs();
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
        out[head.len() + i] = flag(techs.contains(t));
    }
}
