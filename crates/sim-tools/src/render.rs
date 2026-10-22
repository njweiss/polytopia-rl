//! Text renderer (docs/MILESTONE_7 "Text renderer").

use serde::Serialize;
use sim_core::actions::Action;
use sim_core::data::{Tech, UnitType, ALL_TECHS};
use sim_core::economy::income;
use sim_core::grid::Pos;
use sim_core::map::{Resource, Terrain};
use sim_core::score::score;
use sim_core::state::{Improvement, Outcome, Phase, PlayerId, State, TileState};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum View {
    Omniscient,
    Player(PlayerId),
}

/// The snake_case name a data enum serializes to (`UnitType`, `Tech`, `SiteAction`, `Reward`).
pub(crate) fn snake<T: Serialize>(v: &T) -> String {
    match serde_json::to_value(v) {
        Ok(serde_json::Value::String(s)) => s,
        _ => unreachable!("data enums serialize as strings"),
    }
}

pub(crate) fn terrain_name(t: Terrain) -> &'static str {
    match t {
        Terrain::Field => "field",
        Terrain::Forest => "forest",
        Terrain::Mountain => "mountain",
    }
}

pub(crate) fn resource_name(r: Resource) -> &'static str {
    match r {
        Resource::Fruit => "fruit",
        Resource::Crop => "crop",
        Resource::Animal => "animal",
        Resource::Metal => "metal",
    }
}

pub(crate) fn improvement_name(i: Improvement) -> &'static str {
    match i {
        Improvement::LumberHut => "lumber_hut",
        Improvement::Farm => "farm",
        Improvement::Mine => "mine",
    }
}

pub(crate) fn phase_name(p: Phase) -> &'static str {
    match p {
        Phase::Fresh => "Fresh",
        Phase::Moved => "Moved",
        Phase::Attacked => "Attacked",
        Phase::Done => "Done",
    }
}

fn explored(state: &State, view: View, pos: Pos) -> bool {
    match view {
        View::Omniscient => true,
        View::Player(p) => state.is_explored(p, pos),
    }
}

/// The tile's resource as the view sees it: a crop needs Organization, a metal Climbing.
fn visible_resource(state: &State, view: View, t: &TileState) -> Option<Resource> {
    let View::Player(p) = view else {
        return t.resource;
    };
    let techs = &state.players[p as usize].techs;
    t.resource.filter(|r| match r {
        Resource::Crop => techs.contains(Tech::Organization),
        Resource::Metal => techs.contains(Tech::Climbing),
        _ => true,
    })
}

fn cell(state: &State, view: View, pos: Pos) -> String {
    if !explored(state, view, pos) {
        return "????".to_string();
    }
    let t = state.tile(pos);
    let base = match (t.improvement, visible_resource(state, view, t)) {
        (Some(Improvement::LumberHut), _) => 'H',
        (Some(Improvement::Farm), _) => 'F',
        (Some(Improvement::Mine), _) => 'M',
        (None, Some(Resource::Fruit)) => 'o',
        (None, Some(Resource::Crop)) => '+',
        (None, Some(Resource::Animal)) => 'a',
        (None, Some(Resource::Metal)) => '$',
        (None, None) => match t.terrain {
            Terrain::Field => '.',
            Terrain::Forest => 'f',
            Terrain::Mountain => '^',
        },
    };
    let territory = match t.owner_city {
        Some(c) => char::from(b'0' + state.city(c).owner),
        None => '.',
    };
    let site = match t.city {
        Some(c) if state.city(c).is_capital => 'K',
        Some(_) => 'C',
        None if t.village => 'V',
        None => '.',
    };
    let unit = match t.unit.and_then(|id| state.unit(id)) {
        None => '.',
        Some(u) => {
            let c = match u.kind {
                UnitType::Warrior => 'W',
                UnitType::Rider => 'R',
                UnitType::Archer => 'A',
                UnitType::Defender => 'D',
                UnitType::SuperUnit => 'S',
            };
            if u.owner == 0 {
                c
            } else {
                c.to_ascii_lowercase()
            }
        }
    };
    [base, territory, site, unit].iter().collect()
}

fn panel_line(state: &State, view: View, p: PlayerId) -> String {
    let score = score(state, p);
    if view != View::Omniscient && view != View::Player(p) {
        return format!("P{p}: score {score}");
    }
    let pl = &state.players[p as usize];
    let cities = state.cities.iter().filter(|c| c.owner == p).count();
    let units = state.units.iter().filter(|u| u.owner == p).count();
    let techs: Vec<String> = ALL_TECHS
        .iter()
        .filter(|&&t| pl.techs.contains(t))
        .map(snake)
        .collect();
    let techs = if techs.is_empty() {
        "-".to_string()
    } else {
        techs.join(", ")
    };
    format!(
        "P{p}: stars {} | income {} | score {score} | cities {cities} | units {units} | techs {techs}",
        pl.stars,
        income(state, p)
    )
}

pub fn render(state: &State, view: View) -> String {
    let mut lines = Vec::new();
    let header: Vec<String> = (0..state.size).map(|c| format!("{c:<4}")).collect();
    lines.push(format!("   {}", header.join(" ")));
    for r in 0..state.size {
        let cells: Vec<String> = (0..state.size)
            .map(|c| cell(state, view, Pos::new(r, c)))
            .collect();
        lines.push(format!("{r:>2} {}", cells.join(" ")));
    }
    lines.push(String::new());
    let mut status = format!("turn {} | P{} to move", state.turn, state.current);
    match state.outcome {
        Some(Outcome::Win(w)) => status.push_str(&format!(" | game over: P{w} wins")),
        Some(Outcome::Draw) => status.push_str(" | game over: draw"),
        None => {}
    }
    lines.push(status);
    for p in 0..2 {
        lines.push(panel_line(state, view, p));
    }
    let mut out = String::new();
    for l in lines {
        out.push_str(l.trim_end());
        out.push('\n');
    }
    out
}

pub fn describe_tile(state: &State, view: View, pos: Pos) -> String {
    let (row, col) = (pos.row, pos.col);
    if !explored(state, view, pos) {
        return format!("({row},{col}) unexplored");
    }
    let t = state.tile(pos);
    let res = visible_resource(state, view, t).map_or("-", resource_name);
    let imp = t.improvement.map_or("-", improvement_name);
    let ter = match t.owner_city {
        Some(c) => format!("P{}", state.city(c).owner),
        None => "-".to_string(),
    };
    let site = match t.city {
        Some(c) => {
            let city = state.city(c);
            let kind = if city.is_capital { "capital" } else { "city" };
            format!("{kind}#{}(P{})", c.0, city.owner)
        }
        None if t.village => "village".to_string(),
        None => "-".to_string(),
    };
    let unit = match t.unit.and_then(|id| state.unit(id)) {
        None => "-".to_string(),
        Some(u) => {
            let mut parts = vec![
                format!("#{}", u.id.0),
                format!("P{}", u.owner),
                snake(&u.kind),
                "hp".to_string(),
                format!("{}/{}", u.health, u.max_health),
                phase_name(u.phase).to_string(),
            ];
            if u.veteran {
                parts.push("veteran".to_string());
            }
            if u.kills > 0 {
                parts.push(format!("kills {}", u.kills));
            }
            parts.join(" ")
        }
    };
    format!(
        "({row},{col}) {} res={res} imp={imp} territory={ter} site={site} unit={unit}",
        terrain_name(t.terrain)
    )
}

pub fn action_label(action: &Action) -> String {
    match *action {
        Action::Move { unit, to } => format!("Move u{} -> ({},{})", unit.0, to.row, to.col),
        Action::Attack { unit, target } => {
            format!("Attack u{} -> ({},{})", unit.0, target.row, target.col)
        }
        Action::Capture { unit } => format!("Capture u{}", unit.0),
        Action::Recover { unit } => format!("Recover u{}", unit.0),
        Action::Promote { unit } => format!("Promote u{}", unit.0),
        Action::Train { city, kind } => format!("Train c{} {}", city.0, snake(&kind)),
        Action::Site { at, action } => format!("Site ({},{}) {}", at.row, at.col, snake(&action)),
        Action::Research { tech } => format!("Research {}", snake(&tech)),
        Action::ChooseReward { city, reward } => format!("Reward c{} {}", city.0, snake(&reward)),
        Action::EndTurn => "End turn".to_string(),
    }
}
