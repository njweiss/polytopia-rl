//! The Economy bot (docs/MILESTONE_8 "Economy bot").

use sim_core::actions::Action;
use sim_core::data::{Building, Reward, SiteAction, Tech, UnitType};
use sim_core::grid::{all_tiles, Pos};
use sim_core::rng::Rng;
use sim_core::state::Ruleset;
use sim_core::view::PlayerView;

use crate::{
    best_attack, build_monument, building_of_kind, capture, choose_reward, first_legal,
    move_toward, nearest, promote, recover, site_of_kind, Bot,
};

pub struct EconomyBot;

/// Reward preferences for levels 2, 3, 4 and 5+.
const REWARDS: [Reward; 4] = [
    Reward::Workshop,
    Reward::Resources,
    Reward::PopulationGrowth,
    Reward::Park,
];

const SITES: [SiteAction; 5] = [
    SiteAction::Farm,
    SiteAction::Mine,
    SiteAction::HarvestFruit,
    SiteAction::HarvestAnimal,
    SiteAction::LumberHut,
];

const BUILDINGS: [Building; 4] = [
    Building::Market,
    Building::Forge,
    Building::Sawmill,
    Building::Windmill,
];

const RESEARCH: [Tech; 8] = [
    Tech::Hunting,
    Tech::Farming,
    Tech::Climbing,
    Tech::Mining,
    Tech::Forestry,
    Tech::Riding,
    Tech::Strategy,
    Tech::Archery,
];

const RESEARCH_V2: [Tech; 15] = [
    Tech::Hunting,
    Tech::Farming,
    Tech::Climbing,
    Tech::Mining,
    Tech::Forestry,
    Tech::Riding,
    Tech::Strategy,
    Tech::Archery,
    Tech::Roads,
    Tech::Mathematics,
    Tech::Construction,
    Tech::Smithery,
    Tech::Trade,
    Tech::FreeSpirit,
    Tech::Chivalry,
];

impl Bot for EconomyBot {
    fn name(&self) -> &'static str {
        "economy"
    }

    fn choose(&self, view: &PlayerView, _rng: &mut Rng) -> Action {
        let legal = view.legal_actions();
        let v2 = view.ruleset() == Ruleset::V2;
        choose_reward(view, &legal, &REWARDS)
            .or_else(|| promote(&legal))
            .or_else(|| capture(&legal))
            .or_else(|| best_attack(view, &legal, 3))
            .or_else(|| v2.then(|| build_monument(&legal)).flatten())
            .or_else(|| SITES.into_iter().find_map(|k| site_of_kind(&legal, k)))
            .or_else(|| {
                v2.then(|| {
                    BUILDINGS
                        .into_iter()
                        .find_map(|b| building_of_kind(&legal, b))
                })
                .flatten()
            })
            .or_else(|| research(&legal, v2))
            .or_else(|| train(view, &legal, v2))
            .or_else(|| recover(view, &legal))
            .or_else(|| move_toward(view, &legal, |u| village(view, u.pos)))
            .unwrap_or(Action::EndTurn)
    }
}

fn research(legal: &[Action], v2: bool) -> Option<Action> {
    let order: &[Tech] = if v2 { &RESEARCH_V2 } else { &RESEARCH };
    first_legal(legal, order.iter().map(|&tech| Action::Research { tech }))
}

/// For cities in ascending id, the first legal unit in priority order (V1: Warrior;
/// V2: Swordsman, Catapult, Warrior), while units < 2 x cities.
fn train(view: &PlayerView, legal: &[Action], v2: bool) -> Option<Action> {
    let cities = view.my_cities();
    if view.my_units().len() >= 2 * cities.len() {
        return None;
    }
    let order: &[UnitType] = if v2 {
        &[UnitType::Swordsman, UnitType::Catapult, UnitType::Warrior]
    } else {
        &[UnitType::Warrior]
    };
    cities.into_iter().find_map(|c| {
        first_legal(
            legal,
            order.iter().map(|&kind| Action::Train { city: c.id, kind }),
        )
    })
}

/// The nearest known village tile with no unit on it, ties row-major.
fn village(view: &PlayerView, from: Pos) -> Option<Pos> {
    let villages = all_tiles(view.size())
        .filter(|&q| view.tile(q).is_some_and(|t| t.village && t.unit.is_none()));
    nearest(from, villages)
}
