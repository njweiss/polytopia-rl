//! The Economy bot (docs/MILESTONE_8 "Economy bot").

use sim_core::actions::Action;
use sim_core::data::{Reward, SiteAction, Tech, UnitType};
use sim_core::grid::{all_tiles, Pos};
use sim_core::rng::Rng;
use sim_core::view::PlayerView;

use crate::{
    best_attack, capture, choose_reward, first_legal, move_toward, nearest, promote, recover,
    site_of_kind, Bot,
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

impl Bot for EconomyBot {
    fn name(&self) -> &'static str {
        "economy"
    }

    fn choose(&self, view: &PlayerView, _rng: &mut Rng) -> Action {
        let legal = view.legal_actions();
        choose_reward(view, &legal, &REWARDS)
            .or_else(|| promote(&legal))
            .or_else(|| capture(&legal))
            .or_else(|| best_attack(view, &legal, 3))
            .or_else(|| SITES.into_iter().find_map(|k| site_of_kind(&legal, k)))
            .or_else(|| first_legal(&legal, RESEARCH.map(|tech| Action::Research { tech })))
            .or_else(|| train(view, &legal))
            .or_else(|| recover(view, &legal))
            .or_else(|| move_toward(view, &legal, |u| village(view, u.pos)))
            .unwrap_or(Action::EndTurn)
    }
}

/// A Warrior in the first city (ascending id) that can, while units < 2 x cities.
fn train(view: &PlayerView, legal: &[Action]) -> Option<Action> {
    let cities = view.my_cities();
    if view.my_units().len() >= 2 * cities.len() {
        return None;
    }
    first_legal(
        legal,
        cities.into_iter().map(|c| Action::Train {
            city: c.id,
            kind: UnitType::Warrior,
        }),
    )
}

/// The nearest known village tile with no unit on it, ties row-major.
fn village(view: &PlayerView, from: Pos) -> Option<Pos> {
    let villages = all_tiles(view.size())
        .filter(|&q| view.tile(q).is_some_and(|t| t.village && t.unit.is_none()));
    nearest(from, villages)
}
