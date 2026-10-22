//! The Rush bot (docs/MILESTONE_8 "Rush bot").

use sim_core::actions::Action;
use sim_core::data::{Reward, SiteAction, Tech, UnitType};
use sim_core::grid::Pos;
use sim_core::rng::Rng;
use sim_core::view::PlayerView;

use crate::{
    best_attack, capture, choose_reward, first_legal, move_toward, nearest, promote, recover,
    site_of_kind, Bot,
};

pub struct RushBot;

/// Reward preferences for levels 2, 3, 4 and 5+.
const REWARDS: [Reward; 4] = [
    Reward::Explorer,
    Reward::Resources,
    Reward::PopulationGrowth,
    Reward::SuperUnit,
];

impl Bot for RushBot {
    fn name(&self) -> &'static str {
        "rush"
    }

    fn choose(&self, view: &PlayerView, _rng: &mut Rng) -> Action {
        let legal = view.legal_actions();
        choose_reward(view, &legal, &REWARDS)
            .or_else(|| promote(&legal))
            .or_else(|| capture(&legal))
            .or_else(|| best_attack(view, &legal, 0))
            .or_else(|| train(view, &legal))
            .or_else(|| first_legal(&legal, [Action::Research { tech: Tech::Riding }]))
            .or_else(|| site_of_kind(&legal, SiteAction::HarvestFruit))
            .or_else(|| site_of_kind(&legal, SiteAction::HarvestAnimal))
            .or_else(|| recover(view, &legal))
            .or_else(|| move_toward(view, &legal, |u| target(view, u.pos)))
            .unwrap_or(Action::EndTurn)
    }
}

/// For cities in ascending id, a Rider if one is legal for that city, else a Warrior.
fn train(view: &PlayerView, legal: &[Action]) -> Option<Action> {
    view.my_cities().into_iter().find_map(|c| {
        first_legal(
            legal,
            [UnitType::Rider, UnitType::Warrior].map(|kind| Action::Train { city: c.id, kind }),
        )
    })
}

/// The visible enemy capital, else the nearest visible enemy city or unit (ties row-major).
fn target(view: &PlayerView, from: Pos) -> Option<Pos> {
    let me = view.player();
    let cities: Vec<_> = view
        .visible_cities()
        .into_iter()
        .filter(|c| c.owner != me)
        .collect();
    let capitals = cities.iter().filter(|c| c.is_capital).map(|c| c.pos);
    nearest(from, capitals).or_else(|| {
        let units = view
            .visible_units()
            .into_iter()
            .filter(|u| u.owner != me)
            .map(|u| u.pos);
        nearest(from, cities.iter().map(|c| c.pos).chain(units))
    })
}
