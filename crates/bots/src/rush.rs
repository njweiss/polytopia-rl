//! The Rush bot (docs/MILESTONE_8 "Rush bot").

use sim_core::actions::Action;
use sim_core::data::{Reward, SiteAction, Tech, UnitType};
use sim_core::grid::Pos;
use sim_core::rng::Rng;
use sim_core::state::Ruleset;
use sim_core::view::PlayerView;

use crate::{
    best_attack, build_monument, capture, choose_reward, first_legal, move_toward, nearest,
    promote, recover, site_of_kind, Bot,
};

pub struct RushBot;

/// Reward preferences for levels 2, 3, 4 and 5+.
const REWARDS: [Reward; 4] = [
    Reward::Explorer,
    Reward::Resources,
    Reward::PopulationGrowth,
    Reward::SuperUnit,
];

const RESEARCH_V2: [Tech; 3] = [Tech::Riding, Tech::FreeSpirit, Tech::Chivalry];

impl Bot for RushBot {
    fn name(&self) -> &'static str {
        "rush"
    }

    fn choose(&self, view: &PlayerView, _rng: &mut Rng) -> Action {
        let legal = view.legal_actions();
        let v2 = view.ruleset() == Ruleset::V2;
        choose_reward(view, &legal, &REWARDS)
            .or_else(|| promote(&legal))
            .or_else(|| capture(&legal))
            .or_else(|| best_attack(view, &legal, 0))
            .or_else(|| v2.then(|| build_monument(&legal)).flatten())
            .or_else(|| train(view, &legal, v2))
            .or_else(|| research(&legal, v2))
            .or_else(|| site_of_kind(&legal, SiteAction::HarvestFruit))
            .or_else(|| site_of_kind(&legal, SiteAction::HarvestAnimal))
            .or_else(|| recover(view, &legal))
            .or_else(|| move_toward(view, &legal, |u| target(view, u.pos)))
            .unwrap_or(Action::EndTurn)
    }
}

fn research(legal: &[Action], v2: bool) -> Option<Action> {
    if v2 {
        first_legal(legal, RESEARCH_V2.map(|tech| Action::Research { tech }))
    } else {
        first_legal(legal, [Action::Research { tech: Tech::Riding }])
    }
}

/// For cities in ascending id, the first legal unit in priority order (V1: Rider, Warrior;
/// V2: Knight, Rider, Warrior).
fn train(view: &PlayerView, legal: &[Action], v2: bool) -> Option<Action> {
    let order: &[UnitType] = if v2 {
        &[UnitType::Knight, UnitType::Rider, UnitType::Warrior]
    } else {
        &[UnitType::Rider, UnitType::Warrior]
    };
    view.my_cities().into_iter().find_map(|c| {
        first_legal(
            legal,
            order.iter().map(|&kind| Action::Train { city: c.id, kind }),
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
