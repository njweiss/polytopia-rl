//! Score, derived on demand from state (spec: "Score"; docs/MILESTONE_6 "Score").

use crate::data::{UnitType, ALL_TECHS};
use crate::economy::cumulative_population_for_level;
use crate::state::{PlayerId, State};

/// One player's score by term, each already multiplied by its weight.
#[derive(Debug, PartialEq, Eq)]
pub struct ScoreBreakdown {
    pub units: u32,
    pub territory: u32,
    pub exploration: u32,
    pub cities: u32,
    pub parks: u32,
    pub techs: u32,
}

impl ScoreBreakdown {
    pub fn total(&self) -> u32 {
        self.units + self.territory + self.exploration + self.cities + self.parks + self.techs
    }
}

pub fn score_breakdown(state: &State, player: PlayerId) -> ScoreBreakdown {
    let w = &state.config.score_weights;
    let units = state
        .units
        .iter()
        .filter(|u| u.owner == player)
        .map(|u| match u.kind {
            UnitType::SuperUnit => w.super_unit,
            kind => w.unit_per_star * state.tables.unit(kind).cost.expect("trainable unit cost"),
        })
        .sum();
    let tiles = state
        .tiles
        .iter()
        .filter(|t| t.owner_city.is_some_and(|c| state.city(c).owner == player))
        .count() as u32;
    let explored = state.players[player as usize]
        .explored
        .iter()
        .filter(|&&e| e)
        .count() as u32;
    let own_cities = || state.cities.iter().filter(|c| c.owner == player);
    let cities = own_cities()
        .map(|c| {
            let surplus = c
                .population
                .saturating_sub(cumulative_population_for_level(c.level));
            w.city_base + w.city_per_level * (c.level - 1) + w.population_surplus * surplus
        })
        .sum();
    let parks: u32 = own_cities().map(|c| c.parks).sum();
    let techs = state.players[player as usize].techs;
    let tiers: u32 = ALL_TECHS
        .iter()
        .filter(|&&t| techs.contains(t))
        .map(|&t| state.tables.tech(t).tier)
        .sum();
    ScoreBreakdown {
        units,
        territory: w.territory_per_tile * tiles,
        exploration: w.exploration_per_tile * explored,
        cities,
        parks: w.park * parks,
        techs: w.tech_per_tier * tiers,
    }
}

/// `score_breakdown(state, player).total()`.
pub fn score(state: &State, player: PlayerId) -> u32 {
    score_breakdown(state, player).total()
}
