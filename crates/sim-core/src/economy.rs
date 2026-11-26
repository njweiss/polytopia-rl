//! Economy helpers (docs/MILESTONE_4 "Start of turn: income", "Research", "Train",
//! "Population and levels", "Reward menus").

use crate::buildings::market_income;
use crate::data::{Reward, Tech, UnitType};
use crate::grid::all_tiles;
use crate::state::{CityId, PlayerId, Ruleset, State};

/// Sum over the player's cities of `level + (capital ? 1 : 0) + (workshop ? 1 : 0) + parks`,
/// plus under `V2` any Market income from tiles owned by the player's cities.
pub fn income(state: &State, player: PlayerId) -> u32 {
    let base: u32 = state
        .cities
        .iter()
        .filter(|c| c.owner == player)
        .map(|c| c.level + u32::from(c.is_capital) + u32::from(c.workshop) + c.parks)
        .sum();
    if state.config.ruleset != Ruleset::V2 {
        return base;
    }
    let markets: u32 = all_tiles(state.size)
        .filter(|&pos| {
            let t = state.tile(pos);
            t.building.is_some() && t.owner_city.is_some_and(|c| state.city(c).owner == player)
        })
        .map(|pos| market_income(state, pos))
        .sum();
    base + markets
}

/// `L(L+1)/2 - 1`: the cumulative population a city needs for `level` (level 1 -> 0).
pub fn cumulative_population_for_level(level: u32) -> u32 {
    level * (level + 1) / 2 - 1
}

/// Living units whose `origin == city`, never counting Super Units (docs/MILESTONE_5 E6).
pub fn supported_units(state: &State, city: CityId) -> u32 {
    state
        .units
        .iter()
        .filter(|u| u.origin == Some(city) && u.kind != UnitType::SuperUnit)
        .count() as u32
}

/// `tables.tech_cost(tech, cities owned by the player)`.
pub fn research_cost(state: &State, player: PlayerId, tech: Tech) -> u32 {
    let cities = state.cities.iter().filter(|c| c.owner == player).count() as u32;
    state.tables.tech_cost(tech, cities)
}

/// The table menu for the city's current level, without BorderGrowth once it has been used.
pub fn reward_menu(state: &State, city: CityId) -> Vec<Reward> {
    let c = state.city(city);
    state
        .tables
        .reward_menu(c.level)
        .iter()
        .copied()
        .filter(|&r| !(r == Reward::BorderGrowth && c.border_growth_used))
        .collect()
}
