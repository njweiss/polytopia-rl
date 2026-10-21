//! Economy helpers (docs/MILESTONE_4 "Start of turn: income", "Research", "Train",
//! "Population and levels", "Reward menus").

use crate::data::{Reward, Tech};
use crate::state::{CityId, PlayerId, State};

/// Sum over the player's cities of `level + (capital ? 1 : 0) + (workshop ? 1 : 0) + parks`.
pub fn income(state: &State, player: PlayerId) -> u32 {
    let _ = (state, player);
    unimplemented!()
}

/// `L(L+1)/2 - 1`: the cumulative population a city needs for `level` (level 1 -> 0).
pub fn cumulative_population_for_level(level: u32) -> u32 {
    let _ = level;
    unimplemented!()
}

/// Living units whose `origin == city`.
pub fn supported_units(state: &State, city: CityId) -> u32 {
    let _ = (state, city);
    unimplemented!()
}

/// `tables.tech_cost(tech, cities owned by the player)`.
pub fn research_cost(state: &State, player: PlayerId, tech: Tech) -> u32 {
    let _ = (state, player, tech);
    unimplemented!()
}

/// The table menu for the city's current level, without BorderGrowth once it has been used.
pub fn reward_menu(state: &State, city: CityId) -> Vec<Reward> {
    let _ = (state, city);
    unimplemented!()
}
