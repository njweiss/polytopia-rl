//! Roads and city connections under v2 (docs/MILESTONE_16 "Road tiles", "City connections").

use crate::grid::{neighbors, Pos};
use crate::state::{CityId, PlayerId, State};

/// Whether `p` counts as a road for `player`: it has a road or a city, and is not in the other
/// player's territory.
pub fn road_usable(state: &State, player: PlayerId, p: Pos) -> bool {
    let t = state.tile(p);
    (t.road || t.city.is_some()) && t.owner_city.is_none_or(|c| state.city(c).owner == player)
}

/// The path rule: a non-capital city is linked to its owner's capital by 8-adjacent tiles that are
/// all `road_usable` for its owner. Ignores `connected_to`.
pub fn is_connected(state: &State, city: CityId) -> bool {
    let c = state.city(city);
    if c.is_capital {
        return false;
    }
    let Some(capital) = capital_of(state, c.owner) else {
        return false;
    };
    let goal = state.city(capital).pos;
    let mut seen = vec![false; state.tiles.len()];
    seen[state.index(c.pos)] = true;
    let mut stack = vec![c.pos];
    while let Some(at) = stack.pop() {
        for q in neighbors(state.size, at) {
            if q == goal {
                return true;
            }
            let i = state.index(q);
            if !seen[i] && road_usable(state, c.owner, q) {
                seen[i] = true;
                stack.push(q);
            }
        }
    }
    false
}

/// `player`'s capital: its lowest-id city with `is_capital` (unique outside terminal states).
fn capital_of(state: &State, player: PlayerId) -> Option<CityId> {
    state
        .cities
        .iter()
        .find(|c| c.owner == player && c.is_capital)
        .map(|c| c.id)
}

/// The connection update after an applied action under v2 (docs/MILESTONE_16 "Update", R3, R6):
/// per city in ascending id order, a held connection that no longer holds is lost, then an
/// unconnected city on a path connects. Returns the cities whose population rose, ascending, for
/// the caller's level-up check.
pub(crate) fn update_connections(state: &mut State) -> Vec<CityId> {
    let amount = state.tables.v2_rules().connection_population;
    let before: Vec<u32> = state.cities.iter().map(|c| c.population).collect();
    // `state.cities` is kept in ascending id order.
    for i in 0..state.cities.len() {
        let (id, owner) = (state.cities[i].id, state.cities[i].owner);
        if let Some(k) = state.cities[i].connected_to {
            let capital = state.city(k);
            let kept = capital.is_capital && capital.owner == owner && is_connected(state, id);
            if !kept {
                lose(state, id, amount);
                lose(state, k, amount);
                state.cities[i].connected_to = None;
            }
        }
        if state.cities[i].connected_to.is_none() && is_connected(state, id) {
            let capital = capital_of(state, owner).expect("a connected city's owner has a capital");
            state.city_mut(id).population += amount;
            state.city_mut(capital).population += amount;
            state.cities[i].connected_to = Some(capital);
        }
    }
    state
        .cities
        .iter()
        .zip(before)
        .filter(|(c, b)| c.population > *b)
        .map(|(c, _)| c.id)
        .collect()
}

/// Every loss reverses an earlier gain to the same city, so it never goes below 0 (R4).
fn lose(state: &mut State, city: CityId, amount: u32) {
    let c = state.city_mut(city);
    c.population = c
        .population
        .checked_sub(amount)
        .expect("a connection loss reverses an earlier gain");
}
