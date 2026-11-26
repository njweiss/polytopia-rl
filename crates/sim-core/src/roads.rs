//! Roads and city connections under v2 (docs/MILESTONE_16 "Road tiles", "City connections").

use std::collections::VecDeque;

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
    debug_assert!(
        state.outcome.is_some()
            || state
                .cities
                .iter()
                .filter(|c| c.owner == player && c.is_capital)
                .count()
                <= 1,
        "player {player} has multiple capitals in a non-terminal state"
    );
    state
        .cities
        .iter()
        .find(|c| c.owner == player && c.is_capital)
        .map(|c| c.id)
}

/// Single BFS from `player`'s capital over `road_usable` tiles. `reached[i]` is true for every
/// tile 8-adjacent to the capital or to a `road_usable` tile reachable from it.
fn reachable_from_capital(state: &State, player: PlayerId) -> Option<(CityId, Vec<bool>)> {
    let capital = capital_of(state, player)?;
    let start = state.city(capital).pos;
    let mut visited = vec![false; state.tiles.len()];
    let mut reached = vec![false; state.tiles.len()];
    visited[state.index(start)] = true;
    let mut queue = VecDeque::new();
    queue.push_back(start);
    while let Some(at) = queue.pop_front() {
        for q in neighbors(state.size, at) {
            let i = state.index(q);
            reached[i] = true;
            if !visited[i] && road_usable(state, player, q) {
                visited[i] = true;
                queue.push_back(q);
            }
        }
    }
    Some((capital, reached))
}

/// The connection update after an applied action under v2 (docs/MILESTONE_16 "Update", R3, R6):
/// per city in ascending id order, a held connection that no longer holds is lost, then an
/// unconnected city on a path connects.
pub(crate) fn update_connections(state: &mut State) {
    let amount = state.tables.v2_rules().connection_population;
    let nets = [
        reachable_from_capital(state, 0),
        reachable_from_capital(state, 1),
    ];
    // `state.cities` is kept in ascending id order.
    for i in 0..state.cities.len() {
        let c = &state.cities[i];
        let (id, owner, pos, is_capital, connected_to) =
            (c.id, c.owner, c.pos, c.is_capital, c.connected_to);
        let net = nets[owner as usize].as_ref();
        let connected = !is_capital && net.is_some_and(|(_, reached)| reached[state.index(pos)]);
        let mut conn = connected_to;
        if let Some(k) = conn {
            let capital = state.city(k);
            let kept = capital.is_capital && capital.owner == owner && connected;
            if !kept {
                lose(state, id, amount);
                lose(state, k, amount);
                state.cities[i].connected_to = None;
                conn = None;
            }
        }
        if conn.is_none() && connected {
            let (capital, _) = net.expect("a connected city's owner has a capital");
            state.city_mut(id).population += amount as i32;
            state.city_mut(*capital).population += amount as i32;
            state.cities[i].connected_to = Some(*capital);
        }
    }
}

fn lose(state: &mut State, city: CityId, amount: u32) {
    state.city_mut(city).population -= amount as i32;
}
