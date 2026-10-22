//! The Explorer reward's random fog-seeking walk (spec: "Explorer"; docs/MILESTONE_5
//! "Explorer walk").

use std::collections::VecDeque;

use crate::actions::reveal_within;
use crate::grid::{all_tiles, neighbors, Pos};
use crate::movement::terrain_allows;
use crate::rng::Rng;
use crate::state::{CityId, State};

/// Moves the explorer makes before it disappears.
pub const EXPLORER_MOVES: u32 = 15;

/// Farthest BFS distance at which a tile can be a target.
const MAX_TARGET_DIST: u32 = 4;
/// Fog counted per candidate is capped here.
const FOG_CAP: u32 = 4;
const VISITED_PENALTY: f32 = 0.5;

/// Runs the whole walk from `city`'s tile for the city's owner: grows that player's explored mask
/// and draws from `state.rng` only on ties. Used by `ChooseReward { Explorer }`.
pub fn explorer_walk(state: &mut State, city: CityId) {
    let c = state.city(city);
    let (owner, mut cur) = (c.owner, c.pos);
    let passable = |s: &State, q: Pos| terrain_allows(s, owner, q);
    let n = state.size;

    let mut visited = vec![false; n as usize * n as usize];
    visited[state.index(cur)] = true;
    // Explorer vision: Chebyshev 1 with no mountain bonus.
    reveal_within(state, owner, cur, 1);

    for _ in 0..EXPLORER_MOVES {
        let dist = bfs(state, cur, &passable);
        let fog = |q: Pos| {
            neighbors(n, q)
                .chain(std::iter::once(q))
                .filter(|&t| !state.is_explored(owner, t))
                .count() as u32
        };
        let candidates: Vec<(Pos, u32, u32)> = all_tiles(n)
            .filter_map(|q| {
                let d = dist[state.index(q)]?;
                let f = fog(q);
                ((1..=MAX_TARGET_DIST).contains(&d) && f >= 1).then_some((q, d, f))
            })
            .collect();

        let next = if let Some(dmin) = candidates.iter().map(|&(_, d, _)| d).min() {
            let score = |q: Pos, f: u32| {
                let base = f.min(FOG_CAP) as f32;
                if visited[state.index(q)] {
                    base - VISITED_PENALTY
                } else {
                    base
                }
            };
            let nearest: Vec<(Pos, f32)> = candidates
                .iter()
                .filter(|&&(_, d, _)| d == dmin)
                .map(|&(q, _, f)| (q, score(q, f)))
                .collect();
            let best = nearest
                .iter()
                .map(|&(_, s)| s)
                .fold(f32::NEG_INFINITY, f32::max);
            let top: Vec<Pos> = nearest
                .iter()
                .filter(|&&(_, s)| s == best)
                .map(|&(q, _)| q)
                .collect();
            let target = pick(&mut state.rng, &top).expect("at least one candidate");

            let to_target = bfs(state, target, &passable);
            let steps: Vec<Pos> = neighbors(n, cur)
                .filter(|&q| passable(state, q) && to_target[state.index(q)] == Some(dmin - 1))
                .collect();
            pick(&mut state.rng, &steps).expect("a shortest path has a first step")
        } else {
            let open: Vec<Pos> = neighbors(n, cur).filter(|&q| passable(state, q)).collect();
            let unvisited: Vec<Pos> = open
                .iter()
                .copied()
                .filter(|&q| !visited[state.index(q)])
                .collect();
            let choices = if unvisited.is_empty() {
                open
            } else {
                unvisited
            };
            pick(&mut state.rng, &choices).unwrap_or(cur)
        };

        cur = next;
        visited[state.index(cur)] = true;
        reveal_within(state, owner, cur, 1);
    }
}

/// BFS step counts from `from` over passable tiles (8-neighbors), indexed row-major; `None` where
/// unreachable.
fn bfs(state: &State, from: Pos, passable: &dyn Fn(&State, Pos) -> bool) -> Vec<Option<u32>> {
    let mut dist = vec![None; state.tiles.len()];
    dist[state.index(from)] = Some(0);
    let mut queue = VecDeque::from([from]);
    while let Some(p) = queue.pop_front() {
        let d = dist[state.index(p)].expect("queued tiles have a distance");
        for q in neighbors(state.size, p) {
            let i = state.index(q);
            if dist[i].is_none() && passable(state, q) {
                dist[i] = Some(d + 1);
                queue.push_back(q);
            }
        }
    }
    dist
}

/// One of `options` (already row-major): no draw when there is exactly one.
fn pick(rng: &mut Rng, options: &[Pos]) -> Option<Pos> {
    match options.len() {
        0 => None,
        1 => Some(options[0]),
        k => Some(options[rng.gen_range(k as u32) as usize]),
    }
}
