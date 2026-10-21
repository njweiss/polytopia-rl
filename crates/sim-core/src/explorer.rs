//! The Explorer reward's random fog-seeking walk (spec: "Explorer"; docs/MILESTONE_5
//! "Explorer walk").

use std::collections::VecDeque;

use crate::data::Tech;
use crate::grid::{chebyshev, Pos};
use crate::map::Terrain;
use crate::movement::neighbors;
use crate::rng::Rng;
use crate::state::{CityId, PlayerId, State};

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
    let climbing = state.players[owner as usize].techs.contains(Tech::Climbing);
    let passable = |s: &State, q: Pos| s.tile(q).terrain != Terrain::Mountain || climbing;
    let n = state.size;

    let mut visited = vec![false; n as usize * n as usize];
    visited[state.index(cur)] = true;
    reveal_radius_1(state, owner, cur);

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
        reveal_radius_1(state, owner, cur);
    }
}

fn all_tiles(n: u8) -> impl Iterator<Item = Pos> {
    (0..n).flat_map(move |r| (0..n).map(move |c| Pos::new(r, c)))
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

/// Explorer vision: Chebyshev 1 with no mountain bonus. The mask never shrinks.
fn reveal_radius_1(state: &mut State, player: PlayerId, at: Pos) {
    for q in all_tiles(state.size) {
        if chebyshev(at, q) <= 1 {
            let i = state.index(q);
            state.players[player as usize].explored[i] = true;
        }
    }
}
