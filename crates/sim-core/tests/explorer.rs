//! Milestone 5: the Explorer reward's walk (docs/MILESTONE_5 "Explorer walk", E1 to E5; spec
//! "Explorer"). Golden boards are drawn so every move has one best candidate and one first step
//! (or, where a tie is the point, exactly the draws listed); each path below is traced by hand in
//! the comment above it, and the expected mask is the starting mask plus radius 1 around every
//! tile of that path. The differential test compares with `common::exp_explorer_walk`, an
//! independent replay of the brief. The walk only exposes its effect (owner's mask and the rng),
//! so paths are checked through the mask they reveal and the draws they cost.

mod common;

use common::*;
use sim_core::actions::{apply, legal_actions, Action};
use sim_core::data::{Reward, Tech, UnitType};
use sim_core::explorer::{explorer_walk, EXPLORER_MOVES};
use sim_core::grid::Pos;
use sim_core::map::Terrain;
use sim_core::rng::Rng;
use sim_core::state::{CityId, Config, PlayerId, State};
use sim_core::testing::StateBuilder;

// ---------- helpers ----------

/// `size` x `size`, every tile a mountain except `open` (field). No techs, nothing explored.
fn walled(size: u8, open: &[Pos]) -> StateBuilder {
    let mut b = StateBuilder::new(size);
    for q in all_pos(size) {
        if !open.contains(&q) {
            b = b.terrain(q, Terrain::Mountain);
        }
    }
    b
}

fn row0(size: u8) -> Vec<Pos> {
    (0..size).map(|c| p(0, c)).collect()
}

fn diagonal(size: u8) -> Vec<Pos> {
    (0..size).map(|i| p(i, i)).collect()
}

/// `before` with `player`'s mask grown by radius 1 around every tile of `path` (start plus 15
/// moves, each to a neighbor or staying put) and the rng replaced by `rng`.
fn after_path(before: &State, player: PlayerId, path: &[Pos], rng: Rng) -> State {
    assert_eq!(
        path.len(),
        16,
        "test bug: a path is the start plus 15 moves"
    );
    for w in path.windows(2) {
        assert!(cheb(w[0], w[1]) <= 1, "test bug: {w:?} is not a step");
    }
    let mut want = before.clone();
    for &t in path {
        for q in all_pos(want.size) {
            if cheb(q, t) <= 1 {
                let i = idx(&want, q);
                want.players[player as usize].explored[i] = true;
            }
        }
    }
    want.rng = rng;
    want
}

/// Runs the walk for city 0 on a clone and returns the result.
fn walked(before: &State) -> State {
    let mut s = before.clone();
    explorer_walk(&mut s, CityId(0));
    s
}

/// `rng` after `k` draws of `gen_range(bound)`.
fn advanced(rng: &Rng, k: usize, bound: u32) -> Rng {
    let mut r = rng.clone();
    for _ in 0..k {
        r.gen_range(bound);
    }
    r
}

fn explored(s: &State, player: PlayerId, q: Pos) -> bool {
    s.players[player as usize].explored[idx(s, q)]
}

// ---------- goldens without ties (the rng must be unchanged) ----------

/// 19x19: row 0 is open, every other tile a mountain (no Climbing). The city is at (0,1); nothing
/// is explored. Row 0 holds a forest at (0,3) and (0,12), units of both players at (0,4), (0,5),
/// (0,11) and (0,12), a village at (0,7) and the other player's city at (0,9) with a unit on it.
/// Start: reveal rows 0-1 x cols 0-2. Each move from (0,x): (0,x-1) and everything behind has
/// fog 0, (0,x+1) is the only tile at distance 1 with fog (2: (0,x+2) and (1,x+2)), so it is the
/// target and the step. Fifteen moves end on (0,16), which reveals column 17; column 18 stays
/// unexplored (a 14th-move stop would leave column 17 unexplored, a 16th move would reveal 18).
fn corridor(owner: PlayerId) -> State {
    let other = 1 - owner;
    walled(19, &row0(19))
        .city(owner, p(0, 1))
        .terrain(p(0, 3), Terrain::Forest)
        .terrain(p(0, 12), Terrain::Forest)
        .unit(other, UnitType::Warrior, p(0, 4))
        .unit(owner, UnitType::Warrior, p(0, 5))
        .village(p(0, 7))
        .city(other, p(0, 9))
        .unit(other, UnitType::Defender, p(0, 9))
        .unit(owner, UnitType::Rider, p(0, 11))
        .unit(other, UnitType::Archer, p(0, 12))
        .build()
}

#[test]
fn corridor_walk_passes_forests_units_cities_and_villages() {
    for owner in 0..2u8 {
        let before = corridor(owner);
        let path: Vec<Pos> = (1..=16).map(|c| p(0, c)).collect();
        let want = after_path(&before, owner, &path, before.rng.clone());
        let s = walked(&before);
        assert_eq!(s, want, "owner {owner}");
        // Spot checks of the drawn result.
        assert!(explored(&s, owner, p(0, 17)) && explored(&s, owner, p(1, 0)));
        assert!(!explored(&s, owner, p(0, 18)) && !explored(&s, owner, p(2, 5)));
        assert!(s.players[(1 - owner) as usize].explored.iter().all(|e| !e));
    }
}

/// 17x17: row 0 open, the rest mountains (no Climbing). City at (0,14). Explored at the start:
/// every tile in columns 12 to 16 except (1,16).
/// Move 1 from (0,14): (0,15) is at distance 1 with fog 1 ((1,16)); west, (0,13) has fog 0,
/// (0,12) has fog 2 at distance 2 and (0,11) fog 4 at distance 3. The nearest wins: (0,15).
/// Move 2 from (0,15): nothing east has fog; (0,12) (fog 2, distance 3) is nearest, the step is
/// (0,14). Then (0,13), (0,12), and each further move west has exactly one candidate at distance
/// 1 ((0,x-1), fog 2). Move k >= 2 lands on (0, 16 - k), so move 15 ends on (0,1).
#[test]
fn nearest_candidate_beats_a_richer_far_one() {
    let mut b = walled(17, &row0(17)).city(0, p(0, 14));
    let seen: Vec<Pos> = all_pos(17)
        .into_iter()
        .filter(|q| q.col >= 12 && *q != p(1, 16))
        .collect();
    b = b.explore(0, &seen);
    let before = b.build();
    let mut path = vec![p(0, 14), p(0, 15)];
    path.extend((1..=14).rev().map(|c| p(0, c)));
    let want = after_path(&before, 0, &path, before.rng.clone());
    let s = walked(&before);
    assert_eq!(s, want);
    assert!(explored(&s, 0, p(1, 16)), "the near candidate was visited");
    assert!(explored(&s, 0, p(0, 0)), "and then the whole corridor west");
}

/// 20x20: row 0 open, the rest mountains. Explored at the start: every tile in columns 0 to 6.
/// City at (0,2). Fog starts at column 7, so the only candidate is (0,6) (fog 2), at distance
/// exactly 4: the explorer heads east with no draw (a limit of 3 would wander and draw between
/// (0,1) and (0,3)). Move k lands on (0, k + 2); move 15 ends on (0,17), revealing column 18.
#[test]
fn a_candidate_at_distance_four_is_in_reach() {
    let seen: Vec<Pos> = all_pos(20).into_iter().filter(|q| q.col <= 6).collect();
    let before = walled(20, &row0(20))
        .city(0, p(0, 2))
        .explore(0, &seen)
        .build();
    let path: Vec<Pos> = (2..=17).map(|c| p(0, c)).collect();
    let want = after_path(&before, 0, &path, before.rng.clone());
    assert_eq!(walked(&before), want);
}

/// 20x20: the diagonal (i,i) is open, everything else a mountain. P0 has Climbing, and every
/// tile is field except the diagonal (i,i), i >= 1, which is mountain. Explored at the start:
/// everything except the diagonal tiles (i,i), i >= 2. City at (0,0).
/// From (k,k), the only tile at distance 1 with fog is (k+1,k+1) (its neighborhood holds
/// (k+2,k+2)); off-diagonal neighbors see no unexplored tile. So the walk climbs the diagonal:
/// move k lands on (k,k), move 15 on (15,15), which reveals (16,16) (radius 1, no mountain
/// bonus); (17,17) and beyond stay unexplored.
#[test]
fn climbing_lets_the_explorer_walk_on_mountains() {
    let mut b = StateBuilder::new(20).techs(0, &[Tech::Climbing]);
    for i in 1..20 {
        b = b.terrain(p(i, i), Terrain::Mountain);
    }
    let seen: Vec<Pos> = all_pos(20)
        .into_iter()
        .filter(|q| q.row != q.col || q.row < 2)
        .collect();
    let before = b.city(0, p(0, 0)).explore(0, &seen).build();
    let path = diagonal(16);
    let want = after_path(&before, 0, &path, before.rng.clone());
    let s = walked(&before);
    assert_eq!(s, want);
    assert!(explored(&s, 0, p(16, 16)) && !explored(&s, 0, p(17, 17)));
}

/// 6x6: open tiles (0,0) (the city), (0,1), and (0,3) to (0,5) behind the mountain at (0,2);
/// rows 1 to 5 are mountains. Nothing explored.
/// Without Climbing: move 1 goes to (0,1) (fog 2: (0,2), (1,2)). From then on the reachable
/// tiles are (0,0) and (0,1), both fog 0, so the explorer wanders: each tile's only passable
/// neighbor is the other one (visited, so the fallback set; no draw). Odd moves end on (0,1).
/// The mask is rows 0-1 x cols 0-2; (0,3) beyond the mountain stays unexplored.
/// With Climbing every tile is passable: move 1 from (0,0) sees (0,1) and (1,0) with fog 2 each
/// and (1,1) with fog 5 (score 4), so it goes to (1,1) and reveals row 2.
#[test]
fn a_mountain_blocks_without_climbing() {
    let open = [p(0, 0), p(0, 1), p(0, 3), p(0, 4), p(0, 5)];
    let before = walled(6, &open).city(0, p(0, 0)).build();
    let mut path = vec![p(0, 0)];
    for k in 1..=15 {
        path.push(if k % 2 == 1 { p(0, 1) } else { p(0, 0) });
    }
    let want = after_path(&before, 0, &path, before.rng.clone());
    let s = walked(&before);
    assert_eq!(s, want);
    assert!(!explored(&s, 0, p(0, 3)) && !explored(&s, 0, p(2, 2)));

    let mut climber = before.clone();
    climber.players[0].techs.insert(Tech::Climbing);
    let s = walked(&climber);
    assert!(
        explored(&s, 0, p(2, 2)),
        "with Climbing the first move is onto (1,1)"
    );
}

/// 16x16: row 0 open, the rest mountains, everything explored, city at (0,0). No fog, so the
/// explorer wanders: at (0,x) the only unvisited passable neighbor is (0,x+1), so it walks east
/// with no draw, ending on (0,15). Nothing new is revealed.
#[test]
fn wander_prefers_unvisited_neighbors() {
    let before = walled(16, &row0(16)).city(0, p(0, 0)).explore_all().build();
    let path: Vec<Pos> = (0..16).map(|c| p(0, c)).collect();
    let want = after_path(&before, 0, &path, before.rng.clone());
    assert_eq!(walked(&before), want);
}

// ---------- goldens with ties (exactly the listed draws) ----------

/// 5x5: open tiles (0,0) (the city), (0,1) and (0,2); the rest mountains. Nothing explored.
/// Move 1: (0,1) has fog 2 -> go. Move 2: (0,2) has fog 2 ((0,3), (1,3)) -> go. From then on
/// there is no fog in reach. Move 3 from (0,2): its only passable neighbor is (0,1) -> go.
/// Move 4 from (0,1): (0,0) and (0,2) are both visited (the start counts, E1), so the fallback
/// set has two tiles: one `gen_range(2)`. Moves 5, 7, ... return to (0,1) (single neighbor, no
/// draw); moves 4, 6, 8, 10, 12, 14 each draw once: 6 draws. Mask: rows 0-1 x cols 0-3.
#[test]
fn wander_draws_only_between_visited_neighbors() {
    let open = [p(0, 0), p(0, 1), p(0, 2)];
    for seed in 0..4 {
        let before = walled(5, &open)
            .city(0, p(0, 0))
            .rng(Rng::new(seed))
            .build();
        let mut want = before.clone();
        for q in all_pos(5) {
            if q.row <= 1 && q.col <= 3 {
                let i = idx(&want, q);
                want.players[0].explored[i] = true;
            }
        }
        want.rng = advanced(&before.rng, 6, 2);
        assert_eq!(walked(&before), want, "seed {seed}");
    }
}

/// The same pocket with `full_observability`: no fog anywhere, so every move wanders. Move 1:
/// (0,1); move 2: (0,2) (its only unvisited neighbor); then as above, 6 draws. No mask changes.
#[test]
fn full_observability_wanders_and_changes_no_mask() {
    let open = [p(0, 0), p(0, 1), p(0, 2)];
    let cfg = Config {
        full_observability: true,
        ..Config::default()
    };
    let before = walled(5, &open)
        .city(0, p(0, 0))
        .config(cfg)
        .rng(Rng::new(7))
        .build();
    let mut want = before.clone();
    want.rng = advanced(&before.rng, 6, 2);
    assert_eq!(walked(&before), want);
}

/// 20x20: the diagonal (i,i) is open, the rest mountains (no Climbing). City at (2,2). Explored
/// at the start: only (0,2).
/// Move 1: after the start reveal (rows 1-3 x cols 1-3) the two tiles at distance 1 are (1,1),
/// fog 4 ((0,0), (0,1), (1,0), (2,0)), and (3,3), fog 5 ((2,4), (3,4), (4,2), (4,3), (4,4)),
/// which scores min(5, 4) = 4. Tie: one `gen_range(2)` over [(1,1), (3,3)].
/// Draw 0 -> (1,1). Move 2: (0,0) now has fog 0, so (3,3) (distance 2) is the target; the step
/// is (2,2). Move k >= 3 lands on (k,k): move 15 on (15,15).
/// Draw 1 -> (3,3). Move 2: (4,4) at distance 1 beats (1,1) at distance 2. Move k lands on
/// (k+2,k+2): move 15 on (17,17). (0,0), (0,1), (1,0) are never revealed.
/// No other move ties (one passable tile ahead on the diagonal, nothing behind has fog).
#[test]
fn fog_is_capped_at_four_so_five_ties_with_four() {
    let mut both = [false; 2];
    for seed in 0..12 {
        let before = walled(20, &diagonal(20))
            .city(0, p(2, 2))
            .explore(0, &[p(0, 2)])
            .rng(Rng::new(seed))
            .build();
        let draw = before.rng.clone().gen_range(2);
        both[draw as usize] = true;
        let path: Vec<Pos> = if draw == 0 {
            let mut v = vec![p(2, 2), p(1, 1)];
            v.extend((2..=15).map(|i| p(i, i)));
            v
        } else {
            (2..=17).map(|i| p(i, i)).collect()
        };
        let want = after_path(&before, 0, &path, advanced(&before.rng, 1, 2));
        assert_eq!(walked(&before), want, "seed {seed}, draw {draw}");
    }
    assert_eq!(both, [true, true], "test bug: both draws should occur");
}

/// 20x20: row 0 open, the rest mountains. Explored: columns 0 to 6. City at (0,1). The only
/// tile with fog in its neighborhood is (0,6) at distance 5: out of reach, so move 1 wanders
/// between (0,0) and (0,2): one `gen_range(2)`.
/// Draw 0 -> (0,0). Move 2: back to (0,1) (single neighbor). Move 3: (0,6) is still at 5; the
/// only unvisited neighbor is (0,2). Move 4: (0,6) at distance 4 is the target; then east one
/// tile per move: move k >= 3 lands on (0, k - 1), move 15 on (0,14).
/// Draw 1 -> (0,2). Move 2: (0,6) at distance 4; move k lands on (0, k + 1), move 15 on (0,16).
#[test]
fn a_candidate_at_distance_five_is_out_of_reach() {
    let seen: Vec<Pos> = all_pos(20).into_iter().filter(|q| q.col <= 6).collect();
    let mut both = [false; 2];
    for seed in 0..12 {
        let before = walled(20, &row0(20))
            .city(0, p(0, 1))
            .explore(0, &seen)
            .rng(Rng::new(seed))
            .build();
        let draw = before.rng.clone().gen_range(2);
        both[draw as usize] = true;
        let path: Vec<Pos> = if draw == 0 {
            let mut v = vec![p(0, 1), p(0, 0), p(0, 1)];
            v.extend((2..=14).map(|c| p(0, c)));
            v
        } else {
            (1..=16).map(|c| p(0, c)).collect()
        };
        let want = after_path(&before, 0, &path, advanced(&before.rng, 1, 2));
        assert_eq!(walked(&before), want, "seed {seed}, draw {draw}");
    }
    assert_eq!(both, [true, true], "test bug: both draws should occur");
}

// ---------- through apply ----------

#[test]
fn choose_explorer_runs_the_walk_through_apply() {
    let mut before = corridor(0);
    before.cities[0].level = 2;
    before.cities[0].population = 3;
    before.cities[0].pending_reward = true;
    let pick = Action::ChooseReward {
        city: CityId(0),
        reward: Reward::Explorer,
    };
    assert_eq!(
        legal_actions(&before),
        [
            Action::ChooseReward {
                city: CityId(0),
                reward: Reward::Workshop
            },
            pick
        ]
    );
    let mut s = before.clone();
    assert_eq!(apply(&mut s, &pick), Ok(()));
    let path: Vec<Pos> = (1..=16).map(|c| p(0, c)).collect();
    let mut want = after_path(&before, 0, &path, before.rng.clone());
    want.cities[0].pending_reward = false;
    assert_eq!(s, want);
}

#[test]
fn the_level_up_check_runs_after_the_explorer() {
    // Level 2, pending, population 5 = cum(3): resolving any level-2 reward levels up again.
    let mut s = corridor(0);
    s.cities[0].level = 2;
    s.cities[0].population = 5;
    s.cities[0].pending_reward = true;
    let pick = Action::ChooseReward {
        city: CityId(0),
        reward: Reward::Explorer,
    };
    assert_eq!(apply(&mut s, &pick), Ok(()));
    let c = &s.cities[0];
    assert_eq!((c.level, c.pending_reward, c.population), (3, true, 5));
    assert!(explored(&s, 0, p(0, 17)), "the walk ran");
}

// ---------- differential test against the brief's replay ----------

/// A random walk board from the test's own rng: 3x3 to 12x12, mountains 0 to 50%, forests,
/// a city (on field) for each player, a few units, random techs, a random explored mask per player (empty,
/// sparse, blobs, dense or full), sometimes `full_observability`, and a random game rng.
fn walk_board(seed: u64) -> State {
    let mut r = Rng::new(seed ^ 0xe1_9f0e);
    let n = 3 + r.gen_range(10) as u8;
    let mountains = r.gen_range(6);
    let mut b = StateBuilder::new(n);
    for q in all_pos(n) {
        let roll = r.gen_range(10);
        if roll < mountains {
            b = b.terrain(q, Terrain::Mountain);
        } else if roll < mountains + 2 {
            b = b.terrain(q, Terrain::Forest);
        }
    }
    let mut free = all_pos(n);
    let mut take = |r: &mut Rng| free.swap_remove(r.gen_range(free.len() as u32) as usize);
    for owner in 0..2u8 {
        // Cities stand on field tiles.
        let q = take(&mut r);
        b = b.city(owner, q).terrain(q, Terrain::Field);
    }
    for _ in 0..r.gen_range(4) {
        let q = take(&mut r);
        b = b.unit(r.gen_range(2) as u8, UnitType::Warrior, q);
    }
    for owner in 0..2u8 {
        if r.gen_range(2) == 0 {
            b = b.techs(owner, &[Tech::Climbing]);
        }
        let style = r.gen_range(5);
        let tiles: Vec<Pos> = match style {
            0 => Vec::new(),
            1 => all_pos(n)
                .into_iter()
                .filter(|_| r.gen_range(5) == 0)
                .collect(),
            2 => {
                // A few explored squares.
                let mut v = Vec::new();
                for _ in 0..1 + r.gen_range(3) {
                    let c = p(r.gen_range(n as u32) as u8, r.gen_range(n as u32) as u8);
                    let rad = r.gen_range(3);
                    v.extend(all_pos(n).into_iter().filter(|&q| cheb(q, c) <= rad));
                }
                v
            }
            3 => all_pos(n)
                .into_iter()
                .filter(|_| r.gen_range(5) != 0)
                .collect(),
            _ => all_pos(n),
        };
        b = b.explore(owner, &tiles);
    }
    if r.gen_range(8) == 0 {
        b = b.config(Config {
            full_observability: true,
            ..Config::default()
        });
    }
    b.rng(Rng::new(r.next_u32() as u64)).build()
}

#[test]
fn explorer_walk_matches_an_independent_replay() {
    let (mut drew, mut no_draw, mut long_way) = (0, 0, 0);
    for seed in 0..3000u64 {
        let before = walk_board(seed);
        for city in 0..2u32 {
            let mut want = before.clone();
            let path = exp_explorer_walk(&mut want, CityId(city));
            let mut got = before.clone();
            explorer_walk(&mut got, CityId(city));
            assert_eq!(got, want, "seed {seed}, city {city}, replay path {path:?}");
            if want.rng == before.rng {
                no_draw += 1;
            } else {
                drew += 1;
            }
            long_way += usize::from(cheb(path[0], path[15]) >= 5);
        }
    }
    // Vacuity guards (not rules): both kinds of walk and some long ones occur. Open boards
    // almost always tie somewhere; the goldens cover walks without draws.
    assert!(
        drew > 1000 && no_draw > 10,
        "drew {drew}, no draw {no_draw}"
    );
    assert!(
        long_way > 200,
        "only {long_way} walks went 5 or more from the start"
    );
}

// ---------- properties (no reference walk) ----------

#[test]
fn explorer_properties() {
    assert_eq!(EXPLORER_MOVES, 15);
    let mut wander_draws = 0;
    for seed in 10_000..11_500u64 {
        let before = walk_board(seed);
        let city = CityId((seed % 2) as u32);
        let c = before.cities[city.0 as usize].clone();
        let (owner, other) = (c.owner as usize, 1 - c.owner as usize);
        let mut s = before.clone();
        explorer_walk(&mut s, city);
        let ctx = format!("seed {seed}");

        // Only the owner's mask and the rng change; the mask only grows.
        let mut rest = s.clone();
        rest.players[owner].explored = before.players[owner].explored.clone();
        rest.rng = before.rng.clone();
        assert_eq!(rest, before, "{ctx}: the walk changed something else");
        assert_eq!(s.players[other].explored, before.players[other].explored);
        let n = before.size;
        for q in all_pos(n) {
            let i = idx(&s, q);
            assert!(!before.players[owner].explored[i] || s.players[owner].explored[i]);
            // Radius 1 around the start is revealed (E1).
            if cheb(q, c.pos) <= 1 {
                assert!(s.players[owner].explored[i], "{ctx}: start reveal {q:?}");
            }
        }
        // Every newly explored tile is within 1 of a tile the explorer can reach in at most 15
        // steps over passable tiles (in bounds, mountains only with Climbing).
        let climbing = has_tech(&before, c.owner, Tech::Climbing);
        let passable = |q: Pos| before.tile(q).terrain != Terrain::Mountain || climbing;
        let d = bfs(n, c.pos, &passable);
        for q in all_pos(n) {
            let i = idx(&s, q);
            if s.players[owner].explored[i] && !before.players[owner].explored[i] {
                let near = all_pos(n)
                    .into_iter()
                    .any(|t| cheb(t, q) <= 1 && d[idx(&s, t)] <= EXPLORER_MOVES);
                assert!(near, "{ctx}: {q:?} revealed out of reach");
            }
        }
        // Same seed, same result.
        let mut again = before.clone();
        explorer_walk(&mut again, city);
        assert_eq!(again, s, "{ctx}: not deterministic");
        // Nothing in reach: with everything explored the walk wanders, and with two or more
        // passable neighbors at the start the first move must draw.
        let all_seen = before.players[owner].explored.iter().all(|&e| e);
        let open = nbrs(n, c.pos).into_iter().filter(|&q| passable(q)).count();
        if all_seen {
            assert_eq!(s.players[owner].explored, before.players[owner].explored);
            if open >= 2 {
                assert_ne!(s.rng, before.rng, "{ctx}: wandering with a choice draws");
                wander_draws += 1;
            }
        }
    }
    assert!(wander_draws > 50, "only {wander_draws} wandering walks");
}
