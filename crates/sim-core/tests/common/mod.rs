//! Shared helpers for the milestone 3 tests. Every checker here is written from the rules in
//! docs/MILESTONE_3 independently of the code under test: it never calls `reachable_tiles`,
//! `defense_bonus`, `legal_actions` or `apply` to compute an expectation.
#![allow(dead_code)]

use sim_core::actions::Action;
use sim_core::combat::{resolve_attack, Combatant};
use sim_core::data::{Skill, Tables, Tech, UnitType};
use sim_core::grid::Pos;
use sim_core::map::Terrain;
use sim_core::rng::Rng;
use sim_core::state::{Config, Phase, PlayerId, State, TileState, Unit, UnitId};
use sim_core::testing::StateBuilder;

pub fn p(row: u8, col: u8) -> Pos {
    Pos::new(row, col)
}

pub fn cheb(a: Pos, b: Pos) -> u32 {
    let dr = (a.row as i32 - b.row as i32).unsigned_abs();
    let dc = (a.col as i32 - b.col as i32).unsigned_abs();
    dr.max(dc)
}

pub fn all_pos(n: u8) -> Vec<Pos> {
    (0..n)
        .flat_map(|r| (0..n).map(move |c| Pos::new(r, c)))
        .collect()
}

pub fn idx(s: &State, q: Pos) -> usize {
    q.row as usize * s.size as usize + q.col as usize
}

pub fn tile_mut(s: &mut State, q: Pos) -> &mut TileState {
    let i = idx(s, q);
    &mut s.tiles[i]
}

pub fn unit_at(s: &State, q: Pos) -> Option<&Unit> {
    s.units.iter().find(|u| u.pos == q)
}

pub fn unit(s: &State, id: u32) -> &Unit {
    s.units
        .iter()
        .find(|u| u.id == UnitId(id))
        .unwrap_or_else(|| panic!("unit {id} is not alive"))
}

pub fn has_skill(s: &State, kind: UnitType, skill: Skill) -> bool {
    s.tables.unit(kind).skills.contains(&skill)
}

pub fn has_tech(s: &State, player: PlayerId, tech: Tech) -> bool {
    s.players[player as usize].techs.contains(tech)
}

/// Parses a terrain grid: whitespace-separated tokens, `.` field, `f` forest, `m` mountain,
/// `v` village (field). Row 0 is the first string. The board must be square.
pub fn board(rows: &[&str]) -> StateBuilder {
    let n = rows.len() as u8;
    let mut b = StateBuilder::new(n);
    for (r, line) in rows.iter().enumerate() {
        let toks: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(toks.len(), rows.len(), "board must be square: {line:?}");
        for (c, t) in toks.iter().enumerate() {
            let q = p(r as u8, c as u8);
            b = match *t {
                "." => b,
                "f" => b.terrain(q, Terrain::Forest),
                "m" => b.terrain(q, Terrain::Mountain),
                "v" => b.village(q),
                other => panic!("unknown board token {other:?}"),
            };
        }
    }
    b
}

/// Positions marked with `mark` in a grid drawn like `board`, row-major.
pub fn marked(rows: &[&str], mark: &str) -> Vec<Pos> {
    let mut out = Vec::new();
    for (r, line) in rows.iter().enumerate() {
        for (c, t) in line.split_whitespace().enumerate() {
            if t == mark {
                out.push(p(r as u8, c as u8));
            }
        }
    }
    out
}

// ---------- independent rule replays ----------

/// Brute-force reachability from the brief: enumerate every path of at most `movement` steps.
/// A step may enter a tile that is in bounds, explored by the owner (mask before the move), holds
/// no unit, and is not a mountain unless the owner has Climbing. After entering a forest, a
/// mountain, or a tile adjacent to an enemy unit, no further step is taken. The start tile is
/// exempt from those stops. Result row-major, without the start tile.
pub fn bf_reach(s: &State, id: UnitId) -> Vec<Pos> {
    let u = s.units.iter().find(|u| u.id == id).unwrap();
    let budget = s.tables.unit(u.kind).movement;
    let climbing = has_tech(s, u.owner, Tech::Climbing);
    let enemies: Vec<Pos> = s
        .units
        .iter()
        .filter(|e| e.owner != u.owner)
        .map(|e| e.pos)
        .collect();
    let n = s.size as i32;
    let enterable = |q: Pos| {
        s.players[u.owner as usize].explored[idx(s, q)]
            && s.units.iter().all(|o| o.pos != q)
            && (s.tile(q).terrain != Terrain::Mountain || climbing)
    };
    let stops = |q: Pos| {
        matches!(s.tile(q).terrain, Terrain::Forest | Terrain::Mountain)
            || enemies.iter().any(|&e| cheb(e, q) == 1)
    };
    let mut found = vec![false; (n * n) as usize];
    // Depth-first over all paths (budgets are tiny).
    fn walk(
        at: Pos,
        left: u32,
        n: i32,
        enterable: &dyn Fn(Pos) -> bool,
        stops: &dyn Fn(Pos) -> bool,
        found: &mut Vec<bool>,
    ) {
        if left == 0 {
            return;
        }
        for dr in -1i32..=1 {
            for dc in -1i32..=1 {
                if dr == 0 && dc == 0 {
                    continue;
                }
                let (r, c) = (at.row as i32 + dr, at.col as i32 + dc);
                if r < 0 || c < 0 || r >= n || c >= n {
                    continue;
                }
                let q = Pos::new(r as u8, c as u8);
                if !enterable(q) {
                    continue;
                }
                found[(r * n + c) as usize] = true;
                if !stops(q) {
                    walk(q, left - 1, n, enterable, stops, found);
                }
            }
        }
    }
    walk(u.pos, budget, n, &enterable, &stops, &mut found);
    all_pos(s.size)
        .into_iter()
        .filter(|&q| q != u.pos && found[idx(s, q)])
        .collect()
}

/// Defense bonus from the brief: max of 1.0, terrain (forest+Archery or mountain+Climbing, the
/// defender owner's techs) 1.5, own city with Fortify 1.5 or 4.0 with a wall.
pub fn expected_bonus(s: &State, id: UnitId) -> f32 {
    let u = s.units.iter().find(|u| u.id == id).unwrap();
    let t = s.tile(u.pos);
    let mut b = 1.0f32;
    if (t.terrain == Terrain::Forest && has_tech(s, u.owner, Tech::Archery))
        || (t.terrain == Terrain::Mountain && has_tech(s, u.owner, Tech::Climbing))
    {
        b = b.max(1.5);
    }
    if let Some(cid) = t.city {
        let city = s.cities.iter().find(|c| c.id == cid).unwrap();
        if city.owner == u.owner && has_skill(s, u.kind, Skill::Fortify) {
            b = b.max(if city.wall { 4.0 } else { 1.5 });
        }
    }
    b
}

/// Legal actions enumerated straight from the brief, in canonical order. Uses `bf_reach`.
pub fn bf_legal(s: &State) -> Vec<Action> {
    let mut out = Vec::new();
    if s.outcome.is_some() {
        return out;
    }
    let mut ids: Vec<UnitId> = s
        .units
        .iter()
        .filter(|u| u.owner == s.current)
        .map(|u| u.id)
        .collect();
    ids.sort();
    for id in ids {
        let u = s.units.iter().find(|u| u.id == id).unwrap();
        let stats = s.tables.unit(u.kind);
        let dash = stats.skills.contains(&Skill::Dash);
        let escape = stats.skills.contains(&Skill::Escape);
        if u.phase == Phase::Fresh || (u.phase == Phase::Attacked && escape) {
            for to in bf_reach(s, id) {
                out.push(Action::Move { unit: id, to });
            }
        }
        if u.phase == Phase::Fresh || (u.phase == Phase::Moved && dash) {
            for q in all_pos(s.size) {
                let d = cheb(u.pos, q);
                if d >= 1
                    && d <= stats.range
                    && s.players[u.owner as usize].explored[idx(s, q)]
                    && s.units.iter().any(|e| e.pos == q && e.owner != u.owner)
                {
                    out.push(Action::Attack {
                        unit: id,
                        target: q,
                    });
                }
            }
        }
        if u.phase == Phase::Fresh {
            out.push(Action::Recover { unit: id });
        }
        if u.promotion_available && !u.veteran {
            out.push(Action::Promote { unit: id });
        }
    }
    let pending = s
        .cities
        .iter()
        .any(|c| c.owner == s.current && c.pending_reward);
    if !pending {
        out.push(Action::EndTurn);
    }
    out
}

/// Tiles a unit standing on `q` sees: Chebyshev 1, or 2 on a mountain.
pub fn vision(s: &State, q: Pos) -> Vec<Pos> {
    let r = if s.tile(q).terrain == Terrain::Mountain {
        2
    } else {
        1
    };
    all_pos(s.size)
        .into_iter()
        .filter(|&t| cheb(t, q) <= r)
        .collect()
}

pub fn explored_set(s: &State, player: PlayerId) -> Vec<Pos> {
    all_pos(s.size)
        .into_iter()
        .filter(|&q| s.players[player as usize].explored[idx(s, q)])
        .collect()
}

pub fn phase_rank(ph: Phase) -> u8 {
    match ph {
        Phase::Fresh => 0,
        Phase::Moved => 1,
        Phase::Attacked => 2,
        Phase::Done => 3,
    }
}

pub fn sign(x: i32) -> i8 {
    x.signum() as i8
}

/// Healths after an attack, via the milestone 1 combat function (validated by its own goldens)
/// and the test's own defense bonus.
pub fn expected_combat(s: &State, attacker: UnitId, defender: UnitId) -> (u32, u32) {
    let a = s.units.iter().find(|u| u.id == attacker).unwrap();
    let d = s.units.iter().find(|u| u.id == defender).unwrap();
    let (sa, sd) = (s.tables.unit(a.kind), s.tables.unit(d.kind));
    let ca = Combatant {
        attack: sa.attack,
        defense: sa.defense,
        health: a.health,
        max_health: a.max_health,
    };
    let cd = Combatant {
        attack: sd.attack,
        defense: sd.defense,
        health: d.health,
        max_health: d.max_health,
    };
    let in_range = cheb(a.pos, d.pos) <= sd.range;
    let out = resolve_attack(&ca, &cd, expected_bonus(s, defender), in_range);
    (out.attacker_health, out.defender_health)
}

// ---------- random boards ----------

const KINDS: [UnitType; 5] = [
    UnitType::Warrior,
    UnitType::Rider,
    UnitType::Archer,
    UnitType::Defender,
    UnitType::SuperUnit,
];
const TECHS: [Tech; 4] = [Tech::Climbing, Tech::Archery, Tech::Riding, Tech::Hunting];

/// A random scripted board (drawn from the test's own `Rng`): 6x6 to 9x9, random terrain, one
/// city per player (some walled), a few villages, 2 to 4 units per player of every type with
/// random health, kills and veteran status, random techs, and either everything explored or a
/// random partial mask (always including the player's own unit tiles).
pub fn random_board(seed: u64) -> State {
    let mut rng = Rng::new(seed ^ 0x05ee_d0fb_0a2d);
    let tables = Tables::load().unwrap();
    let health = |kind: UnitType| tables.unit(kind).health;
    let n = 6 + rng.gen_range(4) as u8;
    let mut b = StateBuilder::new(n);
    for q in all_pos(n) {
        match rng.gen_range(10) {
            0..=2 => b = b.terrain(q, Terrain::Forest),
            3..=4 => b = b.terrain(q, Terrain::Mountain),
            _ => {}
        }
    }
    let mut free = all_pos(n);
    let mut take = |rng: &mut Rng| free.swap_remove(rng.gen_range(free.len() as u32) as usize);
    for owner in 0..2u8 {
        let q = take(&mut rng);
        let wall = rng.gen_range(2) == 0;
        b = b.city_with(owner, q, |c| c.wall = wall);
        if rng.gen_range(2) == 0 {
            // Leave the city tile free sometimes; put a unit there other times.
            let kind = KINDS[rng.gen_range(4) as usize];
            b = b.unit(owner, kind, q);
        }
    }
    for _ in 0..2 {
        let q = take(&mut rng);
        b = b.village(q);
    }
    let mut unit_tiles: [Vec<Pos>; 2] = [Vec::new(), Vec::new()];
    for owner in 0..2u8 {
        for _ in 0..2 + rng.gen_range(3) {
            let q = take(&mut rng);
            let kind = KINDS[rng.gen_range(5) as usize];
            let max = health(kind) + if rng.gen_range(4) == 0 { 5 } else { 0 };
            let hp = 1 + rng.gen_range(max);
            let kills = rng.gen_range(4);
            let veteran = max > health(kind);
            let promo = kills >= 3 && !veteran && kind != UnitType::SuperUnit;
            b = b.unit_with(owner, kind, q, |u| {
                u.max_health = max;
                u.health = hp;
                u.kills = kills;
                u.veteran = veteran;
                u.promotion_available = promo;
            });
            unit_tiles[owner as usize].push(q);
        }
        let techs: Vec<Tech> = TECHS
            .iter()
            .copied()
            .filter(|_| rng.gen_range(2) == 0)
            .collect();
        b = b.techs(owner, &techs);
    }
    if rng.gen_range(3) == 0 {
        b = b.explore_all();
    } else {
        for owner in 0..2u8 {
            let mut tiles: Vec<Pos> = all_pos(n)
                .into_iter()
                .filter(|_| rng.gen_range(10) < 7)
                .collect();
            tiles.extend(unit_tiles[owner as usize].iter().copied());
            b = b.explore(owner, &tiles);
        }
    }
    b = b.current(rng.gen_range(2) as u8).turn(rng.gen_range(5));
    if rng.gen_range(5) == 0 {
        b = b.config(Config {
            full_observability: true,
            turn_cap: 50,
        });
    }
    b.build()
}

/// Picks an action uniformly among legal actions, except EndTurn is taken with probability
/// 1/8 whenever another action exists (so games advance without ending turns too early).
pub fn pick(rng: &mut Rng, legal: &[Action]) -> Action {
    let others: Vec<Action> = legal
        .iter()
        .copied()
        .filter(|a| *a != Action::EndTurn)
        .collect();
    if others.is_empty() || (legal.contains(&Action::EndTurn) && rng.gen_range(8) == 0) {
        return Action::EndTurn;
    }
    others[rng.gen_range(others.len() as u32) as usize]
}
