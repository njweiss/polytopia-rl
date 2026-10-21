//! Shared helpers for the milestone 3 and 4 tests. Every checker here is written from the rules in
//! docs/MILESTONE_3 and docs/MILESTONE_4 (with the spec's tile-action notes that overrule D4)
//! independently of the code under test: it never calls `reachable_tiles`, `defense_bonus`,
//! `legal_actions`, `apply` or anything in `sim_core::economy` to compute an expectation.
#![allow(dead_code)]

use sim_core::actions::Action;
use sim_core::combat::{resolve_attack, Combatant};
use sim_core::data::{Reward, SiteAction, Skill, Tables, Tech, UnitType, ALL_TECHS, ALL_UNITS};
use sim_core::grid::Pos;
use sim_core::map::Resource;
use sim_core::map::Terrain;
use sim_core::rng::Rng;
use sim_core::state::{
    City, CityId, Config, Improvement, Phase, PlayerId, State, TileState, Unit, UnitId,
};
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

/// Legal actions enumerated straight from the briefs, in canonical order. Uses `bf_reach`.
/// Milestone 4: a pending reward of the current player allows only the lowest-id pending city's
/// ChooseReward (menu order); otherwise units, then Train per city, then site actions per tile,
/// then Research, then EndTurn.
pub fn bf_legal(s: &State) -> Vec<Action> {
    let mut out = Vec::new();
    if s.outcome.is_some() {
        return out;
    }
    let mut mine: Vec<&City> = s.cities.iter().filter(|c| c.owner == s.current).collect();
    mine.sort_by_key(|c| c.id);
    if let Some(c) = mine.iter().find(|c| c.pending_reward) {
        for reward in exp_menu(c) {
            out.push(Action::ChooseReward { city: c.id, reward });
        }
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
    let me = &s.players[s.current as usize];
    for c in &mine {
        for kind in ALL_UNITS {
            let st = s.tables.unit(kind);
            let Some(cost) = st.cost else { continue };
            let unlocked = st.unlocked_by.is_none_or(|t| me.techs.contains(t));
            let supported = s.units.iter().filter(|u| u.origin == Some(c.id)).count() as u32;
            if unlocked
                && s.units.iter().all(|u| u.pos != c.pos)
                && me.stars >= cost
                && supported < c.level + 1
            {
                out.push(Action::Train { city: c.id, kind });
            }
        }
    }
    for q in all_pos(s.size) {
        for action in EXP_SITE_ACTIONS {
            if exp_site_legal(s, q, action) {
                out.push(Action::Site { at: q, action });
            }
        }
    }
    for tech in ALL_TECHS {
        let prereq = s.tables.tech(tech).requires;
        if !me.techs.contains(tech)
            && prereq.is_none_or(|r| me.techs.contains(r))
            && me.stars >= exp_research_cost(s, s.current, tech)
        {
            out.push(Action::Research { tech });
        }
    }
    out.push(Action::EndTurn);
    out
}

// ---------- milestone 4 rules, replayed from the brief ----------

/// The brief's `ALL_SITE_ACTIONS` order, written out.
pub const EXP_SITE_ACTIONS: [SiteAction; 6] = [
    SiteAction::HarvestAnimal,
    SiteAction::HarvestFruit,
    SiteAction::LumberHut,
    SiteAction::ClearForest,
    SiteAction::Farm,
    SiteAction::Mine,
];

/// (cost, tech, population, stars) copied from the brief's Appendix `site_actions.json`.
pub fn exp_site_info(a: SiteAction) -> (u32, Tech, u32, u32) {
    match a {
        SiteAction::HarvestAnimal => (2, Tech::Hunting, 1, 0),
        SiteAction::HarvestFruit => (2, Tech::Organization, 1, 0),
        SiteAction::LumberHut => (3, Tech::Forestry, 1, 0),
        SiteAction::ClearForest => (0, Tech::Forestry, 0, 1),
        SiteAction::Farm => (5, Tech::Farming, 2, 0),
        SiteAction::Mine => (5, Tech::Mining, 2, 0),
    }
}

/// `L(L+1)/2 - 1`.
pub fn exp_cum(level: u32) -> u32 {
    level * (level + 1) / 2 - 1
}

/// The brief's menu table (reduced menus, D6), BorderGrowth dropped once used.
pub fn exp_menu(c: &City) -> Vec<Reward> {
    let base = match c.level {
        0 | 1 => vec![],
        2 => vec![Reward::Workshop],
        3 => vec![Reward::CityWall, Reward::Resources],
        4 => vec![Reward::PopulationGrowth, Reward::BorderGrowth],
        _ => vec![Reward::Park],
    };
    base.into_iter()
        .filter(|r| !(*r == Reward::BorderGrowth && c.border_growth_used))
        .collect()
}

pub fn exp_income(s: &State, player: PlayerId) -> u32 {
    s.cities
        .iter()
        .filter(|c| c.owner == player)
        .map(|c| c.level + u32::from(c.is_capital) + u32::from(c.workshop) + c.parks)
        .sum()
}

pub fn exp_research_cost(s: &State, player: PlayerId, tech: Tech) -> u32 {
    let cities = s.cities.iter().filter(|c| c.owner == player).count() as u32;
    4 + cities * s.tables.tech(tech).tier
}

/// Site-action legality, with the spec's tile-action notes (they overrule the brief's D4 table):
/// LumberHut needs forest without an improvement (an animal may stay); ClearForest needs forest
/// without an improvement (an animal is forfeited); Farm and Mine need their resource and no
/// improvement; harvests need only their resource.
pub fn exp_site_legal(s: &State, q: Pos, a: SiteAction) -> bool {
    let n = s.size;
    if q.row >= n || q.col >= n {
        return false;
    }
    let t = s.tile(q);
    let Some(owner) = t.owner_city else {
        return false;
    };
    let cond = match a {
        SiteAction::HarvestAnimal => t.resource == Some(Resource::Animal),
        SiteAction::HarvestFruit => t.resource == Some(Resource::Fruit),
        SiteAction::LumberHut | SiteAction::ClearForest => {
            t.terrain == Terrain::Forest && t.improvement.is_none()
        }
        SiteAction::Farm => t.resource == Some(Resource::Crop) && t.improvement.is_none(),
        SiteAction::Mine => t.resource == Some(Resource::Metal) && t.improvement.is_none(),
    };
    if !cond {
        return false;
    }
    let city = s.cities.iter().find(|c| c.id == owner).unwrap();
    let (cost, tech, _, _) = exp_site_info(a);
    let me = &s.players[s.current as usize];
    let enemy_here = s.units.iter().any(|u| u.pos == q && u.owner != s.current);
    city.owner == s.current && !enemy_here && me.techs.contains(tech) && me.stars >= cost
}

/// The level-up check: one level at a time, only while no reward is pending.
pub fn exp_level_check(c: &mut City) {
    if !c.pending_reward && c.population >= exp_cum(c.level + 1) {
        c.level += 1;
        c.pending_reward = true;
    }
}

fn exp_city_mut(s: &mut State, id: CityId) -> &mut City {
    s.cities.iter_mut().find(|c| c.id == id).unwrap()
}

/// The state after a legal milestone 4 action (Train, Site, Research, ChooseReward or EndTurn),
/// replayed from the brief. Panics on unit actions.
pub fn exp_after(prev: &State, a: &Action) -> State {
    let mut s = prev.clone();
    let cur = s.current as usize;
    match *a {
        Action::Train { city, kind } => {
            let cost = s.tables.unit(kind).cost.unwrap();
            let health = s.tables.unit(kind).health;
            let pos = s.cities.iter().find(|c| c.id == city).unwrap().pos;
            s.players[cur].stars -= cost;
            let id = UnitId(s.next_unit_id);
            s.next_unit_id += 1;
            s.units.push(Unit {
                id,
                kind,
                owner: s.current,
                pos,
                health,
                max_health: health,
                origin: Some(city),
                kills: 0,
                veteran: false,
                promotion_available: false,
                phase: Phase::Done,
                last_dir: None,
            });
            tile_mut(&mut s, pos).unit = Some(id);
        }
        Action::Site { at, action } => {
            let (cost, _, pop, stars) = exp_site_info(action);
            s.players[cur].stars = s.players[cur].stars - cost + stars;
            let t = tile_mut(&mut s, at);
            match action {
                SiteAction::HarvestAnimal | SiteAction::HarvestFruit => t.resource = None,
                SiteAction::LumberHut => t.improvement = Some(Improvement::LumberHut),
                SiteAction::ClearForest => {
                    t.terrain = Terrain::Field;
                    t.resource = None;
                }
                SiteAction::Farm => t.improvement = Some(Improvement::Farm),
                SiteAction::Mine => t.improvement = Some(Improvement::Mine),
            }
            let owner = t.owner_city.unwrap();
            let c = exp_city_mut(&mut s, owner);
            c.population += pop;
            exp_level_check(c);
        }
        Action::Research { tech } => {
            s.players[cur].stars -= exp_research_cost(prev, prev.current, tech);
            s.players[cur].techs.insert(tech);
        }
        Action::ChooseReward { city, reward } => {
            let c = exp_city_mut(&mut s, city);
            match reward {
                Reward::Workshop => c.workshop = true,
                Reward::CityWall => c.wall = true,
                Reward::Park => c.parks += 1,
                Reward::Resources => {}
                Reward::PopulationGrowth => c.population += 3,
                Reward::BorderGrowth => c.border_growth_used = true,
            }
            c.pending_reward = false;
            exp_level_check(c);
            if reward == Reward::Resources {
                s.players[cur].stars += 5;
            }
            if reward == Reward::BorderGrowth {
                let at = prev.cities.iter().find(|c| c.id == city).unwrap().pos;
                for q in all_pos(s.size) {
                    let i = idx(&s, q);
                    let t = &mut s.tiles[i];
                    if cheb(q, at) <= 2 && t.owner_city.is_none() && !t.village && t.city.is_none()
                    {
                        t.owner_city = Some(city);
                        s.players[cur].explored[i] = true;
                    }
                }
            }
        }
        Action::EndTurn => {
            s.current = 1 - s.current;
            if s.current == 0 {
                s.turn += 1;
            }
            let now = s.current;
            if s.turn >= 1 {
                s.players[now as usize].stars += exp_income(prev, now);
            }
            for u in s.units.iter_mut().filter(|u| u.owner == now) {
                u.phase = Phase::Fresh;
            }
        }
        _ => panic!("exp_after handles only economy actions and EndTurn, got {a:?}"),
    }
    s
}

/// Milestone 4 invariants on a single state (brief "Tests" item 6).
pub fn check_economy_invariants(s: &State, ctx: &str) {
    for c in &s.cities {
        let supported = s.units.iter().filter(|u| u.origin == Some(c.id)).count() as u32;
        assert!(
            supported <= c.level + 1,
            "{ctx}: city {:?} supports {supported} units at level {}",
            c.id,
            c.level
        );
        assert!(c.level >= 1, "{ctx}: {c:?}");
        if c.pending_reward {
            assert!(
                c.population >= exp_cum(c.level),
                "{ctx}: pending but below its level: {c:?}"
            );
            assert_eq!(
                c.owner, s.current,
                "{ctx}: pending reward of a non-current player"
            );
            assert!(c.level >= 2, "{ctx}: {c:?}");
        } else {
            assert!(
                c.population < exp_cum(c.level + 1),
                "{ctx}: missed level-up: {c:?}"
            );
        }
    }
    for pl in &s.players {
        // `stars` is unsigned; a wrapped subtraction would show up as a huge number.
        assert!(pl.stars < 1_000_000, "{ctx}: stars {}", pl.stars);
    }
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

/// A random scripted board with an economy (drawn from the test's own `Rng`): like
/// `random_board`, plus resources and improvements consistent with the terrain, one or two cities
/// per player at levels 1 to 6 with populations that satisfy the D5 invariants (pending rewards
/// only for the current player's cities, sometimes with enough population to cascade), buildings,
/// stars, random techs, and units whose origins respect each city's capacity.
pub fn random_economy_board(seed: u64) -> State {
    let mut rng = Rng::new(seed ^ 0x0ec0_0b0a_2d00);
    let tables = Tables::load().unwrap();
    let n = 6 + rng.gen_range(4) as u8;
    let current = rng.gen_range(2) as u8;
    let mut b = StateBuilder::new(n).current(current);
    let mut free = all_pos(n);
    let mut take = |rng: &mut Rng| free.swap_remove(rng.gen_range(free.len() as u32) as usize);

    // Cities first (on field tiles), so terrain and resources go around them.
    let mut cities: Vec<(PlayerId, Pos, u32)> = Vec::new(); // owner, pos, level
    for owner in 0..2u8 {
        for k in 0..1 + rng.gen_range(2) {
            let q = take(&mut rng);
            let level = 1 + rng.gen_range(6);
            let pending = owner == current && level >= 2 && rng.gen_range(4) == 0;
            let pop = if pending {
                exp_cum(level) + rng.gen_range(level + 3)
            } else {
                exp_cum(level) + rng.gen_range(level + 1)
            };
            let (workshop, wall, used) = (
                rng.gen_range(3) == 0,
                rng.gen_range(3) == 0,
                rng.gen_range(3) == 0,
            );
            let parks = if level >= 5 { rng.gen_range(3) } else { 0 };
            b = b.city_with(owner, q, |c| {
                c.level = level;
                c.population = pop;
                c.pending_reward = pending;
                c.is_capital = k == 0;
                c.workshop = workshop;
                c.wall = wall;
                c.border_growth_used = used;
                c.parks = parks;
            });
            cities.push((owner, q, level));
        }
    }
    let mut special: Vec<Pos> = cities.iter().map(|c| c.1).collect();
    for _ in 0..1 + rng.gen_range(2) {
        let q = take(&mut rng);
        b = b.village(q);
        special.push(q);
    }
    for q in all_pos(n) {
        if special.contains(&q) {
            continue; // city or village tile: plain field
        }
        let roll = rng.gen_range(20);
        match rng.gen_range(10) {
            0..=2 => {
                b = b.terrain(q, Terrain::Forest);
                if roll < 8 {
                    b = b.resource(q, Resource::Animal);
                }
                if roll.is_multiple_of(5) {
                    b = b.improvement(q, Improvement::LumberHut);
                }
            }
            3..=4 => {
                b = b.terrain(q, Terrain::Mountain);
                if roll < 8 {
                    b = b.resource(q, Resource::Metal);
                    if roll < 2 {
                        b = b.improvement(q, Improvement::Mine);
                    }
                }
            }
            _ => {
                if roll < 5 {
                    b = b.resource(q, Resource::Fruit);
                } else if roll < 10 {
                    b = b.resource(q, Resource::Crop);
                    if roll < 7 {
                        b = b.improvement(q, Improvement::Farm);
                    }
                }
            }
        }
    }

    // Units: some on city tiles, origins only where the city still has capacity.
    let mut supported = vec![0u32; cities.len()];
    let mut unit_tiles: [Vec<Pos>; 2] = [Vec::new(), Vec::new()];
    for owner in 0..2u8 {
        for _ in 0..2 + rng.gen_range(3) {
            let on_city = rng.gen_range(4) == 0;
            let own: Vec<usize> = (0..cities.len())
                .filter(|&i| {
                    cities[i].0 == owner && unit_tiles.iter().flatten().all(|&t| t != cities[i].1)
                })
                .collect();
            let q = if on_city && !own.is_empty() {
                cities[own[rng.gen_range(own.len() as u32) as usize]].1
            } else {
                take(&mut rng)
            };
            let kind = if rng.gen_range(12) == 0 {
                UnitType::SuperUnit
            } else {
                KINDS[rng.gen_range(4) as usize]
            };
            let home = rng.gen_range(cities.len() as u32) as usize;
            let origin = (kind != UnitType::SuperUnit
                && cities[home].0 == owner
                && supported[home] < cities[home].2 + 1
                && rng.gen_range(4) != 0)
                .then(|| {
                    supported[home] += 1;
                    CityId(home as u32)
                });
            let max = tables.unit(kind).health;
            let hp = 1 + rng.gen_range(max);
            b = b.unit_with(owner, kind, q, |u| {
                u.health = hp;
                u.origin = origin;
                u.phase = [Phase::Fresh, Phase::Moved, Phase::Attacked, Phase::Done]
                    [rng.gen_range(4) as usize];
            });
            unit_tiles[owner as usize].push(q);
        }
        let techs: Vec<Tech> = ALL_TECHS
            .iter()
            .copied()
            .filter(|_| rng.gen_range(2) == 0)
            .collect();
        let stars = rng.gen_range(40);
        b = b.techs(owner, &techs).stars(owner, stars);
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
    b.turn(rng.gen_range(5)).build()
}

/// Milestone 4 picker. ChooseReward (when pending, the only kind) uniformly. Otherwise EndTurn
/// with probability 1/8 when something else exists, else a category among the non-empty ones
/// with weights units 4, Train 1, Site 2, Research 1, then uniformly inside it.
pub fn pick_weighted(rng: &mut Rng, legal: &[Action]) -> Action {
    let mut groups: [Vec<Action>; 4] = Default::default();
    let mut rewards = Vec::new();
    for &a in legal {
        match a {
            Action::Move { .. }
            | Action::Attack { .. }
            | Action::Recover { .. }
            | Action::Promote { .. } => groups[0].push(a),
            Action::Train { .. } => groups[1].push(a),
            Action::Site { .. } => groups[2].push(a),
            Action::Research { .. } => groups[3].push(a),
            Action::ChooseReward { .. } => rewards.push(a),
            Action::EndTurn => {}
        }
    }
    if !rewards.is_empty() {
        return rewards[rng.gen_range(rewards.len() as u32) as usize];
    }
    let weights = [4u32, 1, 2, 1];
    let total: u32 = (0..4)
        .filter(|&i| !groups[i].is_empty())
        .map(|i| weights[i])
        .sum();
    if total == 0 || (legal.contains(&Action::EndTurn) && rng.gen_range(8) == 0) {
        return Action::EndTurn;
    }
    let mut r = rng.gen_range(total);
    for i in 0..4 {
        if groups[i].is_empty() {
            continue;
        }
        if r < weights[i] {
            return groups[i][rng.gen_range(groups[i].len() as u32) as usize];
        }
        r -= weights[i];
    }
    unreachable!()
}
