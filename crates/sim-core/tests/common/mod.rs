//! Shared helpers for the milestone 3 to 6 tests. Every checker here is written from the rules
//! in docs/MILESTONE_3, docs/MILESTONE_4 (with the spec's tile-action notes that overrule D4),
//! docs/MILESTONE_5 and docs/MILESTONE_6 independently of the code under test: it never calls
//! `reachable_tiles`, `defense_bonus`, `legal_actions`, `apply`, anything in `sim_core::economy`,
//! `sim_core::explorer`, `sim_core::spawn` or `sim_core::score` to compute an expectation.
#![allow(dead_code)]

use sim_core::actions::Action;
use sim_core::combat::{resolve_attack, Combatant};
use sim_core::data::{Reward, SiteAction, Skill, Tables, Tech, UnitType, ALL_TECHS, ALL_UNITS};
use sim_core::grid::Pos;
use sim_core::map::Resource;
use sim_core::map::Terrain;
use sim_core::rng::Rng;
use sim_core::state::{
    City, CityId, Config, Dir, Improvement, Outcome, Phase, PlayerId, State, TileState, Unit,
    UnitId,
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
/// ChooseReward (menu order); otherwise units (per unit Move, Attack, Capture (milestone 6),
/// Recover, Promote), then Train per city, then site actions per tile, then Research, then
/// EndTurn. Nothing once the game has an outcome.
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
        if exp_capture_legal(s, u) {
            out.push(Action::Capture { unit: id });
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
            let supported = exp_supported(s, c.id);
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
        // docs/MILESTONE_15 "Data files" (v2 only).
        SiteAction::BurnForest => (3, Tech::Construction, 0, 0),
    }
}

/// `L(L+1)/2 - 1`.
pub fn exp_cum(level: u32) -> u32 {
    level * (level + 1) / 2 - 1
}

/// The milestone 5 menu table (replaces D6 of milestone 4), BorderGrowth dropped once used.
pub fn exp_menu(c: &City) -> Vec<Reward> {
    let base = match c.level {
        0 | 1 => vec![],
        2 => vec![Reward::Workshop, Reward::Explorer],
        3 => vec![Reward::CityWall, Reward::Resources],
        4 => vec![Reward::PopulationGrowth, Reward::BorderGrowth],
        _ => vec![Reward::Park, Reward::SuperUnit],
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

/// Living units with `origin == city`, Super Units excluded (milestone 5 rule 3).
pub fn exp_supported(s: &State, city: CityId) -> u32 {
    s.units
        .iter()
        .filter(|u| u.origin == Some(city) && u.kind != UnitType::SuperUnit)
        .count() as u32
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
        // Never legal before milestone 17 (docs/MILESTONE_15 V4).
        SiteAction::BurnForest => false,
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

/// The state after a legal milestone 4 action (Train, Site, Research, ChooseReward or EndTurn,
/// with milestone 6's turn cap) or a milestone 6 Capture, replayed from the briefs. Panics on the
/// other unit actions.
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
                SiteAction::BurnForest => unreachable!("never legal before milestone 17"),
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
                Reward::Explorer | Reward::SuperUnit => {}
            }
            c.pending_reward = false;
            exp_level_check(c);
            if reward == Reward::Resources {
                s.players[cur].stars += 5;
            }
            // The walk and the spawn touch no city field, so running them after the level-up
            // check gives the same state as the brief's order.
            if reward == Reward::Explorer {
                exp_explorer_walk(&mut s, city);
            }
            if reward == Reward::SuperUnit {
                exp_spawn_super_unit(&mut s, city);
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
        Action::Capture { unit } => exp_capture(&mut s, unit),
        Action::EndTurn if prev.current == 1 && prev.turn + 1 == prev.config.turn_cap => {
            // Milestone 6 turn cap: no income, no unit reset; the higher score wins.
            s.turn += 1;
            s.current = 0;
            s.outcome = Some(exp_outcome_by_score(prev));
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
        _ => panic!("exp_after handles only economy actions, Capture and EndTurn, got {a:?}"),
    }
    s
}

/// Milestone 4 invariants on a single state (brief "Tests" item 6); capacity ignores Super Units
/// (milestone 5 rule 3).
pub fn check_economy_invariants(s: &State, ctx: &str) {
    for c in &s.cities {
        let supported = exp_supported(s, c.id);
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

// ---------- milestone 5 rules, replayed from the brief ----------

/// In-bounds 8-neighbors of `q`, row-major.
pub fn nbrs(n: u8, q: Pos) -> Vec<Pos> {
    let mut out = Vec::new();
    for dr in -1i32..=1 {
        for dc in -1i32..=1 {
            let (r, c) = (q.row as i32 + dr, q.col as i32 + dc);
            if (dr, dc) != (0, 0) && r >= 0 && c >= 0 && r < n as i32 && c < n as i32 {
                out.push(p(r as u8, c as u8));
            }
        }
    }
    out
}

/// Breadth-first step counts from `from` over tiles where `ok` holds (8-neighbors, each step 1),
/// row-major indexed; `u32::MAX` where unreachable.
pub fn bfs(n: u8, from: Pos, ok: &dyn Fn(Pos) -> bool) -> Vec<u32> {
    let at = |q: Pos| q.row as usize * n as usize + q.col as usize;
    let mut d = vec![u32::MAX; n as usize * n as usize];
    d[at(from)] = 0;
    let mut frontier = vec![from];
    let mut k = 0;
    while !frontier.is_empty() {
        k += 1;
        let mut next = Vec::new();
        for &f in &frontier {
            for q in nbrs(n, f) {
                if ok(q) && d[at(q)] == u32::MAX {
                    d[at(q)] = k;
                    next.push(q);
                }
            }
        }
        frontier = next;
    }
    d
}

/// The brief's tie rule: no draw for a single option, else `rng.gen_range(len)` over the
/// (row-major) list.
fn exp_pick(rng: &mut Rng, options: &[Pos]) -> Pos {
    if options.len() == 1 {
        options[0]
    } else {
        options[rng.gen_range(options.len() as u32) as usize]
    }
}

/// The Explorer walk replayed from docs/MILESTONE_5 "Explorer walk" (steps 0 to 7, E1 to E5).
/// Mutates the city owner's explored mask and `s.rng` only. Returns the tiles stood on: the start,
/// then one per move.
pub fn exp_explorer_walk(s: &mut State, city: CityId) -> Vec<Pos> {
    let c = s.cities.iter().find(|c| c.id == city).unwrap().clone();
    let owner = c.owner as usize;
    let n = s.size;
    let at = |q: Pos| q.row as usize * n as usize + q.col as usize;
    let climbing = has_tech(s, c.owner, Tech::Climbing);
    let passable: Vec<bool> = all_pos(n)
        .iter()
        .map(|&q| s.tile(q).terrain != Terrain::Mountain || climbing)
        .collect();
    let ok = |q: Pos| passable[at(q)];
    let see = |s: &mut State, from: Pos| {
        for q in all_pos(n) {
            if cheb(q, from) <= 1 {
                s.players[owner].explored[at(q)] = true;
            }
        }
    };
    let fog = |s: &State, t: Pos| {
        nbrs(n, t)
            .into_iter()
            .chain([t])
            .filter(|&q| !s.players[owner].explored[at(q)])
            .count() as u32
    };

    let mut cur = c.pos;
    let mut visited = vec![false; n as usize * n as usize];
    visited[at(cur)] = true;
    see(s, cur);
    let mut path = vec![cur];
    for _ in 0..15 {
        let d = bfs(n, cur, &ok);
        let cands: Vec<Pos> = all_pos(n)
            .into_iter()
            .filter(|&t| ok(t) && d[at(t)] >= 1 && d[at(t)] <= 4 && fog(s, t) >= 1)
            .collect();
        let next = if let Some(dmin) = cands.iter().map(|&t| d[at(t)]).min() {
            let score = |t: Pos| fog(s, t).min(4) as f32 - if visited[at(t)] { 0.5 } else { 0.0 };
            let near: Vec<Pos> = cands.into_iter().filter(|&t| d[at(t)] == dmin).collect();
            let best = near.iter().map(|&t| score(t)).fold(f32::MIN, f32::max);
            let top: Vec<Pos> = near.into_iter().filter(|&t| score(t) == best).collect();
            let target = exp_pick(&mut s.rng, &top);
            let dt = bfs(n, target, &ok);
            let steps: Vec<Pos> = nbrs(n, cur)
                .into_iter()
                .filter(|&q| ok(q) && dt[at(q)] == d[at(target)] - 1)
                .collect();
            exp_pick(&mut s.rng, &steps)
        } else {
            let open: Vec<Pos> = nbrs(n, cur).into_iter().filter(|&q| ok(q)).collect();
            let fresh: Vec<Pos> = open.iter().copied().filter(|&q| !visited[at(q)]).collect();
            let set = if fresh.is_empty() { open } else { fresh };
            if set.is_empty() {
                cur
            } else {
                exp_pick(&mut s.rng, &set)
            }
        };
        cur = next;
        visited[at(cur)] = true;
        see(s, cur);
        path.push(cur);
    }
    path
}

/// The brief's clockwise ring: N, NE, E, SE, S, SW, W, NW as (dr, dc).
pub const RING: [(i8, i8); 8] = [
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
];

/// Push direction from the brief: own `last_dir`, the opposite of an enemy's, or toward the center
/// `(size - 1) / 2` by the sign per axis (south on the exact center).
pub fn exp_push_dir(s: &State, occupant: &Unit, city_owner: PlayerId) -> Dir {
    match occupant.last_dir {
        Some(d) if occupant.owner == city_owner => d,
        Some(d) => Dir {
            dr: -d.dr,
            dc: -d.dc,
        },
        None => {
            let center = (s.size as i32 - 1) / 2;
            let dr = sign(center - occupant.pos.row as i32);
            let dc = sign(center - occupant.pos.col as i32);
            if (dr, dc) == (0, 0) {
                Dir { dr: 1, dc: 0 }
            } else {
                Dir { dr, dc }
            }
        }
    }
}

/// Probe order around the ring from the intended direction: 0, -1 (ccw), +1 (cw), -2, +2, -3, +3,
/// 4 (opposite).
pub fn exp_probe_order(d: Dir) -> Vec<Dir> {
    let i = RING.iter().position(|&r| r == (d.dr, d.dc)).unwrap() as i32;
    [0, -1, 1, -2, 2, -3, 3, 4]
        .iter()
        .map(|k| {
            let (dr, dc) = RING[(i + k).rem_euclid(8) as usize];
            Dir { dr, dc }
        })
        .collect()
}

/// `ChooseReward { SuperUnit }`'s spawn replayed from the brief: push (or remove) the occupant,
/// then create the Super Unit. Returns its id.
pub fn exp_spawn_super_unit(s: &mut State, city: CityId) -> UnitId {
    let c = s.cities.iter().find(|c| c.id == city).unwrap().clone();
    if let Some(occ) = s.units.iter().find(|u| u.pos == c.pos).cloned() {
        let climbing = has_tech(s, occ.owner, Tech::Climbing);
        let n = s.size as i32;
        let dest = exp_probe_order(exp_push_dir(s, &occ, c.owner))
            .into_iter()
            .map(|d| {
                (
                    c.pos.row as i32 + d.dr as i32,
                    c.pos.col as i32 + d.dc as i32,
                )
            })
            .filter(|&(r, col)| r >= 0 && col >= 0 && r < n && col < n)
            .map(|(r, col)| p(r as u8, col as u8))
            .find(|&q| {
                s.units.iter().all(|u| u.pos != q)
                    && (s.tile(q).terrain != Terrain::Mountain || climbing)
            });
        tile_mut(s, c.pos).unit = None;
        match dest {
            Some(q) => {
                tile_mut(s, q).unit = Some(occ.id);
                s.units.iter_mut().find(|u| u.id == occ.id).unwrap().pos = q;
                for t in vision(s, q) {
                    let i = idx(s, t);
                    s.players[occ.owner as usize].explored[i] = true;
                }
            }
            None => s.units.retain(|u| u.id != occ.id),
        }
    }
    let health = s.tables.unit(UnitType::SuperUnit).health;
    let id = UnitId(s.next_unit_id);
    s.next_unit_id += 1;
    s.units.push(Unit {
        id,
        kind: UnitType::SuperUnit,
        owner: c.owner,
        pos: c.pos,
        health,
        max_health: health,
        origin: Some(city),
        kills: 0,
        veteran: false,
        promotion_available: false,
        phase: Phase::Done,
        last_dir: None,
    });
    tile_mut(s, c.pos).unit = Some(id);
    id
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

// ---------- milestone 6 rules, replayed from the brief ----------

/// Unit cost in stars from the spec's unit table (Super Unit: none).
pub fn exp_unit_cost(kind: UnitType) -> Option<u32> {
    match kind {
        UnitType::Warrior => Some(2),
        UnitType::Rider | UnitType::Archer | UnitType::Defender => Some(3),
        UnitType::SuperUnit => None,
        // docs/MILESTONE_15 "Data files" (v2 only).
        UnitType::Swordsman => Some(5),
        UnitType::Catapult | UnitType::Knight => Some(8),
    }
}

/// Tech tier from the spec's tech table.
pub fn exp_tier(tech: Tech) -> u32 {
    match tech {
        Tech::Climbing | Tech::Hunting | Tech::Organization | Tech::Riding => 1,
        Tech::Archery | Tech::Forestry | Tech::Farming | Tech::Strategy | Tech::Mining => 2,
        // docs/MILESTONE_15 "Data files" (v2 only).
        Tech::Roads | Tech::FreeSpirit => 2,
        Tech::Trade | Tech::Chivalry | Tech::Mathematics | Tech::Construction | Tech::Smithery => 3,
    }
}

/// The brief's score terms `[units, territory, exploration, cities, parks, techs]`, with the
/// weights from `s.config.score_weights`.
pub fn exp_score_terms(s: &State, player: PlayerId) -> [u32; 6] {
    let w = s.config.score_weights;
    let units = s
        .units
        .iter()
        .filter(|u| u.owner == player)
        .map(|u| match exp_unit_cost(u.kind) {
            Some(cost) => w.unit_per_star * cost,
            None => w.super_unit,
        })
        .sum();
    let owner_of = |id: CityId| s.cities.iter().find(|c| c.id == id).unwrap().owner;
    let tiles = s
        .tiles
        .iter()
        .filter(|t| t.owner_city.is_some_and(|c| owner_of(c) == player))
        .count() as u32;
    let seen = s.players[player as usize]
        .explored
        .iter()
        .filter(|&&e| e)
        .count() as u32;
    let mine: Vec<&City> = s.cities.iter().filter(|c| c.owner == player).collect();
    let cities = mine
        .iter()
        .map(|c| {
            let surplus = c.population.saturating_sub(exp_cum(c.level));
            w.city_base + w.city_per_level * (c.level - 1) + w.population_surplus * surplus
        })
        .sum();
    let parks: u32 = mine.iter().map(|c| c.parks).sum();
    let techs = ALL_TECHS
        .iter()
        .filter(|&&t| s.players[player as usize].techs.contains(t))
        .map(|&t| w.tech_per_tier * exp_tier(t))
        .sum();
    [
        units,
        w.territory_per_tile * tiles,
        w.exploration_per_tile * seen,
        cities,
        w.park * parks,
        techs,
    ]
}

pub fn exp_score(s: &State, player: PlayerId) -> u32 {
    exp_score_terms(s, player).iter().sum()
}

/// `Win(p)` for the strictly higher score, `Draw` on equal scores.
pub fn exp_outcome_by_score(s: &State) -> Outcome {
    let (a, b) = (exp_score(s, 0), exp_score(s, 1));
    if a > b {
        Outcome::Win(0)
    } else if b > a {
        Outcome::Win(1)
    } else {
        Outcome::Draw
    }
}

/// Capture legality apart from the outcome, pending-reward and current-player checks: phase
/// Fresh, and the unit's tile is a village or holds a city of the other player.
pub fn exp_capture_legal(s: &State, u: &Unit) -> bool {
    let t = s.tile(u.pos);
    let enemy_city = t
        .city
        .is_some_and(|id| s.cities.iter().any(|c| c.id == id && c.owner != u.owner));
    u.phase == Phase::Fresh && (t.village || enemy_city)
}

/// The brief's Capture effect, applied to `s` in place.
pub fn exp_capture(s: &mut State, unit: UnitId) {
    let u = s.units.iter().find(|u| u.id == unit).unwrap().clone();
    let at = u.pos;
    let city = if s.tile(at).village {
        let id = CityId(s.cities.len() as u32);
        s.cities.push(City {
            id,
            owner: u.owner,
            pos: at,
            level: 1,
            population: 0,
            is_capital: false,
            workshop: false,
            parks: 0,
            wall: false,
            border_growth_used: false,
            pending_reward: false,
        });
        let t = tile_mut(s, at);
        t.village = false;
        t.city = Some(id);
        for q in all_pos(s.size) {
            let t = tile_mut(s, q);
            if cheb(q, at) <= 1 && t.owner_city.is_none() {
                t.owner_city = Some(id);
            }
        }
        id
    } else {
        let id = s.tile(at).city.unwrap();
        s.cities.iter_mut().find(|c| c.id == id).unwrap().owner = u.owner;
        for o in s.units.iter_mut().filter(|o| o.origin == Some(id)) {
            o.origin = None;
        }
        id
    };
    let me = s.units.iter_mut().find(|o| o.id == unit).unwrap();
    me.origin = Some(city);
    me.phase = Phase::Done;
    for q in all_pos(s.size) {
        if s.tile(q).owner_city == Some(city) {
            let i = idx(s, q);
            s.players[u.owner as usize].explored[i] = true;
        }
    }
    if s.cities.iter().any(|c| c.id == city && c.is_capital) {
        s.outcome = Some(Outcome::Win(u.owner));
    }
}

/// Tiles whose `owner_city` is `city`, row-major.
pub fn territory_of(s: &State, city: CityId) -> Vec<Pos> {
    all_pos(s.size)
        .into_iter()
        .filter(|&q| s.tile(q).owner_city == Some(city))
        .collect()
}

/// One of every action kind, for "nothing is legal after the end" checks (ids taken from `s`
/// where it has them).
pub fn every_action_kind(s: &State) -> Vec<Action> {
    let unit = s.units.first().map_or(UnitId(0), |u| u.id);
    let city = s.cities.first().map_or(CityId(0), |c| c.id);
    let at = s.units.first().map_or(Pos::new(0, 0), |u| u.pos);
    vec![
        Action::Move { unit, to: at },
        Action::Attack { unit, target: at },
        Action::Capture { unit },
        Action::Recover { unit },
        Action::Promote { unit },
        Action::Train {
            city,
            kind: UnitType::Warrior,
        },
        Action::Site {
            at,
            action: SiteAction::HarvestFruit,
        },
        Action::Research {
            tech: Tech::Hunting,
        },
        Action::ChooseReward {
            city,
            reward: Reward::Workshop,
        },
        Action::EndTurn,
    ]
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
            ..Config::default()
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
            // Super Units keep their origin but never use capacity (milestone 5 rule 3).
            let super_unit = kind == UnitType::SuperUnit;
            let origin = (cities[home].0 == owner
                && (super_unit || supported[home] < cities[home].2 + 1)
                && rng.gen_range(4) != 0)
                .then(|| {
                    supported[home] += u32::from(!super_unit);
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
            | Action::Capture { .. }
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
