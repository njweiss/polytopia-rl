//! Shared helpers for the milestone 7 tests. Every checker here is written from docs/MILESTONE_7
//! independently of the code under test: it never calls anything in `sim_tools` to compute an
//! expectation. Game states come from plain `sim_core` calls (`new_game_from_seed`, `apply`).
#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};

use sim_core::actions::{apply, legal_actions, Action};
use sim_core::data::ALL_TECHS;
use sim_core::grid::Pos;
use sim_core::map::{MapParams, Resource, Terrain};
use sim_core::rng::Rng;
use sim_core::setup::new_game_from_seed;
use sim_core::state::{
    City, CityId, Config, Dir, Improvement, Outcome, Phase, PlayerState, ScoreWeights, State,
    TileState, Unit, UnitId,
};
use sim_tools::replay::ReplayLog;

/// The CLI's policy stream: `N XOR 0x9e3779b97f4a7c15` (docs/MILESTONE_7 "CLI").
pub const POLICY_SEED_XOR: u64 = 0x9e37_79b9_7f4a_7c15;

pub fn p(row: u8, col: u8) -> Pos {
    Pos::new(row, col)
}

pub fn cfg(turn_cap: u32, full_observability: bool) -> Config {
    Config {
        full_observability,
        turn_cap,
        score_weights: ScoreWeights::default(),
    }
}

// ---------------------------------------------------------------------------------------------
// Independent state digest: FNV-1a 64 over the byte stream of docs/MILESTONE_7 "State digest".
// ---------------------------------------------------------------------------------------------

pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

struct Enc(Vec<u8>);

impl Enc {
    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    fn i8(&mut self, v: i8) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u16(&mut self, v: u16) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn bool(&mut self, v: bool) {
        self.0.push(v as u8);
    }
    fn len(&mut self, n: usize) {
        self.u32(n as u32);
    }
    fn pos(&mut self, q: Pos) {
        self.u8(q.row);
        self.u8(q.col);
    }
    fn opt_u32(&mut self, v: Option<u32>) {
        match v {
            None => self.u8(0),
            Some(x) => {
                self.u8(1);
                self.u32(x);
            }
        }
    }
    fn opt_u8(&mut self, v: Option<u8>) {
        match v {
            None => self.u8(0),
            Some(x) => {
                self.u8(1);
                self.u8(x);
            }
        }
    }
}

fn terrain_index(t: Terrain) -> u8 {
    match t {
        Terrain::Field => 0,
        Terrain::Forest => 1,
        Terrain::Mountain => 2,
    }
}

fn resource_index(r: Resource) -> u8 {
    match r {
        Resource::Fruit => 0,
        Resource::Crop => 1,
        Resource::Animal => 2,
        Resource::Metal => 3,
    }
}

fn improvement_index(i: Improvement) -> u8 {
    match i {
        Improvement::LumberHut => 0,
        Improvement::Farm => 1,
        Improvement::Mine => 2,
    }
}

fn phase_index(ph: Phase) -> u8 {
    match ph {
        Phase::Fresh => 0,
        Phase::Moved => 1,
        Phase::Attacked => 2,
        Phase::Done => 3,
    }
}

/// `TechSet(u16)`: the one field, bit `i` set for the `i`-th declared tech (Q6).
fn techset_bits(state: &State, player: usize) -> u16 {
    let set = state.players[player].techs;
    ALL_TECHS
        .iter()
        .filter(|&&t| set.contains(t))
        .map(|&t| 1u16 << (t as u16))
        .sum()
}

/// The byte stream the digest hashes. Assumptions (docs/QUESTIONS.md Q6): the fixed-size
/// `players: [PlayerState; 2]` array has no length prefix; `TechSet` is its single `u16`;
/// `Outcome` is its variant index as `u8` followed by the `Win` payload.
pub fn digest_bytes(s: &State) -> Vec<u8> {
    // Exhaustive destructuring: adding a field to any of these structs breaks compilation here.
    let State {
        size,
        turn,
        current,
        rng,
        config,
        tables: _,
        tiles,
        units,
        cities,
        players,
        next_unit_id,
        outcome,
    } = s;
    let mut e = Enc(Vec::new());
    e.u8(*size);
    e.u32(*turn);
    e.u8(*current);
    let mut r = rng.clone();
    e.u32(r.next_u32());
    e.u32(r.next_u32());
    let Config {
        full_observability,
        turn_cap,
        score_weights,
    } = config;
    e.bool(*full_observability);
    e.u32(*turn_cap);
    let ScoreWeights {
        unit_per_star,
        super_unit,
        territory_per_tile,
        exploration_per_tile,
        city_base,
        city_per_level,
        population_surplus,
        park,
        tech_per_tier,
    } = score_weights;
    for w in [
        unit_per_star,
        super_unit,
        territory_per_tile,
        exploration_per_tile,
        city_base,
        city_per_level,
        population_surplus,
        park,
        tech_per_tier,
    ] {
        e.u32(*w);
    }
    e.len(tiles.len());
    for t in tiles {
        let TileState {
            terrain,
            resource,
            improvement,
            village,
            city,
            owner_city,
            unit,
        } = t;
        e.u8(terrain_index(*terrain));
        e.opt_u8(resource.map(resource_index));
        e.opt_u8(improvement.map(improvement_index));
        e.bool(*village);
        e.opt_u32(city.map(|c| c.0));
        e.opt_u32(owner_city.map(|c| c.0));
        e.opt_u32(unit.map(|u| u.0));
    }
    e.len(units.len());
    for u in units {
        let Unit {
            id,
            kind,
            owner,
            pos,
            health,
            max_health,
            origin,
            kills,
            veteran,
            promotion_available,
            phase,
            last_dir,
        } = u;
        e.u32(id.0);
        e.u8(*kind as u8);
        e.u8(*owner);
        e.pos(*pos);
        e.u32(*health);
        e.u32(*max_health);
        e.opt_u32(origin.map(|c| c.0));
        e.u32(*kills);
        e.bool(*veteran);
        e.bool(*promotion_available);
        e.u8(phase_index(*phase));
        match last_dir {
            None => e.u8(0),
            Some(Dir { dr, dc }) => {
                e.u8(1);
                e.i8(*dr);
                e.i8(*dc);
            }
        }
    }
    e.len(cities.len());
    for c in cities {
        let City {
            id,
            owner,
            pos,
            level,
            population,
            is_capital,
            workshop,
            parks,
            wall,
            border_growth_used,
            pending_reward,
        } = c;
        e.u32(id.0);
        e.u8(*owner);
        e.pos(*pos);
        e.u32(*level);
        e.u32(*population);
        e.bool(*is_capital);
        e.bool(*workshop);
        e.u32(*parks);
        e.bool(*wall);
        e.bool(*border_growth_used);
        e.bool(*pending_reward);
    }
    for (i, pl) in players.iter().enumerate() {
        let PlayerState {
            stars,
            techs: _,
            explored,
        } = pl;
        e.u32(*stars);
        e.u16(techset_bits(s, i));
        e.len(explored.len());
        for &x in explored {
            e.bool(x);
        }
    }
    e.u32(*next_unit_id);
    match outcome {
        None => e.u8(0),
        Some(Outcome::Win(w)) => {
            e.u8(1);
            e.u8(0);
            e.u8(*w);
        }
        Some(Outcome::Draw) => {
            e.u8(1);
            e.u8(1);
        }
    }
    e.0
}

pub fn my_digest(s: &State) -> u64 {
    fnv1a64(&digest_bytes(s))
}

pub fn my_digest_hex(s: &State) -> String {
    format!("{:016x}", my_digest(s))
}

// ---------------------------------------------------------------------------------------------
// Independent replica of the random policy (docs/MILESTONE_7 "Random policy").
// ---------------------------------------------------------------------------------------------

/// Q7 (assumption): with a pending reward `EndTurn` is not legal, so the brief's rule would return
/// an illegal action. Then the pick is `legal[gen_range(legal.len())]` with no `gen_f32` draw.
pub fn my_random_action(s: &State, rng: &mut Rng) -> Action {
    let legal = legal_actions(s);
    assert!(!legal.is_empty(), "no legal actions");
    if !legal.contains(&Action::EndTurn) {
        return legal[rng.gen_range(legal.len() as u32) as usize];
    }
    let others: Vec<Action> = legal
        .into_iter()
        .filter(|a| *a != Action::EndTurn)
        .collect();
    if others.is_empty() {
        return Action::EndTurn;
    }
    let u = rng.gen_f32();
    if u < 0.15 {
        Action::EndTurn
    } else {
        others[rng.gen_range(others.len() as u32) as usize]
    }
}

/// Plays a game with `my_random_action` driven by `Rng::new(policy_seed)`, stopping when terminal
/// or after `max_actions`. Returns every state (initial first) and the actions.
pub fn play_independent(
    seed: u64,
    params: &MapParams,
    config: &Config,
    max_actions: usize,
    policy_seed: u64,
) -> (Vec<State>, Vec<Action>) {
    let mut s = new_game_from_seed(seed, params, config);
    let mut rng = Rng::new(policy_seed);
    let mut states = vec![s.clone()];
    let mut actions = Vec::new();
    while !s.is_terminal() && actions.len() < max_actions {
        let a = my_random_action(&s, &mut rng);
        apply(&mut s, &a).expect("replica policy picks legal actions");
        actions.push(a);
        states.push(s.clone());
    }
    (states, actions)
}

/// Rebuilds every state of a log with plain `sim_core` calls (no verification).
pub fn resimulate(log: &ReplayLog) -> Vec<State> {
    let mut s = new_game_from_seed(log.seed, &log.map_params, &log.config);
    let mut out = vec![s.clone()];
    for a in &log.actions {
        apply(&mut s, a).expect("log action is legal");
        out.push(s.clone());
    }
    out
}

/// The log `record_game` must produce for this game, built by hand from the brief.
pub fn expected_log(
    seed: u64,
    params: &MapParams,
    config: &Config,
    states: &[State],
    actions: &[Action],
) -> ReplayLog {
    ReplayLog {
        version: 1,
        seed,
        map_params: params.clone(),
        config: config.clone(),
        actions: actions.to_vec(),
        digests: states.iter().map(my_digest_hex).collect(),
        outcome: states.last().unwrap().outcome,
    }
}

// ---------------------------------------------------------------------------------------------
// Temp files (no `tempfile` crate: unique names under `std::env::temp_dir()`).
// ---------------------------------------------------------------------------------------------

static COUNTER: AtomicU32 = AtomicU32::new(0);

pub struct TempFile(pub PathBuf);

impl TempFile {
    pub fn new(name: &str) -> TempFile {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let path = std::env::temp_dir().join(format!(
            "polytopia_m7_{}_{}_{}",
            std::process::id(),
            n,
            name
        ));
        let _ = std::fs::remove_file(&path);
        TempFile(path)
    }

    pub fn arg(&self) -> String {
        self.0.to_str().expect("utf-8 temp path").to_string()
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

pub fn unit_id(id: u32) -> UnitId {
    UnitId(id)
}

pub fn city_id(id: u32) -> CityId {
    CityId(id)
}
