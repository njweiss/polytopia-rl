//! Shared helpers for the milestone 9 tests. Every checker here is written from docs/MILESTONE_9
//! independently of `sim_env`: layouts, encoders and decoders are re-derived from the brief's
//! tables, and game states come from plain `sim_core` / `bots` calls.
#![allow(dead_code)]

use bots::{make_bot, BotKind};
use sim_core::actions::{apply, legal_actions, Action};
use sim_core::data::{Reward, SiteAction, Tech, UnitType};
use sim_core::economy::income;
use sim_core::grid::Pos;
use sim_core::map::{MapParams, Resource, Terrain};
use sim_core::rng::Rng;
use sim_core::score::score;
use sim_core::setup::new_game_from_seed;
use sim_core::state::{Config, Improvement, Phase, PlayerId, State};
use sim_core::view::PlayerView;
use sim_env::actions::ActionType;
use sim_env::env::{AgentSeat, EnvConfig, Opponent, RewardConfig};

/// `bot_rng` salts (docs/MILESTONE_8 "Public API").
pub const SALT: [u64; 2] = [0x9e37_79b9_7f4a_7c15, 0xc2b2_ae3d_27d4_eb4f];

pub const KINDS: [BotKind; 3] = [BotKind::Random, BotKind::Rush, BotKind::Economy];

/// Board channel offsets, from the brief's table: group sizes are the enum sizes in declaration
/// order (Terrain 3, Resource 4, Improvement 3, UnitType 5, Phase 4).
pub mod ch {
    pub const EXPLORED: usize = 0;
    pub const TERRAIN: usize = 1; // field, forest, mountain
    pub const RESOURCE: usize = 4; // fruit, crop, animal, metal
    pub const IMPROVEMENT: usize = 8; // lumber_hut, farm, mine
    pub const VILLAGE: usize = 11;
    pub const TERRITORY_MINE: usize = 12;
    pub const TERRITORY_ENEMY: usize = 13;
    pub const CITY_MINE: usize = 14;
    pub const CITY_ENEMY: usize = 15;
    pub const CITY_LEVEL: usize = 16;
    pub const CITY_CAPITAL: usize = 17;
    pub const MY_UNIT: usize = 18; // warrior, rider, archer, defender, super_unit
    pub const ENEMY_UNIT: usize = 23;
    pub const HEALTH: usize = 28;
    pub const PHASE: usize = 29; // fresh, moved, attacked, done
    pub const VETERAN: usize = 33;
    pub const CAN_PROMOTE: usize = 34;
    pub const CAN_HARVEST: usize = 35;
    pub const CAN_BUILD: usize = 36;
    pub const COUNT: usize = 37;
}

/// Global vector offsets, from the brief.
pub mod gl {
    pub const STARS: usize = 0;
    pub const INCOME: usize = 1;
    pub const TURN: usize = 2;
    pub const PENDING: usize = 3;
    pub const MY_SCORE: usize = 4;
    pub const OPP_SCORE: usize = 5;
    pub const SCORE_DIFF: usize = 6;
    pub const MY_CITIES: usize = 7;
    pub const MY_UNITS: usize = 8;
    pub const TECH: usize = 9; // climbing .. mining, 9 entries
    pub const COUNT: usize = 18;
}

pub fn terrain_i(t: Terrain) -> usize {
    match t {
        Terrain::Field => 0,
        Terrain::Forest => 1,
        Terrain::Mountain => 2,
    }
}

pub fn resource_i(r: Resource) -> usize {
    match r {
        Resource::Fruit => 0,
        Resource::Crop => 1,
        Resource::Animal => 2,
        Resource::Metal => 3,
    }
}

pub fn improvement_i(i: Improvement) -> usize {
    match i {
        Improvement::LumberHut => 0,
        Improvement::Farm => 1,
        Improvement::Mine => 2,
    }
}

pub fn unit_i(u: UnitType) -> usize {
    match u {
        UnitType::Warrior => 0,
        UnitType::Rider => 1,
        UnitType::Archer => 2,
        UnitType::Defender => 3,
        UnitType::SuperUnit => 4,
    }
}

pub fn phase_i(p: Phase) -> usize {
    match p {
        Phase::Fresh => 0,
        Phase::Moved => 1,
        Phase::Attacked => 2,
        Phase::Done => 3,
    }
}

pub const UNITS: [UnitType; 5] = [
    UnitType::Warrior,
    UnitType::Rider,
    UnitType::Archer,
    UnitType::Defender,
    UnitType::SuperUnit,
];

pub const TECHS: [Tech; 9] = [
    Tech::Climbing,
    Tech::Hunting,
    Tech::Organization,
    Tech::Riding,
    Tech::Archery,
    Tech::Forestry,
    Tech::Farming,
    Tech::Strategy,
    Tech::Mining,
];

pub const REWARDS: [Reward; 8] = [
    Reward::Workshop,
    Reward::Explorer,
    Reward::CityWall,
    Reward::Resources,
    Reward::PopulationGrowth,
    Reward::BorderGrowth,
    Reward::Park,
    Reward::SuperUnit,
];

pub const SITES: [SiteAction; 6] = [
    SiteAction::HarvestAnimal,
    SiteAction::HarvestFruit,
    SiteAction::LumberHut,
    SiteAction::ClearForest,
    SiteAction::Farm,
    SiteAction::Mine,
];

pub fn p(row: u8, col: u8) -> Pos {
    Pos::new(row, col)
}

pub fn all_pos(n: u8) -> Vec<Pos> {
    (0..n)
        .flat_map(|r| (0..n).map(move |c| Pos::new(r, c)))
        .collect()
}

pub fn cheb(a: Pos, b: Pos) -> u32 {
    let dr = (a.row as i32 - b.row as i32).unsigned_abs();
    let dc = (a.col as i32 - b.col as i32).unsigned_abs();
    dr.max(dc)
}

/// Tile index `row * size + col`.
pub fn ti(size: u8, q: Pos) -> usize {
    q.row as usize * size as usize + q.col as usize
}

pub fn tile_of(size: u8, i: usize) -> Pos {
    Pos::new((i / size as usize) as u8, (i % size as usize) as u8)
}

/// Flat index of `(channel, row, col)` in a `(C, H, W)` buffer of an `n`-wide board.
pub fn bi(n: u8, c: usize, q: Pos) -> usize {
    (c * n as usize + q.row as usize) * n as usize + q.col as usize
}

pub fn cfg(turn_cap: u32) -> Config {
    Config {
        turn_cap,
        ..Config::default()
    }
}

pub fn env_cfg(opponent: Opponent, seat: AgentSeat, turn_cap: u32) -> EnvConfig {
    EnvConfig {
        map: MapParams::default(),
        config: cfg(turn_cap),
        opponent,
        agent_seat: seat,
        reward: RewardConfig::default(),
        max_agent_actions: 5000,
    }
}

pub fn bot(kind: BotKind, epsilon: f32) -> Opponent {
    Opponent::Bot { kind, epsilon }
}

pub fn approx(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1e-6
}

pub fn assert_approx(actual: f32, expected: f32, what: &str) {
    assert!(
        approx(actual, expected),
        "{what}: got {actual}, expected {expected}"
    );
}

pub fn assert_slices_approx(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(approx(*a, *e), "{what}: index {i}: got {a}, expected {e}");
    }
}

/// docs/MILESTONE_7 "Random policy" (with Q7): EndTurn with probability 0.15 when other actions
/// exist (`gen_f32` first, then `gen_range`); with EndTurn illegal a single `gen_range`. Returns an
/// index into `legal`.
pub fn random_index(legal: &[Action], rng: &mut Rng) -> usize {
    assert!(!legal.is_empty(), "no legal actions");
    let Some(end) = legal.iter().position(|a| *a == Action::EndTurn) else {
        return rng.gen_range(legal.len() as u32) as usize;
    };
    if legal.len() == 1 {
        return end;
    }
    if rng.gen_f32() < 0.15 {
        return end;
    }
    let others: Vec<usize> = (0..legal.len()).filter(|&i| i != end).collect();
    others[rng.gen_range(others.len() as u32) as usize]
}

/// Every `every`-th state (the terminal state included) of a game where both seats play the
/// random policy driven by `Rng::new(policy_seed)`.
pub fn random_playout_states(
    seed: u64,
    config: &Config,
    policy_seed: u64,
    every: usize,
) -> Vec<State> {
    let mut s = new_game_from_seed(seed, &MapParams::default(), config);
    let mut rng = Rng::new(policy_seed);
    let mut out = Vec::new();
    let mut n = 0;
    while !s.is_terminal() {
        if n % every == 0 {
            out.push(s.clone());
        }
        let legal = legal_actions(&s);
        let a = legal[random_index(&legal, &mut rng)];
        apply(&mut s, &a).expect("random policy picks legal actions");
        n += 1;
    }
    out.push(s);
    out
}

/// Every `every`-th state (the terminal state included) of a bot game.
pub fn bot_playout_states(
    seed: u64,
    kinds: [BotKind; 2],
    epsilon: f32,
    config: &Config,
    every: usize,
) -> Vec<State> {
    let bots = [make_bot(kinds[0], epsilon), make_bot(kinds[1], epsilon)];
    let mut rngs = [Rng::new(seed ^ SALT[0]), Rng::new(seed ^ SALT[1])];
    let mut s = new_game_from_seed(seed, &MapParams::default(), config);
    let mut out = Vec::new();
    let mut n = 0;
    while !s.is_terminal() {
        if n % every == 0 {
            out.push(s.clone());
        }
        let cur = s.current as usize;
        let a = bots[cur].choose(&PlayerView::new(&s, s.current), &mut rngs[cur]);
        apply(&mut s, &a).expect("bots choose legal actions");
        n += 1;
    }
    out.push(s);
    out
}

/// A mixed corpus: random games (fog and full observability) and bot games of every kind.
pub fn corpus(seeds: std::ops::Range<u64>, every: usize) -> Vec<State> {
    let mut out = Vec::new();
    for seed in seeds {
        out.extend(random_playout_states(seed, &cfg(12), seed + 1000, every));
        let full = Config {
            full_observability: true,
            ..cfg(8)
        };
        out.extend(random_playout_states(seed, &full, seed + 2000, every));
        let a = KINDS[seed as usize % 3];
        let b = KINDS[(seed as usize + 1) % 3];
        out.extend(bot_playout_states(seed, [a, b], 0.2, &cfg(30), every));
    }
    out
}

/// docs/MILESTONE_9 "Observation" board encoding, re-derived from the ground-truth state for the
/// viewer `v` (an independent reading of the table; the code under test reads a `PlayerView`).
pub fn my_encode_board(s: &State, v: PlayerId) -> Vec<f32> {
    let n = s.size;
    let mut out = vec![0.0f32; ch::COUNT * n as usize * n as usize];
    let techs = s.players[v as usize].techs;
    let legal = if s.current == v {
        legal_actions(s)
    } else {
        Vec::new()
    };
    for q in all_pos(n) {
        if !s.is_explored(v, q) {
            continue;
        }
        let mut set = |c: usize, x: f32| out[bi(n, c, q)] = x;
        let t = s.tile(q);
        set(ch::EXPLORED, 1.0);
        set(ch::TERRAIN + terrain_i(t.terrain), 1.0);
        if let Some(r) = t.resource {
            let shown = match r {
                Resource::Crop => techs.contains(Tech::Organization),
                Resource::Metal => techs.contains(Tech::Climbing),
                Resource::Fruit | Resource::Animal => true,
            };
            if shown {
                set(ch::RESOURCE + resource_i(r), 1.0);
            }
        }
        if let Some(i) = t.improvement {
            set(ch::IMPROVEMENT + improvement_i(i), 1.0);
        }
        if t.village {
            set(ch::VILLAGE, 1.0);
        }
        if let Some(c) = t.owner_city {
            let mine = s.city(c).owner == v;
            set(
                if mine {
                    ch::TERRITORY_MINE
                } else {
                    ch::TERRITORY_ENEMY
                },
                1.0,
            );
        }
        if let Some(c) = t.city {
            let c = s.city(c);
            set(
                if c.owner == v {
                    ch::CITY_MINE
                } else {
                    ch::CITY_ENEMY
                },
                1.0,
            );
            set(ch::CITY_LEVEL, c.level.min(10) as f32 / 10.0);
            if c.is_capital {
                set(ch::CITY_CAPITAL, 1.0);
            }
        }
        if let Some(id) = t.unit {
            let u = s.unit(id).expect("occupant is alive");
            if u.owner == v {
                set(ch::MY_UNIT + unit_i(u.kind), 1.0);
                set(ch::PHASE + phase_i(u.phase), 1.0);
            } else {
                set(ch::ENEMY_UNIT + unit_i(u.kind), 1.0);
            }
            set(ch::HEALTH, u.health as f32 / u.max_health as f32);
            if u.veteran {
                set(ch::VETERAN, 1.0);
            }
        }
    }
    for a in &legal {
        match *a {
            Action::Promote { unit } => {
                let q = s.unit(unit).expect("living unit").pos;
                out[bi(n, ch::CAN_PROMOTE, q)] = 1.0;
            }
            Action::Site { at, action } => {
                let c = match action {
                    SiteAction::HarvestAnimal | SiteAction::HarvestFruit => ch::CAN_HARVEST,
                    _ => ch::CAN_BUILD,
                };
                out[bi(n, c, at)] = 1.0;
            }
            _ => {}
        }
    }
    out
}

/// docs/MILESTONE_9 "Observation" global vector, re-derived from the ground-truth state.
pub fn my_encode_globals(s: &State, v: PlayerId) -> Vec<f32> {
    let me = &s.players[v as usize];
    let (my, opp) = (score(s, v), score(s, 1 - v));
    let cities = s.cities.iter().filter(|c| c.owner == v).count() as u32;
    let units = s.units.iter().filter(|u| u.owner == v).count() as u32;
    let pending = s.cities.iter().any(|c| c.owner == v && c.pending_reward);
    let mut out = vec![
        me.stars.min(50) as f32 / 50.0,
        income(s, v).min(20) as f32 / 20.0,
        s.turn as f32 / s.config.turn_cap as f32,
        if pending { 1.0 } else { 0.0 },
        my.min(2000) as f32 / 2000.0,
        opp.min(2000) as f32 / 2000.0,
        ((my as f32 - opp as f32) / 1000.0).clamp(-1.0, 1.0),
        cities.min(8) as f32 / 8.0,
        units.min(16) as f32 / 16.0,
    ];
    for t in TECHS {
        out.push(if me.techs.contains(t) { 1.0 } else { 0.0 });
    }
    out
}

pub fn kind_i(k: ActionType) -> usize {
    match k {
        ActionType::Move => 0,
        ActionType::Attack => 1,
        ActionType::Capture => 2,
        ActionType::Recover => 3,
        ActionType::Promote => 4,
        ActionType::Train => 5,
        ActionType::ChooseReward => 6,
        ActionType::Site => 7,
        ActionType::Research => 8,
        ActionType::EndTurn => 9,
    }
}

pub const KIND_ORDER: [ActionType; 10] = [
    ActionType::Move,
    ActionType::Attack,
    ActionType::Capture,
    ActionType::Recover,
    ActionType::Promote,
    ActionType::Train,
    ActionType::ChooseReward,
    ActionType::Site,
    ActionType::Research,
    ActionType::EndTurn,
];

/// The brief's `arg_len`, re-derived.
pub fn my_arg_len(size: u8, k: ActionType) -> usize {
    let n = size as usize * size as usize;
    match k {
        ActionType::Move | ActionType::Attack => n,
        ActionType::Train => 5,
        ActionType::ChooseReward => 8,
        ActionType::Site => 6,
        ActionType::Research => 9,
        ActionType::Capture | ActionType::Recover | ActionType::Promote | ActionType::EndTurn => 0,
    }
}

/// `(actor, kind, arg)` of an action, re-derived from the brief's "Factored action space".
pub fn my_encode(s: &State, a: &Action) -> (usize, ActionType, usize) {
    let n = s.size as usize * s.size as usize;
    let unit_tile = |id| ti(s.size, s.unit(id).expect("living unit").pos);
    let city_tile = |id| n + ti(s.size, s.city(id).pos);
    match *a {
        Action::Move { unit, to } => (unit_tile(unit), ActionType::Move, ti(s.size, to)),
        Action::Attack { unit, target } => {
            (unit_tile(unit), ActionType::Attack, ti(s.size, target))
        }
        Action::Capture { unit } => (unit_tile(unit), ActionType::Capture, 0),
        Action::Recover { unit } => (unit_tile(unit), ActionType::Recover, 0),
        Action::Promote { unit } => (unit_tile(unit), ActionType::Promote, 0),
        Action::Train { city, kind } => (city_tile(city), ActionType::Train, unit_i(kind)),
        Action::ChooseReward { city, reward } => (
            city_tile(city),
            ActionType::ChooseReward,
            REWARDS.iter().position(|&r| r == reward).unwrap(),
        ),
        Action::Site { at, action } => (
            2 * n + ti(s.size, at),
            ActionType::Site,
            SITES.iter().position(|&x| x == action).unwrap(),
        ),
        Action::Research { tech } => (
            3 * n,
            ActionType::Research,
            TECHS.iter().position(|&t| t == tech).unwrap(),
        ),
        Action::EndTurn => (3 * n, ActionType::EndTurn, 0),
    }
}

/// The brief's decoding, re-derived; `None` for every failure case.
pub fn my_decode(s: &State, actor: usize, k: ActionType, arg: usize) -> Option<Action> {
    let n = s.size as usize * s.size as usize;
    if actor > 3 * n || arg >= my_arg_len(s.size, k).max(1) {
        return None;
    }
    let family = actor / n; // 0 unit, 1 city, 2 site, 3 global
    let tile = tile_of(s.size, actor % n);
    match k {
        ActionType::Move
        | ActionType::Attack
        | ActionType::Capture
        | ActionType::Recover
        | ActionType::Promote => {
            if family != 0 {
                return None;
            }
            let u = s.unit(s.tile(tile).unit?)?;
            if u.owner != s.current {
                return None;
            }
            Some(match k {
                ActionType::Move => Action::Move {
                    unit: u.id,
                    to: tile_of(s.size, arg),
                },
                ActionType::Attack => Action::Attack {
                    unit: u.id,
                    target: tile_of(s.size, arg),
                },
                ActionType::Capture => Action::Capture { unit: u.id },
                ActionType::Recover => Action::Recover { unit: u.id },
                _ => Action::Promote { unit: u.id },
            })
        }
        ActionType::Train | ActionType::ChooseReward => {
            if family != 1 {
                return None;
            }
            let c = s.city(s.tile(tile).city?);
            if c.owner != s.current {
                return None;
            }
            Some(if k == ActionType::Train {
                Action::Train {
                    city: c.id,
                    kind: UNITS[arg],
                }
            } else {
                Action::ChooseReward {
                    city: c.id,
                    reward: REWARDS[arg],
                }
            })
        }
        ActionType::Site => (family == 2).then_some(Action::Site {
            at: tile,
            action: SITES[arg],
        }),
        ActionType::Research => (family == 3).then_some(Action::Research { tech: TECHS[arg] }),
        ActionType::EndTurn => (family == 3).then_some(Action::EndTurn),
    }
}

/// `t_s` for seat `s` (docs/MILESTONE_9 "Reward").
pub fn terminal_sign(s: &State, seat: PlayerId) -> f32 {
    match s.outcome {
        Some(sim_core::state::Outcome::Win(w)) if w == seat => 1.0,
        Some(sim_core::state::Outcome::Win(_)) => -1.0,
        _ => 0.0,
    }
}

/// `phi_s` (docs/MILESTONE_9 "Reward").
pub fn potential(s: &State, seat: PlayerId, scale: f32) -> f32 {
    if s.is_terminal() {
        return 0.0;
    }
    (score(s, seat) as f32 - score(s, 1 - seat) as f32) / scale
}

/// The brief's per-seat reward for a step from `before` to `after`.
pub fn my_rewards(before: &State, after: &State, r: &RewardConfig) -> [f32; 2] {
    [0u8, 1].map(|seat| {
        r.terminal_weight * terminal_sign(after, seat)
            + r.shaping_weight
                * (r.gamma * potential(after, seat, r.potential_scale)
                    - potential(before, seat, r.potential_scale))
    })
}
