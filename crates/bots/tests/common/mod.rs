//! Shared helpers for the milestone 8 bot tests. Every checker here is written from
//! docs/MILESTONE_8 independently of the code under test: expectations never come from calling
//! a bot, `bot_rng`, `play_game`, `run_match` or `MatchStats` (game states come from plain
//! `sim_core` calls; bots are only called as the thing being checked or to drive games).
#![allow(dead_code)]

use bots::{make_bot, Bot, BotKind};
use sim_core::actions::{apply, legal_actions, Action};
use sim_core::grid::Pos;
use sim_core::map::{MapParams, Terrain};
use sim_core::rng::Rng;
use sim_core::score::score;
use sim_core::setup::new_game_from_seed;
use sim_core::state::{Config, Outcome, State};
use sim_core::testing::StateBuilder;
use sim_core::view::PlayerView;

/// `bot_rng` salts (docs/MILESTONE_8 "Public API").
pub const SALT: [u64; 2] = [0x9e37_79b9_7f4a_7c15, 0xc2b2_ae3d_27d4_eb4f];

pub const KINDS: [BotKind; 3] = [BotKind::Random, BotKind::Rush, BotKind::Economy];

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

/// An `n`x`n` all-field board both players have fully explored.
pub fn open(n: u8) -> StateBuilder {
    StateBuilder::new(n).explore_all()
}

/// The tiles with `r0 <= row <= r1` and `c0 <= col <= c1`.
pub fn rect(r0: u8, r1: u8, c0: u8, c1: u8) -> Vec<Pos> {
    (r0..=r1)
        .flat_map(|r| (c0..=c1).map(move |c| Pos::new(r, c)))
        .collect()
}

pub fn cfg(turn_cap: u32) -> Config {
    Config {
        turn_cap,
        ..Config::default()
    }
}

/// The bot's choice for the player to move, with a throwaway rng.
pub fn choice(bot: &dyn Bot, s: &State) -> Action {
    bot.choose(&PlayerView::new(s, s.current), &mut Rng::new(0))
}

/// Asserts that the hand-derived `expected` is legal (a check on the scripted board itself, via
/// `sim_core::actions::legal_actions`) and that the bot picks exactly it.
pub fn assert_choice(bot: &dyn Bot, s: &State, expected: Action) {
    assert!(
        legal_actions(s).contains(&expected),
        "test board is wrong: {expected:?} is not legal"
    );
    assert_eq!(choice(bot, s), expected);
}

/// docs/MILESTONE_7 "Random policy" with Q7, applied to a given legal list: EndTurn with
/// probability 0.15 when other actions exist (`gen_f32` first, then `gen_range`); with EndTurn
/// illegal a single `gen_range` over `legal`.
pub fn my_random_from(legal: &[Action], rng: &mut Rng) -> Action {
    assert!(!legal.is_empty(), "no legal actions");
    if !legal.contains(&Action::EndTurn) {
        return legal[rng.gen_range(legal.len() as u32) as usize];
    }
    let others: Vec<Action> = legal
        .iter()
        .copied()
        .filter(|a| *a != Action::EndTurn)
        .collect();
    if others.is_empty() {
        return Action::EndTurn;
    }
    if rng.gen_f32() < 0.15 {
        Action::EndTurn
    } else {
        others[rng.gen_range(others.len() as u32) as usize]
    }
}

/// Collects states from a random playout (`my_random_from` driven by `Rng::new(policy_seed)`),
/// keeping every `every`-th state before an action.
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
        let a = my_random_from(&legal_actions(&s), &mut rng);
        apply(&mut s, &a).expect("replica policy picks legal actions");
        n += 1;
    }
    out
}

/// Collects states from a game driven by the bots under test (only to generate inputs).
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
    out
}

/// docs/MILESTONE_8 "captured_capital": the game was won and the winner owns every capital.
pub fn my_captured_capital(s: &State) -> bool {
    match s.outcome {
        Some(Outcome::Win(w)) => s
            .cities
            .iter()
            .filter(|c| c.is_capital)
            .all(|c| c.owner == w),
        _ => false,
    }
}

/// One player's counts, counted directly from the state.
pub fn my_counts(s: &State, player: u8) -> (u32, u32, u32, u32) {
    let cities = s.cities.iter().filter(|c| c.owner == player).count() as u32;
    let units = s.units.iter().filter(|u| u.owner == player).count() as u32;
    (
        cities,
        units,
        s.players[player as usize].stars,
        score(s, player),
    )
}

/// A snapshot as `(turn, cities, units, stars, score)` per player.
pub type MySnapshot = (u32, [u32; 2], [u32; 2], [u32; 2], [u32; 2]);

pub fn my_snapshot(s: &State, turn: u32) -> MySnapshot {
    let a = my_counts(s, 0);
    let b = my_counts(s, 1);
    (turn, [a.0, b.0], [a.1, b.1], [a.2, b.2], [a.3, b.3])
}

/// An independent replica of docs/MILESTONE_8 `play_game`. Returns the final state, the actions,
/// and the snapshots taken at the first moment `turn == t && current == 0` for t in 10, 20, 30
/// (checked on every state, the initial and the terminal state included).
pub struct MyGame {
    pub state: State,
    pub actions: Vec<Action>,
    pub snapshots: Vec<MySnapshot>,
}

pub fn my_play_game(
    seed: u64,
    kinds: [BotKind; 2],
    eps: [f32; 2],
    map: &MapParams,
    config: &Config,
) -> MyGame {
    let bots = [make_bot(kinds[0], eps[0]), make_bot(kinds[1], eps[1])];
    let mut rngs = [Rng::new(seed ^ SALT[0]), Rng::new(seed ^ SALT[1])];
    let mut s = new_game_from_seed(seed, map, config);
    let mut actions = Vec::new();
    let mut snapshots: Vec<MySnapshot> = Vec::new();
    loop {
        for t in [10, 20, 30] {
            if s.turn == t && s.current == 0 && !snapshots.iter().any(|x| x.0 == t) {
                snapshots.push(my_snapshot(&s, t));
            }
        }
        if s.is_terminal() {
            break;
        }
        let cur = s.current as usize;
        let a = bots[cur].choose(&PlayerView::new(&s, s.current), &mut rngs[cur]);
        apply(&mut s, &a).expect("bots choose legal actions");
        actions.push(a);
    }
    MyGame {
        state: s,
        actions,
        snapshots,
    }
}

/// Vision gain of `t` for `player` (docs/MILESTONE_8 "Common bot machinery"): unexplored in-bounds
/// tiles within radius 2 of a mountain, 1 otherwise.
pub fn my_vision_gain(s: &State, player: u8, t: Pos) -> u32 {
    let r = if s.tile(t).terrain == Terrain::Mountain {
        2
    } else {
        1
    };
    all_pos(s.size)
        .into_iter()
        .filter(|&q| cheb(q, t) <= r && !s.is_explored(player, q))
        .count() as u32
}
