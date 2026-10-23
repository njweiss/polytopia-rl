//! Match running and statistics (docs/MILESTONE_8 "Match running and statistics").

use std::ops::Range;

use sim_core::actions::{apply, Action};
use sim_core::map::MapParams;
use sim_core::score::score;
use sim_core::setup::new_game_from_seed;
use sim_core::state::{Config, Outcome, State};
use sim_core::view::PlayerView;

use crate::{bot_rng, make_bot, BotKind};

/// The turns at which `play_game` takes snapshots.
const SNAPSHOT_TURNS: [u32; 3] = [10, 20, 30];

/// Per-player counts at the first moment `state.turn == turn && state.current == 0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Snapshot {
    pub turn: u32,
    pub cities: [u32; 2],
    pub units: [u32; 2],
    pub stars: [u32; 2],
    pub score: [u32; 2],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GameResult {
    pub seed: u64,
    /// By player id.
    pub bots: [BotKind; 2],
    pub outcome: Outcome,
    pub turns: u32,
    pub scores: [u32; 2],
    pub captured_capital: bool,
    /// At turns 10, 20, 30 (those that were reached).
    pub snapshots: Vec<Snapshot>,
}

/// Both seats' bots with their `bot_rng` streams: the current player's choice from its own view.
pub fn bot_policy(
    seed: u64,
    bots: [BotKind; 2],
    epsilons: [f32; 2],
) -> impl FnMut(&State) -> Action {
    let players = [
        make_bot(bots[0], epsilons[0]),
        make_bot(bots[1], epsilons[1]),
    ];
    let mut rngs = [bot_rng(seed, 0), bot_rng(seed, 1)];
    move |s: &State| {
        let cur = s.current as usize;
        players[cur].choose(&PlayerView::new(s, s.current), &mut rngs[cur])
    }
}

pub fn play_game(
    seed: u64,
    bots: [BotKind; 2],
    epsilons: [f32; 2],
    map: &MapParams,
    config: &Config,
) -> GameResult {
    let mut policy = bot_policy(seed, bots, epsilons);
    let mut state = new_game_from_seed(seed, map, config);
    let mut snapshots: Vec<Snapshot> = Vec::new();
    loop {
        let t = state.turn;
        if state.current == 0
            && SNAPSHOT_TURNS.contains(&t)
            && !snapshots.iter().any(|x| x.turn == t)
        {
            snapshots.push(snapshot(&state));
        }
        if state.is_terminal() {
            break;
        }
        let action = policy(&state);
        apply(&mut state, &action).expect("bots choose legal actions");
    }
    let outcome = state.outcome.expect("the loop ends on a terminal state");
    // A capture win leaves the winner owning every capital; the cap never does (Q12).
    let captured_capital = match outcome {
        Outcome::Win(w) => state
            .cities
            .iter()
            .filter(|c| c.is_capital)
            .all(|c| c.owner == w),
        Outcome::Draw => false,
    };
    GameResult {
        seed,
        bots,
        outcome,
        turns: state.turn,
        scores: [score(&state, 0), score(&state, 1)],
        captured_capital,
        snapshots,
    }
}

fn snapshot(state: &State) -> Snapshot {
    let count = |p: u8| {
        (
            state.cities.iter().filter(|c| c.owner == p).count() as u32,
            state.units.iter().filter(|u| u.owner == p).count() as u32,
        )
    };
    let (c0, u0) = count(0);
    let (c1, u1) = count(1);
    Snapshot {
        turn: state.turn,
        cities: [c0, c1],
        units: [u0, u1],
        stars: [state.players[0].stars, state.players[1].stars],
        score: [score(state, 0), score(state, 1)],
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlayerMeans {
    pub cities: f64,
    pub units: f64,
    pub stars: f64,
    pub score: f64,
}

/// Means over the games that reached `turn`.
#[derive(Debug, Clone, PartialEq)]
pub struct CheckpointStats {
    pub turn: u32,
    pub games: u32,
    pub a: PlayerMeans,
    pub b: PlayerMeans,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatchStats {
    pub a: BotKind,
    pub b: BotKind,
    pub games: u32,
    pub a_wins: u32,
    pub b_wins: u32,
    pub draws: u32,
    pub a_capture_wins: u32,
    pub b_capture_wins: u32,
    /// Mean of `a`'s final score minus `b`'s.
    pub mean_score_diff: f64,
    pub checkpoints: Vec<CheckpointStats>,
}

impl MatchStats {
    /// Each result is attributed to `a` and `b` through `result.bots` (`a` is the first seat
    /// holding kind `a`).
    pub fn from_results(a: BotKind, b: BotKind, results: &[GameResult]) -> MatchStats {
        let mut m = MatchStats {
            a,
            b,
            games: 0,
            a_wins: 0,
            b_wins: 0,
            draws: 0,
            a_capture_wins: 0,
            b_capture_wins: 0,
            mean_score_diff: 0.0,
            checkpoints: Vec::new(),
        };
        let mut diff = 0.0;
        let seats: Vec<usize> = results
            .iter()
            .map(|r| r.bots.iter().position(|&k| k == a).expect("a played"))
            .collect();
        for (r, &a_seat) in results.iter().zip(&seats) {
            m.games += 1;
            match r.outcome {
                Outcome::Win(w) if w as usize == a_seat => {
                    m.a_wins += 1;
                    m.a_capture_wins += u32::from(r.captured_capital);
                }
                Outcome::Win(_) => {
                    m.b_wins += 1;
                    m.b_capture_wins += u32::from(r.captured_capital);
                }
                Outcome::Draw => m.draws += 1,
            }
            diff += r.scores[a_seat] as f64 - r.scores[1 - a_seat] as f64;
        }
        if m.games > 0 {
            m.mean_score_diff = diff / m.games as f64;
        }
        for turn in SNAPSHOT_TURNS {
            let reached: Vec<(&Snapshot, usize)> = results
                .iter()
                .zip(&seats)
                .filter_map(|(r, &a_seat)| {
                    r.snapshots
                        .iter()
                        .find(|x| x.turn == turn)
                        .map(|x| (x, a_seat))
                })
                .collect();
            if reached.is_empty() {
                continue;
            }
            let n = reached.len() as f64;
            let means = |seat: fn(usize) -> usize| {
                let mean = |f: fn(&Snapshot, usize) -> u32| {
                    reached
                        .iter()
                        .map(|&(s, a_seat)| f(s, seat(a_seat)) as f64)
                        .sum::<f64>()
                        / n
                };
                PlayerMeans {
                    cities: mean(|s, i| s.cities[i]),
                    units: mean(|s, i| s.units[i]),
                    stars: mean(|s, i| s.stars[i]),
                    score: mean(|s, i| s.score[i]),
                }
            };
            m.checkpoints.push(CheckpointStats {
                turn,
                games: reached.len() as u32,
                a: means(|s| s),
                b: means(|s| 1 - s),
            });
        }
        m
    }

    /// `a_wins / games` (draws count as no win).
    pub fn win_rate_a(&self) -> f64 {
        self.a_wins as f64 / self.games as f64
    }

    /// `b_wins / games` (draws count as no win).
    pub fn win_rate_b(&self) -> f64 {
        self.b_wins as f64 / self.games as f64
    }
}

/// Plays each seed twice: `a` as player 0, then `b` as player 0, epsilons 0.
pub fn run_match(
    a: BotKind,
    b: BotKind,
    seeds: Range<u64>,
    map: &MapParams,
    config: &Config,
) -> MatchStats {
    let results: Vec<GameResult> = seeds
        .flat_map(|seed| {
            [[a, b], [b, a]].map(|bots| play_game(seed, bots, [0.0, 0.0], map, config))
        })
        .collect();
    MatchStats::from_results(a, b, &results)
}
