//! Milestone 8: match runner and statistics (docs/MILESTONE_8 "Match running and statistics";
//! "Tests" item 4). `from_results` is checked on hand-made results with hand-computed means;
//! `play_game` and `run_match` against `common::my_play_game`, an independent loop that counts
//! everything itself.

mod common;

use bots::{
    play_game, run_match, BotKind, CheckpointStats, GameResult, MatchStats, PlayerMeans, Snapshot,
};
use common::*;
use sim_core::map::MapParams;
use sim_core::score::score;
use sim_core::state::Outcome;

const RUSH: BotKind = BotKind::Rush;
const ECON: BotKind = BotKind::Economy;
const RAND: BotKind = BotKind::Random;

fn snap(
    turn: u32,
    cities: [u32; 2],
    units: [u32; 2],
    stars: [u32; 2],
    score: [u32; 2],
) -> Snapshot {
    Snapshot {
        turn,
        cities,
        units,
        stars,
        score,
    }
}

fn result(
    seed: u64,
    bots: [BotKind; 2],
    outcome: Outcome,
    turns: u32,
    scores: [u32; 2],
    captured_capital: bool,
    snapshots: Vec<Snapshot>,
) -> GameResult {
    GameResult {
        seed,
        bots,
        outcome,
        turns,
        scores,
        captured_capital,
        snapshots,
    }
}

/// Five games of Rush (a) against Economy (b), seats alternating.
fn hand_results() -> Vec<GameResult> {
    vec![
        // Rush is player 0 and wins by capture: diff 900 - 600 = 300.
        result(
            1,
            [RUSH, ECON],
            Outcome::Win(0),
            18,
            [900, 600],
            true,
            vec![snap(10, [1, 2], [3, 2], [4, 6], [700, 650])],
        ),
        // Rush is player 1 and wins at the cap: diff 1200 - 1100 = 100.
        result(
            1,
            [ECON, RUSH],
            Outcome::Win(1),
            30,
            [1100, 1200],
            false,
            vec![
                snap(10, [2, 1], [2, 4], [8, 2], [800, 690]),
                snap(20, [3, 1], [3, 5], [10, 1], [1000, 900]),
                snap(30, [3, 2], [4, 5], [12, 3], [1100, 1200]),
            ],
        ),
        // Draw: diff 0.
        result(
            2,
            [RUSH, ECON],
            Outcome::Draw,
            30,
            [1000, 1000],
            false,
            vec![
                snap(10, [1, 1], [2, 2], [5, 5], [600, 600]),
                snap(20, [2, 2], [3, 3], [6, 6], [800, 800]),
                snap(30, [2, 2], [3, 3], [7, 7], [1000, 1000]),
            ],
        ),
        // Rush is player 1 and wins by capture: diff 1500 - 700 = 800.
        result(
            2,
            [ECON, RUSH],
            Outcome::Win(1),
            25,
            [700, 1500],
            true,
            vec![
                snap(10, [1, 2], [1, 4], [3, 0], [600, 750]),
                snap(20, [1, 3], [1, 6], [2, 1], [650, 1200]),
            ],
        ),
        // Economy is player 0 and wins by capture: diff 400 - 800 = -400.
        result(
            3,
            [ECON, RUSH],
            Outcome::Win(0),
            12,
            [800, 400],
            true,
            vec![snap(10, [2, 1], [3, 2], [1, 1], [700, 500])],
        ),
    ]
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

fn assert_means(got: &PlayerMeans, want: [f64; 4], what: &str) {
    let g = [got.cities, got.units, got.stars, got.score];
    for (i, name) in ["cities", "units", "stars", "score"].iter().enumerate() {
        assert!(
            close(g[i], want[i]),
            "{what} {name}: got {}, want {}",
            g[i],
            want[i]
        );
    }
}

#[test]
fn from_results_attributes_through_bots() {
    let m = MatchStats::from_results(RUSH, ECON, &hand_results());
    assert_eq!((m.a, m.b), (RUSH, ECON));
    assert_eq!(m.games, 5);
    assert_eq!((m.a_wins, m.b_wins, m.draws), (3, 1, 1));
    assert_eq!((m.a_capture_wins, m.b_capture_wins), (2, 1));
    // (300 + 100 + 0 + 800 - 400) / 5
    assert!(close(m.mean_score_diff, 160.0), "{}", m.mean_score_diff);
    assert!(close(m.win_rate_a(), 0.6));
    assert!(close(m.win_rate_b(), 0.2));
}

#[test]
fn from_results_swapped_roles_flip_everything() {
    let m = MatchStats::from_results(ECON, RUSH, &hand_results());
    assert_eq!((m.a, m.b), (ECON, RUSH));
    assert_eq!((m.a_wins, m.b_wins, m.draws), (1, 3, 1));
    assert_eq!((m.a_capture_wins, m.b_capture_wins), (1, 2));
    assert!(close(m.mean_score_diff, -160.0), "{}", m.mean_score_diff);
    assert!(close(m.win_rate_a(), 0.2));
    assert!(close(m.win_rate_b(), 0.6));
}

#[test]
fn checkpoint_means_cover_only_games_that_reached_the_turn() {
    let m = MatchStats::from_results(RUSH, ECON, &hand_results());
    let turns: Vec<(u32, u32)> = m.checkpoints.iter().map(|c| (c.turn, c.games)).collect();
    assert_eq!(turns, vec![(10, 5), (20, 3), (30, 2)]);
    let c: &[CheckpointStats] = &m.checkpoints;
    // Turn 10, all five games. Rush (a): cities 1,1,1,2,1; units 3,4,2,4,2; stars 4,2,5,0,1;
    // score 700,690,600,750,500. Economy (b): cities 2,2,1,1,2; units 2,2,2,1,3; stars
    // 6,8,5,3,1; score 650,800,600,600,700.
    assert_means(&c[0].a, [1.2, 3.0, 2.4, 648.0], "turn 10 a");
    assert_means(&c[0].b, [1.6, 2.0, 4.6, 670.0], "turn 10 b");
    // Turn 20: games 2, 3, 4. a: cities 1,2,3; units 5,3,6; stars 1,6,1; score 900,800,1200.
    // b: cities 3,2,1; units 3,3,1; stars 10,6,2; score 1000,800,650.
    assert_means(
        &c[1].a,
        [2.0, 14.0 / 3.0, 8.0 / 3.0, 2900.0 / 3.0],
        "turn 20 a",
    );
    assert_means(&c[1].b, [2.0, 7.0 / 3.0, 6.0, 2450.0 / 3.0], "turn 20 b");
    // Turn 30: games 2, 3. a: cities 2,2; units 5,3; stars 3,7; score 1200,1000.
    // b: cities 3,2; units 4,3; stars 12,7; score 1100,1000.
    assert_means(&c[2].a, [2.0, 4.0, 5.0, 1100.0], "turn 30 a");
    assert_means(&c[2].b, [2.5, 3.5, 9.5, 1050.0], "turn 30 b");
}

#[test]
fn checkpoints_only_for_turns_some_game_reached() {
    let all = hand_results();
    let short = vec![all[0].clone(), all[4].clone()];
    let m = MatchStats::from_results(RUSH, ECON, &short);
    assert_eq!(m.games, 2);
    assert_eq!(m.checkpoints.len(), 1);
    assert_eq!((m.checkpoints[0].turn, m.checkpoints[0].games), (10, 2));
    // a: (1, 3, 4, 700) and (1, 2, 1, 500); b: (2, 2, 6, 650) and (2, 3, 1, 700).
    assert_means(&m.checkpoints[0].a, [1.0, 2.5, 2.5, 600.0], "a");
    assert_means(&m.checkpoints[0].b, [2.0, 2.5, 3.5, 675.0], "b");
}

fn my_snapshots(g: &MyGame) -> Vec<Snapshot> {
    g.snapshots
        .iter()
        .map(|&(turn, cities, units, stars, score)| snap(turn, cities, units, stars, score))
        .collect()
}

/// `play_game` against the independent loop, field by field.
fn check_play_game(seed: u64, kinds: [BotKind; 2], eps: [f32; 2], turn_cap: u32) -> MyGame {
    let config = cfg(turn_cap);
    let got = play_game(seed, kinds, eps, &MapParams::default(), &config);
    let want = my_play_game(seed, kinds, eps, &MapParams::default(), &config);
    let s = &want.state;
    let what = format!("seed {seed} {kinds:?} eps {eps:?} cap {turn_cap}");
    assert_eq!(got.seed, seed, "{what}");
    assert_eq!(got.bots, kinds, "{what}");
    assert_eq!(Some(got.outcome), s.outcome, "{what}");
    assert_eq!(got.turns, s.turn, "{what}");
    assert_eq!(got.scores, [score(s, 0), score(s, 1)], "{what}");
    assert_eq!(got.captured_capital, my_captured_capital(s), "{what}");
    assert_eq!(got.snapshots, my_snapshots(&want), "{what}");
    want
}

#[test]
fn play_game_matches_an_independent_loop() {
    for (seed, kinds, eps, cap) in [
        (0, [RUSH, ECON], [0.0, 0.0], 30),
        (1, [ECON, RUSH], [0.0, 0.0], 30),
        (2, [RUSH, RAND], [0.0, 0.0], 30),
        (3, [RAND, ECON], [0.0, 0.0], 25),
        (4, [ECON, ECON], [0.1, 0.2], 21),
        (5, [RUSH, RUSH], [0.3, 0.0], 12),
        (6, [RAND, RAND], [0.0, 0.0], 30),
    ] {
        check_play_game(seed, kinds, eps, cap);
    }
}

#[test]
fn play_game_is_deterministic() {
    let config = cfg(30);
    for seed in [7, 8] {
        let a = play_game(
            seed,
            [RUSH, ECON],
            [0.1, 0.1],
            &MapParams::default(),
            &config,
        );
        let b = play_game(
            seed,
            [RUSH, ECON],
            [0.1, 0.1],
            &MapParams::default(),
            &config,
        );
        assert_eq!(a, b);
    }
}

#[test]
fn captured_capital_on_capture_and_cap_ended_games() {
    // Rush against random over a fixed seed range must produce both kinds of ending.
    let (mut captures, mut capped) = (0, 0);
    for seed in 0..12 {
        for kinds in [[RUSH, RAND], [RAND, RUSH]] {
            let g = check_play_game(seed, kinds, [0.0, 0.0], 30);
            let s = &g.state;
            if my_captured_capital(s) {
                captures += 1;
                assert!(s.turn < 30, "a capture ends the game before the cap");
            } else {
                capped += 1;
                assert_eq!(s.turn, 30, "seed {seed}: no capture means the cap ended it");
            }
        }
    }
    assert!(captures > 0, "no capital capture in 24 Rush games");
    assert!(capped > 0, "no cap-ended game in 24 games");
}

#[test]
fn a_one_turn_game_ends_at_the_cap_without_snapshots() {
    let g = play_game(3, [RUSH, ECON], [0.0, 0.0], &MapParams::default(), &cfg(1));
    assert_eq!(g.turns, 1);
    assert!(!g.captured_capital);
    assert!(g.snapshots.is_empty());
}

#[test]
fn snapshot_turns() {
    // Cap 25: at most turns 10 and 20. Cap 30 with no capture: the terminal state (turn 30,
    // player 0 to move) is the first moment of turn 30, so 10, 20 and 30 (assumption, see
    // docs/QUESTIONS.md).
    let g = play_game(
        11,
        [ECON, ECON],
        [0.0, 0.0],
        &MapParams::default(),
        &cfg(25),
    );
    let turns: Vec<u32> = g.snapshots.iter().map(|s| s.turn).collect();
    assert!(turns.iter().all(|&t| t == 10 || t == 20), "{turns:?}");
    let g = play_game(
        11,
        [ECON, ECON],
        [0.0, 0.0],
        &MapParams::default(),
        &cfg(30),
    );
    if !g.captured_capital {
        let turns: Vec<u32> = g.snapshots.iter().map(|s| s.turn).collect();
        assert_eq!(turns, vec![10, 20, 30]);
    }
    let g = play_game(11, [ECON, ECON], [0.0, 0.0], &MapParams::default(), &cfg(9));
    assert!(g.snapshots.is_empty());
}

/// docs/MILESTONE_8 `run_match`, aggregated independently from `my_play_game`.
fn my_match(a: BotKind, b: BotKind, seeds: std::ops::Range<u64>, turn_cap: u32) -> MatchStats {
    let config = cfg(turn_cap);
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
    // (turn, games, a sums, b sums)
    let mut sums: Vec<(u32, u32, [f64; 4], [f64; 4])> = Vec::new();
    let mut diff = 0.0;
    for seed in seeds {
        for (kinds, a_seat) in [([a, b], 0usize), ([b, a], 1usize)] {
            let g = my_play_game(seed, kinds, [0.0, 0.0], &MapParams::default(), &config);
            let s = &g.state;
            m.games += 1;
            let captured = my_captured_capital(s);
            match s.outcome.unwrap() {
                Outcome::Win(w) if w as usize == a_seat => {
                    m.a_wins += 1;
                    m.a_capture_wins += captured as u32;
                }
                Outcome::Win(_) => {
                    m.b_wins += 1;
                    m.b_capture_wins += captured as u32;
                }
                Outcome::Draw => m.draws += 1,
            }
            let b_seat = 1 - a_seat;
            diff += score(s, a_seat as u8) as f64 - score(s, b_seat as u8) as f64;
            for &(turn, cities, units, stars, sc) in &g.snapshots {
                let i = match sums.iter().position(|x| x.0 == turn) {
                    Some(i) => i,
                    None => {
                        sums.push((turn, 0, [0.0; 4], [0.0; 4]));
                        sums.len() - 1
                    }
                };
                sums[i].1 += 1;
                for (k, v) in [cities, units, stars, sc].iter().enumerate() {
                    sums[i].2[k] += v[a_seat] as f64;
                    sums[i].3[k] += v[b_seat] as f64;
                }
            }
        }
    }
    m.mean_score_diff = diff / m.games as f64;
    sums.sort_by_key(|x| x.0);
    for (turn, games, sa, sb) in sums {
        let n = games as f64;
        let means = |s: [f64; 4]| PlayerMeans {
            cities: s[0] / n,
            units: s[1] / n,
            stars: s[2] / n,
            score: s[3] / n,
        };
        m.checkpoints.push(CheckpointStats {
            turn,
            games,
            a: means(sa),
            b: means(sb),
        });
    }
    m
}

fn assert_stats_eq(got: &MatchStats, want: &MatchStats) {
    assert_eq!((got.a, got.b, got.games), (want.a, want.b, want.games));
    assert_eq!(
        (
            got.a_wins,
            got.b_wins,
            got.draws,
            got.a_capture_wins,
            got.b_capture_wins
        ),
        (
            want.a_wins,
            want.b_wins,
            want.draws,
            want.a_capture_wins,
            want.b_capture_wins
        )
    );
    assert!(close(got.mean_score_diff, want.mean_score_diff));
    assert_eq!(got.checkpoints.len(), want.checkpoints.len());
    for (g, w) in got.checkpoints.iter().zip(&want.checkpoints) {
        assert_eq!((g.turn, g.games), (w.turn, w.games));
        assert_means(&g.a, [w.a.cities, w.a.units, w.a.stars, w.a.score], "a");
        assert_means(&g.b, [w.b.cities, w.b.units, w.b.stars, w.b.score], "b");
    }
}

#[test]
fn run_match_plays_each_seed_with_seats_swapped() {
    let got = run_match(RUSH, RAND, 0..3, &MapParams::default(), &cfg(15));
    assert_eq!(got.games, 6);
    assert_stats_eq(&got, &my_match(RUSH, RAND, 0..3, 15));
    let got = run_match(ECON, RUSH, 3..5, &MapParams::default(), &cfg(22));
    assert_eq!(got.games, 4);
    assert_stats_eq(&got, &my_match(ECON, RUSH, 3..5, 22));
}
