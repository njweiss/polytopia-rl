//! Milestone 8: the `play` and `tournament` subcommands and the RandomBot parity check
//! (docs/MILESTONE_8 "CLI additions", H4; "Tests" item 6). Expected logs come from an
//! independent bot loop (`my_bot_game`, with the brief's salted seeds) and `common::expected_log`;
//! the expected tournament table is built from an independent `run_match` call with the brief's
//! format strings.

mod common;

use bots::{make_bot, run_match, Bot, BotKind, RandomBot};
use common::*;
use sim_core::actions::{apply, Action};
use sim_core::map::MapParams;
use sim_core::rng::Rng;
use sim_core::setup::new_game_from_seed;
use sim_core::state::{Config, Outcome, State};
use sim_core::view::PlayerView;
use sim_tools::policy::random_action;
use sim_tools::replay::{replay, ReplayLog};

/// `bot_rng` salts (docs/MILESTONE_8 "Public API").
const SALT: [u64; 2] = [0x9e37_79b9_7f4a_7c15, 0xc2b2_ae3d_27d4_eb4f];

struct Run {
    code: i32,
    out: String,
    err: String,
}

fn run(args: &[&str]) -> Run {
    let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    let mut out = Vec::new();
    let mut err = Vec::new();
    let code = sim_tools::cli::run_cli(&args, &mut out, &mut err);
    Run {
        code,
        out: String::from_utf8(out).expect("utf-8 out"),
        err: String::from_utf8(err).expect("utf-8 err"),
    }
}

fn outcome_text(o: Option<Outcome>) -> String {
    match o {
        Some(Outcome::Win(p)) => format!("Win({p})"),
        Some(Outcome::Draw) => "Draw".to_string(),
        None => "unfinished".to_string(),
    }
}

fn read_log(path: &std::path::Path) -> ReplayLog {
    let text = std::fs::read_to_string(path).expect("log file written");
    serde_json::from_str(&text).expect("log file is a ReplayLog")
}

/// A bot game with epsilon 0 and the `play_game` rng derivation, stopping when terminal or after
/// `max_actions`. Returns every state (initial first) and the actions.
fn my_bot_game(
    seed: u64,
    kinds: [BotKind; 2],
    config: &Config,
    max_actions: usize,
) -> (Vec<State>, Vec<Action>) {
    let bots = [make_bot(kinds[0], 0.0), make_bot(kinds[1], 0.0)];
    let mut rngs = [Rng::new(seed ^ SALT[0]), Rng::new(seed ^ SALT[1])];
    let mut s = new_game_from_seed(seed, &MapParams::default(), config);
    let mut states = vec![s.clone()];
    let mut actions = Vec::new();
    while !s.is_terminal() && actions.len() < max_actions {
        let cur = s.current as usize;
        let a = bots[cur].choose(&PlayerView::new(&s, s.current), &mut rngs[cur]);
        apply(&mut s, &a).expect("bots choose legal actions");
        actions.push(a);
        states.push(s.clone());
    }
    (states, actions)
}

fn kind(name: &str) -> BotKind {
    match name {
        "random" => BotKind::Random,
        "rush" => BotKind::Rush,
        "economy" => BotKind::Economy,
        _ => panic!("{name}"),
    }
}

#[test]
fn play_writes_the_bot_game_log_and_message() {
    for (bots, seed, cap) in [("rush,economy", 3u64, 6u32), ("random,rush", 8, 5)] {
        let f = TempFile::new("bots_play.json");
        let r = run(&[
            "play",
            "--bots",
            bots,
            "--seed",
            &seed.to_string(),
            "--out",
            &f.arg(),
            "--turn-cap",
            &cap.to_string(),
        ]);
        assert_eq!(r.code, 0, "err: {}", r.err);
        assert_eq!(r.err, "");
        let log = read_log(&f.0);
        let names: Vec<&str> = bots.split(',').collect();
        let config = cfg(cap, false);
        let (states, actions) = my_bot_game(seed, [kind(names[0]), kind(names[1])], &config, 5000);
        assert!(states.last().unwrap().is_terminal());
        assert_eq!(
            log,
            expected_log(seed, &MapParams::default(), &config, &states, &actions)
        );
        assert_eq!(
            r.out.trim_end_matches('\n'),
            format!(
                "wrote {}: {} actions, outcome {}",
                f.arg(),
                actions.len(),
                outcome_text(log.outcome)
            )
        );
        let v = run(&["verify", &f.arg()]);
        assert_eq!(v.code, 0, "verify: {}", v.err);
    }
}

#[test]
fn play_max_actions_stops_early() {
    let f = TempFile::new("bots_play_max.json");
    let r = run(&[
        "play",
        "--max-actions",
        "7",
        "--bots",
        "economy,random",
        "--out",
        &f.arg(),
        "--seed",
        "4",
        "--turn-cap",
        "10",
    ]);
    assert_eq!(r.code, 0, "err: {}", r.err);
    let log = read_log(&f.0);
    let config = cfg(10, false);
    let (states, actions) = my_bot_game(4, [BotKind::Economy, BotKind::Random], &config, 7);
    assert_eq!(actions.len(), 7);
    assert_eq!(
        log,
        expected_log(4, &MapParams::default(), &config, &states, &actions)
    );
    assert_eq!(
        r.out.trim_end_matches('\n'),
        format!("wrote {}: 7 actions, outcome unfinished", f.arg())
    );
    assert!(replay(&log).is_ok());
}

/// The brief's table for `--bots LIST --games N --turn-cap K --seed S`, from `run_match`.
fn expected_table(names: &[&str], n: u64, k: u32, s: u64) -> String {
    let config = cfg(k, false);
    let mut out = format!(
        "tournament: {n} seeds per pair ({} games), turn cap {k}\n",
        2 * n
    );
    for i in 0..names.len() {
        for j in i + 1..names.len() {
            let (x, y) = (names[i], names[j]);
            let m = run_match(kind(x), kind(y), s..s + n, &MapParams::default(), &config);
            out.push_str(&format!(
                "{x} vs {y}: {x} {} win, {y} {} win, {} draw | by capture {}/{} | mean score diff {:+.1}\n",
                m.a_wins, m.b_wins, m.draws, m.a_capture_wins, m.b_capture_wins, m.mean_score_diff
            ));
            for c in &m.checkpoints {
                out.push_str(&format!(
                    "  turn {}: {x} cities {:.1} units {:.1} stars {:.1} score {:.1} | {y} cities {:.1} units {:.1} stars {:.1} score {:.1}\n",
                    c.turn,
                    c.a.cities, c.a.units, c.a.stars, c.a.score,
                    c.b.cities, c.b.units, c.b.stars, c.b.score
                ));
            }
        }
    }
    out
}

#[test]
fn tournament_prints_the_table() {
    let r = run(&[
        "tournament",
        "--bots",
        "rush,economy",
        "--games",
        "2",
        "--turn-cap",
        "12",
        "--seed",
        "4",
    ]);
    assert_eq!(r.code, 0, "err: {}", r.err);
    assert_eq!(r.err, "");
    let want = expected_table(&["rush", "economy"], 2, 12, 4);
    assert!(
        want.contains("  turn 10: "),
        "the table has a checkpoint line"
    );
    assert_eq!(r.out, want);
}

#[test]
fn tournament_runs_every_pair_in_list_order_with_seed_0_by_default() {
    let r = run(&[
        "tournament",
        "--turn-cap",
        "10",
        "--bots",
        "random,rush,economy",
        "--games",
        "1",
    ]);
    assert_eq!(r.code, 0, "err: {}", r.err);
    assert_eq!(
        r.out,
        expected_table(&["random", "rush", "economy"], 1, 10, 0)
    );
}

#[test]
fn tournament_gate() {
    let gate = |bots: &str, min: &str| {
        run(&[
            "tournament",
            "--bots",
            bots,
            "--games",
            "1",
            "--turn-cap",
            "4",
            "--gate-min",
            min,
        ])
    };
    for bots in ["random,rush", "economy,random", "rush,economy,random"] {
        let r = gate(bots, "1.01");
        assert_eq!(r.code, 1, "{bots}: a win rate above 1 is impossible");
        assert!(
            format!("{}{}", r.out, r.err).contains("GATE FAILED"),
            "{bots}: {} {}",
            r.out,
            r.err
        );
        let r = gate(bots, "0.0");
        assert_eq!(r.code, 0, "{bots}: err {}", r.err);
        assert!(!format!("{}{}", r.out, r.err).contains("GATE FAILED"));
    }
    // Only pairs of random against rush or economy are gated.
    let r = gate("rush,economy", "1.01");
    assert_eq!(r.code, 0, "err {}", r.err);
    assert!(!format!("{}{}", r.out, r.err).contains("GATE FAILED"));
}

struct TempDir(std::path::PathBuf);

impl TempDir {
    fn new(name: &str) -> TempDir {
        let f = TempFile::new(name);
        let path = f.0.clone();
        drop(f);
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("temp dir");
        TempDir(path)
    }

    fn arg(&self) -> String {
        self.0.to_str().expect("utf-8 temp path").to_string()
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn tournament_saves_the_first_games_of_each_pair() {
    let dir = TempDir::new("bots_replays");
    let r = run(&[
        "tournament",
        "--bots",
        "rush,random",
        "--games",
        "2",
        "--turn-cap",
        "5",
        "--seed",
        "6",
        "--save-replays",
        &dir.arg(),
        "--save-count",
        "3",
    ]);
    assert_eq!(r.code, 0, "err: {}", r.err);
    let config = cfg(5, false);
    // Game order: seed 6 with rush as player 0, seed 6 swapped, seed 7 with rush as player 0.
    let games = [
        (6, [BotKind::Rush, BotKind::Random]),
        (6, [BotKind::Random, BotKind::Rush]),
        (7, [BotKind::Rush, BotKind::Random]),
    ];
    for (i, (seed, kinds)) in games.into_iter().enumerate() {
        let path = dir.0.join(format!("rush-vs-random-{i}.json"));
        assert!(path.exists(), "{path:?}");
        let log = read_log(&path);
        assert!(replay(&log).is_ok(), "{path:?} verifies");
        let (states, actions) = my_bot_game(seed, kinds, &config, usize::MAX);
        assert_eq!(
            log,
            expected_log(seed, &MapParams::default(), &config, &states, &actions),
            "{path:?}"
        );
    }
    assert!(!dir.0.join("rush-vs-random-3.json").exists());
    let names: Vec<String> = std::fs::read_dir(&dir.0)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    assert_eq!(names.len(), 3, "{names:?}");
}

#[test]
fn tournament_save_names_per_pair() {
    let dir = TempDir::new("bots_replays_pairs");
    let r = run(&[
        "tournament",
        "--bots",
        "rush,random,economy",
        "--games",
        "1",
        "--turn-cap",
        "3",
        "--save-replays",
        &dir.arg(),
        "--save-count",
        "1",
    ]);
    assert_eq!(r.code, 0, "err: {}", r.err);
    let mut names: Vec<String> = std::fs::read_dir(&dir.0)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    assert_eq!(
        names,
        vec![
            "random-vs-economy-0.json",
            "rush-vs-economy-0.json",
            "rush-vs-random-0.json"
        ]
    );
    for n in &names {
        assert!(replay(&read_log(&dir.0.join(n))).is_ok(), "{n}");
    }
}

#[test]
fn usage_errors_exit_2() {
    let missing = TempFile::new("bots_never_written.json");
    let out = missing.arg();
    let cases: Vec<Vec<&str>> = vec![
        vec!["play", "--seed", "1", "--out", &out],
        vec!["play", "--bots", "rush", "--seed", "1", "--out", &out],
        vec!["play", "--bots", "rush,foo", "--seed", "1", "--out", &out],
        vec![
            "play",
            "--bots",
            "rush,economy,random",
            "--seed",
            "1",
            "--out",
            &out,
        ],
        vec!["play", "--bots", "rush,economy", "--out", &out],
        vec!["play", "--bots", "rush,economy", "--seed", "1"],
        vec![
            "play",
            "--bots",
            "rush,economy",
            "--seed",
            "x",
            "--out",
            &out,
        ],
        vec![
            "play",
            "--bots",
            "rush,economy",
            "--seed",
            "1",
            "--out",
            &out,
            "--turn-cap",
            "z",
        ],
        vec![
            "play",
            "--bots",
            "rush,economy",
            "--seed",
            "1",
            "--out",
            &out,
            "--max-actions",
        ],
        vec![
            "play",
            "--bots",
            "rush,economy",
            "--seed",
            "1",
            "--out",
            &out,
            "--bogus",
        ],
        vec!["play", "--bots"],
        vec!["tournament"],
        vec!["tournament", "--bots", "rush"],
        vec!["tournament", "--bots", "rush,nope"],
        vec!["tournament", "--bots", "rush,economy", "--games", "x"],
        vec!["tournament", "--bots", "rush,economy", "--turn-cap", "x"],
        vec!["tournament", "--bots", "rush,economy", "--seed", "-3"],
        vec!["tournament", "--bots", "rush,economy", "--gate-min", "high"],
        vec!["tournament", "--bots", "rush,economy", "--bogus"],
    ];
    for args in cases {
        let r = run(&args);
        assert_eq!(r.code, 2, "{args:?}");
        assert!(!r.err.is_empty(), "{args:?}: usage goes to err");
        assert!(
            r.out.is_empty(),
            "{args:?}: nothing on out, got {:?}",
            r.out
        );
    }
    assert!(
        !missing.0.exists(),
        "a usage error must not write the output file"
    );
}

#[test]
fn usage_text_lists_the_new_subcommands() {
    let r = run(&[]);
    assert_eq!(r.code, 2);
    assert!(r.err.contains("polytopia play --bots"), "{}", r.err);
    assert!(r.err.contains("polytopia tournament --bots"), "{}", r.err);
}

#[test]
fn random_bot_and_random_action_agree() {
    // H4: same states, equal rngs -> same action and same draws.
    let mut checked = 0;
    for seed in 0..10u64 {
        let config = cfg(if seed % 2 == 0 { 8 } else { 30 }, seed == 3);
        let (states, _) = play_independent(seed, &MapParams::default(), &config, 1500, seed + 77);
        for (i, s) in states.iter().enumerate() {
            if s.is_terminal() {
                continue;
            }
            let mut a = Rng::new(seed * 10_000 + i as u64);
            let mut b = a.clone();
            let want = random_action(s, &mut a);
            let got = RandomBot.choose(&PlayerView::new(s, s.current), &mut b);
            assert_eq!(got, want, "seed {seed} state {i}");
            assert_eq!(b, a, "seed {seed} state {i}: draws");
            checked += 1;
        }
    }
    assert!(checked > 1000, "{checked}");
}
