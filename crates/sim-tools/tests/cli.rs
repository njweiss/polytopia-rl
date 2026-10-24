//! Milestone 7: the `polytopia` CLI through `run_cli` (docs/MILESTONE_7 "CLI", G3, G7, "Tests"
//! item 6). Files live under unique `std::env::temp_dir()` names and are removed on drop.
//! Expected logs come from `common::play_independent` with the CLI's policy seed; expected views
//! from `render` of independently re-simulated states.

mod common;

use common::*;
use sim_core::map::MapParams;
use sim_core::state::{Outcome, State};
use sim_tools::html::html_page;
use sim_tools::render::{action_label, render, View};
use sim_tools::replay::ReplayLog;

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

fn read_log(f: &TempFile) -> ReplayLog {
    let text = std::fs::read_to_string(&f.0).expect("log file written");
    serde_json::from_str(&text).expect("log file is a ReplayLog")
}

/// Writes a log with `play-random` and returns it with its re-simulated states.
fn play(f: &TempFile, seed: u64, turn_cap: u32) -> (ReplayLog, Vec<State>) {
    let r = run(&[
        "play-random",
        "--seed",
        &seed.to_string(),
        "--out",
        &f.arg(),
        "--turn-cap",
        &turn_cap.to_string(),
    ]);
    assert_eq!(r.code, 0, "err: {}", r.err);
    let log = read_log(f);
    let states = resimulate(&log);
    (log, states)
}

#[test]
fn play_random_writes_the_expected_log_and_message() {
    let f = TempFile::new("play.json");
    let r = run(&[
        "play-random",
        "--seed",
        "7",
        "--out",
        &f.arg(),
        "--turn-cap",
        "3",
    ]);
    assert_eq!(r.code, 0, "err: {}", r.err);
    assert_eq!(r.err, "");
    let log = read_log(&f);
    let config = cfg(3, false);
    let (states, actions) =
        play_independent(7, &MapParams::default(), &config, 5000, 7 ^ POLICY_SEED_XOR);
    assert_eq!(
        log,
        expected_log(7, &MapParams::default(), &config, &states, &actions)
    );
    assert!(log.outcome.is_some());
    assert_eq!(
        r.out.trim_end_matches('\n'),
        format!(
            "wrote {}: {} actions, outcome {}",
            f.arg(),
            actions.len(),
            outcome_text(log.outcome)
        )
    );
}

#[test]
fn play_random_options_max_actions_and_full_observability() {
    let f = TempFile::new("opts.json");
    let r = run(&[
        "play-random",
        "--full-observability",
        "--max-actions",
        "25",
        "--out",
        &f.arg(),
        "--seed",
        "3",
    ]);
    assert_eq!(r.code, 0, "err: {}", r.err);
    let log = read_log(&f);
    let config = cfg(30, true);
    let (states, actions) =
        play_independent(3, &MapParams::default(), &config, 25, 3 ^ POLICY_SEED_XOR);
    assert_eq!(actions.len(), 25);
    assert_eq!(
        log,
        expected_log(3, &MapParams::default(), &config, &states, &actions)
    );
    assert_eq!(
        r.out.trim_end_matches('\n'),
        format!("wrote {}: 25 actions, outcome unfinished", f.arg())
    );
}

#[test]
fn play_random_defaults_are_turn_cap_30_and_5000_actions() {
    let f = TempFile::new("defaults.json");
    let r = run(&["play-random", "--seed", "12", "--out", &f.arg()]);
    assert_eq!(r.code, 0, "err: {}", r.err);
    let log = read_log(&f);
    let config = cfg(30, false);
    assert_eq!(log.config, config);
    assert_eq!(log.map_params, MapParams::default());
    let (states, actions) = play_independent(
        12,
        &MapParams::default(),
        &config,
        5000,
        12 ^ POLICY_SEED_XOR,
    );
    assert_eq!(
        log,
        expected_log(12, &MapParams::default(), &config, &states, &actions)
    );
}

#[test]
fn verify_accepts_a_good_log() {
    let f = TempFile::new("verify.json");
    let (log, _) = play(&f, 21, 2);
    let r = run(&["verify", &f.arg()]);
    assert_eq!(r.code, 0, "err: {}", r.err);
    assert_eq!(r.err, "");
    assert_eq!(
        r.out.trim_end_matches('\n'),
        format!(
            "OK: {} actions, outcome {}",
            log.actions.len(),
            outcome_text(log.outcome)
        )
    );
}

#[test]
fn verify_rejects_a_corrupted_log_with_exit_1() {
    let f = TempFile::new("corrupt.json");
    let (mut log, _) = play(&f, 22, 2);
    log.digests[3] = "0000000000000000".into();
    std::fs::write(&f.0, serde_json::to_string(&log).unwrap()).unwrap();
    let r = run(&["verify", &f.arg()]);
    assert_eq!(r.code, 1);
    assert!(!r.err.is_empty(), "the error goes to err");
    assert!(!r.out.contains("OK"), "out: {}", r.out);

    std::fs::write(&f.0, "this is not a replay").unwrap();
    let r = run(&["verify", &f.arg()]);
    assert_eq!(r.code, 1);
    assert!(!r.err.is_empty());
}

#[test]
fn view_step_prints_the_render_of_that_state() {
    let f = TempFile::new("view.json");
    let (log, states) = play(&f, 23, 2);
    let n = log.actions.len();
    for i in [0, 1, n / 2, n] {
        let r = run(&["view", &f.arg(), "--step", &i.to_string()]);
        assert_eq!(r.code, 0, "err: {}", r.err);
        assert_eq!(r.out, render(&states[i], View::Omniscient), "step {i}");
        for pl in [0u8, 1] {
            let r = run(&[
                "view",
                &f.arg(),
                "--player",
                &pl.to_string(),
                "--step",
                &i.to_string(),
            ]);
            assert_eq!(r.code, 0, "err: {}", r.err);
            assert_eq!(
                r.out,
                render(&states[i], View::Player(pl)),
                "step {i} P{pl}"
            );
        }
    }
    // Default: the last state.
    let r = run(&["view", &f.arg()]);
    assert_eq!(r.code, 0);
    assert_eq!(r.out, render(&states[n], View::Omniscient));
    let r = run(&["view", &f.arg(), "--player", "1"]);
    assert_eq!(r.code, 0);
    assert_eq!(r.out, render(&states[n], View::Player(1)));
}

fn expected_all(log: &ReplayLog, states: &[State], view: View) -> String {
    let mut s = String::new();
    for (i, st) in states.iter().enumerate() {
        let label = if i == 0 {
            "start".to_string()
        } else {
            action_label(&log.actions[i - 1])
        };
        s.push_str(&format!("== step {i}: {label} ==\n"));
        s.push_str(&render(st, view));
    }
    s
}

#[test]
fn view_all_prints_every_state_with_headers() {
    let f = TempFile::new("all.json");
    let (log, states) = play(&f, 24, 1);
    let r = run(&["view", &f.arg(), "--all"]);
    assert_eq!(r.code, 0, "err: {}", r.err);
    assert_eq!(
        r.out.lines().filter(|l| l.starts_with("== step ")).count(),
        log.actions.len() + 1
    );
    assert!(r.out.starts_with("== step 0: start ==\n"));
    assert_eq!(r.out, expected_all(&log, &states, View::Omniscient));
    let r = run(&["view", &f.arg(), "--all", "--player", "0"]);
    assert_eq!(r.code, 0);
    assert_eq!(r.out, expected_all(&log, &states, View::Player(0)));
}

#[test]
fn html_writes_the_page() {
    let f = TempFile::new("html.json");
    let out = TempFile::new("page.html");
    let (log, _) = play(&f, 25, 2);
    let r = run(&["html", &f.arg(), &out.arg()]);
    assert_eq!(r.code, 0, "err: {}", r.err);
    let page = std::fs::read_to_string(&out.0).expect("page written");
    assert_eq!(page, html_page(&log).unwrap());
    assert!(page.contains("const REPLAY ="));
}

#[test]
fn usage_errors_exit_2_and_write_only_to_err() {
    let f = TempFile::new("usage.json");
    let (log, _) = play(&f, 26, 1);
    let file = f.arg();
    let missing_out = TempFile::new("never_written.json");
    let too_far = (log.actions.len() + 1).to_string();
    let cases: Vec<Vec<&str>> = vec![
        vec![],
        vec!["bogus"],
        vec!["--seed", "1"],
        vec!["play-random"],
        vec!["play-random", "--seed", "1"],
        vec!["play-random", "--out", &missing_out.0.to_str().unwrap()],
        vec![
            "play-random",
            "--seed",
            "x",
            "--out",
            missing_out.0.to_str().unwrap(),
        ],
        vec![
            "play-random",
            "--seed",
            "-1",
            "--out",
            missing_out.0.to_str().unwrap(),
        ],
        vec!["play-random", "--seed", "1", "--out"],
        vec![
            "play-random",
            "--seed",
            "1",
            "--out",
            missing_out.0.to_str().unwrap(),
            "--turn-cap",
        ],
        vec![
            "play-random",
            "--seed",
            "1",
            "--out",
            missing_out.0.to_str().unwrap(),
            "--turn-cap",
            "ten",
        ],
        vec![
            "play-random",
            "--seed",
            "1",
            "--out",
            missing_out.0.to_str().unwrap(),
            "--max-actions",
            "many",
        ],
        vec![
            "play-random",
            "--seed",
            "1",
            "--out",
            missing_out.0.to_str().unwrap(),
            "--frobnicate",
        ],
        vec!["verify"],
        vec!["view"],
        vec!["view", &file, "--step"],
        vec!["view", &file, "--step", "abc"],
        vec!["view", &file, "--step", &too_far],
        vec!["view", &file, "--player", "2"],
        vec!["view", &file, "--player"],
        vec!["view", &file, "--bogus"],
        vec!["html"],
        vec!["html", &file],
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
        !missing_out.0.exists(),
        "a usage error must not write the output file"
    );
}
