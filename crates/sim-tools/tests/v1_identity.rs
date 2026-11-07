//! Milestone 15 "Tests" items 1 and 4 (v1 byte-identity): under `ruleset: v1` digests, legal-action
//! lists and replay files are exactly what the pre-milestone-15 build produced.
//!
//! The two recorded values below were produced by the build of commit 580ea42 (main before
//! milestone 15, no `ruleset` anywhere), never by milestone 15 code:
//! - `fixtures/v1_seed5_cap10.json` is the file `polytopia play-random --seed 5 --turn-cap 10`
//!   wrote (68 actions, Win(1)), byte for byte.
//! - `PLAYOUT_FOLD` is this file's `playout_fold()` run on that build.

mod common;

use common::*;
use sim_core::actions::legal_actions;
use sim_core::map::MapParams;
use sim_tools::digest::state_digest;
use sim_tools::replay::{replay, ReplayLog};

const V1_REPLAY: &str = include_str!("fixtures/v1_seed5_cap10.json");

/// Recorded with the pre-milestone-15 build (see the module comment).
const PLAYOUT_FOLD: &str = "13a381e302b30a06";

/// FNV-1a over, for every state of sixteen random v1 playouts: the state's digest as 16 hex digits,
/// then the JSON of its legal-action list, then a newline.
fn playout_fold() -> String {
    let mut bytes = Vec::new();
    let mut states_seen = 0usize;
    for seed in 0..16u64 {
        let (states, _) = play_independent(
            seed,
            &MapParams::default(),
            &cfg(8 + (seed as u32 % 6), seed % 2 == 1),
            3000,
            seed + 77,
        );
        for s in &states {
            bytes.extend_from_slice(format!("{:016x}", state_digest(s)).as_bytes());
            bytes.extend_from_slice(serde_json::to_string(&legal_actions(s)).unwrap().as_bytes());
            bytes.push(b'\n');
            states_seen += 1;
        }
    }
    assert!(
        states_seen > 1000,
        "the playouts are long enough to matter ({states_seen})"
    );
    format!("{:016x}", fnv1a64(&bytes))
}

#[test]
fn v1_digests_and_legal_actions_of_random_playouts_are_unchanged() {
    assert_eq!(playout_fold(), PLAYOUT_FOLD);
}

#[test]
fn a_recorded_v1_replay_file_still_verifies_and_reserializes_byte_for_byte() {
    let log = ReplayLog::from_json(V1_REPLAY).expect("the v1 replay parses");
    assert_eq!(log.actions.len(), 68);
    assert_eq!(log.config.turn_cap, 10);
    let end = replay(&log).expect("the v1 replay verifies");
    assert_eq!(end.outcome, log.outcome);
    assert_eq!(log.to_json(), V1_REPLAY.trim_end());
    assert!(!V1_REPLAY.contains("ruleset"));

    // The CLI accepts it unchanged.
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/v1_seed5_cap10.json");
    let args = vec!["verify".to_string(), path.to_str().unwrap().to_string()];
    let (mut out, mut err) = (Vec::new(), Vec::new());
    let code = sim_tools::cli::run_cli(&args, &mut out, &mut err);
    assert_eq!(code, 0, "{}", String::from_utf8_lossy(&err));
    assert_eq!(
        String::from_utf8(out).unwrap(),
        "OK: 68 actions, outcome Win(1)\n"
    );
}
