//! Milestone 15 "Tests" item 1 (v1 byte-identity, the Env side): under `ruleset: v1` the recorded
//! hashes, observations, masks, rewards and replay JSON are exactly what the pre-milestone-15 build
//! produced.
//!
//! `rules_hash` and `obs_hash(11)` are the values the brief records (docs/MILESTONE_15 "Hashes",
//! also docs/RL_REPORT.md). `EPISODE_FOLD` is this file's `episode_fold()` run on the build of
//! commit 580ea42 (main before milestone 15), never on milestone 15 code.

mod common;

use bots::BotKind;
use common::*;
use sim_env::env::{AgentSeat, Env};
use sim_env::hash::{obs_hash, rules_hash};
use sim_env::obs::{encode_board_with_legal, encode_globals, obs_spec};

/// Recorded with the pre-milestone-15 build (see the module comment).
const EPISODE_FOLD: &str = "20ea3ea53c74464b";

fn fnv(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h ^= b as u64;
        *h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

/// FNV-1a over four Env episodes (agent seat 0 against Rush, the agent playing Economy through
/// `bot_action`): per step the board and globals (f32 bits, little-endian), the actor mask and the
/// reward bits; per episode the replay JSON.
fn episode_fold() -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let size = 11;
    let spec = obs_spec(size);
    let mut steps = 0;
    for seed in 0..4u64 {
        let mut env = Env::new(env_cfg(bot(BotKind::Rush, 0.0), AgentSeat::Fixed(0), 12));
        env.reset(seed);
        loop {
            let mut board = vec![0.0f32; spec.channels * size as usize * size as usize];
            let mut globals = vec![0.0f32; spec.globals];
            encode_board_with_legal(&env.view(), env.legal_actions(), &mut board);
            encode_globals(&env.view(), &mut globals);
            for x in board.iter().chain(&globals) {
                fnv(&mut h, &x.to_bits().to_le_bytes());
            }
            for &m in env.masks().actor() {
                fnv(&mut h, &[m as u8]);
            }
            if env.is_terminal() {
                break;
            }
            let f = env.bot_action(BotKind::Economy).unwrap();
            let out = env.step_factored(&f).unwrap();
            fnv(&mut h, &out.reward.to_bits().to_le_bytes());
            steps += 1;
            if out.terminated || out.truncated {
                break;
            }
        }
        fnv(&mut h, env.replay_json().as_bytes());
    }
    assert!(
        steps > 100,
        "the episodes are long enough to matter ({steps})"
    );
    format!("{h:016x}")
}

#[test]
fn v1_rules_and_obs_hashes_are_the_recorded_values() {
    assert_eq!(rules_hash(), "a8b52b91eae34503");
    assert_eq!(obs_hash(11), "d5f9b1033cfd55ca");
}

#[test]
fn v1_env_episodes_are_unchanged() {
    assert_eq!(episode_fold(), EPISODE_FOLD);
}
