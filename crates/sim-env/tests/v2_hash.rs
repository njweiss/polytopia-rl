//! Milestone 15 "Tests" items 6 and 7: `rules_hash_for` (docs/MILESTONE_15 "Hashes") and the Env
//! refusing v2 until milestone 20 (docs/MILESTONE_15 "Env"; the Rust form is docs/QUESTIONS.md
//! Q43). The v2 golden is the test's own FNV-1a over the brief's canonical bytes, built from the
//! files on disk: the v1 canonical bytes (milestone 11 layout, `RULES_VERSION` included), then
//! for each `data/v2/*.json` in file-name order `"v2/<file>"`, `\0`, its contents, then
//! `RULESET`, `\0`, `v2`.

mod common;

use std::fs;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};

use common::*;
use sim_core::state::{Config, Ruleset};
use sim_env::env::{AgentSeat, Env, EnvConfig, Opponent};
use sim_env::hash::{
    embedded_data_files, embedded_v2_data_files, rules_hash, rules_hash_for, RULES_VERSION,
};

fn my_fnv(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data")
}

/// `(prefix + file name, contents)` of every `*.json` directly in `dir`, in file-name order.
fn json_files(dir: &Path, prefix: &str) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = fs::read_dir(dir)
        .expect("data dir")
        .map(|e| e.expect("dir entry").path())
        .filter(|p| p.is_file() && p.extension().is_some_and(|x| x == "json"))
        .map(|p| {
            let name = p.file_name().unwrap().to_str().unwrap().to_string();
            (format!("{prefix}{name}"), fs::read_to_string(&p).unwrap())
        })
        .collect();
    out.sort();
    out
}

fn my_v1_bytes() -> Vec<u8> {
    let mut out = Vec::new();
    for (name, contents) in json_files(&data_dir(), "") {
        out.extend_from_slice(name.as_bytes());
        out.push(0);
        out.extend_from_slice(contents.as_bytes());
    }
    out.extend_from_slice(b"RULES_VERSION\x00");
    out.extend_from_slice(RULES_VERSION.to_string().as_bytes());
    out
}

fn my_v2_bytes(v2_files: &[(String, String)]) -> Vec<u8> {
    let mut out = my_v1_bytes();
    for (name, contents) in v2_files {
        out.extend_from_slice(name.as_bytes());
        out.push(0);
        out.extend_from_slice(contents.as_bytes());
    }
    out.extend_from_slice(b"RULESET\x00v2");
    out
}

#[test]
fn rules_hash_for_v1_is_the_recorded_v1_hash() {
    assert_eq!(rules_hash_for(Ruleset::V1), rules_hash());
    assert_eq!(rules_hash_for(Ruleset::V1), "a8b52b91eae34503");
    assert_eq!(
        format!("{:016x}", my_fnv(&my_v1_bytes())),
        "a8b52b91eae34503"
    );
    // The v1 file list does not pick up the v2 directory.
    let names: Vec<&str> = embedded_data_files().iter().map(|f| f.0).collect();
    assert_eq!(
        names,
        [
            "rewards.json",
            "rules.json",
            "site_actions.json",
            "techs.json",
            "units.json"
        ]
    );
}

#[test]
fn embedded_v2_data_files_are_the_v2_dir_in_name_order() {
    let disk = json_files(&data_dir().join("v2"), "v2/");
    let names: Vec<&str> = disk.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(
        names,
        [
            "v2/buildings.json",
            "v2/monuments.json",
            "v2/rules.json",
            "v2/site_actions.json",
            "v2/techs.json",
            "v2/units.json"
        ]
    );
    let emb = embedded_v2_data_files();
    assert_eq!(emb.len(), disk.len());
    for ((en, ec), (dn, dc)) in emb.iter().zip(&disk) {
        assert_eq!(en, dn);
        assert_eq!(ec, dc, "{en}: embedded bytes equal the file");
    }
}

#[test]
fn rules_hash_for_v2_is_the_documented_canonical() {
    let disk = json_files(&data_dir().join("v2"), "v2/");
    let golden = format!("{:016x}", my_fnv(&my_v2_bytes(&disk)));
    let h = rules_hash_for(Ruleset::V2);
    assert_eq!(h, golden);
    assert_eq!(h, rules_hash_for(Ruleset::V2), "deterministic");
    assert_ne!(h, rules_hash());
    // The v1 hash is not moved by the v2 files.
    assert_eq!(rules_hash(), "a8b52b91eae34503");
}

#[test]
fn the_v2_canonical_changes_with_one_v2_byte() {
    let disk = json_files(&data_dir().join("v2"), "v2/");
    let base = my_fnv(&my_v2_bytes(&disk));
    assert_eq!(rules_hash_for(Ruleset::V2), format!("{base:016x}"));
    for i in 0..disk.len() {
        let mut changed = disk.clone();
        let c = &mut changed[i].1;
        let at = c.find(|ch: char| ch.is_ascii_digit()).expect("a digit");
        let mut b = c.clone().into_bytes();
        b[at] = if b[at] == b'9' { b'8' } else { b[at] + 1 };
        *c = String::from_utf8(b).unwrap();
        assert_ne!(my_fnv(&my_v2_bytes(&changed)), base, "{}", disk[i].0);
    }
}

fn v2_env_cfg() -> EnvConfig {
    EnvConfig {
        config: Config {
            ruleset: Ruleset::V2,
            ..cfg(10)
        },
        ..env_cfg(Opponent::TwoSeat, AgentSeat::Fixed(0), 10)
    }
}

fn panic_message(e: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = e.downcast_ref::<String>() {
        s.clone()
    } else if let Some(s) = e.downcast_ref::<&str>() {
        s.to_string()
    } else {
        String::new()
    }
}

#[test]
fn env_refuses_v2_naming_milestone_20() {
    let r = catch_unwind(AssertUnwindSafe(|| Env::new(v2_env_cfg())));
    let msg = panic_message(r.err().expect("Env::new refuses a v2 config"));
    assert!(msg.contains("milestone 20"), "{msg:?}");
    // v1 is fine.
    let mut env = Env::new(env_cfg(Opponent::TwoSeat, AgentSeat::Fixed(0), 10));
    env.reset(0);
    assert!(!env.is_terminal());
}
