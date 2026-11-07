//! Spec hashes stored in checkpoints (docs/MILESTONE_11 "Hashes and checkpoints").

use sim_core::data::{ALL_REWARDS, ALL_SITE_ACTIONS, ALL_TECHS, ALL_UNITS};
use sim_core::state::Ruleset;

use crate::actions::{arg_len, ActionType};
use crate::obs::{obs_spec, snake_name};

/// Bumped by hand for a rules change that lives in code rather than in `data/*.json`; part of
/// `rules_hash` (docs/MILESTONE_11 L9).
pub const RULES_VERSION: u32 = 1;

/// 64-bit FNV-1a: offset basis `0xcbf29ce484222325`, prime `0x100000001b3`.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// `h` as 16 lowercase hex digits (`{:016x}`).
pub fn hex64(h: u64) -> String {
    format!("{h:016x}")
}

/// The string `obs_hash(size)` hashes: `size`, then `channel_names`, `global_names`, the 10
/// action-type names, `arg_len` per action type (decimal), and the names of the unit types,
/// rewards, techs and site actions, all as one flat list of items joined with `\n` (no trailing
/// newline).
pub fn obs_canonical(size: u8) -> String {
    let spec = obs_spec(size);
    let mut items = vec![size.to_string()];
    items.extend(spec.channel_names);
    items.extend(spec.global_names);
    items.extend(ActionType::ALL.iter().map(|&k| snake_name(k)));
    items.extend(
        ActionType::ALL
            .iter()
            .map(|&k| arg_len(size, k).to_string()),
    );
    items.extend(ALL_UNITS.iter().map(|&x| snake_name(x)));
    items.extend(ALL_REWARDS.iter().map(|&x| snake_name(x)));
    items.extend(ALL_TECHS.iter().map(|&x| snake_name(x)));
    items.extend(ALL_SITE_ACTIONS.iter().map(|&x| snake_name(x)));
    items.join("\n")
}

/// `hex64(fnv1a64(obs_canonical(size)))`.
pub fn obs_hash(size: u8) -> String {
    hex64(fnv1a64(obs_canonical(size).as_bytes()))
}

/// Every `data/*.json` the simulator embeds, as `(file name, contents)` in file-name order.
pub fn embedded_data_files() -> Vec<(&'static str, &'static str)> {
    vec![
        ("rewards.json", include_str!("../../../data/rewards.json")),
        ("rules.json", include_str!("../../../data/rules.json")),
        (
            "site_actions.json",
            include_str!("../../../data/site_actions.json"),
        ),
        ("techs.json", include_str!("../../../data/techs.json")),
        ("units.json", include_str!("../../../data/units.json")),
    ]
}

/// For each `(name, contents)` in order: the name's bytes, a `\0`, the contents' bytes; then the
/// bytes of `RULES_VERSION`, a `\0` and `rules_version` in decimal.
pub fn rules_canonical(files: &[(&str, &str)], rules_version: u32) -> Vec<u8> {
    let mut out = Vec::new();
    for (name, contents) in files {
        out.extend_from_slice(name.as_bytes());
        out.push(0);
        out.extend_from_slice(contents.as_bytes());
    }
    out.extend_from_slice(b"RULES_VERSION");
    out.push(0);
    out.extend_from_slice(rules_version.to_string().as_bytes());
    out
}

/// `hex64(fnv1a64(&rules_canonical(&embedded_data_files(), RULES_VERSION)))`.
pub fn rules_hash() -> String {
    hex64(fnv1a64(&rules_canonical(
        &embedded_data_files(),
        RULES_VERSION,
    )))
}

/// Every `data/v2/*.json` the simulator embeds, as `("v2/<file name>", contents)` in file-name
/// order.
pub fn embedded_v2_data_files() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            "v2/buildings.json",
            include_str!("../../../data/v2/buildings.json"),
        ),
        (
            "v2/monuments.json",
            include_str!("../../../data/v2/monuments.json"),
        ),
        ("v2/rules.json", include_str!("../../../data/v2/rules.json")),
        (
            "v2/site_actions.json",
            include_str!("../../../data/v2/site_actions.json"),
        ),
        ("v2/techs.json", include_str!("../../../data/v2/techs.json")),
        ("v2/units.json", include_str!("../../../data/v2/units.json")),
    ]
}

/// `V1`: `rules_hash()`. `V2`: FNV-1a 64 of the v1 canonical bytes, then for each
/// `embedded_v2_data_files()` entry its name, `\0`, its contents, then `RULESET`, `\0`, `v2`
/// (docs/MILESTONE_15 "Hashes").
pub fn rules_hash_for(ruleset: Ruleset) -> String {
    match ruleset {
        Ruleset::V1 => rules_hash(),
        Ruleset::V2 => unimplemented!("milestone 15: rules_hash_for(V2)"),
    }
}
