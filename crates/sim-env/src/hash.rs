//! Spec hashes stored in checkpoints (docs/MILESTONE_11 "Hashes and checkpoints").

/// Bumped by hand for a rules change that lives in code rather than in `data/*.json`; part of
/// `rules_hash` (docs/MILESTONE_11 L9).
pub const RULES_VERSION: u32 = 1;

/// 64-bit FNV-1a: offset basis `0xcbf29ce484222325`, prime `0x100000001b3`.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let _ = bytes;
    unimplemented!()
}

/// `h` as 16 lowercase hex digits (`{:016x}`).
pub fn hex64(h: u64) -> String {
    let _ = h;
    unimplemented!()
}

/// The string `obs_hash(size)` hashes: `size`, then `channel_names`, `global_names`, the 10
/// action-type names, `arg_len` per action type (decimal), and the names of the unit types,
/// rewards, techs and site actions, all as one flat list of items joined with `\n` (no trailing
/// newline).
pub fn obs_canonical(size: u8) -> String {
    let _ = size;
    unimplemented!()
}

/// `hex64(fnv1a64(obs_canonical(size)))`.
pub fn obs_hash(size: u8) -> String {
    let _ = size;
    unimplemented!()
}

/// Every `data/*.json` the simulator embeds, as `(file name, contents)` in file-name order.
pub fn embedded_data_files() -> Vec<(&'static str, &'static str)> {
    unimplemented!()
}

/// For each `(name, contents)` in order: the name's bytes, a `\0`, the contents' bytes; then the
/// bytes of `RULES_VERSION`, a `\0` and `rules_version` in decimal.
pub fn rules_canonical(files: &[(&str, &str)], rules_version: u32) -> Vec<u8> {
    let _ = (files, rules_version);
    unimplemented!()
}

/// `hex64(fnv1a64(&rules_canonical(&embedded_data_files(), RULES_VERSION)))`.
pub fn rules_hash() -> String {
    unimplemented!()
}
