//! Milestone 11 "Tests" item 1: `sim_env::hash`.
//!
//! FNV-1a 64 test vectors are the published ones (Fowler/Noll/Vo reference test suite):
//! `""` -> cbf29ce484222325, `"a"` -> af63dc4c8601ec8c, `"foobar"` -> 85944171f73967e8. The
//! golden canonical string for `size = 3` is written out by hand from the brief's item order and
//! the milestone 9 name tables (docs/MILESTONE_9, docs/QUESTIONS.md Q15; the same lists as
//! python/tests/test_env.py): 37 channels, 18 globals, 10 action types, the arg lengths
//! `[9, 9, 0, 0, 0, 5, 8, 6, 9, 0]` for a 3x3 board, 5 unit types, 8 rewards, 9 techs and 6 site
//! actions. The canonical-format details (flat item list, no trailing newline, `RULES_VERSION`
//! entry after the files) are docs/QUESTIONS.md Q22. Expected hashes are computed with the
//! test's own FNV-1a below, never with the code under test.

use std::fs;
use std::path::PathBuf;

use sim_env::hash::{
    embedded_data_files, fnv1a64, hex64, obs_canonical, obs_hash, rules_canonical, rules_hash,
    RULES_VERSION,
};

const GOLDEN_3: &str = "\
    3\nexplored\nterrain.field\nterrain.forest\n\
    terrain.mountain\nresource.fruit\nresource.crop\nresource.animal\n\
    resource.metal\nimprovement.lumber_hut\nimprovement.farm\nimprovement.mine\n\
    village\nterritory.mine\nterritory.enemy\ncity.mine\n\
    city.enemy\ncity.level\ncity.capital\nmy_unit.warrior\n\
    my_unit.rider\nmy_unit.archer\nmy_unit.defender\nmy_unit.super_unit\n\
    enemy_unit.warrior\nenemy_unit.rider\nenemy_unit.archer\nenemy_unit.defender\n\
    enemy_unit.super_unit\nhealth\nphase.fresh\nphase.moved\n\
    phase.attacked\nphase.done\nveteran\ncan_promote\n\
    can_harvest\ncan_build\nstars\nincome\n\
    turn\npending_reward\nmy_score\nopp_score\n\
    score_diff\nmy_cities\nmy_units\ntech.climbing\n\
    tech.hunting\ntech.organization\ntech.riding\ntech.archery\n\
    tech.forestry\ntech.farming\ntech.strategy\ntech.mining\n\
    move\nattack\ncapture\nrecover\n\
    promote\ntrain\nchoose_reward\nsite\n\
    research\nend_turn\n9\n9\n\
    0\n0\n0\n5\n\
    8\n6\n9\n0\n\
    warrior\nrider\narcher\ndefender\n\
    super_unit\nworkshop\nexplorer\ncity_wall\n\
    resources\npopulation_growth\nborder_growth\npark\n\
    super_unit\nclimbing\nhunting\norganization\n\
    riding\narchery\nforestry\nfarming\n\
    strategy\nmining\nharvest_animal\nharvest_fruit\n\
    lumber_hut\nclear_forest\nfarm\nmine";

/// Independent FNV-1a 64.
fn my_fnv(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn my_hex(bytes: &[u8]) -> String {
    format!("{:016x}", my_fnv(bytes))
}

fn is_hex16(s: &str) -> bool {
    s.len() == 16
        && s.chars()
            .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c))
}

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data")
}

/// `(file name, contents)` of every `data/*.json` on disk, in file-name order.
fn disk_files() -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = fs::read_dir(data_dir())
        .expect("data dir")
        .map(|e| e.expect("dir entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .map(|p| {
            let name = p.file_name().unwrap().to_str().unwrap().to_string();
            (name, fs::read_to_string(&p).expect("readable"))
        })
        .collect();
    out.sort();
    out
}

/// Independent build of the rules byte string (brief "Public API" plus Q22).
fn my_rules_bytes(files: &[(String, String)], version: u32) -> Vec<u8> {
    let mut out = Vec::new();
    for (name, contents) in files {
        out.extend_from_slice(name.as_bytes());
        out.push(0);
        out.extend_from_slice(contents.as_bytes());
    }
    out.extend_from_slice(b"RULES_VERSION");
    out.push(0);
    out.extend_from_slice(version.to_string().as_bytes());
    out
}

#[test]
fn fnv1a64_known_vectors() {
    assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
    assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
    assert_eq!(fnv1a64(b"foobar"), 0x8594_4171_f739_67e8);
    // And it agrees with the independent implementation on arbitrary bytes.
    let bytes: Vec<u8> = (0..=255u8).chain((0..=255u8).rev()).collect();
    assert_eq!(fnv1a64(&bytes), my_fnv(&bytes));
}

#[test]
fn hex64_is_16_lowercase_digits() {
    assert_eq!(hex64(0), "0000000000000000");
    assert_eq!(hex64(0xabc), "0000000000000abc");
    assert_eq!(hex64(u64::MAX), "ffffffffffffffff");
    assert_eq!(hex64(0xcbf2_9ce4_8422_2325), "cbf29ce484222325");
}

#[test]
fn obs_canonical_for_size_3_matches_the_golden_string() {
    assert_eq!(obs_canonical(3), GOLDEN_3);
    assert_eq!(GOLDEN_3.split('\n').count(), 104);
    assert!(!GOLDEN_3.ends_with('\n'));
}

#[test]
fn obs_hash_is_the_fnv_of_the_canonical_string() {
    assert_eq!(obs_hash(3), my_hex(GOLDEN_3.as_bytes()));
    for size in [6u8, 8, 11] {
        let h = obs_hash(size);
        assert!(is_hex16(&h), "{h:?}");
        assert_eq!(h, my_hex(obs_canonical(size).as_bytes()));
        assert_eq!(h, obs_hash(size), "deterministic");
    }
}

#[test]
fn obs_canonical_for_size_11_differs_only_in_size_and_tile_arg_lengths() {
    // Items 0 (size) and the Move/Attack arg lengths (3 * 3 = 9 -> 11 * 11 = 121) depend on size.
    let small: Vec<&str> = GOLDEN_3.split('\n').collect();
    let canon = obs_canonical(11);
    let big: Vec<&str> = canon.split('\n').collect();
    assert_eq!(big.len(), small.len());
    // 1 size item + 37 channels + 18 globals + 10 action types precede the arg lengths.
    let args = 1 + 37 + 18 + 10;
    for (i, (a, b)) in small.iter().zip(&big).enumerate() {
        match i {
            0 => assert_eq!((*a, *b), ("3", "11")),
            i if i == args || i == args + 1 => assert_eq!((*a, *b), ("9", "121")),
            _ => assert_eq!(a, b, "item {i}"),
        }
    }
    assert_ne!(obs_hash(11), obs_hash(3));
    assert_ne!(obs_hash(11), obs_hash(8));
}

/// The hash of `GOLDEN_3` with one ingredient changed.
fn modified(f: impl Fn(&mut Vec<String>)) -> String {
    let mut items: Vec<String> = GOLDEN_3.split('\n').map(str::to_string).collect();
    f(&mut items);
    hex64(fnv1a64(items.join("\n").as_bytes()))
}

#[test]
fn obs_hash_changes_when_any_ingredient_changes() {
    let base = obs_hash(3);
    let args = 1 + 37 + 18 + 10;
    let units = args + 10;
    let changes: Vec<(&str, String)> = vec![
        ("size", modified(|v| v[0] = "4".into())),
        ("a channel name", modified(|v| v[1] = "explored2".into())),
        ("channel order", modified(|v| v.swap(2, 3))),
        ("a global name", modified(|v| v[38] = "stars2".into())),
        ("action-type order", modified(|v| v.swap(56, 57))),
        ("an arg length", modified(|v| v[args + 5] = "6".into())),
        ("unit order", modified(|v| v.swap(units, units + 1))),
        ("reward order", modified(|v| v.swap(units + 5, units + 6))),
        ("tech order", modified(|v| v.swap(units + 13, units + 14))),
        (
            "site-action order",
            modified(|v| v.swap(units + 22, units + 23)),
        ),
        ("an extra site action", modified(|v| v.push("fish".into()))),
    ];
    assert_eq!(
        modified(|_| {}),
        base,
        "the unmodified list hashes to obs_hash(3)"
    );
    for (what, h) in &changes {
        assert_ne!(*h, base, "{what}");
    }
}

#[test]
fn embedded_data_files_are_the_data_dir_in_name_order() {
    let disk = disk_files();
    let names: Vec<&str> = disk.iter().map(|(n, _)| n.as_str()).collect();
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
    let emb = embedded_data_files();
    assert_eq!(emb.len(), disk.len());
    for ((en, ec), (dn, dc)) in emb.iter().zip(&disk) {
        assert_eq!(en, dn);
        assert_eq!(ec, dc, "{en}: embedded bytes equal the file");
    }
}

#[test]
fn rules_canonical_layout() {
    let files = [("a.json", "{}"), ("b.json", "x")];
    assert_eq!(
        rules_canonical(&files, 7),
        b"a.json\0{}b.json\0xRULES_VERSION\x007".to_vec()
    );
    assert_eq!(rules_canonical(&[], 12), b"RULES_VERSION\x0012".to_vec());
}

#[test]
fn rules_hash_is_the_fnv_of_the_data_files_and_version() {
    assert_eq!(RULES_VERSION, 1);
    let h = rules_hash();
    assert!(is_hex16(&h), "{h:?}");
    assert_eq!(h, my_hex(&my_rules_bytes(&disk_files(), RULES_VERSION)));
    assert_eq!(h, rules_hash(), "deterministic");
}

#[test]
fn rules_hash_changes_with_one_data_byte_or_the_version() {
    let disk = disk_files();
    let files: Vec<(&str, &str)> = disk.iter().map(|(n, c)| (n.as_str(), c.as_str())).collect();
    let base = hex64(fnv1a64(&rules_canonical(&files, RULES_VERSION)));
    assert_eq!(base, rules_hash());
    // One byte of units.json (a digit somewhere in the stats) changed.
    let units = &disk[4].1;
    let at = units.find(|c: char| c.is_ascii_digit()).expect("a digit");
    let mut changed = units.clone().into_bytes();
    changed[at] = if changed[at] == b'9' {
        b'8'
    } else {
        changed[at] + 1
    };
    let changed = String::from_utf8(changed).unwrap();
    let mut files2 = files.clone();
    files2[4].1 = &changed;
    assert_ne!(
        hex64(fnv1a64(&rules_canonical(&files2, RULES_VERSION))),
        base
    );
    assert_ne!(
        hex64(fnv1a64(&rules_canonical(&files, RULES_VERSION + 1))),
        base
    );
    // The file names are part of the bytes.
    let mut files3 = files.clone();
    files3[0].0 = "reward.json";
    assert_ne!(
        hex64(fnv1a64(&rules_canonical(&files3, RULES_VERSION))),
        base
    );
}
