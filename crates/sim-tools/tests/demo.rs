//! Milestone 7: `scripts/demo.sh` exists (docs/MILESTONE_7 "scripts/demo.sh", "Tests" item 8).
//! Running it is part of the gate and the review phase (it builds in release), not of
//! `cargo test`.

use std::os::unix::fs::PermissionsExt;
use std::path::Path;

#[test]
fn demo_script_exists_is_executable_and_writes_the_demo_paths() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/demo.sh");
    let meta = std::fs::metadata(&path).expect("scripts/demo.sh exists");
    assert!(
        meta.permissions().mode() & 0o111 != 0,
        "demo.sh is executable"
    );
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.contains("target/demo/replay.json"));
    assert!(text.contains("target/demo/replay.html"));
}
