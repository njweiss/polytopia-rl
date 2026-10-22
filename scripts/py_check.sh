#!/usr/bin/env bash
# Milestone 9: build polytopia_sim into the active Python env (the pixi env: `pixi run`) and run
# the pytest suite. Called by check.sh when maturin and python3 are on the PATH.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo build -q --release -p sim-tools --bin polytopia
(cd python && maturin develop -q --release)
POLYTOPIA_BIN="$PWD/target/release/polytopia" python3 -m pytest -q python/tests
