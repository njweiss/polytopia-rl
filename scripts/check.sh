#!/usr/bin/env bash
# The one gate. An agent's work is not done until this exits 0.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
echo "check.sh: OK"
