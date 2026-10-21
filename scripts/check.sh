#!/usr/bin/env bash
# The one gate. An agent's work is not done until this exits 0.
set -euo pipefail
cd "$(dirname "$0")/.."
# Test lock (set by /implement N): tests must be byte-identical to the test commit.
lock="$(git rev-parse --git-dir)/tests-lock"
if [ -f "$lock" ]; then
  base="$(cat "$lock")"
  if ! git diff --quiet "$base" -- 'crates/*/tests/*' \
     || [ -n "$(git ls-files --others --exclude-standard -- 'crates/*/tests/*')" ]; then
    echo "check.sh: FAIL: files under crates/*/tests/ differ from test commit $base" >&2
    git diff --stat "$base" -- 'crates/*/tests/*' >&2
    exit 1
  fi
fi
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
echo "check.sh: OK"
