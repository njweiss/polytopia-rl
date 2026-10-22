#!/usr/bin/env bash
# The one gate. An agent's work is not done until this exits 0.
set -euo pipefail
cd "$(dirname "$0")/.."
# Test lock (set by /implement N): tests must be byte-identical to the test commit.
# Locked: crates/*/tests/ and python/tests/rl/ (milestone 10).
lock="$(git rev-parse --git-dir)/tests-lock"
if [ -f "$lock" ]; then
  base="$(cat "$lock")"
  if ! git diff --quiet "$base" -- 'crates/*/tests/*' 'python/tests/rl/*' \
     || [ -n "$(git ls-files --others --exclude-standard -- 'crates/*/tests/*' 'python/tests/rl/*')" ]; then
    echo "check.sh: FAIL: files under crates/*/tests/ or python/tests/rl/ differ from test commit $base" >&2
    git diff --stat "$base" -- 'crates/*/tests/*' 'python/tests/rl/*' >&2
    exit 1
  fi
fi
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
if command -v maturin >/dev/null && command -v python3 >/dev/null; then
  ./scripts/py_check.sh
else
  echo "SKIPPED python tests (maturin not found)"
fi
echo "check.sh: OK"
