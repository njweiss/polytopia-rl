#!/usr/bin/env bash
# Milestone 9: build polytopia_sim into the active Python env (the pixi env: `pixi run`) and run
# the pytest suite. Called by check.sh when maturin and python3 are on the PATH.
# In a git worktree, build a wheel into target/wheels and unpack it to target/pyext instead of
# running `maturin develop` into the shared Python env.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo build -q --release -p sim-tools --bin polytopia
git_dir="$(cd "$(git rev-parse --git-dir)" && pwd -P)"
common_dir="$(cd "$(git rev-parse --git-common-dir)" && pwd -P)"
if [ "$git_dir" != "$common_dir" ]; then
  rm -rf target/wheels target/pyext
  mkdir -p target/wheels target/pyext
  (cd python && maturin build -q --release -i "$(command -v python3)" -o ../target/wheels)
  python3 -m zipfile -e target/wheels/*.whl target/pyext
  export PYTHONPATH="$PWD/target/pyext${PYTHONPATH:+:$PYTHONPATH}"
else
  (cd python && maturin develop -q --release)
fi
POLYTOPIA_BIN="$PWD/target/release/polytopia" python3 -m pytest -q python/tests --ignore=python/tests/rl
