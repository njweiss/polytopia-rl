#!/usr/bin/env bash
# Milestone 10: the PPO pytest suite (CPU). Called by check.sh after py_check.sh has built
# polytopia_sim into the active Python env (or target/pyext in a git worktree), and only when
# torch is importable.
set -euo pipefail
cd "$(dirname "$0")/.."
git_dir="$(cd "$(git rev-parse --git-dir)" && pwd -P)"
common_dir="$(cd "$(git rev-parse --git-common-dir)" && pwd -P)"
if [ "$git_dir" != "$common_dir" ]; then
  export PYTHONPATH="$PWD/target/pyext${PYTHONPATH:+:$PYTHONPATH}"
fi
python3 -m pytest -q python/tests/rl
