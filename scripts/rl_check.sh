#!/usr/bin/env bash
# Milestone 10: the PPO pytest suite (CPU). Called by check.sh after py_check.sh has built
# polytopia_sim into the active Python env, and only when torch is importable.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m pytest -q python/tests/rl
