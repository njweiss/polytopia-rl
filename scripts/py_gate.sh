#!/usr/bin/env bash
# Milestone 9 gate (run inside the pixi env: `pixi run ./scripts/py_gate.sh`): a release wheel,
# unpacked into a throwaway directory that shadows any `maturin develop` build, then the pytest
# suite and the random-agent benchmark (2000 complete games).
set -euo pipefail
cd "$(dirname "$0")/.."
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
(cd python && maturin build -q --release -o "$tmp/wheels")
python3 -m zipfile -e "$tmp"/wheels/*.whl "$tmp/site"
cargo build -q --release -p sim-tools --bin polytopia
export PYTHONPATH="$tmp/site"
python3 -c 'import polytopia_sim, sys; assert polytopia_sim.__file__.startswith(sys.argv[1])' "$tmp/site"
POLYTOPIA_BIN="$PWD/target/release/polytopia" python3 -m pytest -q python/tests
python3 python/bench/random_agent.py
