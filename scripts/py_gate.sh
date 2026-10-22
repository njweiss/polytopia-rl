#!/usr/bin/env bash
# Milestone 9 gate: a release wheel in a throwaway venv, the pytest suite, then the random-agent
# benchmark (2000 complete games).
set -euo pipefail
cd "$(dirname "$0")/.."
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
python3 -m venv "$tmp/venv"
maturin build -q --release -m python/pyproject.toml -o "$tmp/wheels"
wheel="$(ls "$tmp"/wheels/*.whl)"
"$tmp/venv/bin/pip" install -q "$wheel[dev]"
cargo build -q --release -p sim-tools --bin polytopia
POLYTOPIA_BIN="$PWD/target/release/polytopia" "$tmp/venv/bin/python" -m pytest -q python/tests
"$tmp/venv/bin/python" python/bench/random_agent.py
