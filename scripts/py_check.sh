#!/usr/bin/env bash
# Milestone 9: build polytopia_sim into a venv under target/ and run the pytest suite.
# Called by check.sh when maturin and python3 are on the PATH.
set -euo pipefail
cd "$(dirname "$0")/.."
venv="$PWD/target/py-venv"
[ -x "$venv/bin/python" ] || python3 -m venv "$venv"
cargo build -q --release -p sim-tools --bin polytopia
VIRTUAL_ENV="$venv" PATH="$venv/bin:$PATH" maturin develop -q --release --extras dev \
  -m python/pyproject.toml
POLYTOPIA_BIN="$PWD/target/release/polytopia" "$venv/bin/python" -m pytest -q python/tests
