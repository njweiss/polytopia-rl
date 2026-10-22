#!/usr/bin/env bash
# Builds a sample replay (seed 1, turn cap 20), verifies it and writes its HTML page
# (docs/MILESTONE_7 "scripts/demo.sh").
set -euo pipefail
cd "$(dirname "$0")/.."
cargo build --release -p sim-tools
bin=target/release/polytopia
mkdir -p target/demo
"$bin" play-random --seed 1 --turn-cap 20 --out target/demo/replay.json
"$bin" verify target/demo/replay.json
"$bin" html target/demo/replay.json target/demo/replay.html
echo "replay: target/demo/replay.json"
echo "page:   target/demo/replay.html"
