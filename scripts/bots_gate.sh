#!/usr/bin/env bash
# Milestone 8 gate: Rush and Economy each beat random in at least 80% of side-swapped games.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo build --release -p sim-tools
mkdir -p target/bots
./target/release/polytopia tournament --bots random,rush,economy --games 100 --turn-cap 30 \
  --gate-min 0.8 --save-replays target/bots --save-count 2
