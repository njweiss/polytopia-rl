#!/usr/bin/env bash
# Milestone 11: the fixed evaluation suite (seeds from python/configs/suite.json). Inside the pixi env:
#   pixi run scripts/suite.sh [CHECKPOINT] [--out DIR] [--seeds N]   (default runs/curriculum_seed1/latest.pt)
# Writes DIR/suite.json and DIR/suite.md (default DIR: <checkpoint dir>/suite_<checkpoint name>).
#   pixi run scripts/suite.sh --compare A.json B.json
set -euo pipefail
cd "$(dirname "$0")/.."
(cd python && maturin develop -q --release)
if [ "${1:-}" = "--compare" ]; then
  PYTHONPATH=python exec python3 -m polytopia_rl.suite "$@"
fi
ckpt="runs/curriculum_seed1/latest.pt"
if [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; then
  ckpt="$1"
  shift
fi
out="$(dirname "$ckpt")/suite_$(basename "$ckpt" .pt)"
PYTHONPATH=python python3 -m polytopia_rl.suite --checkpoint "$ckpt" --out "$out" "$@"
