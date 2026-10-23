#!/usr/bin/env bash
# Milestone 10: build polytopia_sim (release) and train. Run inside the pixi env:
#   pixi run scripts/train.sh python/configs/stage1_random.json [key=value ...] [--run-dir DIR]
set -euo pipefail
cd "$(dirname "$0")/.."
config="$1"
shift
(cd python && maturin develop -q --release)
PYTHONPATH=python python3 -m polytopia_rl.train --config "$config" "$@"
