#!/usr/bin/env bash
# Milestone 11: snapshot-pool self-play from the M10 checkpoint. Inside the pixi env:
#   pixi run scripts/selfplay.sh python/configs/selfplay.json [key=value ...] [--run-dir DIR]
#     [--init-checkpoint CKPT]   (default runs/curriculum_seed1/latest.pt)
#     [--resume CKPT]            (continue a self-play run instead)
set -euo pipefail
cd "$(dirname "$0")/.."
config="$1"
shift
init=(--init-checkpoint runs/curriculum_seed1/latest.pt)
for arg in "$@"; do
  case "$arg" in --init-checkpoint|--init-checkpoint=*|--resume|--resume=*) init=() ;; esac
done
(cd python && maturin develop -q --release)
# One CPU thread per pool: the many tiny CPU tensor ops otherwise keep 11 idle worker threads
# spinning against the main thread (~20% slower; docs/QUESTIONS.md Q30). Override by setting it.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
PYTHONPATH=python python3 -m polytopia_rl.train --config "$config" "${init[@]}" "$@"
