#!/usr/bin/env bash
# Milestone 11: round robin of a self-play run's pool (RUNDIR/pool/*.pt) and the three bots,
# written to RUNDIR/tournament.json and RUNDIR/tournament.md. Inside the pixi env:
#   pixi run scripts/tournament.sh RUNDIR [--seeds 100]
set -euo pipefail
cd "$(dirname "$0")/.."
run="$1"
shift
(cd python && maturin develop -q --release)
PYTHONPATH=python python3 -m polytopia_rl.tournament --run "$run" "$@"
