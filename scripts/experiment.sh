#!/usr/bin/env bash
# Milestone 12: multi-seed experiments (arms x seeds) from a spec file. Inside the pixi env:
#   pixi run scripts/experiment.sh run SPEC [--parallel K] [--only ARM]
#   pixi run scripts/experiment.sh report SPEC
# Runs live in runs/exp/<name>/<arm>/seed<seed>; `run` can be repeated after a crash.
set -euo pipefail
cd "$(dirname "$0")/.."
(cd python && maturin develop -q --release)
cmd="$1"
shift
parallel=()
if [ "$cmd" = "run" ]; then
  parallel=(--parallel 1)  # default from the 12a concurrency measurement (docs/RL_REPORT.md)
  for arg in "$@"; do
    case "$arg" in --parallel|--parallel=*) parallel=() ;; esac
  done
fi
export PYTHONPATH=python
exec python3 -m polytopia_rl.experiment "$cmd" "$@" "${parallel[@]}"
