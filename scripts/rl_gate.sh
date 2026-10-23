#!/usr/bin/env bash
# Milestone 10b gate (docs/MILESTONE_10 K2), inside the pixi env:
#   pixi run scripts/rl_gate.sh [CHECKPOINT]   (default runs/curriculum_seed1/best.pt, else latest.pt)
# Greedy, 200 evaluation seeds side-swapped (400 games) against random, economy and rush.
# Passes with >= 90% wins vs random and >= 60% vs economy; rush is reported only.
set -euo pipefail
cd "$(dirname "$0")/.."
ckpt="${1:-runs/curriculum_seed1/best.pt}"
[ -f "$ckpt" ] || ckpt="$(dirname "$ckpt")/latest.pt"
(cd python && maturin develop -q --release)
PYTHONPATH=python python3 - "$ckpt" <<'PY'
import sys

from polytopia_rl.evaluate import env_kwargs_of, evaluate, net_policy
from polytopia_rl.train import load_policy, resolve_device

device = resolve_device("auto")
net, cfg = load_policy(sys.argv[1], device)
print(f"checkpoint {sys.argv[1]}")
gates = {"random": 0.9, "economy": 0.6, "rush": None}
ok = True
for opp, need in gates.items():
    r = evaluate(net_policy(net, device=device), opp, 200, env_kwargs_of(cfg), greedy=True)
    verdict = "reported" if need is None else ("PASS" if r.win_rate >= need else "FAIL")
    ok &= verdict != "FAIL"
    print(
        f"{opp:8s} W-L-D {r.wins}-{r.losses}-{r.draws}  win rate {r.win_rate:.3f} "
        f"[{r.wilson_lo:.3f}, {r.wilson_hi:.3f}]  seat0 {r.by_seat[0]:.3f} seat1 {r.by_seat[1]:.3f}  "
        f"len {r.mean_length:.1f}  score diff {r.mean_score_diff:+.1f}  {verdict}"
        + ("" if need is None else f" (need >= {need})"),
        flush=True,
    )
print("rl_gate: OK" if ok else "rl_gate: FAIL")
sys.exit(0 if ok else 1)
PY
