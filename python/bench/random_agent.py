"""Milestone 9 gate benchmark: 500 complete games per mode with a random legal-index agent.

Modes `random`, `rush` and `economy` play the agent against that bot; `none` is two-seat mode
with the random agent driving both seats. Every episode must end `terminated`.
"""

import time

import numpy as np

import polytopia_sim as ps

GAMES = 500
MODES = ["random", "rush", "economy", "none"]


def run(mode, rng):
    env = ps.Env(opponent=mode, agent_seat="seed_parity")
    steps = 0
    start = time.perf_counter()
    for seed in range(GAMES):
        env.reset(seed)
        while True:
            _, _, terminated, truncated = env.step_legal(int(rng.integers(env.legal_count())))
            steps += 1
            if terminated or truncated:
                break
        if not terminated:
            raise SystemExit(f"{mode} seed {seed}: episode truncated, not terminated")
    return steps, time.perf_counter() - start


def main():
    rng = np.random.default_rng(0)
    print(f"{'mode':<8} {'games':>6} {'agent steps':>12} {'seconds':>8} {'steps/s':>10} {'games/s':>8}")
    for mode in MODES:
        steps, secs = run(mode, rng)
        print(f"{mode:<8} {GAMES:>6} {steps:>12} {secs:>8.2f} {steps / secs:>10.0f} {GAMES / secs:>8.1f}")


if __name__ == "__main__":
    main()
