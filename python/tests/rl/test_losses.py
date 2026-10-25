"""Milestone 11 tests 4: loss review ("Loss review").

Expected records come from replaying every seed pair here on an independent `polytopia_sim.Env`
with the same scripted policy (a pure function of seed, seat and actions so far), with the loss type
read from the end state (before the cap and without an own capital on the board: captured; at the
cap: score) and the first-city-lost turn from the agent's own observations (docs/QUESTIONS.md Q24).
Replay files must pass `polytopia verify` (milestone 7), so they hold the opponent's moves too.
"""

import json
import os
import pathlib
import subprocess

import polytopia_sim as ps
import pytest
from sp_helpers import REPO, has_own_capital, keyed_random_policy, play_out

from polytopia_rl.losses import LossRecord, collect_losses
from polytopia_rl.suite import SUITE_SEED_BASE

KW = dict(size=8, turn_cap=10)
OPPONENT = "rush"
SEEDS = 6


def polytopia_cli():
    exe = os.environ.get("POLYTOPIA_BIN")
    if exe:
        return [exe]
    return ["cargo", "run", "-q", "--release", "-p", "sim-tools", "--bin", "polytopia", "--"]


def expected_losses(policy, seeds):
    out = []
    for j in range(seeds):
        seed = SUITE_SEED_BASE + j
        for seat in (0, 1):
            env = ps.Env(**KW, opponent=OPPONENT, opponent_epsilon=0.0, agent_seat=seat)
            env.reset(seed)
            _, first_lost = play_out(env, policy, greedy=True)
            info = env.info()
            if info["outcome"] != f"win_{1 - seat}":
                continue
            at_cap = info["turn"] >= KW["turn_cap"]
            assert at_cap or not has_own_capital(env), "a loss before the cap is a capital capture"
            out.append(
                dict(
                    seed=seed,
                    seat=seat,
                    final_turn=info["turn"],
                    loss_type="score" if at_cap else "capital_captured",
                    scores=tuple(info["scores"]),
                    first_city_lost_turn=first_lost,
                )
            )
    return out


@pytest.fixture(scope="module")
def expected():
    return expected_losses(keyed_random_policy(), SEEDS)


def as_dict(r: LossRecord):
    return dict(
        seed=r.seed,
        seat=r.seat,
        final_turn=r.final_turn,
        loss_type=r.loss_type,
        scores=tuple(r.scores),
        first_city_lost_turn=r.first_city_lost_turn,
    )


def test_records_match_the_replayed_games(expected, tmp_path):
    # Guard: the scripted games include a capital capture and a loss on score at the cap. (The
    # scripted agent never holds a second city, so its first lost city is always the capital.)
    assert {e["loss_type"] for e in expected} == {"capital_captured", "score"}, expected
    assert any(e["loss_type"] == "capital_captured" and e["first_city_lost_turn"] == e["final_turn"] for e in expected)
    assert any(e["loss_type"] == "score" and e["first_city_lost_turn"] is None for e in expected)
    recs = collect_losses(keyed_random_policy(), OPPONENT, SEEDS, KW, str(tmp_path), limit=100)
    assert [as_dict(r) for r in recs] == expected
    for r in recs:
        p = pathlib.Path(r.path)
        assert p.name == f"{OPPONENT}-{r.seed}-seat{r.seat}.json"
        assert p.parent == tmp_path and p.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(pathlib.Path(r.path).name for r in recs)
    # Seat shares add up to all losses.
    by_seat = [sum(r.seat == s for r in recs) for s in (0, 1)]
    assert sum(by_seat) == len(recs)


def test_limit_keeps_the_first_losses(expected, tmp_path):
    recs = collect_losses(keyed_random_policy(), OPPONENT, SEEDS, KW, str(tmp_path), limit=2)
    assert [as_dict(r) for r in recs] == expected[:2]
    assert len(list(tmp_path.iterdir())) == 2


def test_replay_files_verify_and_hold_both_sides(expected, tmp_path):
    recs = collect_losses(keyed_random_policy(), OPPONENT, SEEDS, KW, str(tmp_path), limit=100)
    for kind in ("capital_captured", "score"):
        r = next(r for r in recs if r.loss_type == kind)
        log = json.loads(pathlib.Path(r.path).read_text())
        assert log["seed"] == r.seed
        assert log["config"]["turn_cap"] == KW["turn_cap"]
        assert log["outcome"] == {"Win": 1 - r.seat}
        # A log without the opponent's moves fails verify (illegal action or digest mismatch).
        res = subprocess.run(polytopia_cli() + ["verify", r.path], cwd=REPO, capture_output=True, text=True)
        assert res.returncode == 0, res.stderr
        assert res.stdout.strip() == f"OK: {len(log['actions'])} actions, outcome Win({1 - r.seat})"
