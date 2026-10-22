"""Milestone 10 tests 9: side-swapped evaluation ("Evaluation").

Wilson 95% intervals (z = 1.96) from the closed form, which agrees with published tables:
8/10 -> (0.4902, 0.9433), 0/10 -> (0, 0.2775), 10/10 -> (0.7225, 1), 50/100 -> (0.4038, 0.5962).
The scripted policies' expectations: an agent that only ends its turn never trains or builds, so
the economy bot out-scores it; an agent imitating the random bot against the random bot wins about
half of a side-swapped set by symmetry.
"""

import numpy as np
import polytopia_sim as ps
import pytest
import torch
from rl_helpers import end_turn_path, random_bot_path

from polytopia_rl.evaluate import EVAL_SEED_BASE, evaluate, net_policy, wilson_interval
from polytopia_rl.model import PolicyNet


@pytest.mark.parametrize(
    "wins,n,lo,hi",
    [(8, 10, 0.4902, 0.9433), (0, 10, 0.0, 0.2775), (10, 10, 0.7225, 1.0), (50, 100, 0.4038, 0.5962)],
)
def test_wilson_interval_on_known_counts(wins, n, lo, hi):
    got = wilson_interval(wins, n)
    assert got[0] == pytest.approx(lo, abs=1e-3)
    assert got[1] == pytest.approx(hi, abs=1e-3)


class Recorder:
    """Scripted EndTurn-first policy that records what each game looked like."""

    def __init__(self):
        self.calls = {}
        self.first_terrain = {}

    def __call__(self, env, greedy):
        info = env.info()
        key = (info["seed"], info["agent_seat"])
        self.calls[key] = self.calls.get(key, 0) + 1
        if key not in self.first_terrain:
            board, _ = env.observation()
            names = env.spec()["channel_names"]
            idx = [i for i, n in enumerate(names) if n.startswith("terrain.")]
            self.first_terrain[key] = board[idx].copy()
        return end_turn_path(env)


def check_counts(res, seeds):
    assert res.games == 2 * seeds
    assert res.wins + res.losses + res.draws == res.games
    assert res.win_rate == pytest.approx(res.wins / res.games)
    lo, hi = wilson_interval(res.wins, res.games)
    assert (res.wilson_lo, res.wilson_hi) == pytest.approx((lo, hi))
    pg = res.per_game
    assert len(pg) == res.games
    assert [g["seed"] for g in pg] == [EVAL_SEED_BASE + j for j in range(seeds) for _ in range(2)]
    assert [g["seat"] for g in pg] == [0, 1] * seeds
    assert sum(g["outcome"] == "win" for g in pg) == res.wins
    assert sum(g["outcome"] == "loss" for g in pg) == res.losses
    assert sum(g["outcome"] == "draw" for g in pg) == res.draws
    for seat in (0, 1):
        mine = [g for g in pg if g["seat"] == seat]
        assert res.by_seat[seat] == pytest.approx(sum(g["outcome"] == "win" for g in mine) / len(mine))
    assert res.mean_length == pytest.approx(np.mean([g["length"] for g in pg]))
    assert res.mean_score_diff == pytest.approx(np.mean([g["score_diff"] for g in pg]))


def test_each_seed_twice_with_swapped_seats_on_the_same_map():
    rec = Recorder()
    seeds = 4
    res = evaluate(rec, "random", seeds, dict(size=8, turn_cap=4, full_observability=True), greedy=True)
    check_counts(res, seeds)
    expected = {(EVAL_SEED_BASE + j, s) for j in range(seeds) for s in (0, 1)}
    assert set(rec.calls) == expected
    for j in range(seeds):
        s = EVAL_SEED_BASE + j
        # Full observability: the terrain both seats see first is the same map.
        np.testing.assert_array_equal(rec.first_terrain[(s, 0)], rec.first_terrain[(s, 1)])
    for g in res.per_game:
        assert g["seed"] >= EVAL_SEED_BASE
        assert g["length"] == rec.calls[(g["seed"], g["seat"])]


def test_end_turn_policy_loses_almost_every_game_to_economy():
    res = evaluate(lambda env, greedy: end_turn_path(env), "economy", 10, dict(turn_cap=15), greedy=True)
    check_counts(res, 10)
    assert res.losses >= 18, f"{res.wins}-{res.losses}-{res.draws}"
    assert res.mean_score_diff < 0


def test_random_imitation_wins_about_half_against_random():
    # 60 games: the win rate's sd is at most sqrt(0.25 / 60) = 0.065; the band is about 4.3 sd.
    rng = np.random.default_rng(0)
    res = evaluate(lambda env, greedy: random_bot_path(env, rng), "random", 30, dict(size=8, turn_cap=15), greedy=False)
    check_counts(res, 30)
    assert 0.22 <= res.win_rate <= 0.78, f"{res.wins}-{res.losses}-{res.draws}"


def test_opponent_epsilon_is_forced_to_zero():
    pol = lambda env, greedy: end_turn_path(env)  # noqa: E731
    a = evaluate(pol, "random", 3, dict(size=8, turn_cap=6), greedy=True)
    b = evaluate(pol, "random", 3, dict(size=8, turn_cap=6, opponent_epsilon=0.5), greedy=True)
    assert a.per_game == b.per_game


def test_greedy_net_evaluation_is_deterministic():
    spec = ps.Env(size=8).spec()
    torch.manual_seed(0)
    net = PolicyNet(spec, width=8, blocks=1)
    kw = dict(size=8, turn_cap=4)
    a = evaluate(net_policy(net), "random", 2, kw, greedy=True)
    b = evaluate(net_policy(net), "random", 2, kw, greedy=True)
    check_counts(a, 2)
    assert a.per_game == b.per_game
    # Sampled play with equally seeded generators is repeatable too.
    c = evaluate(net_policy(net, torch.Generator().manual_seed(5)), "random", 2, kw, greedy=False)
    d = evaluate(net_policy(net, torch.Generator().manual_seed(5)), "random", 2, kw, greedy=False)
    assert c.per_game == d.per_game


def test_by_seat_separates_the_seats():
    # Random-bot imitation as seat 0 (wins about half vs random), EndTurn-only as seat 1 (never
    # builds anything, so it should essentially never win).
    rng = np.random.default_rng(1)

    def pol(env, greedy):
        return random_bot_path(env, rng) if env.info()["agent_seat"] == 0 else end_turn_path(env)

    res = evaluate(pol, "random", 10, dict(size=8, turn_cap=15), greedy=False)
    check_counts(res, 10)
    assert res.by_seat[0] >= 0.2 and res.by_seat[1] <= 0.1, res.by_seat
