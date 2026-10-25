"""Milestone 11 tests 3: the evaluation suite ("Evaluation suite", "Seed ranges").

Every game the suite reports is replayed here on an independent `polytopia_sim.Env` built from the
brief's table (opponent, epsilon, full observability, seat) with the same scripted policy, which is
a pure function of (seed, seat, actions so far); outcome, length, score difference and loss type
must agree. Loss type: a lost game that ended before the turn cap was a capital capture, one that
reached the cap was decided by score (spec "Win condition"; the cap ends at `turn == turn_cap`).

`compare` hand values (two-proportion z test with the pooled proportion):
- 150/400 vs 210/400: p = 0.45, se = sqrt(0.45 * 0.55 * (2 / 400)) = 0.035178, z = 0.15 / se = 4.264.
- 200/400 vs 204/400: p = 0.505, se = 0.035354, z = 0.01 / se = 0.2829.
"""

import re

import numpy as np
import polytopia_sim as ps
import pytest
from sp_helpers import bot_policy, end_turn_policy, keyed_random_policy

from polytopia_rl.evaluate import wilson_interval
from polytopia_rl.suite import (
    DEFAULT_SUITE,
    SUITE_SEED_BASE,
    SUITE_SEED_LIMIT,
    Comparison,
    Condition,
    ConditionResult,
    SuiteResult,
    compare,
    run_suite,
    suite_markdown,
)
from polytopia_rl.tournament import RATING_SEED_BASE
from polytopia_rl.vec_env import TRAIN_SEED_LIMIT

KW = dict(size=8, turn_cap=6)
TABLE = [
    ("random_e0", "random", 0.0, False, True),
    ("economy_e0", "economy", 0.0, False, True),
    ("rush_e0", "rush", 0.0, False, True),
    ("economy_e20", "economy", 0.2, False, True),
    ("rush_e20", "rush", 0.2, False, True),
    ("economy_fullobs", "economy", 0.0, True, True),
    ("rush_fullobs", "rush", 0.0, True, True),
    ("rush_e0_sampled", "rush", 0.0, False, False),
]


def test_default_suite_is_the_table():
    got = [(c.name, c.opponent, c.epsilon, c.full_observability, c.greedy) for c in DEFAULT_SUITE]
    assert got == TABLE


def test_seed_ranges_do_not_overlap_and_the_suite_range_is_enforced():
    assert TRAIN_SEED_LIMIT == 1_000_000_000
    assert SUITE_SEED_BASE == 1_000_000_000 and SUITE_SEED_LIMIT == 1_500_000_000
    assert RATING_SEED_BASE == 2_000_000_000
    assert TRAIN_SEED_LIMIT <= SUITE_SEED_BASE < SUITE_SEED_LIMIT <= RATING_SEED_BASE
    cond = [Condition("x", "random")]
    with pytest.raises(ValueError):
        run_suite(end_turn_policy, cond, SUITE_SEED_LIMIT - SUITE_SEED_BASE + 1, KW)


class Recorder:
    """keyed_random_policy that records the greedy flag and whether the board is fully explored."""

    def __init__(self):
        self.inner = keyed_random_policy()
        self.greedy = set()
        self.explored_all = set()

    def __call__(self, env, greedy):
        self.greedy.add(greedy)
        board, _ = env.observation()
        e = list(env.spec()["channel_names"]).index("explored")
        self.explored_all.add(bool((board[e] == 1).all()))
        return self.inner(env, greedy)


def replay(cond, seed, seat, policy):
    env = ps.Env(
        **KW,
        opponent=cond.opponent,
        opponent_epsilon=cond.epsilon,
        full_observability=cond.full_observability,
        agent_seat=seat,
    )
    env.reset(seed)
    length = 0
    while True:
        _, _, term, trunc = env.step_factored(*policy(env, cond.greedy))
        length += 1
        if term or trunc:
            break
    info = env.info()
    outcome = {f"win_{seat}": "win", f"win_{1 - seat}": "loss"}.get(info["outcome"], "draw")
    loss_type = None
    if outcome == "loss":
        loss_type = "score" if info["turn"] >= KW["turn_cap"] else "capital_captured"
    scores = info["scores"]
    return dict(outcome=outcome, length=length, score_diff=scores[seat] - scores[1 - seat], loss_type=loss_type)


def check_condition(r: ConditionResult, seeds):
    pg = r.per_game
    assert r.games == len(pg) == 2 * seeds
    assert [g["seed"] for g in pg] == [SUITE_SEED_BASE + j for j in range(seeds) for _ in range(2)]
    assert [g["seat"] for g in pg] == [0, 1] * seeds
    wins = sum(g["outcome"] == "win" for g in pg)
    losses = sum(g["outcome"] == "loss" for g in pg)
    assert (r.wins, r.losses, r.draws) == (wins, losses, r.games - wins - losses)
    assert r.win_rate == pytest.approx(wins / r.games)
    assert (r.wilson_lo, r.wilson_hi) == pytest.approx(wilson_interval(wins, r.games))
    for seat in (0, 1):
        mine = [g for g in pg if g["seat"] == seat]
        assert r.by_seat[seat] == pytest.approx(sum(g["outcome"] == "win" for g in mine) / len(mine))
    assert r.mean_length == pytest.approx(np.mean([g["length"] for g in pg]))
    assert r.mean_score_diff == pytest.approx(np.mean([g["score_diff"] for g in pg]))
    cap = sum(g["loss_type"] == "capital_captured" for g in pg)
    score = sum(g["loss_type"] == "score" for g in pg)
    assert cap + score == losses
    assert r.loss_capital_share == pytest.approx(cap / losses if losses else 0.0)
    assert r.loss_score_share == pytest.approx(score / losses if losses else 0.0)


@pytest.mark.parametrize("cond", [Condition(*row) for row in TABLE], ids=lambda c: c.name)
def test_each_condition_plays_its_table_row(cond):
    seeds = 2
    rec = Recorder()
    res = run_suite(rec, [cond], seeds, KW)
    assert list(res.conditions) == [cond.name]
    r = res.conditions[cond.name]
    assert r.name == cond.name
    check_condition(r, seeds)
    assert rec.greedy == {cond.greedy}
    assert rec.explored_all == ({True} if cond.full_observability else {False})
    policy = keyed_random_policy()
    for g in r.per_game:
        want = replay(cond, g["seed"], g["seat"], policy)
        got = {k: g[k] for k in want}
        assert got == want, (g["seed"], g["seat"])


def test_whole_suite_runs_every_condition_on_the_same_seeds_and_embeds_hashes():
    res = run_suite(end_turn_policy, DEFAULT_SUITE, 1, KW, step=1234)
    assert list(res.conditions) == [c.name for c in DEFAULT_SUITE]
    spec = ps.Env(size=KW["size"]).spec()
    assert res.obs_hash == spec["obs_hash"] and res.rules_hash == spec["rules_hash"]
    assert res.step == 1234
    for r in res.conditions.values():
        check_condition(r, 1)
    # Guard for the per-condition replay check: with the keyed policy, epsilon 0.2 really plays
    # different games than epsilon 0.
    policy = keyed_random_policy()
    e0, e20 = Condition(*TABLE[1]), Condition(*TABLE[3])
    a = [replay(e0, SUITE_SEED_BASE + j, s, policy) for j in range(2) for s in (0, 1)]
    b = [replay(e20, SUITE_SEED_BASE + j, s, policy) for j in range(2) for s in (0, 1)]
    assert a != b


def test_end_turn_policy_loses_nearly_everything():
    # Doing nothing keeps the starting 515 points; every bot gains score by exploring or building.
    res = run_suite(end_turn_policy, DEFAULT_SUITE, 3, dict(size=8, turn_cap=12))
    for name, r in res.conditions.items():
        check_condition(r, 3)
        assert r.wins == 0 and r.losses >= 5, f"{name}: {r.wins}-{r.losses}-{r.draws}"
        assert r.mean_score_diff < 0, name


@pytest.mark.parametrize("bot", ["economy", "rush"])
def test_bot_imitation_against_the_same_bot_mirrors_the_seats(bot):
    # The agent plays exactly the bot's moves, so seed j as seat 0 and as seat 1 is the same game
    # with the sides swapped: every win is matched by a loss on the other seat.
    cond = Condition(f"{bot}_mirror", bot)
    seeds = 4
    res = run_suite(bot_policy(bot), [cond], seeds, dict(size=8, turn_cap=12))
    r = res.conditions[cond.name]
    check_condition(r, seeds)
    flip = {"win": "loss", "loss": "win", "draw": "draw"}
    pg = r.per_game
    for j in range(seeds):
        g0, g1 = pg[2 * j], pg[2 * j + 1]
        assert g1["outcome"] == flip[g0["outcome"]]
        assert g1["score_diff"] == -g0["score_diff"]
        assert g1["length"] > 0 and g0["length"] > 0
    assert r.wins == r.losses
    assert 0.3 <= r.win_rate + 0.5 * r.draws / r.games <= 0.7  # exactly 0.5, loose by design
    # Seat 0's wins plus seat 1's wins cover every seed that was not drawn.
    assert r.by_seat[0] + r.by_seat[1] == pytest.approx(1 - r.draws / 2 / seeds)


def cr(name, wins, games):
    lo, hi = wilson_interval(wins, games)
    return ConditionResult(
        name=name,
        games=games,
        wins=wins,
        losses=games - wins,
        draws=0,
        win_rate=wins / games,
        wilson_lo=lo,
        wilson_hi=hi,
        by_seat={0: wins / games, 1: wins / games},
        mean_length=10.0,
        mean_score_diff=0.0,
        loss_capital_share=0.0,
        loss_score_share=1.0,
        per_game=[],
    )


def suite_of(*crs, step=None):
    return SuiteResult(conditions={c.name: c for c in crs}, obs_hash="0" * 16, rules_hash="1" * 16, step=step)


def test_compare_against_hand_computed_z():
    a = suite_of(cr("big", 150, 400), cr("tiny", 200, 400), cr("same", 37, 50), cr("zero", 0, 50), cr("only_a", 1, 2))
    b = suite_of(cr("tiny", 204, 400), cr("big", 210, 400), cr("same", 37, 50), cr("zero", 0, 50))
    out = compare(a, b)
    assert all(isinstance(c, Comparison) for c in out)
    assert [c.name for c in out] == ["big", "tiny", "same", "zero"]
    big, tiny, same, zero = out
    assert (big.win_rate_a, big.win_rate_b) == pytest.approx((0.375, 0.525))
    assert big.diff == pytest.approx(0.15)
    assert big.z == pytest.approx(4.264, abs=1e-3) and big.significant
    assert tiny.diff == pytest.approx(0.01)
    assert tiny.z == pytest.approx(0.2829, abs=1e-3) and not tiny.significant
    assert same.z == pytest.approx(0.0) and not same.significant
    assert zero.z == 0.0 and not zero.significant
    # Reversed order flips the sign.
    rev = {c.name: c for c in compare(b, a)}
    assert rev["big"].z == pytest.approx(-4.264, abs=1e-3) and rev["big"].significant


def test_compare_threshold_is_1_96():
    # 50/100 vs 64/100: p = 0.57, se = sqrt(0.57 * 0.43 * 0.02) = 0.070014, z = 1.9996 (flagged).
    # 50/100 vs 63/100: p = 0.565, se = 0.070107, z = 1.8543 (not flagged).
    out = compare(suite_of(cr("x", 50, 100), cr("y", 50, 100)), suite_of(cr("x", 64, 100), cr("y", 63, 100)))
    assert out[0].z == pytest.approx(1.9996, abs=1e-3) and out[0].significant
    assert out[1].z == pytest.approx(1.8543, abs=1e-3) and not out[1].significant


def test_markdown_lists_each_condition_once_with_hashes_and_step():
    res = suite_of(*[cr(c.name, 5, 10) for c in DEFAULT_SUITE], step=4242)
    md = suite_markdown(res)
    for c in DEFAULT_SUITE:
        assert len(re.findall(rf"\b{c.name}\b", md)) == 1, c.name
    assert "0" * 16 in md and "1" * 16 in md and "4242" in md
