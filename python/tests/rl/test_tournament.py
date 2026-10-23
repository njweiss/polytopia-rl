"""Milestone 11 tests 6: side-swapped matches and round robins ("Ratings", "Seed ranges").

Each reported game is replayed here on an independent `polytopia_sim.Env`. A greedy net against
itself plays every seed twice with the sides swapped, so the two games are the same game and the
first name's wins equal the second's exactly. Expected ordering of the three scripted players: the
economy bot out-scores the random bot, and both out-score a player that only ends its turn (it keeps
the starting score). Ratings are checked against the independent fit in sp_helpers.bt_reference on
the reported matrices with draws as half wins.
"""

import numpy as np
import polytopia_sim as ps
import pytest
import torch
from sp_helpers import bt_reference, end_turn_policy

from polytopia_rl.evaluate import net_policy
from polytopia_rl.model import PolicyNet
from polytopia_rl.tournament import (
    RATING_SEED_BASE,
    PairResult,
    Player,
    TournamentResult,
    play_pair,
    round_robin,
    tournament_markdown,
)

KW = dict(size=8, turn_cap=10)


def winner_of(info, a_seat):
    out = info["outcome"]
    if out == f"win_{a_seat}":
        return "a"
    if out == f"win_{1 - a_seat}":
        return "b"
    return None


def check_pair(res: PairResult, seeds):
    pg = res.per_game
    assert res.games == len(pg) == 2 * seeds
    assert [g["seed"] for g in pg] == [RATING_SEED_BASE + j for j in range(seeds) for _ in range(2)]
    assert [g["a_seat"] for g in pg] == [0, 1] * seeds
    assert all(g["seed"] >= 2_000_000_000 for g in pg)
    assert res.a_wins == sum(g["winner"] == "a" for g in pg)
    assert res.b_wins == sum(g["winner"] == "b" for g in pg)
    assert res.a_wins + res.b_wins + res.draws == res.games
    seat0 = sum((g["winner"] == "a") == (g["a_seat"] == 0) for g in pg if g["winner"] is not None)
    assert res.seat_wins == {0: seat0, 1: res.a_wins + res.b_wins - seat0}


def test_bot_against_scripted_policy_replays():
    seeds = 3
    res = play_pair(Player("economy", bot="economy"), Player("idle", policy_fn=end_turn_policy), seeds, KW)
    assert (res.a, res.b) == ("economy", "idle")
    check_pair(res, seeds)
    for g in res.per_game:
        idle_seat = 1 - g["a_seat"]
        env = ps.Env(**KW, opponent="economy", opponent_epsilon=0.0, agent_seat=idle_seat)
        env.reset(g["seed"])
        done = False
        while not done:
            _, _, term, trunc = env.step_factored(*end_turn_policy(env, True))
            done = term or trunc
        assert g["winner"] == winner_of(env.info(), g["a_seat"]), g
    assert res.a_wins >= 5


def test_bot_against_bot_replays():
    seeds = 2
    res = play_pair(Player("rush", bot="rush"), Player("economy", bot="economy"), seeds, KW)
    check_pair(res, seeds)
    for g in res.per_game:
        env = ps.Env(**KW, opponent="none")
        env.reset(g["seed"])
        bots = {g["a_seat"]: "rush", 1 - g["a_seat"]: "economy"}
        done = False
        while not done:
            _, _, term, trunc = env.step_factored(*env.bot_action(bots[env.info()["current_player"]]))
            done = term or trunc
        assert g["winner"] == winner_of(env.info(), g["a_seat"]), g


def test_greedy_net_against_itself_is_an_exact_mirror():
    spec = ps.Env(size=KW["size"]).spec()
    torch.manual_seed(0)
    net = PolicyNet(spec, width=8, blocks=1)
    seeds = 3
    res = play_pair(Player("n1", policy_fn=net_policy(net)), Player("n2", policy_fn=net_policy(net)), seeds, KW)
    check_pair(res, seeds)
    assert res.a_wins == res.b_wins
    for j in range(seeds):
        g0, g1 = res.per_game[2 * j], res.per_game[2 * j + 1]
        assert g0["length"] == g1["length"]
        assert g1["winner"] == {"a": "b", "b": "a", None: None}[g0["winner"]]
    # Seat win counts are reported and cover the decisive games.
    assert res.seat_wins[0] + res.seat_wins[1] == res.games - res.draws


def half_policy(env, greedy):
    """The economy bot's moves as seat 0, idle as seat 1: against the idle player it wins as seat 0
    and mostly draws as seat 1 (two idle players usually keep equal scores), so the pair holds draws
    between unequal players."""
    if env.info()["current_player"] == 0:
        return env.bot_action("economy")
    return end_turn_policy(env, greedy)


@pytest.fixture(scope="module")
def rr():
    players = [
        Player("idle", policy_fn=end_turn_policy),
        Player("random", bot="random"),
        Player("economy", bot="economy"),
        Player("half", policy_fn=half_policy),
    ]
    return round_robin(players, 4, KW)


def test_round_robin_matrices_are_consistent(rr):
    assert isinstance(rr, TournamentResult)
    assert rr.players == ["idle", "random", "economy", "half"]
    n = 4
    for i in range(n):
        assert rr.games[i, i] == 0 and rr.wins[i, i] == 0
        for j in range(n):
            if i != j:
                assert rr.games[i, j] == rr.games[j, i] == 8
                assert rr.draws[i, j] == rr.draws[j, i]
                assert rr.wins[i, j] + rr.wins[j, i] + rr.draws[i, j] == rr.games[i, j]
    total = rr.games.sum() / 2
    assert 0 <= rr.seat_win_rate[0] <= 1 and 0 <= rr.seat_win_rate[1] <= 1
    decisive = (rr.games.sum() - rr.draws.sum()) / 2
    assert (rr.seat_win_rate[0] + rr.seat_win_rate[1]) * total == pytest.approx(decisive)


def test_round_robin_agrees_with_play_pair(rr):
    res = play_pair(Player("idle", policy_fn=end_turn_policy), Player("economy", bot="economy"), 4, KW)
    assert (rr.wins[0, 2], rr.wins[2, 0], rr.draws[0, 2]) == (res.a_wins, res.b_wins, res.draws)


def test_round_robin_ratings_and_ordering(rr):
    # Guard: half beats idle as seat 0 and the pair has draws, so draws carry weight in the fit.
    assert rr.wins[3, 0] >= 4 and rr.draws[3, 0] >= 2
    assert rr.ratings[1] == pytest.approx(0.0, abs=1e-9)  # the random bot is the anchor
    want = bt_reference(rr.wins + 0.5 * rr.draws, rr.games, anchor=1)
    np.testing.assert_allclose(rr.ratings, want, atol=1e-3)
    idle, rnd, eco, _ = rr.ratings
    assert eco > rnd > idle


def test_markdown_names_every_player(rr):
    md = tournament_markdown(rr)
    for name in rr.players:
        assert name in md
