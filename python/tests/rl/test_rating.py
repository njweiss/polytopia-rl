"""Milestone 11 tests 5: Bradley-Terry ratings in Elo units ("Ratings").

Hand values: with two players the maximum-likelihood strength ratio is the (prior-adjusted) win
ratio, so `r_0 - r_1 = 400 * log10((w_01 + 0.5) / (w_10 + 0.5))` with the brief's one virtual
0.5-0.5 game per played pair. `win_prob(400, 0) = 1 / (1 + 10^-1) = 10/11`. Larger cases are checked
against noise-free expected-count matrices (true ratings known) and against the independent fit in
sp_helpers.bt_reference.
"""

import math

import numpy as np
import pytest
from sp_helpers import bt_reference

from polytopia_rl.rating import fit_ratings, win_prob


def two(w01, w10):
    return np.array([[0.0, w01], [w10, 0.0]]), np.array([[0.0, w01 + w10], [w01 + w10, 0.0]])


def test_win_prob_identities():
    assert win_prob(0.0, 0.0) == pytest.approx(0.5)
    assert win_prob(1234.0, 1234.0) == pytest.approx(0.5)
    assert win_prob(400.0, 0.0) == pytest.approx(10 / 11)
    assert win_prob(0.0, 400.0) == pytest.approx(1 / 11)
    assert win_prob(1600.0, 1200.0) == pytest.approx(10 / 11)
    for a, b in [(0, 100), (-50, 300), (800, 0)]:
        assert win_prob(a, b) + win_prob(b, a) == pytest.approx(1.0)
    assert win_prob(800.0, 0.0) == pytest.approx(100 / 101)


def test_two_players_closed_form_with_the_prior():
    wins, games = two(7.5, 2.5)
    r = fit_ratings(wins, games, anchor=1)
    assert r[1] == pytest.approx(0.0, abs=1e-9)
    assert r[0] == pytest.approx(400 * math.log10(8 / 3), abs=1e-3)  # 170.39, not 400*ln


def test_anchor_and_anchor_rating():
    wins, games = two(7.5, 2.5)
    r = fit_ratings(wins, games, anchor=0, anchor_rating=1500.0)
    assert r[0] == pytest.approx(1500.0, abs=1e-9)
    assert r[1] == pytest.approx(1500.0 - 400 * math.log10(8 / 3), abs=1e-3)


def test_all_wins_stay_finite_thanks_to_the_prior():
    wins, games = two(10.0, 0.0)
    r = fit_ratings(wins, games, anchor=1)
    assert np.all(np.isfinite(r))
    assert r[0] == pytest.approx(400 * math.log10(10.5 / 0.5), abs=1e-3)  # 528.9


def test_draws_count_as_half_wins():
    # Ten draws: 5 for each side.
    wins, games = two(5.0, 5.0)
    np.testing.assert_allclose(fit_ratings(wins, games, anchor=0), [0.0, 0.0], atol=1e-6)
    # 6 wins, 2 losses, 2 draws -> 7 vs 3 half-win counts.
    wins, games = two(7.0, 3.0)
    r = fit_ratings(wins, games, anchor=1)
    assert r[0] == pytest.approx(400 * math.log10(7.5 / 3.5), abs=1e-3)


def expected_counts(ratings, pairs, n):
    k = len(ratings)
    wins, games = np.zeros((k, k)), np.zeros((k, k))
    for i, j in pairs:
        games[i, j] = games[j, i] = n
        wins[i, j] = n * win_prob(ratings[i], ratings[j])
        wins[j, i] = n * win_prob(ratings[j], ratings[i])
    return wins, games


TRUE = np.array([0.0, 150.0, -200.0, 420.0, 35.0])
COMPLETE = [(i, j) for i in range(5) for j in range(i + 1, 5)]
SPARSE = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 3)]


@pytest.mark.parametrize("pairs", [COMPLETE, SPARSE], ids=["complete", "sparse"])
def test_recovers_known_ratings_from_expected_counts(pairs):
    # 10^6 games per pair: the prior moves each rate by at most 0.5 / 10^6, far below 1 Elo.
    wins, games = expected_counts(TRUE, pairs, 1_000_000)
    r = fit_ratings(wins, games, anchor=0)
    np.testing.assert_allclose(r, TRUE, atol=1.0)


def test_matches_the_independent_fit_on_small_counts():
    wins = np.array(
        [
            [0, 3, 1, 0, 2],
            [5, 0, 4.5, 2, 0],
            [1, 1.5, 0, 6, 0],
            [0, 0, 2, 0, 7],
            [8, 0, 0, 1, 0],
        ],
        dtype=float,
    )
    games = wins + wins.T
    r = fit_ratings(wins, games, anchor=2, anchor_rating=0.0)
    np.testing.assert_allclose(r, bt_reference(wins, games, anchor=2), atol=1e-3)
    assert r[2] == pytest.approx(0.0, abs=1e-9)


def test_permuting_players_permutes_ratings():
    wins, games = expected_counts(TRUE, SPARSE, 40)
    wins = np.round(wins)  # integer counts, not exactly the BT model
    games = wins + wins.T
    perm = np.array([2, 0, 3, 1, 4])  # new player i is old player perm[i]
    r = fit_ratings(wins, games, anchor=0)
    rp = fit_ratings(wins[np.ix_(perm, perm)], games[np.ix_(perm, perm)], anchor=int(np.flatnonzero(perm == 0)[0]))
    np.testing.assert_allclose(rp, r[perm], atol=1e-3)


def test_transitive_chain_orders_correctly():
    # A beats B 70-30, B beats C 70-30, A and C never meet.
    wins = np.array([[0, 70, 0], [30, 0, 70], [0, 30, 0]], dtype=float)
    games = wins + wins.T
    r = fit_ratings(wins, games, anchor=2)
    assert r[0] > r[1] > r[2]
    np.testing.assert_allclose(r, bt_reference(wins, games, anchor=2), atol=1e-3)


def test_player_without_games_raises():
    wins = np.array([[0, 3, 0], [2, 0, 0], [0, 0, 0]], dtype=float)
    with pytest.raises(ValueError):
        fit_ratings(wins, wins + wins.T, anchor=0)


def test_disconnected_graph_raises():
    wins = np.zeros((4, 4))
    wins[0, 1], wins[1, 0], wins[2, 3], wins[3, 2] = 3, 2, 4, 1
    with pytest.raises(ValueError):
        fit_ratings(wins, wins + wins.T, anchor=0)
