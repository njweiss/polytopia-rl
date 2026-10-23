"""Bradley-Terry ratings in Elo units (docs/MILESTONE_11 "Ratings")."""

from __future__ import annotations

import numpy as np


def win_prob(r_a, r_b) -> float:
    """`1 / (1 + 10 ** ((r_b - r_a) / 400))`."""
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))


def fit_ratings(wins: np.ndarray, games: np.ndarray, anchor: int, anchor_rating: float = 0.0) -> np.ndarray:
    """Maximum-likelihood Bradley-Terry ratings, `400 * log10(strength)`, shifted so that
    `ratings[anchor] == anchor_rating`.

    `wins[i, j]`: games `i` won against `j` (a draw counts 0.5 for each side); `games[i, j]`: games
    between them (symmetric). Prior: every pair with `games[i, j] > 0` gets one virtual game split
    0.5-0.5. MM updates until the largest change is below 1e-9 (at most 10,000 iterations). A player
    without games, or a disconnected graph of played pairs, raises `ValueError`.

    The prior keeps a player that won (or lost) every game at a finite rating; with many games per
    pair its effect is negligible."""
    wins = np.asarray(wins, dtype=np.float64)
    games = np.asarray(games, dtype=np.float64)
    k = len(wins)
    played = games > 0
    if not played.any(axis=1).all():
        raise ValueError("a player has no games")
    seen, todo = {0}, [0]
    while todo:
        i = todo.pop()
        for j in np.flatnonzero(played[i]):
            if j not in seen:
                seen.add(int(j))
                todo.append(int(j))
    if len(seen) < k:
        raise ValueError("the graph of played pairs is disconnected")
    w = np.where(played, wins + 0.5, 0.0)
    n = np.where(played, games + 1.0, 0.0)
    total = w.sum(axis=1)
    # MM update (Hunter 2004): p_i <- W_i / sum_j n_ij / (p_i + p_j), then normalized.
    p = np.ones(k)
    for _ in range(10_000):
        new = total / (n / (p[:, None] + p[None, :])).sum(axis=1)
        new /= np.exp(np.log(new).mean())
        change = np.abs(new - p).max()
        p = new
        if change < 1e-9:
            break
    r = 400.0 * np.log10(p)
    return r - r[anchor] + anchor_rating
