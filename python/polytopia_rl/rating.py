"""Bradley-Terry ratings in Elo units (docs/MILESTONE_11 "Ratings")."""

from __future__ import annotations

import numpy as np


def win_prob(r_a, r_b) -> float:
    """`1 / (1 + 10 ** ((r_b - r_a) / 400))`."""
    raise NotImplementedError


def fit_ratings(wins: np.ndarray, games: np.ndarray, anchor: int, anchor_rating: float = 0.0) -> np.ndarray:
    """Maximum-likelihood Bradley-Terry ratings, `400 * log10(strength)`, shifted so that
    `ratings[anchor] == anchor_rating`.

    `wins[i, j]`: games `i` won against `j` (a draw counts 0.5 for each side); `games[i, j]`: games
    between them (symmetric). Prior: every pair with `games[i, j] > 0` gets one virtual game split
    0.5-0.5. MM updates until the largest change is below 1e-9 (at most 10,000 iterations). A player
    without games, or a disconnected graph of played pairs, raises `ValueError`."""
    raise NotImplementedError
