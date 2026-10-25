"""Shared helpers for the milestone 12 tests. Nothing here imports the code under test.

- `t_sf_ref`: the Student t upper tail written independently of the brief's continued fraction:
  closed forms for df 1 (Cauchy) and 2, otherwise Simpson integration of the density.
- `welch_ref`: the brief's Welch formulas, with `t_sf_ref` for p.
- `elo_ref`: the brief's `elo_vs_init` of one head-to-head (clipped score).
"""

import math

import numpy as np


def t_pdf(x, df):
    c = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
    return c * (1 + x * x / df) ** (-(df + 1) / 2)


def t_sf_ref(t, df, intervals=20_000):
    """P(T > t), t >= 0."""
    if df == 1:
        return 0.5 - math.atan(t) / math.pi
    if df == 2:
        return 0.5 - t / (2 * math.sqrt(2 + t * t))
    xs = np.linspace(0.0, t, intervals + 1)
    ys = np.array([t_pdf(x, df) for x in xs])
    h = t / intervals
    integral = h / 3 * (ys[0] + ys[-1] + 4 * ys[1:-1:2].sum() + 2 * ys[2:-1:2].sum())
    return 0.5 - integral


def welch_ref(a, b):
    """(mean_a, mean_b, diff, t, df, p) by the brief's formulas (nonzero variances)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    t = (b.mean() - a.mean()) / math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    p = 2 * t_sf_ref(abs(t), df)
    return a.mean(), b.mean(), b.mean() - a.mean(), t, df, p


def elo_ref(wins, draws, games):
    """400 log10(s / (1 - s)) of s = (wins + draws / 2) / games clipped to [0.5 / n, 1 - 0.5 / n]."""
    s = (wins + 0.5 * draws) / games
    s = min(max(s, 0.5 / games), 1 - 0.5 / games)
    return 400 * math.log10(s / (1 - s))
