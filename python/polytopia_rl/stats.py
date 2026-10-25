"""Welch's t-test across seeds (docs/MILESTONE_12 "Welch's t-test"), with numpy/math only."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class Welch:
    mean_a: float
    mean_b: float
    diff: float  # mean_b - mean_a
    t: float
    df: float  # Welch-Satterthwaite
    p: float  # two-sided: 2 * t_sf(|t|, df)
    significant: bool  # p < alpha


def welch(a: Sequence[float], b: Sequence[float], alpha: float = 0.05) -> Welch:
    """Welch's t-test of `b` against `a` with sample variances (`ddof=1`). Both variances 0: `t = 0,
    p = 1` if the means are equal, else `t = +-inf, p = 0`. Fewer than 2 values in either raises
    `ValueError`."""
    raise NotImplementedError("milestone 12: welch")


def t_sf(t: float, df: float) -> float:
    """`P(T > t)` for Student's t with `df` degrees of freedom (`df` may be non-integer):
    `0.5 * I_{df / (df + t^2)}(df / 2, 1 / 2)` for `t >= 0`, the regularized incomplete beta function
    by a continued fraction (Lentz's method), absolute error below 1e-10."""
    raise NotImplementedError("milestone 12: t_sf")
