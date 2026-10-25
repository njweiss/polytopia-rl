"""Welch's t-test across seeds (docs/MILESTONE_12 "Welch's t-test"), with numpy/math only."""

from __future__ import annotations

import math
import statistics
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
    if len(a) < 2 or len(b) < 2:
        raise ValueError(f"Welch's t-test needs at least 2 values per arm (got {len(a)} and {len(b)})")
    # statistics computes in exact arithmetic, so identical values give a variance of exactly 0.
    mean_a, mean_b = statistics.mean(map(float, a)), statistics.mean(map(float, b))
    va = statistics.variance(map(float, a)) / len(a)  # s_a^2 / n_a, ddof 1
    vb = statistics.variance(map(float, b)) / len(b)
    diff = mean_b - mean_a
    if va + vb == 0:
        t = 0.0 if diff == 0 else math.copysign(math.inf, diff)
        df = math.nan
        p = 1.0 if diff == 0 else 0.0
    else:
        t = diff / math.sqrt(va + vb)
        df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
        p = 2 * t_sf(abs(t), df)
    return Welch(mean_a=mean_a, mean_b=mean_b, diff=diff, t=t, df=df, p=p, significant=bool(p < alpha))


def t_sf(t: float, df: float) -> float:
    """`P(T > t)` for Student's t with `df` degrees of freedom (`df` may be non-integer):
    `0.5 * I_{df / (df + t^2)}(df / 2, 1 / 2)` for `t >= 0`, the regularized incomplete beta function
    by a continued fraction (Lentz's method), absolute error below 1e-10."""
    if t < 0:
        return 1.0 - t_sf(-t, df)
    if math.isinf(t):
        return 0.0
    # x = df / (df + t^2) and 1 - x, each computed directly so neither loses digits.
    return 0.5 * _betainc(df / 2, 0.5, df / (df + t * t), t * t / (df + t * t))


def _betainc(a: float, b: float, x: float, y: float) -> float:
    """The regularized incomplete beta function `I_x(a, b)`, with `y = 1 - x`."""
    if x <= 0.0:
        return 0.0
    if y <= 0.0:
        return 1.0
    # The continued fraction converges fast for x < (a + 1) / (a + b + 2); otherwise use the
    # symmetry I_x(a, b) = 1 - I_{1-x}(b, a).
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _betainc(b, a, y, x)
    log_front = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(y)
    return math.exp(log_front) * _betacf(a, b, x) / a


def _betacf(a: float, b: float, x: float, eps: float = 1e-15, max_iter: int = 1_000_000) -> float:
    """The continued fraction of `I_x(a, b)` by the modified Lentz method."""
    tiny = 1e-300
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, max_iter + 1):
        # Even step: d_{2m} = m (b - m) x / ((a + 2m - 1)(a + 2m)).
        num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        d = 1.0 + num * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + num / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        # Odd step: d_{2m+1} = -(a + m)(a + b + m) x / ((a + 2m)(a + 2m + 1)).
        num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + num / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    raise ArithmeticError(f"incomplete beta continued fraction did not converge (a={a}, b={b}, x={x})")
