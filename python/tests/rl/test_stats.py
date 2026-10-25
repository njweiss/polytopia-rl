"""Milestone 12 tests 4: Welch's t-test and the Student t tail ("Welch's t-test").

Known values: t_sf(0, df) = 0.5 by symmetry; df 1 is the Cauchy distribution, P(T > t) = 1/2 -
atan(t) / pi, so t_sf(1, 1) = 0.25; df 2 has P(T > t) = 1/2 - t / (2 sqrt(2 + t^2)); two-sided 5%
critical values from the standard t table: 12.7062 (df 1), 2.776445 (df 4), 1.959964 (normal).

Hand example (exact fractions): a = [1, 2, 3, 4, 5] (mean 3, s^2 = 5/2, n 5), b = [2, 4, 6] (mean 4,
s^2 = 4, n 3). s_a^2 / n_a = 1/2, s_b^2 / n_b = 4/3, sum 11/6:
t = 1 / sqrt(11/6) = sqrt(6/11) = 0.7385489, df = (11/6)^2 / ((1/2)^2 / 4 + (4/3)^2 / 2) = 484/137 =
3.5328467, p = 2 * P(T_3.5328 > 0.7385) = 0.506214 (Simpson integration of the density,
exp_helpers.t_sf_ref). Student's pooled t would give t = 0.7906 with df 6; ddof 0 gives t = 0.9045.
"""

import math

import pytest
from exp_helpers import t_sf_ref, welch_ref

from polytopia_rl.stats import Welch, t_sf, welch


@pytest.mark.parametrize("df", [1, 2, 3.5, 4, 30, 1e7])
def test_t_sf_at_zero_is_one_half(df):
    assert t_sf(0.0, df) == pytest.approx(0.5, abs=1e-10)


@pytest.mark.parametrize("t", [0.1, 0.5, 1.0, 2.0, 5.0, 40.0])
def test_t_sf_matches_the_closed_forms_for_df_1_and_2(t):
    assert t_sf(t, 1) == pytest.approx(0.5 - math.atan(t) / math.pi, abs=1e-10)
    assert t_sf(t, 2) == pytest.approx(0.5 - t / (2 * math.sqrt(2 + t * t)), abs=1e-10)


def test_t_sf_known_values_from_the_brief():
    assert t_sf(1.0, 1) == pytest.approx(0.25, abs=1e-10)
    assert 2 * t_sf(2.776445, 4) == pytest.approx(0.05, abs=1e-5)
    assert 2 * t_sf(12.7062, 1) == pytest.approx(0.05, abs=1e-5)
    # Large df approaches the normal distribution.
    assert 2 * t_sf(1.959964, 1e7) == pytest.approx(0.05, abs=1e-5)


@pytest.mark.parametrize("t,df", [(0.7385, 3.53), (1.5, 3.5), (2.2, 4.7), (0.3, 9.25), (3.1, 17.0)])
def test_t_sf_at_non_integer_df_matches_numerical_integration(t, df):
    assert t_sf(t, df) == pytest.approx(t_sf_ref(t, df), abs=1e-9)


def test_t_sf_of_a_negative_t_is_the_complement():
    # P(T > -t) = 1 - P(T > t) by symmetry (the function's definition, P(T > t)).
    assert t_sf(-1.0, 1) == pytest.approx(0.75, abs=1e-10)
    assert t_sf(-1.5, 3.5) == pytest.approx(1 - t_sf_ref(1.5, 3.5), abs=1e-9)


def test_t_sf_decreases_in_t():
    xs = [0.0, 0.2, 0.7, 1.3, 2.5, 6.0]
    ys = [t_sf(x, 4.4) for x in xs]
    assert all(y1 > y2 for y1, y2 in zip(ys, ys[1:]))


def test_welch_on_the_hand_example():
    w = welch([1, 2, 3, 4, 5], [2, 4, 6])
    assert isinstance(w, Welch)
    assert (w.mean_a, w.mean_b) == pytest.approx((3.0, 4.0))
    assert w.diff == pytest.approx(1.0)
    assert w.t == pytest.approx(math.sqrt(6 / 11), abs=1e-9)
    assert w.df == pytest.approx(484 / 137, abs=1e-9)
    assert w.p == pytest.approx(0.506214, abs=1e-5)
    assert w.p == pytest.approx(welch_ref([1, 2, 3, 4, 5], [2, 4, 6])[5], abs=1e-9)
    assert not w.significant


def test_welch_three_seeds_against_three_seeds():
    # Three seeds per arm, as in 12b: rush_e0 win rates of two arms.
    a, b = [0.90, 0.94, 0.86], [0.80, 0.92, 0.83]
    w = welch(a, b)
    mean_a, mean_b, diff, t, df, p = welch_ref(a, b)
    assert (w.mean_a, w.mean_b, w.diff) == pytest.approx((mean_a, mean_b, diff))
    assert w.t == pytest.approx(t, abs=1e-9) and w.t < 0
    assert w.df == pytest.approx(df, abs=1e-9)
    assert w.p == pytest.approx(p, abs=1e-8)
    assert not w.significant


def test_welch_significance_threshold_is_alpha():
    a, b = [0.0, 1.0, 2.0], [10.0, 11.0, 12.5]
    p = welch_ref(a, b)[5]
    assert 1e-4 < p < 0.01
    assert welch(a, b).significant
    assert not welch(a, b, alpha=p * 0.99).significant
    assert welch(a, b, alpha=p * 1.01).significant
    w = welch([1, 2, 3, 4, 5], [2, 4, 6], alpha=0.6)  # p = 0.506
    assert w.significant


def test_swapping_the_arms_negates_t_and_diff_and_keeps_p():
    a, b = [1, 2, 3, 4, 5], [2, 4, 6]
    ab, ba = welch(a, b), welch(b, a)
    assert ba.t == pytest.approx(-ab.t)
    assert ba.diff == pytest.approx(-ab.diff)
    assert (ba.mean_a, ba.mean_b) == pytest.approx((ab.mean_b, ab.mean_a))
    assert ba.df == pytest.approx(ab.df)
    assert ba.p == pytest.approx(ab.p)


def test_zero_variance_cases():
    w = welch([0.9, 0.9, 0.9], [0.9, 0.9])
    assert w.t == 0.0 and w.p == 1.0 and not w.significant
    assert w.diff == pytest.approx(0.0)
    w = welch([0.5, 0.5], [0.75, 0.75, 0.75])
    assert w.t == math.inf and w.p == 0.0 and w.significant
    assert w.diff == pytest.approx(0.25)
    w = welch([0.75, 0.75, 0.75], [0.5, 0.5])
    assert w.t == -math.inf and w.p == 0.0 and w.significant


def test_one_zero_variance_uses_the_formula():
    # s_a^2 = 0: t = (mean_b - mean_a) / sqrt(s_b^2 / n_b), df = n_b - 1 exactly.
    a, b = [1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]
    w = welch(a, b)
    assert w.t == pytest.approx(1.5 / math.sqrt((5 / 3) / 4), abs=1e-9)
    assert w.df == pytest.approx(3.0, abs=1e-9)
    assert w.p == pytest.approx(2 * t_sf_ref(abs(w.t), 3), abs=1e-8)


@pytest.mark.parametrize("a,b", [([1.0], [1.0, 2.0]), ([1.0, 2.0], [3.0]), ([], [1.0, 2.0])])
def test_fewer_than_two_values_raise(a, b):
    with pytest.raises(ValueError):
        welch(a, b)
