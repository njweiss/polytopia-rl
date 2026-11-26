"""Milestone 16b tests 7: `PairedT`, `paired_t`, and paired comparisons in `experiment.report`.

Golden example (docs/MILESTONE_16b.md, Rule 7 & Tests item 7):
  a = [1.0, 2.0, 3.0, 4.0], b = [2.0, 4.0, 4.0, 6.0]
  d = [1, 2, 1, 2], mean_a = 2.5, mean_b = 4.0, diff = 1.5,
  sd_diff = sqrt(1/3), t = 1.5 / sqrt(1/12) = 3 * sqrt(3) ~ 5.1961524,
  df = 3.0, p = 2 * t_sf(3 * sqrt(3), 3) ~ 0.0138468.
  (ddof=0 gives sd_diff=0.5, t=6.0; df=2n-2=6 gives p ~ 0.00202.)
"""

import json
import math
import os

import numpy as np
import pytest
import test_experiment_report as m12
from exp_helpers import t_sf_ref

from polytopia_rl import experiment
from polytopia_rl.stats import PairedT, paired_t, welch

OLD_REPORT_KEYS = {"name", "arms", "seeds", "runs", "tables", "comparisons", "n_flagged", "elo", "obs_hash"}


def paired_ref(a, b):
    """Independent paired t-test in float64 using `exp_helpers.t_sf_ref`."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = b - a
    n = len(d)
    diff = float(d.mean())
    sd = float(d.std(ddof=1))
    df = float(n - 1)
    if sd == 0.0:
        t = 0.0 if diff == 0.0 else math.copysign(math.inf, diff)
        p = 1.0 if diff == 0.0 else 0.0
    else:
        t = diff / (sd / math.sqrt(n))
        p = 2 * t_sf_ref(abs(t), df)
    return float(a.mean()), float(b.mean()), diff, sd, t, df, p


def test_paired_t_golden_hand_worked_example():
    a = [1.0, 2.0, 3.0, 4.0]
    b = [2.0, 4.0, 4.0, 6.0]
    pt = paired_t(a, b)
    assert isinstance(pt, PairedT)
    assert pt.mean_a == pytest.approx(2.5, abs=1e-12)
    assert pt.mean_b == pytest.approx(4.0, abs=1e-12)
    assert pt.diff == pytest.approx(1.5, abs=1e-12)
    assert pt.sd_diff == pytest.approx(math.sqrt(1.0 / 3.0), abs=1e-12)
    expected_t = 3.0 * math.sqrt(3.0)
    assert pt.t == pytest.approx(expected_t, abs=1e-10)
    assert pt.df == pytest.approx(3.0, abs=1e-12)
    expected_p = 2.0 * t_sf_ref(expected_t, 3.0)
    assert pt.p == pytest.approx(expected_p, abs=1e-9)
    assert pt.significant is True


def test_paired_t_alpha_threshold():
    a = [1.0, 2.0, 3.0, 4.0]
    b = [2.0, 4.0, 4.0, 6.0]
    p = paired_ref(a, b)[6]
    assert 0.01 < p < 0.02
    assert paired_t(a, b, alpha=p * 0.99).significant is False
    assert paired_t(a, b, alpha=p * 1.01).significant is True


def test_paired_t_seed_shift_invariance():
    a = [0.70, 0.72, 0.68, 0.74]
    b = [0.76, 0.77, 0.75, 0.80]
    shifts = [100.0, -50.0, 25.0, -75.0]
    a2 = [x + c for x, c in zip(a, shifts)]
    b2 = [y + c for y, c in zip(b, shifts)]

    pt1 = paired_t(a, b)
    pt2 = paired_t(a2, b2)
    assert pt2.diff == pytest.approx(pt1.diff, abs=1e-10)
    assert pt2.sd_diff == pytest.approx(pt1.sd_diff, abs=1e-10)
    assert pt2.t == pytest.approx(pt1.t, abs=1e-9)
    assert pt2.df == pytest.approx(pt1.df, abs=1e-12)
    assert pt2.p == pytest.approx(pt1.p, abs=1e-9)
    assert pt2.significant == pt1.significant

    # Contrast with Welch: huge per-seed shifts destroy Welch's t.
    assert abs(welch(a2, b2).t) < abs(welch(a, b).t) * 0.05


def test_paired_t_swapped_arms_negate_diff_and_t_and_keep_p():
    a = [0.85, 0.91, 0.88, 0.94]
    b = [0.89, 0.93, 0.90, 0.99]
    ab = paired_t(a, b)
    ba = paired_t(b, a)
    assert (ba.mean_a, ba.mean_b) == pytest.approx((ab.mean_b, ab.mean_a))
    assert ba.diff == pytest.approx(-ab.diff)
    assert ba.sd_diff == pytest.approx(ab.sd_diff)
    assert ba.t == pytest.approx(-ab.t)
    assert ba.df == pytest.approx(ab.df)
    assert ba.p == pytest.approx(ab.p)
    assert ba.significant == ab.significant


def test_paired_t_zero_variance_edge_cases():
    # Identical arms: d = [0, 0, 0] -> t = 0.0, p = 1.0, df = 2.0.
    z = paired_t([0.5, 0.75, 1.0], [0.5, 0.75, 1.0])
    assert z.diff == 0.0 and z.sd_diff == 0.0 and z.df == 2.0
    assert z.t == 0.0 and z.p == 1.0 and z.significant is False

    # Constant positive shift (exact dyadic floats so d_i == 0.25 exactly): t = +inf, p = 0.0.
    pos = paired_t([0.5, 0.75, 1.0], [0.75, 1.0, 1.25])
    assert pos.diff == pytest.approx(0.25) and pos.sd_diff == 0.0 and pos.df == 2.0
    assert pos.t == math.inf and pos.p == 0.0 and pos.significant is True

    # Constant negative shift: t = -inf, p = 0.0.
    neg = paired_t([0.75, 1.0, 1.25], [0.5, 0.75, 1.0])
    assert neg.diff == pytest.approx(-0.25) and neg.sd_diff == 0.0 and neg.df == 2.0
    assert neg.t == -math.inf and neg.p == 0.0 and neg.significant is True


@pytest.mark.parametrize(
    "a,b",
    [
        ([1.0, 2.0], [1.0, 2.0, 3.0]),
        ([1.0, 2.0, 3.0], [1.0, 2.0]),
        ([], []),
        ([1.0], [2.0]),
        ([1.0], [1.0, 2.0]),
    ],
)
def test_paired_t_invalid_lengths_raise_value_error(a, b):
    with pytest.raises(ValueError):
        paired_t(a, b)


def test_experiment_report_includes_paired_stats_and_markdown_table(tmp_path):
    cwd = os.getcwd()
    try:
        spec = m12.build(tmp_path, init=True)
        rep = experiment.report(spec)
    finally:
        os.chdir(cwd)

    out = tmp_path / "runs" / "exp" / "r"
    raw = json.loads((out / "report.json").read_text())
    assert set(raw) == OLD_REPORT_KEYS | {"cost"}

    got = {(c["a"], c["b"], c["metric"]): c for c in rep.comparisons}
    assert got
    for (a, b, m), c in got.items():
        assert "paired" in c, (a, b, m)
        pr = c["paired"]
        assert set(pr) == {"t", "df", "p", "significant"}
        va = [m12.expected_run_values(a, s)[2][m] for s in m12.SEEDS]
        vb = [m12.expected_run_values(b, s)[2][m] for s in m12.SEEDS]
        _, _, _, sd, t, df, p = paired_ref(va, vb)
        if math.isinf(t):
            assert pr["t"] == t, (a, b, m)
        elif sd < 1e-12:
            # Near-zero float64 ULP variance (e.g. constant decimal shift 0.01): exact vs float64
            # sd can differ at the ULP level, giving huge t with the same sign and p ~ 0.
            assert math.copysign(1.0, pr["t"]) == math.copysign(1.0, t), (a, b, m)
            assert abs(pr["t"]) > 1e10, (a, b, m)
        else:
            assert pr["t"] == pytest.approx(t, rel=1e-6), (a, b, m)
        assert pr["df"] == pytest.approx(df, abs=1e-12), (a, b, m)
        assert pr["p"] == pytest.approx(p, abs=1e-6), (a, b, m)
        if c["primary"]:
            assert pr["significant"] is bool(p < 0.05), (a, b, m)
        else:
            assert pr["significant"] is None, f"exploratory metric {m} must not be flagged"

    # Exploratory metric economy_e0 has a tiny paired p-value but must still remain unflagged.
    eco = got[("base", "pbot40", "economy_e0")]["paired"]
    assert eco["p"] < 0.05 and eco["significant"] is None

    # Markdown places the paired table directly beneath the Welch table (Clarification B2).
    md = (out / "report.md").read_text()
    welch_hdr = "## Comparisons (Welch, b minus a)"
    paired_hdr = "## Comparisons (paired t, b minus a)"
    rr_hdr = "## Round robin"
    assert welch_hdr in md and paired_hdr in md and rr_hdr in md
    assert md.index(welch_hdr) < md.index(paired_hdr) < md.index(rr_hdr)

    paired_section = md[md.index(paired_hdr) : md.index(rr_hdr)]
    assert "significant" in paired_section
    assert "exploratory" in paired_section
    assert "base" in paired_section and "pbot40" in paired_section and "rush_e0" in paired_section


def test_experiment_report_paired_detects_effect_masked_by_seed_variance(tmp_path):
    """Four seeds with large between-seed spread and a consistent +0.05 lift on `rush_e0`:
    Welch is not detected (p > 0.5), while paired t is significant (p < 0.05)."""
    from polytopia_rl.suite import DEFAULT_SUITE, SuiteResult, suite_to_json

    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        arms = {"ctrl": {}, "treat": {"p_bot": 0.4}}
        seeds = [1, 2, 3, 4]
        spec_d = dict(
            name="paired_power",
            base_config="python/configs/selfplay.json",
            seeds=seeds,
            arms=arms,
            primary=["rush_e0"],
            suite_seeds=50,
            tournament_seeds=10,
            eval_checkpoints=1,
            h2h_seeds=100,
        )
        (tmp_path / "spec.json").write_text(json.dumps(spec_d))
        spec = experiment.load_spec(str(tmp_path / "spec.json"))
        base_wins = {1: 20, 2: 40, 3: 60, 4: 80}
        lift = {1: 4, 2: 6, 3: 4, 4: 6}  # d = [0.04, 0.06, 0.04, 0.06] -> t = 8.66, df = 3, p ~ 0.003
        for arm in arms:
            for seed in seeds:
                rd = tmp_path / "runs" / "exp" / "paired_power" / arm / f"seed{seed}"
                (rd / "suite_1000").mkdir(parents=True)
                crs = []
                for c in DEFAULT_SUITE:
                    w = base_wins[seed] + (lift[seed] if arm == "treat" else 0)
                    crs.append(m12.cond_result(c.name, w, 100))
                res = SuiteResult(conditions={c.name: c for c in crs}, obs_hash=m12.OBS, rules_hash=m12.RULES, step=1000)
                (rd / "suite_1000" / "suite.json").write_text(suite_to_json(res))
        rep = experiment.report(spec)
    finally:
        os.chdir(cwd)

    rush = next(c for c in rep.comparisons if c["metric"] == "rush_e0")
    assert rush["p"] > 0.5 and rush["significant"] is False
    assert rush["paired"]["p"] < 0.05 and rush["paired"]["significant"] is True
    assert rush["paired"]["df"] == pytest.approx(3.0)
