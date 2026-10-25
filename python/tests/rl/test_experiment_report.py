"""Milestone 12 tests 7: `report` on hand-written run directories ("Report").

Every expected number is computed here from the table that wrote the files, with the brief's rules:
a run's value for a suite condition is the mean of its checkpoints' win rates (not the pooled games:
one checkpoint of `pbot40/seed2` has 40 games instead of 100 so the two differ); `elo_vs_init` per
checkpoint is `400 log10(s / (1 - s))` of the clipped head-to-head score (exp_helpers.elo_ref), then
the mean over checkpoints; per arm the per-seed values, their mean, `ddof=1` sd, min and max; Welch
per pair of arms in spec order (first arm as `a`), flagged only for `primary` metrics.

Clipping hand values: 200/200 -> s = 1 - 0.5/200 = 0.9975, 400 log10(399) = +1040.39; 0/200 ->
-1040.39; 100 wins + 50 draws of 200 -> s = 0.625, 400 log10(5/3) = +88.74.

The design makes `rush_e0` (primary) separate base from pbot40 clearly, and `economy_e0` (not
primary) separate them even more: its comparison must still carry no flag.
"""

import dataclasses
import json
import math
import os

import numpy as np
import pytest
from exp_helpers import elo_ref, welch_ref

from polytopia_rl import experiment
from polytopia_rl.evaluate import wilson_interval
from polytopia_rl.suite import DEFAULT_SUITE, ConditionResult, SuiteResult, suite_to_json

ARMS = {"base": {}, "pbot40": {"p_bot": 0.4}, "ent003": {"ent_coef": 0.003}}
SEEDS = [1, 2, 3]
STEPS = [3000, 6000, 9000]
CONDS = [c.name for c in DEFAULT_SUITE]
OBS, RULES = "a" * 16, "b" * 16
# Base win rate (percent) per arm and condition, in DEFAULT_SUITE order.
RATE = {
    "base": [99, 60, 80, 55, 80, 62, 88, 85],
    "pbot40": [99, 95, 96, 70, 84, 61, 90, 87],
    "ent003": [98, 61, 89, 57, 81, 64, 87, 86],
}
SEED_DELTA = {1: 0, 2: -2, 3: 1}
CKPT_DELTA = [-1, 2, 0]


def games_of(arm, seed, k):
    return 40 if (arm, seed, k) == ("pbot40", 2, 2) else 100


def wins_of(arm, seed, ci, k):
    pct = min(100, max(0, RATE[arm][ci] + SEED_DELTA[seed] + CKPT_DELTA[k] + (ci % 3) * (seed - 2)))
    return round(pct * games_of(arm, seed, k) / 100)


def h2h_of(arm, seed, k):
    """(a_wins, b_wins, draws) of 200 games against the init checkpoint."""
    if (arm, seed, k) == ("base", 1, 2):
        return 200, 0, 0
    if (arm, seed, k) == ("ent003", 3, 0):
        return 0, 200, 0
    if (arm, seed, k) == ("pbot40", 1, 1):
        return 100, 50, 50
    a = 110 + 15 * list(ARMS).index(arm) + 7 * seed + 3 * k
    return a, 200 - a - 4, 4


def cond_result(name, wins, games):
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


def build(root, init=True, obs_override=None, arms=ARMS, seeds=SEEDS):
    """Hand-written experiment `r` under `root`; returns the spec."""
    os.chdir(root)
    d = dict(
        name="r",
        base_config="python/configs/selfplay.json",
        init_checkpoint="init.pt" if init else None,
        seeds=seeds,
        arms=arms,
        primary=["rush_e0", "elo_vs_init"] if init else ["rush_e0"],
        suite_seeds=50,
        tournament_seeds=10,
        eval_checkpoints=3,
        h2h_seeds=100,
    )
    (root / "spec.json").write_text(json.dumps(d))
    spec = experiment.load_spec(str(root / "spec.json"))
    finals = []
    for arm in arms:
        for seed in seeds:
            rd = root / "runs" / "exp" / "r" / arm / f"seed{seed}"
            rd.mkdir(parents=True)
            for name in ["ckpt_1000.pt"] + [f"ckpt_{s}.pt" for s in STEPS] + ["ckpt_final.pt", "latest.pt"]:
                (rd / name).write_bytes(b"")  # placeholders: report reads no checkpoint
            for k, step in enumerate(STEPS):
                crs = [cond_result(c, wins_of(arm, seed, ci, k), games_of(arm, seed, k)) for ci, c in enumerate(CONDS)]
                obs = obs_override.get((arm, seed, step), OBS) if obs_override else OBS
                res = SuiteResult(conditions={c.name: c for c in crs}, obs_hash=obs, rules_hash=RULES, step=step)
                (rd / f"suite_{step}").mkdir()
                (rd / f"suite_{step}" / "suite.json").write_text(suite_to_json(res))
                if init:
                    a, b, dr = h2h_of(arm, seed, k)
                    h = dict(a=f"{arm}_seed{seed}", b="init", games=200, a_wins=a, b_wins=b, draws=dr, seat_wins={"0": a, "1": b}, per_game=[])
                    (rd / f"h2h_{step}.json").write_text(json.dumps(h))
            rows = []
            for step in range(1000, 10000, 1000):
                sub = {c: {"win_rate": 0.5 + 0.004 * step / 100 - 0.01 * seed + 0.02 * list(arms).index(arm)} for c in ("economy_e0", "rush_e0", "rush_e20")}
                rows.append({"type": "suite", "global_step": step, "suite": sub, "regression": False})
                if step % 2000 == 0:
                    rows.append({"type": "rating", "global_step": step, "rating": {"learner": 921.3, "snapshot_0": 900.0}, "gap_snapshot_0": 21.3})
            (rd / "eval.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            finals.append(f"{arm}_seed{seed}")
    players = ["random", "economy", "rush"] + (["init"] if init else []) + finals
    n = len(players)
    games = np.full((n, n), 20.0) - 20.0 * np.eye(n)
    wins = np.triu(np.full((n, n), 12.0), 1) + np.tril(np.full((n, n), 8.0), -1)
    t = dict(
        players=players,
        wins=wins.tolist(),
        draws=np.zeros((n, n)).tolist(),
        games=games.tolist(),
        seat_win_rate={"0": 0.5, "1": 0.5},
        ratings=[0.0] + [37.5 * i + 0.25 for i in range(1, n)],
    )
    (root / "runs" / "exp" / "r" / "tournament.json").write_text(json.dumps(t))
    (root / "runs" / "exp" / "r" / "tournament.md").write_text("| player | Elo |\n")
    return spec


def expected_run_values(arm, seed, init=True):
    suite = {c: [wins_of(arm, seed, ci, k) / games_of(arm, seed, k) for k in range(len(STEPS))] for ci, c in enumerate(CONDS)}
    out = {c: float(np.mean(v)) for c, v in suite.items()}
    elo = None
    if init:
        elo = [elo_ref(a, dr, 200) for a, _, dr in (h2h_of(arm, seed, k) for k in range(len(STEPS)))]
        out["elo_vs_init"] = float(np.mean(elo))
    return suite, elo, out


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    root = tmp_path_factory.mktemp("rep")
    cwd = os.getcwd()
    try:
        spec = build(root)
        rep = experiment.report(spec)
    finally:
        os.chdir(cwd)
    return root, rep


def test_per_run_values_are_means_over_the_checkpoints(made):
    # The reference against the hand values in the module docstring.
    assert elo_ref(200, 0, 200) == pytest.approx(400 * math.log10(399), abs=1e-9)
    assert elo_ref(200, 0, 200) == pytest.approx(1040.39, abs=0.01)
    assert elo_ref(0, 0, 200) == pytest.approx(-1040.39, abs=0.01)
    assert elo_ref(100, 50, 200) == pytest.approx(88.74, abs=0.01)
    _, rep = made
    assert isinstance(rep, experiment.ExperimentReport)
    assert [(r["arm"], r["seed"]) for r in rep.runs] == [(a, s) for a in ARMS for s in SEEDS]
    for r in rep.runs:
        suite, elo, means = expected_run_values(r["arm"], r["seed"])
        assert list(r["steps"]) == STEPS
        for c in CONDS:
            assert r["suite"][c] == pytest.approx(suite[c]), (r["arm"], r["seed"], c)
            assert r["suite_mean"][c] == pytest.approx(means[c]), (r["arm"], r["seed"], c)
        assert r["elo_vs_init"] == pytest.approx(elo)
        assert r["elo_vs_init_mean"] == pytest.approx(means["elo_vs_init"])
        assert all(math.isfinite(x) for x in r["elo_vs_init"]), "clipped, never infinite"
        assert (r["obs_hash"], r["rules_hash"]) == (OBS, RULES)
    # The 40-game checkpoint: mean of rates, not pooled games.
    r = next(x for x in rep.runs if (x["arm"], x["seed"]) == ("pbot40", 2))
    ci = CONDS.index("economy_e20")
    pooled = sum(wins_of("pbot40", 2, ci, k) for k in range(3)) / sum(games_of("pbot40", 2, k) for k in range(3))
    assert r["suite_mean"]["economy_e20"] != pytest.approx(pooled, abs=1e-6)


def test_per_seed_tables_mean_sd_min_max(made):
    _, rep = made
    assert list(rep.tables) == CONDS + ["elo_vs_init"]
    for metric, table in rep.tables.items():
        assert list(table) == list(ARMS), metric
        for arm, row in table.items():
            vals = [expected_run_values(arm, s)[2][metric] for s in SEEDS]
            assert row["per_seed"] == pytest.approx(vals), (metric, arm)
            assert row["mean"] == pytest.approx(np.mean(vals))
            assert row["sd"] == pytest.approx(np.std(vals, ddof=1))
            assert row["min"] == pytest.approx(min(vals)) and row["max"] == pytest.approx(max(vals))


def test_welch_per_pair_with_flags_only_on_primary_metrics(made):
    _, rep = made
    pairs = [("base", "pbot40"), ("base", "ent003"), ("pbot40", "ent003")]
    metrics = CONDS + ["elo_vs_init"]
    got = {(c["a"], c["b"], c["metric"]): c for c in rep.comparisons}
    assert len(rep.comparisons) == len(got) == len(pairs) * len(metrics)
    assert set(got) == {(a, b, m) for a, b in pairs for m in metrics}
    for (a, b, m), c in got.items():
        va = [expected_run_values(a, s)[2][m] for s in SEEDS]
        vb = [expected_run_values(b, s)[2][m] for s in SEEDS]
        mean_a, mean_b, diff, t, df, p = welch_ref(va, vb)
        assert (c["mean_a"], c["mean_b"], c["diff"]) == pytest.approx((mean_a, mean_b, diff)), (a, b, m)
        assert c["t"] == pytest.approx(t, rel=1e-6) and c["df"] == pytest.approx(df, rel=1e-6), (a, b, m)
        assert c["p"] == pytest.approx(p, abs=1e-6), (a, b, m)
        primary = m in ("rush_e0", "elo_vs_init")
        assert c["primary"] == primary
        if primary:
            assert c["significant"] == (p < 0.05), (a, b, m)
        else:
            assert c["significant"] is None, (a, b, m)
    assert rep.n_flagged == len(pairs) * 2
    # The design: rush_e0 separates base from pbot40 (flagged); economy_e0 separates them more but
    # is exploratory.
    flagged = got[("base", "pbot40", "rush_e0")]["significant"]
    assert flagged is not None and bool(flagged)
    assert got[("base", "pbot40", "economy_e0")]["p"] < got[("base", "pbot40", "rush_e0")]["p"]
    assert got[("base", "pbot40", "economy_e0")]["significant"] is None


def test_round_robin_elo_and_written_files(made):
    root, rep = made
    t = json.loads((root / "runs" / "exp" / "r" / "tournament.json").read_text())
    assert rep.elo == pytest.approx(dict(zip(t["players"], t["ratings"])))
    assert rep.obs_hash == OBS
    out = root / "runs" / "exp" / "r"
    assert json.loads((out / "report.json").read_text()) == json.loads(json.dumps(dataclasses.asdict(rep)))
    md = (out / "report.md").read_text()
    for name in list(ARMS) + CONDS + ["elo_vs_init"]:
        assert name in md, name
    png = out / "report_curves.png"
    assert png.exists() and png.stat().st_size > 0
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_without_init_there_is_no_elo_vs_init(tmp_path):
    cwd = os.getcwd()
    try:
        arms = {"base": {}, "pbot40": {"p_bot": 0.4}}
        spec = build(tmp_path, init=False, arms=arms, seeds=[1, 2, 3])
        rep = experiment.report(spec)
    finally:
        os.chdir(cwd)
    assert list(rep.tables) == CONDS
    assert all(r["elo_vs_init"] is None and r["elo_vs_init_mean"] is None for r in rep.runs)
    assert {c["metric"] for c in rep.comparisons} == set(CONDS)
    assert rep.n_flagged == 1


def test_mixed_obs_hash_is_refused(tmp_path):
    cwd = os.getcwd()
    try:
        spec = build(tmp_path, obs_override={("ent003", 2, 6000): "c" * 16})
        with pytest.raises(ValueError):
            experiment.report(spec)
    finally:
        os.chdir(cwd)
