"""Milestone 14 tests 3 (report side) and 5: `elo_vs_ref` and the cost columns, on hand-written run
directories ("Rules").

The run directories are those of test_experiment_report.py (same suite tables and head-to-head
scores, from its `wins_of`, `games_of` and `h2h_of`), except that:
- the spec has a `reference`, and each `h2h_{step}.json` is a game against it (`b` = "ref");
- the arms are networks of different sizes, each run's `ckpt_final.pt` is a real tiny checkpoint of
  its arm's `width` and `blocks` (6x6 map), and the reference is a fourth size;
- each run has a hand-written `metrics.jsonl` (the `SPS` table below).

Expected values: `elo_vs_ref` per checkpoint is `exp_helpers.elo_ref` of the head-to-head (computed
exactly like `elo_vs_init`, clipping included: 200/200 -> +1040.39, 0/200 -> -1040.39), the run value
is the mean over the checkpoints, the per-arm table and the Welch comparisons are as for
`elo_vs_init`. The parameter count of an arm is `sum(p.numel() for p in PolicyNet(spec, width,
blocks).parameters())`, built here; the mean sps of a run is the mean of the `sps` field over its
`metrics.jsonl` rows, then `np.mean` and `np.std(ddof=1)` over seeds. Names and shapes of the
report fields: docs/QUESTIONS.md Q39 (cost) and Q40 (reference).
"""

import dataclasses
import json
import math
import os

import numpy as np
import polytopia_sim as ps
import pytest
import test_experiment_report as m12
import torch
from exp_helpers import elo_ref, welch_ref

from polytopia_rl import experiment
from polytopia_rl.checkpoint import save_checkpoint
from polytopia_rl.config import PPOConfig
from polytopia_rl.model import PolicyNet
from polytopia_rl.suite import SuiteResult, suite_to_json

SPEC6 = ps.Env(size=6).spec()
# The milestone 12 arm names (its fixture tables are keyed by them), with sizes as overrides.
SIZES = {"base": (8, 1), "pbot40": (16, 2), "ent003": (12, 1)}
ARMS = {arm: {"width": w, "blocks": b} for arm, (w, b) in SIZES.items()}
REF_SIZE = (4, 3)
SEEDS = m12.SEEDS
STEPS = m12.STEPS
CONDS = m12.CONDS


def sps_rows(arm, seed):
    """Hand-written `sps` values of run (arm, seed): 4 to 6 updates, arm-dependent speed."""
    base = {"base": 1300.0, "pbot40": 700.0, "ent003": 950.0}[arm]
    n = 3 + seed
    return [base + 37.0 * seed - 11.0 * i + (5.5 if i % 2 else 0.0) for i in range(n)]


def write_ckpt(path, width, blocks, seed):
    cfg = PPOConfig(size=6, turn_cap=3, width=width, blocks=blocks, device="cpu", seed=seed)
    torch.manual_seed(seed)
    net = PolicyNet(SPEC6, width=width, blocks=blocks)
    save_checkpoint(str(path), model=net, optimizer=None, cfg=cfg, global_step=9000, stage=0, seed_index=None, rng_state=None, spec=SPEC6)


def build(root, init=False):
    """Experiment `r` with a reference under `root` (also the working directory); returns the spec."""
    os.chdir(root)
    d = dict(
        name="r",
        base_config="python/configs/selfplay.json",
        seeds=SEEDS,
        arms=ARMS,
        reference="ref.pt",
        primary=["rush_e0", "elo_vs_ref"],
        suite_seeds=50,
        tournament_seeds=10,
        eval_checkpoints=3,
        h2h_seeds=100,
    )
    if init:
        d["init_checkpoint"] = "init.pt"
        write_ckpt(root / "init.pt", 8, 1, 99)
    write_ckpt(root / "ref.pt", *REF_SIZE, 77)
    (root / "spec.json").write_text(json.dumps(d))
    spec = experiment.load_spec(str(root / "spec.json"))
    finals = []
    for arm in ARMS:
        for seed in SEEDS:
            rd = root / "runs" / "exp" / "r" / arm / f"seed{seed}"
            rd.mkdir(parents=True)
            for name in ["ckpt_1000.pt"] + [f"ckpt_{s}.pt" for s in STEPS] + ["latest.pt"]:
                (rd / name).write_bytes(b"")  # placeholders
            write_ckpt(rd / "ckpt_final.pt", *SIZES[arm], seed)
            for k, step in enumerate(STEPS):
                crs = [m12.cond_result(c, m12.wins_of(arm, seed, ci, k), m12.games_of(arm, seed, k)) for ci, c in enumerate(CONDS)]
                res = SuiteResult(conditions={c.name: c for c in crs}, obs_hash=m12.OBS, rules_hash=m12.RULES, step=step)
                (rd / f"suite_{step}").mkdir()
                (rd / f"suite_{step}" / "suite.json").write_text(suite_to_json(res))
                a, b, dr = m12.h2h_of(arm, seed, k)
                h = dict(a=f"{arm}_seed{seed}", b="ref", games=200, a_wins=a, b_wins=b, draws=dr, seat_wins={"0": a, "1": b}, per_game=[])
                (rd / f"h2h_{step}.json").write_text(json.dumps(h))
            rows = []
            for i, v in enumerate(sps_rows(arm, seed)):
                rows.append({"global_step": 4096 * (i + 1), "update": i + 1, "sps": v, "rollout_sps": 2 * v, "update_sps": 3 * v, "stage": 0, "lr": 1e-4})
            (rd / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            finals.append(f"{arm}_seed{seed}")
    players = ["random", "economy", "rush", "ref"] + finals
    n = len(players)
    t = dict(
        players=players,
        wins=(np.triu(np.full((n, n), 12.0), 1) + np.tril(np.full((n, n), 8.0), -1)).tolist(),
        draws=np.zeros((n, n)).tolist(),
        games=(np.full((n, n), 20.0) - 20.0 * np.eye(n)).tolist(),
        seat_win_rate={"0": 0.5, "1": 0.5},
        ratings=[0.0] + [37.5 * i + 0.25 for i in range(1, n)],
    )
    (root / "runs" / "exp" / "r" / "tournament.json").write_text(json.dumps(t))
    (root / "runs" / "exp" / "r" / "tournament.md").write_text("| player | Elo |\n")
    return spec


def expected_elo(arm, seed):
    per = [elo_ref(a, dr, 200) for a, _, dr in (m12.h2h_of(arm, seed, k) for k in range(len(STEPS)))]
    return per, float(np.mean(per))


def expected_value(arm, seed, metric):
    if metric == "elo_vs_ref":
        return expected_elo(arm, seed)[1]
    ci = CONDS.index(metric)
    return float(np.mean([m12.wins_of(arm, seed, ci, k) / m12.games_of(arm, seed, k) for k in range(len(STEPS))]))


def params_of(width, blocks):
    return sum(p.numel() for p in PolicyNet(SPEC6, width=width, blocks=blocks).parameters())


def made_in(root, init=False):
    cwd = os.getcwd()
    try:
        spec = build(root, init=init)
        rep = experiment.report(spec)
    finally:
        os.chdir(cwd)
    return rep


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    root = tmp_path_factory.mktemp("ref")
    return root, made_in(root)


def test_metrics_are_the_conditions_then_elo_vs_ref(made):
    _, rep = made
    assert list(rep.tables) == CONDS + ["elo_vs_ref"]
    assert "elo_vs_init" not in rep.tables


def test_elo_vs_ref_per_run_with_clipping(made):
    _, rep = made
    assert [(r["arm"], r["seed"]) for r in rep.runs] == [(a, s) for a in ARMS for s in SEEDS]
    clipped = []
    for r in rep.runs:
        per, mean = expected_elo(r["arm"], r["seed"])
        assert r["elo_vs_ref"] == pytest.approx(per), (r["arm"], r["seed"])
        assert r["elo_vs_ref_mean"] == pytest.approx(mean), (r["arm"], r["seed"])
        assert all(math.isfinite(x) for x in r["elo_vs_ref"]), "clipped, never infinite"
        clipped += [x for x in r["elo_vs_ref"] if abs(x) > 1000]
    # base/seed1 at 9000 won 200 of 200, ent003/seed3 at 3000 lost 200 of 200 (m12.h2h_of).
    assert sorted(clipped) == pytest.approx([-1040.39, 1040.39], abs=0.01)


def test_elo_vs_ref_table_and_flagged_comparisons(made):
    _, rep = made
    for arm, row in rep.tables["elo_vs_ref"].items():
        vals = [expected_value(arm, s, "elo_vs_ref") for s in SEEDS]
        assert row["per_seed"] == pytest.approx(vals), arm
        assert row["mean"] == pytest.approx(np.mean(vals))
        assert row["sd"] == pytest.approx(np.std(vals, ddof=1))
        assert (row["min"], row["max"]) == pytest.approx((min(vals), max(vals)))
    pairs = [("base", "pbot40"), ("base", "ent003"), ("pbot40", "ent003")]
    got = {(c["a"], c["b"], c["metric"]): c for c in rep.comparisons}
    assert set(got) == {(a, b, m) for a, b in pairs for m in CONDS + ["elo_vs_ref"]}
    for a, b in pairs:
        c = got[a, b, "elo_vs_ref"]
        _, _, diff, t, df, p = welch_ref([expected_value(a, s, "elo_vs_ref") for s in SEEDS], [expected_value(b, s, "elo_vs_ref") for s in SEEDS])
        assert c["diff"] == pytest.approx(diff) and c["t"] == pytest.approx(t, rel=1e-6) and c["df"] == pytest.approx(df, rel=1e-6)
        assert c["p"] == pytest.approx(p, abs=1e-6)
        assert c["primary"] is True and c["significant"] == (p < 0.05)
    assert rep.n_flagged == len(pairs) * 2


def test_round_robin_with_ref_is_read(made):
    root, rep = made
    t = json.loads((root / "runs" / "exp" / "r" / "tournament.json").read_text())
    assert rep.elo is not None, "the round robin's players (bots, ref, runs) are the experiment's"
    assert rep.elo == pytest.approx(dict(zip(t["players"], t["ratings"])))


def test_report_files_with_a_reference(made):
    root, rep = made
    out = root / "runs" / "exp" / "r"
    assert json.loads((out / "report.json").read_text()) == json.loads(json.dumps(dataclasses.asdict(rep)))
    md = (out / "report.md").read_text()
    assert "elo_vs_ref" in md
    assert "elo_vs_init" not in md


def test_reference_beside_an_init_checkpoint_gives_only_elo_vs_ref(tmp_path):
    rep = made_in(tmp_path, init=True)
    assert list(rep.tables) == CONDS + ["elo_vs_ref"]
    for r in rep.runs:
        assert r["elo_vs_ref"] == pytest.approx(expected_elo(r["arm"], r["seed"])[0])
    assert rep.elo is not None, "the round robin has ref and no init"


# ---------------------------------------------------------------- cost columns


def test_parameter_count_of_each_arm(made):
    _, rep = made
    assert list(rep.cost) == list(ARMS)
    want = {arm: params_of(*SIZES[arm]) for arm in ARMS}
    assert len(set(want.values())) == len(want), "three different sizes"
    assert params_of(*REF_SIZE) not in want.values()
    for arm in ARMS:
        assert rep.cost[arm]["params"] == want[arm], arm


def test_mean_sps_per_run_then_over_seeds(made):
    _, rep = made
    for arm in ARMS:
        per_seed = [float(np.mean(sps_rows(arm, s))) for s in SEEDS]
        row = rep.cost[arm]["sps"]
        assert row["per_seed"] == pytest.approx(per_seed), arm
        assert row["mean"] == pytest.approx(np.mean(per_seed)), arm
        assert row["sd"] == pytest.approx(np.std(per_seed, ddof=1)), arm


def test_cost_columns_in_the_markdown(made):
    root, rep = made
    md = (root / "runs" / "exp" / "r" / "report.md").read_text()
    for arm in ARMS:
        n = params_of(*SIZES[arm])
        assert f"{n}" in md or f"{n:,}" in md, (arm, n)
    assert "sps" in md


# ---------------------------------------------------------------- specs without the new fields

OLD_RUN_KEYS = {"arm", "seed", "steps", "suite", "suite_mean", "elo_vs_init", "elo_vs_init_mean", "obs_hash", "rules_hash"}
OLD_REPORT_KEYS = {"name", "arms", "seeds", "runs", "tables", "comparisons", "n_flagged", "elo", "obs_hash"}


@pytest.mark.parametrize("init", [True, False])
def test_milestone_12_report_keys_are_unchanged(tmp_path, init):
    cwd = os.getcwd()
    try:
        spec = m12.build(tmp_path, init=init)
        rep = experiment.report(spec)
    finally:
        os.chdir(cwd)
    assert (spec.reference, spec.init_from) == (None, None)
    assert list(rep.tables) == CONDS + (["elo_vs_init"] if init else [])
    for r in rep.runs:
        assert set(r) == OLD_RUN_KEYS
    assert {c["metric"] for c in rep.comparisons} == set(CONDS) | ({"elo_vs_init"} if init else set())
    assert rep.elo is not None
    got = set(json.loads((tmp_path / "runs" / "exp" / "r" / "report.json").read_text()))
    assert got == OLD_REPORT_KEYS | {"cost"}
    md = (tmp_path / "runs" / "exp" / "r" / "report.md").read_text()
    assert "elo_vs_ref" not in md
