"""Milestone 11 tests 9-10: self-play training end to end (tiny, CPU), warm start, resume, the
checkpoint CLIs, and the shipped configs and scripts.

Tiny config: batch 4 x 32 = 128 steps, 4 updates (512 steps); snapshots, suite evaluations and
checkpoints every 256 steps, a rating every 512, one seed wherever seeds are counted. Expected LR per
update is the milestone 10 linear annealing from `cfg.lr` (fresh schedule on warm start); a fresh
Adam has taken `updates * update_epochs * num_minibatches` steps at the end. The warm-start
checkpoint's optimizer has 3 steps already, so reusing it would show up in that count.
"""

import json
import math
import os
import pathlib
import re

import numpy as np
import polytopia_sim as ps
import pytest
import torch
from sp_helpers import REPO

from polytopia_rl import losses, suite, tournament, train
from polytopia_rl.checkpoint import SpecMismatch, load_checkpoint, save_checkpoint
from polytopia_rl.config import PPOConfig, load_config
from polytopia_rl.model import PolicyNet

BATCH = 128
UPDATES = 4
INIT_STEP = 5_000_000
SUBSET = {"economy_e0", "rush_e0", "rush_e20"}
DOC_KEYS = [
    "global_step",
    "update",
    "sps",
    "lr",
    "pg_loss",
    "v_loss",
    "entropy",
    "approx_kl",
    "clipfrac",
    "explained_variance",
    "mean_return",
    "train_win_rate",
    "mean_episode_len",
    "kind_freq",
    "end_turn_freq",
    "pool_size",
]


def sp_cfg(**kw):
    base = dict(
        size=6,
        turn_cap=3,
        num_envs=4,
        num_steps=32,
        num_minibatches=4,
        update_epochs=2,
        width=8,
        blocks=1,
        total_steps=UPDATES * BATCH,
        eval_every=2 * BATCH,
        eval_seeds=1,
        device="cpu",
        seed=3,
        opponents=["economy", "rush"],
        stages=[{"opponents": ["economy", "rush"]}],
        selfplay=True,
        snapshot_every=2 * BATCH,
        rating_every=4 * BATCH,
        rating_seeds=1,
        tournament_seeds=1,
        p_bot=0.5,
    )
    return PPOConfig(**{**base, **kw})


def spec():
    return dict(ps.Env(size=6).spec())


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def same_state(a, b):
    assert a.keys() == b.keys()
    return all(torch.equal(a[k], b[k]) for k in a)


def jsonl(path):
    return [json.loads(x) for x in pathlib.Path(path).read_text().splitlines()]


def untimed(row):
    return {k: v for k, v in row.items() if "sps" not in k and "time" not in k}


def make_init(path, obs_hash=None):
    """An M10-style warm-start checkpoint: tiny net (seed 11), Adam with 3 steps taken."""
    torch.manual_seed(11)
    net = PolicyNet(spec(), width=8, blocks=1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(3):
        opt.zero_grad()
        sum(p.sum() for p in net.parameters()).backward()
        opt.step()
    sp = spec()
    if obs_hash is not None:
        sp["obs_hash"] = obs_hash
    save_checkpoint(str(path), model=net, optimizer=opt, cfg=sp_cfg(selfplay=False), global_step=INIT_STEP, stage=2, seed_index=None, rng_state=None, spec=sp)
    return net


@pytest.fixture(scope="module")
def run1(tmp_path_factory):
    d = tmp_path_factory.mktemp("sp")
    init = make_init(d / "init.pt")
    cfg_path = d / "tiny.json"
    cfg_path.write_text(sp_cfg().to_json())
    run_dir = d / "run"
    train.main(["--config", str(cfg_path), "--run-dir", str(run_dir), "--init-checkpoint", str(d / "init.pt")])
    return run_dir, init, d / "init.pt"


def finite(x):
    if isinstance(x, dict):
        return all(finite(v) for v in x.values())
    if isinstance(x, bool) or x is None:
        return True
    return math.isfinite(x)


def test_metrics_documented_keys_and_fresh_lr_schedule(run1):
    d, _, _ = run1
    rows = jsonl(d / "metrics.jsonl")
    assert [r["update"] for r in rows] == [1, 2, 3, 4]
    assert [r["global_step"] for r in rows] == [BATCH * u for u in (1, 2, 3, 4)]
    assert rows[0]["lr"] == pytest.approx(3e-4, rel=1e-9)
    sizes = []
    for r in rows:
        for k in DOC_KEYS:
            assert k in r, k
        assert finite(r), r
        assert r["lr"] == pytest.approx(3e-4 * (1 - (r["update"] - 1) / UPDATES), rel=1e-6)
        sizes.append(r["pool_size"])
    assert sizes == sorted(sizes) and 1 <= sizes[0] and sizes[-1] <= 3


def test_fresh_optimizer_and_final_checkpoint(run1):
    d, _, _ = run1
    ck = load(d / "ckpt_final.pt")
    assert ck["global_step"] == UPDATES * BATCH
    assert ck["spec"]["obs_hash"] == spec()["obs_hash"] and ck["spec"]["rules_hash"] == spec()["rules_hash"]
    steps = {int(torch.as_tensor(s["step"]).item()) for s in ck["optimizer"]["state"].values()}
    assert steps == {UPDATES * 2 * 4}, "a fresh Adam, not the warm-start one (which had 3 steps)"
    for name in ["ckpt_256.pt", "ckpt_512.pt", "latest.pt"]:
        assert (d / name).exists(), name


def test_pool_snapshots_on_schedule_and_snapshot_0_is_the_warm_start(run1):
    d, init, _ = run1
    names = sorted(p.name for p in (d / "pool").iterdir())
    assert names == ["snap_0.pt", "snap_256.pt", "snap_512.pt"]
    snaps = {n: load_checkpoint(str(d / "pool" / n), spec=spec()) for n in names}
    assert [snaps[n].global_step for n in names] == [0, 256, 512]
    assert same_state(snaps["snap_0.pt"].model, init.state_dict())
    assert not same_state(snaps["snap_256.pt"].model, init.state_dict()), "the learner trained"
    assert same_state(snaps["snap_512.pt"].model, load(d / "ckpt_final.pt")["model"])


def test_eval_log_suite_subset_regression_flag_and_ratings(run1):
    d, _, _ = run1
    rows = jsonl(d / "eval.jsonl")
    suites = [r for r in rows if r["type"] == "suite"]
    ratings = [r for r in rows if r["type"] == "rating"]
    assert [r["global_step"] for r in suites] == [256, 512]
    assert [r["global_step"] for r in ratings] == [512]
    floor = sp_cfg().regression_floor
    for r in suites:
        assert set(r["suite"]) == SUBSET
        rates = {k: v["win_rate"] for k, v in r["suite"].items()}
        assert all(0.0 <= x <= 1.0 for x in rates.values())
        assert r["regression"] == any(x < floor for x in rates.values())
    for r in ratings:
        assert set(r["rating"]) >= {"learner", "snapshot_0"}
        assert finite(r["rating"])


def test_final_tournament_and_train_log(run1):
    d, _, _ = run1
    t = json.loads((d / "tournament.json").read_text())
    assert set(t["players"]) == {"random", "economy", "rush", "snap_0", "snap_256", "snap_512"}
    wins, draws, games = (np.array(t[k], dtype=float) for k in ("wins", "draws", "games"))
    n = len(t["players"])
    assert wins.shape == draws.shape == games.shape == (n, n)
    np.testing.assert_array_equal(wins + wins.T + draws, games)
    assert np.all(games[~np.eye(n, dtype=bool)] == 2)  # every pair, one seed, seats swapped
    assert t["ratings"][t["players"].index("random")] == pytest.approx(0.0, abs=1e-9)
    assert set(t["seat_win_rate"]) == {"0", "1"}
    md = (d / "tournament.md").read_text()
    for name in t["players"]:
        assert name in md
    assert (d / "train.log").read_text().strip()


def test_tournament_cli_rebuilds_the_round_robin(run1, tmp_path):
    d, _, _ = run1
    import shutil

    run = tmp_path / "run"
    shutil.copytree(d / "pool", run / "pool")
    tournament.main(["--run", str(run), "--seeds", "1"])
    t = json.loads((run / "tournament.json").read_text())
    assert set(t["players"]) == {"random", "economy", "rush", "snap_0", "snap_256", "snap_512"}
    assert (run / "tournament.md").exists()


def test_resume_continues_pool_rng_and_seeds_exactly(run1, tmp_path):
    d, _, _ = run1
    train.run(sp_cfg(), str(tmp_path), resume=str(d / "ckpt_256.pt"))
    rows = jsonl(tmp_path / "metrics.jsonl")
    assert [r["update"] for r in rows] == [3, 4]
    assert [untimed(r) for r in rows] == [untimed(r) for r in jsonl(d / "metrics.jsonl")[2:]]
    ev = [untimed(r) for r in jsonl(tmp_path / "eval.jsonl")]
    assert ev == [untimed(r) for r in jsonl(d / "eval.jsonl") if r["global_step"] > 256]
    assert same_state(load(tmp_path / "ckpt_final.pt")["model"], load(d / "ckpt_final.pt")["model"])
    for name in ["snap_0.pt", "snap_256.pt", "snap_512.pt"]:
        assert same_state(load(tmp_path / "pool" / name)["model"], load(d / "pool" / name)["model"]), name


def test_warm_start_with_a_wrong_obs_hash_fails(tmp_path):
    make_init(tmp_path / "bad.pt", obs_hash="0123456789abcdef")
    with pytest.raises(SpecMismatch):
        train.run(sp_cfg(), str(tmp_path / "run"), init_checkpoint=str(tmp_path / "bad.pt"))


# ---------------------------------------------------------------- checkpoint CLIs


def first_line_has_hashes_and_step(out, step):
    first = out.splitlines()[0]
    sp = spec()
    assert sp["obs_hash"] in first and sp["rules_hash"] in first, first
    assert re.search(rf"\b{step}\b", first), first


def test_suite_cli_reports_hashes_writes_and_compares(run1, tmp_path, capsys):
    d, _, _ = run1
    ckpt = str(d / "ckpt_final.pt")
    suite.main(["--checkpoint", ckpt, "--seeds", "1", "--out", str(tmp_path / "a")])
    first_line_has_hashes_and_step(capsys.readouterr().out, 512)
    res = json.loads((tmp_path / "a" / "suite.json").read_text())
    assert res["obs_hash"] == spec()["obs_hash"] and res["step"] == 512
    md = (tmp_path / "a" / "suite.md").read_text()
    for c in suite.DEFAULT_SUITE:
        assert c.name in md
    suite.main(["--checkpoint", ckpt, "--seeds", "1", "--out", str(tmp_path / "b")])
    capsys.readouterr()
    suite.main(["--compare", str(tmp_path / "a" / "suite.json"), str(tmp_path / "b" / "suite.json")])
    out = capsys.readouterr().out
    for c in suite.DEFAULT_SUITE:
        assert c.name in out


def test_losses_cli_reports_hashes_and_writes_replays(run1, tmp_path, capsys):
    d, _, _ = run1
    losses.main(["--checkpoint", str(d / "ckpt_final.pt"), "--opponent", "rush", "--count", "2", "--out", str(tmp_path)])
    first_line_has_hashes_and_step(capsys.readouterr().out, 512)
    files = list(tmp_path.iterdir())
    assert len(files) <= 2
    for f in files:
        assert re.fullmatch(r"rush-\d+-seat[01]\.json", f.name)


# ---------------------------------------------------------------- shipped configs and scripts


def test_selfplay_config_and_defaults_match_the_brief():
    d = PPOConfig.from_json("{}")
    assert d.selfplay is False
    assert (d.snapshot_every, d.pool_max, d.p_bot, d.p_latest) == (1_000_000, 40, 0.2, 0.5)
    assert (d.opponent_greedy, d.rating_every, d.regression_floor) == (False, 2_000_000, 0.9)
    cfg = load_config(str(REPO / "python" / "configs" / "selfplay.json"))
    assert cfg.selfplay is True
    assert cfg.total_steps == 20_000_000
    assert (cfg.snapshot_every, cfg.pool_max) == (1_000_000, 40)
    assert (cfg.p_bot, cfg.p_latest, cfg.opponent_greedy) == (0.4, 0.5, False)
    assert (cfg.eval_every, cfg.eval_seeds, cfg.regression_floor) == (1_000_000, 50, 0.9)
    assert (cfg.rating_every, cfg.rating_seeds) == (2_000_000, 50)


def test_suite_config_and_scripts_exist():
    s = json.loads((REPO / "python" / "configs" / "suite.json").read_text())
    assert s["seeds"] == 200
    for name in ["suite.sh", "selfplay.sh", "tournament.sh"]:
        p = REPO / "scripts" / name
        assert p.exists() and os.access(p, os.X_OK), name
