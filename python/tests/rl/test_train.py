"""Milestone 10 tests 10-11: train driver, checkpoints and resume, curriculum env rebuild, plots.

Tiny CPU config: batch 4 x 32 = 128 steps, 4 updates, an eval (and checkpoint) every 256 steps.
Expected `lr` per update is CleanRL's linear annealing `lr * (1 - (update - 1) / num_updates)` with
`num_updates = total_steps // (num_envs * num_steps)`; the Adam step count is
`updates * update_epochs * num_minibatches` when nothing stops early.
"""

import json
import math
import pathlib

import numpy as np
import polytopia_sim as ps
import pytest
import torch

from polytopia_rl import train
from polytopia_rl.config import PPOConfig
from polytopia_rl.plot import plot_run

BATCH = 128
UPDATES = 4

DOC_KEYS = [
    "global_step",
    "update",
    "sps",
    "stage",
    "lr",
    "pg_loss",
    "v_loss",
    "entropy",
    "entropy_actor",
    "entropy_kind",
    "entropy_arg",
    "approx_kl",
    "clipfrac",
    "explained_variance",
    "mean_return",
    "train_win_rate",
    "mean_episode_len",
    "kind_freq",
    "end_turn_freq",
    "adv_mean",
    "adv_std",
]
CKPT_KEYS = ["model", "optimizer", "config", "global_step", "stage", "vec_env", "rng", "spec"]


def tiny_cfg(**kw):
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
        opponents=["random"],
        stages=[{"opponents": ["random"]}],
    )
    return PPOConfig(**{**base, **kw})


def metrics(run_dir):
    return [json.loads(line) for line in (pathlib.Path(run_dir) / "metrics.jsonl").read_text().splitlines()]


def untimed(row):
    return {k: v for k, v in row.items() if "sps" not in k and "time" not in k}


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def same_state(a, b):
    assert a.keys() == b.keys()
    return all(torch.equal(a[k], b[k]) for k in a)


@pytest.fixture(scope="module")
def run1(tmp_path_factory):
    d = tmp_path_factory.mktemp("run1")
    net = train.run(tiny_cfg(), str(d))
    return d, net


def finite_numbers(x):
    if isinstance(x, dict):
        return all(finite_numbers(v) for v in x.values())
    if isinstance(x, bool) or x is None:
        return True
    return math.isfinite(x)


def test_metrics_have_documented_keys_and_finite_values(run1):
    d, _ = run1
    rows = metrics(d)
    assert [r["update"] for r in rows] == [1, 2, 3, 4]
    assert [r["global_step"] for r in rows] == [BATCH * u for u in (1, 2, 3, 4)]
    for r in rows:
        for k in DOC_KEYS:
            assert k in r, k
        assert finite_numbers(r), r
        assert r["stage"] == 0
        assert r["lr"] == pytest.approx(3e-4 * (1 - (r["update"] - 1) / UPDATES), rel=1e-6)
        assert set(r["kind_freq"]) == set(ps.Env(size=6).spec()["action_types"])
        assert sum(r["kind_freq"].values()) == pytest.approx(1.0, abs=1e-6)
        assert r["end_turn_freq"] == pytest.approx(r["kind_freq"]["end_turn"], abs=1e-6)
        assert r["entropy"] == pytest.approx(r["entropy_actor"] + r["entropy_kind"] + r["entropy_arg"], abs=1e-4)
    assert rows[-1]["mean_return"] is not None and rows[-1]["train_win_rate"] is not None
    assert 0.0 <= rows[-1]["train_win_rate"] <= 1.0


def test_checkpoints_and_eval_log_written(run1):
    d, _ = run1
    for name in ["ckpt_256.pt", "ckpt_512.pt", "latest.pt", "best.pt"]:
        assert (d / name).exists(), name
    ck = load(d / "ckpt_256.pt")
    for k in CKPT_KEYS:
        assert k in ck, k
    assert ck["global_step"] == 256 and ck["stage"] == 0
    assert ck["optimizer"]["state"], "optimizer state (Adam moments) is saved"
    assert PPOConfig.from_json(ck["config"]) == tiny_cfg()
    evals = [json.loads(x) for x in (d / "eval.jsonl").read_text().splitlines()]
    assert len(evals) == 2


def test_save_load_gives_the_same_outputs(run1):
    d, net = run1
    net2, cfg2 = train.load_policy(str(d / "latest.pt"))
    assert cfg2 == tiny_cfg()
    env = ps.Env(size=6, turn_cap=3, opponent="random")
    env.reset(12345)
    board, glob = (torch.from_numpy(x)[None] for x in env.observation())
    net.eval()
    net2.eval()
    with torch.no_grad():
        f1, f2 = net.trunk(board, glob), net2.trunk(board, glob)
        assert torch.equal(net.actor_logits(f1), net2.actor_logits(f2))
        assert torch.equal(net.value(f1), net2.value(f2))
    assert same_state(net.state_dict(), net2.state_dict())


def test_cpu_run_is_bit_for_bit_repeatable(run1, tmp_path):
    d, _ = run1
    train.run(tiny_cfg(), str(tmp_path))
    assert [untimed(r) for r in metrics(tmp_path)] == [untimed(r) for r in metrics(d)]
    assert same_state(load(tmp_path / "latest.pt")["model"], load(d / "latest.pt")["model"])


def test_resume_matches_an_uninterrupted_run(run1, tmp_path):
    d, _ = run1
    train.run(tiny_cfg(), str(tmp_path), resume=str(d / "ckpt_256.pt"))
    rows = metrics(tmp_path)
    assert [r["update"] for r in rows] == [3, 4]
    assert [untimed(r) for r in rows] == [untimed(r) for r in metrics(d)[2:]]
    a, b = load(tmp_path / "latest.pt"), load(d / "latest.pt")
    assert same_state(a["model"], b["model"])
    assert a["global_step"] == b["global_step"] == UPDATES * BATCH
    assert a["vec_env"] == b["vec_env"]  # the same next training seeds and in-progress episodes


def test_curriculum_advance_keeps_policy_and_optimizer(tmp_path):
    # Advance as soon as one episode has finished after >= one update in stage 0.
    cfg = tiny_cfg(
        stages=[{"opponents": ["random"]}, {"opponents": ["economy"]}],
        min_stage_steps=BATCH,
        stage_window=1,
        stage_win_rate=0.0,
        eval_every=10**9,
    )
    train.run(cfg, str(tmp_path))
    rows = metrics(tmp_path)
    stages = [r["stage"] for r in rows]
    assert stages[0] == 0 and stages[-1] == 1 and stages == sorted(stages)
    for r in rows:  # the LR schedule continues across the advance
        assert r["lr"] == pytest.approx(3e-4 * (1 - (r["update"] - 1) / UPDATES), rel=1e-6)
    ck = load(tmp_path / "latest.pt")
    assert ck["stage"] == 1
    steps = {int(torch.as_tensor(s["step"]).item()) for s in ck["optimizer"]["state"].values()}
    assert steps == {UPDATES * cfg.update_epochs * cfg.num_minibatches}


def test_cli_and_plot(tmp_path):
    cfg_path = tmp_path / "tiny.json"
    cfg_path.write_text(tiny_cfg(total_steps=BATCH * 8).to_json())
    run_dir = tmp_path / "run"
    train.main(["--config", str(cfg_path), "--run-dir", str(run_dir), "total_steps=256", "seed=4"])
    rows = metrics(run_dir)
    assert [r["update"] for r in rows] == [1, 2]
    assert load(run_dir / "latest.pt")["global_step"] == 256
    assert PPOConfig.from_json(load(run_dir / "latest.pt")["config"]).seed == 4
    paths = plot_run(str(run_dir))
    assert paths
    for p in paths:
        p = pathlib.Path(p)
        assert p.exists() and p.stat().st_size > 0 and p.suffix == ".png"
        assert run_dir in p.parents
    assert np.isfinite(rows[-1]["pg_loss"])
