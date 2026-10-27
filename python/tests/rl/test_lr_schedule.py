"""Milestone 12c tests 1-6: the LR schedule keys `anneal_lr` and `lr_anneal_steps`, the load-time check,
resume, and the shipped `selfplay12c.json` ("Public API", "Rules", "Experiment (12c)").

Tiny CPU config (as test_train.py): batch 4 x 32 = 128 steps, total 512 steps, so 4 updates; an eval
and checkpoint every 256 steps. Expected LRs are hand computations of the brief's rule: update `u`
(1-based) uses `lr * (1 - (u - 1) / H)` with `H = (lr_anneal_steps or total_steps) // 128`, or `lr`
for every update when `anneal_lr` is false. With lr 3e-4:
  - default, H = 512 // 128 = 4: 3e-4 * [1, 3/4, 2/4, 1/4]
  - lr_anneal_steps 1024, H = 8: 3e-4 * [1, 7/8, 6/8, 5/8]; the last (1.875e-4) is just above lr / 2
  - lr_anneal_steps 1000, H = 1000 // 128 = 7 (floor): 3e-4 * [1, 6/7, 5/7, 4/7]
The LR actually used is read from the optimizer each time `train_update` is called, and from the `lr`
column of `metrics.jsonl`.
"""

import json
import pathlib

import pytest
import torch
from sp_helpers import REPO

from polytopia_rl import train
from polytopia_rl.config import PPOConfig, load_config
from polytopia_rl.experiment import load_spec

LR = 3e-4
BATCH = 128
TOTAL = 4 * BATCH
CONST = [LR] * 4
DEFAULT_SCHEDULE = [LR * f for f in (1, 3 / 4, 2 / 4, 1 / 4)]
HALVING = [LR * f for f in (1, 7 / 8, 6 / 8, 5 / 8)]
FLOOR7 = [LR * f for f in (1, 6 / 7, 5 / 7, 4 / 7)]


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
        lr=LR,
        total_steps=TOTAL,
        eval_every=2 * BATCH,
        eval_seeds=1,
        device="cpu",
        seed=3,
        opponents=["random"],
        stages=[{"opponents": ["random"]}],
    )
    return PPOConfig(**{**base, **kw})


def metrics_lrs(run_dir):
    rows = [json.loads(line) for line in (pathlib.Path(run_dir) / "metrics.jsonl").read_text().splitlines()]
    return [r["update"] for r in rows], [r["lr"] for r in rows]


def run_recording(monkeypatch, cfg, run_dir, resume=None):
    """Run the trainer; return the LR of every optimizer param group at each `train_update` call."""
    seen = []
    real = train.train_update

    def spy(net, optimizer, *args, **kwargs):
        seen.append([g["lr"] for g in optimizer.param_groups])
        return real(net, optimizer, *args, **kwargs)

    monkeypatch.setattr(train, "train_update", spy)
    train.run(cfg, str(run_dir), resume=resume)
    monkeypatch.setattr(train, "train_update", real)
    return seen


def assert_schedule(seen, run_dir, expected, updates=(1, 2, 3, 4)):
    assert len(seen) == len(expected)
    for groups, want in zip(seen, expected):
        assert groups and all(g == pytest.approx(want, rel=1e-9) for g in groups), (groups, want)
    got_updates, lrs = metrics_lrs(run_dir)
    assert got_updates == list(updates)
    assert lrs == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------- 1. defaults


def test_new_keys_default_to_todays_schedule():
    for cfg in (PPOConfig(), PPOConfig.from_json("{}")):
        assert cfg.anneal_lr is True
        assert cfg.lr_anneal_steps is None


def test_new_keys_round_trip_through_json():
    cfg = PPOConfig.from_json(json.dumps({"total_steps": 1000, "anneal_lr": False, "lr_anneal_steps": 2000}))
    assert cfg.anneal_lr is False and cfg.lr_anneal_steps == 2000
    assert PPOConfig.from_json(cfg.to_json()) == cfg


@pytest.mark.parametrize("lr_anneal_steps", [None, TOTAL])
def test_default_schedule_is_todays_formula(monkeypatch, tmp_path, lr_anneal_steps):
    # lr_anneal_steps == total_steps is not "below" total_steps, so it loads, and gives H = 4 as well.
    seen = run_recording(monkeypatch, tiny_cfg(lr_anneal_steps=lr_anneal_steps), tmp_path)
    assert_schedule(seen, tmp_path, DEFAULT_SCHEDULE)


# ---------------------------------------------------------------- 2. a longer horizon


def test_horizon_twice_the_run_halves_the_lr_linearly(monkeypatch, tmp_path):
    seen = run_recording(monkeypatch, tiny_cfg(lr_anneal_steps=2 * TOTAL), tmp_path)
    assert_schedule(seen, tmp_path, HALVING)
    last = seen[-1][0]
    assert LR / 2 < last < LR * 3 / 4  # just above lr / 2


def test_horizon_is_floored_to_whole_updates(monkeypatch, tmp_path):
    seen = run_recording(monkeypatch, tiny_cfg(lr_anneal_steps=1000), tmp_path)
    assert_schedule(seen, tmp_path, FLOOR7)


# ---------------------------------------------------------------- 3. no annealing


@pytest.mark.parametrize("lr_anneal_steps", [None, 2 * TOTAL])
def test_anneal_lr_false_keeps_lr_constant(monkeypatch, tmp_path, lr_anneal_steps):
    seen = run_recording(monkeypatch, tiny_cfg(anneal_lr=False, lr_anneal_steps=lr_anneal_steps), tmp_path)
    assert_schedule(seen, tmp_path, CONST)
    ck = torch.load(tmp_path / "latest.pt", map_location="cpu", weights_only=False)
    assert all(g["lr"] == LR for g in ck["optimizer"]["param_groups"])


# ---------------------------------------------------------------- 4. the load-time check


def test_horizon_below_total_steps_raises_from_from_json():
    with pytest.raises(ValueError):
        PPOConfig.from_json(json.dumps({"total_steps": 1000, "lr_anneal_steps": 999}))
    with pytest.raises(ValueError):
        PPOConfig.from_json(json.dumps({"lr_anneal_steps": 1_000_000}))  # default total_steps 20M
    # Equal, above, or unset is fine.
    assert PPOConfig.from_json(json.dumps({"total_steps": 1000, "lr_anneal_steps": 1000})).lr_anneal_steps == 1000
    assert PPOConfig.from_json(json.dumps({"total_steps": 1000, "lr_anneal_steps": 3000})).lr_anneal_steps == 3000
    assert PPOConfig.from_json(json.dumps({"total_steps": 1000, "lr_anneal_steps": None})).lr_anneal_steps is None


def test_horizon_below_total_steps_raises_from_load_config(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"total_steps": 1000, "lr_anneal_steps": 999}))
    with pytest.raises(ValueError):
        load_config(str(path))
    # The check sees the config after the overrides.
    path.write_text(json.dumps({"total_steps": 1000, "lr_anneal_steps": 1000}))
    assert load_config(str(path)).lr_anneal_steps == 1000
    with pytest.raises(ValueError):
        load_config(str(path), ["total_steps=2000"])
    with pytest.raises(ValueError):
        load_config(str(path), ["lr_anneal_steps=500"])
    cfg = load_config(str(path), ["total_steps=500"])
    assert cfg.total_steps == 500 and cfg.lr_anneal_steps == 1000
    assert load_config(str(path), ["total_steps=2000", "lr_anneal_steps=null"]).lr_anneal_steps is None


# ---------------------------------------------------------------- 5. resume


@pytest.mark.parametrize(
    "kw, expected",
    [(dict(lr_anneal_steps=2 * TOTAL), HALVING), (dict(lr_anneal_steps=1000), FLOOR7), (dict(anneal_lr=False), CONST)],
)
def test_resume_keeps_the_schedule(monkeypatch, tmp_path, kw, expected):
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    cfg = tiny_cfg(**kw)
    seen_full = run_recording(monkeypatch, cfg, full)
    assert_schedule(seen_full, full, expected)
    seen_resumed = run_recording(monkeypatch, cfg, resumed, resume=str(full / "ckpt_256.pt"))
    assert_schedule(seen_resumed, resumed, expected[2:], updates=(3, 4))
    assert seen_resumed == seen_full[2:]
    assert metrics_lrs(resumed)[1] == metrics_lrs(full)[1][2:]


# ---------------------------------------------------------------- 6. the shipped experiment


SELFPLAY12C = REPO / "python" / "experiments" / "selfplay12c.json"
ARMS = {"decay10": {}, "decay20": {"lr_anneal_steps": 20_000_000}, "const": {"anneal_lr": False}}


def test_selfplay12c_spec_is_the_owners_experiment():
    assert SELFPLAY12C.exists()
    s = load_spec(str(SELFPLAY12C))
    assert s.name == "selfplay12c"
    assert s.base_config == "python/configs/selfplay.json"
    assert s.init_checkpoint == "runs/curriculum_seed1/latest.pt"
    assert list(s.seeds) == [1, 2, 3]
    assert s.total_steps == 10_000_000
    assert s.arms == ARMS
    assert list(s.arms) == ["decay10", "decay20", "const"]
    assert sorted(s.primary) == ["elo_vs_init", "rush_e0"]
    assert s.suite_seeds == 200
    assert s.h2h_seeds == 100
    assert s.tournament_seeds == 100
    assert s.eval_checkpoints == 3


def test_selfplay12c_arms_load_with_the_intended_schedule():
    # Each arm's trainer config as the launcher builds it: base config, then seed and total_steps, then
    # the arm's overrides (experiment.train_command).
    assert SELFPLAY12C.exists()
    s = load_spec(str(SELFPLAY12C))
    got = {}
    for arm, overrides in s.arms.items():
        argv = [f"seed={seed}" for seed in s.seeds[:1]] + [f"total_steps={s.total_steps}"]
        argv += [f"{k}={json.dumps(v)}" for k, v in overrides.items()]
        cfg = load_config(str(REPO / s.base_config), argv)
        got[arm] = (cfg.anneal_lr, cfg.lr_anneal_steps, cfg.total_steps, cfg.lr, cfg.p_bot)
    assert got == {
        "decay10": (True, None, 10_000_000, 1e-4, 0.4),
        "decay20": (True, 20_000_000, 10_000_000, 1e-4, 0.4),
        "const": (False, None, 10_000_000, 1e-4, 0.4),
    }
