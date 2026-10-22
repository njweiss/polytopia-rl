"""Milestone 10 tests 10 (config part): defaults table, JSON round trip, `key=value` overrides,
and the shipped configs ("Default hyperparameters", "Curriculum")."""

import json
import pathlib

import pytest

from polytopia_rl.config import PPOConfig, load_config

CONFIGS = pathlib.Path(__file__).resolve().parents[2] / "configs"
THREE_STAGES = [{"opponents": ["random"]}, {"opponents": ["economy"]}, {"opponents": ["economy", "rush"]}]

DEFAULTS = dict(
    num_envs=32,
    num_steps=128,
    num_minibatches=8,
    update_epochs=4,
    lr=3e-4,
    max_grad_norm=0.5,
    width=64,
    blocks=4,
    shaping_weight=1.0,
    terminal_weight=1.0,
    seed=1,
    gamma=0.999,
    gae_lambda=0.95,
    clip_eps=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    target_kl=None,
    norm_adv=True,
    potential_scale=1000,
    total_steps=20_000_000,
    device="auto",
    size=11,
    turn_cap=30,
    full_observability=False,
    stage_window=200,
    stage_win_rate=0.8,
    eval_every=500_000,
    eval_seeds=50,
    stages=THREE_STAGES,
)


def test_defaults_match_the_brief():
    cfg = PPOConfig.from_json("{}")
    for k, v in DEFAULTS.items():
        assert getattr(cfg, k) == v, k


def test_json_round_trip():
    cfg = PPOConfig.from_json(
        json.dumps({"num_envs": 4, "target_kl": 0.02, "opponents": ["economy", "rush"], "stages": THREE_STAGES[:2]})
    )
    assert cfg.num_envs == 4 and cfg.target_kl == 0.02 and cfg.opponents == ["economy", "rush"]
    again = PPOConfig.from_json(cfg.to_json())
    assert again == cfg
    assert json.loads(cfg.to_json())["num_envs"] == 4


def test_unknown_json_key_raises():
    with pytest.raises(ValueError):
        PPOConfig.from_json('{"num_env": 4}')


def test_load_config_with_overrides(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"num_envs": 4, "lr": 0.01}))
    cfg = load_config(
        str(path),
        ["num_envs=8", "lr=0.001", "device=cuda", "target_kl=0.02", 'opponents=["economy","rush"]', "norm_adv=false", "total_steps=1000"],
    )
    assert cfg.num_envs == 8 and isinstance(cfg.num_envs, int)
    assert cfg.lr == 0.001
    assert cfg.device == "cuda"
    assert cfg.target_kl == 0.02
    assert cfg.opponents == ["economy", "rush"]
    assert cfg.norm_adv is False
    assert cfg.total_steps == 1000
    assert cfg.num_steps == 128  # untouched default
    assert load_config(str(path), ["target_kl=null"]).target_kl is None
    assert load_config(str(path)).num_envs == 4


@pytest.mark.parametrize("bad", ["num_env=8", "num_envs", "=3"])
def test_bad_override_raises(tmp_path, bad):
    path = tmp_path / "c.json"
    path.write_text("{}")
    with pytest.raises(ValueError):
        load_config(str(path), [bad])


def test_shipped_configs():
    stage1 = load_config(str(CONFIGS / "stage1_random.json"))
    assert stage1.stages == [{"opponents": ["random"]}]
    cur = load_config(str(CONFIGS / "curriculum.json"))
    assert cur.stages == THREE_STAGES
    assert cur.stage_win_rate == 0.8 and cur.stage_window == 200
