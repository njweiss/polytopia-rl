"""Milestone 14 test 1: the two optional spec fields and the shipped scaling specs ("Public API",
"Rules", "Experiment").

Expected values are the brief's: `reference` and `init_from` default to None; `init_from` together
with `init_checkpoint` raises `ValueError` on load; the arms `w32b4`, `w64b4`, `w128b4`, `w64b8`
(width x blocks 32x4, 64x4, 128x4, 64x8) in both phases, seeds 1 to 3, `total_steps` 10M, reference
`runs/curriculum_seed1/latest.pt`, primary `elo_vs_ref` and `rush_e0`; phase 1 from
`python/configs/curriculum.json` with no init, phase 2 from `python/configs/selfplay.json` with
`init_from` `scale14_bots`; evaluation over the last 3 checkpoints ("Decisions already made").
"""

import json

import pytest
from sp_helpers import REPO

from polytopia_rl.config import load_config
from polytopia_rl.experiment import Spec, load_spec

BASE = {
    "name": "scale",
    "base_config": "python/configs/selfplay.json",
    "seeds": [1, 2],
    "arms": {"w32b4": {"width": 32, "blocks": 4}, "w64b4": {"width": 64, "blocks": 4}},
    "primary": ["elo_vs_ref", "rush_e0"],
    "suite_seeds": 200,
    "tournament_seeds": 100,
}

ARMS = {
    "w32b4": {"width": 32, "blocks": 4},
    "w64b4": {"width": 64, "blocks": 4},
    "w128b4": {"width": 128, "blocks": 4},
    "w64b8": {"width": 64, "blocks": 8},
}
REFERENCE = "runs/curriculum_seed1/latest.pt"


def write(tmp_path, d, name="spec.json"):
    p = tmp_path / name
    p.write_text(json.dumps(d))
    return str(p)


def test_new_fields_default_to_none(tmp_path):
    s = load_spec(write(tmp_path, BASE))
    assert isinstance(s, Spec)
    assert s.reference is None
    assert s.init_from is None
    assert s.init_checkpoint is None


def test_new_fields_load(tmp_path):
    s = load_spec(write(tmp_path, {**BASE, "reference": "ref.pt", "init_from": "scale14_bots"}))
    assert s.reference == "ref.pt"
    assert s.init_from == "scale14_bots"
    assert s.init_checkpoint is None
    assert list(s.primary) == ["elo_vs_ref", "rush_e0"], "primary may name elo_vs_ref"


def test_reference_with_init_checkpoint_loads(tmp_path):
    # Only init_from excludes init_checkpoint; a reference may sit next to an init checkpoint.
    s = load_spec(write(tmp_path, {**BASE, "reference": "ref.pt", "init_checkpoint": "init.pt"}))
    assert (s.reference, s.init_checkpoint) == ("ref.pt", "init.pt")


def test_explicit_nulls_load(tmp_path):
    s = load_spec(write(tmp_path, {**BASE, "reference": None, "init_from": None, "init_checkpoint": "init.pt"}))
    assert (s.reference, s.init_from, s.init_checkpoint) == (None, None, "init.pt")


def test_init_from_with_init_checkpoint_raises(tmp_path):
    with pytest.raises(ValueError):
        load_spec(write(tmp_path, {**BASE, "init_from": "scale14_bots", "init_checkpoint": "runs/curriculum_seed1/latest.pt"}))


def test_init_from_with_init_checkpoint_raises_with_a_reference_too(tmp_path):
    with pytest.raises(ValueError):
        load_spec(write(tmp_path, {**BASE, "init_from": "scale14_bots", "init_checkpoint": "init.pt", "reference": "ref.pt"}))


# ---------------------------------------------------------------- shipped files


def shipped(name):
    return load_spec(str(REPO / "python" / "experiments" / f"{name}.json"))


@pytest.mark.parametrize("name", ["scale14_bots", "scale14_sp"])
def test_scale14_specs_common_settings(name):
    s = shipped(name)
    assert s.name == name
    assert s.arms == ARMS
    assert list(s.arms) == list(ARMS), "arms in the brief's order"
    assert list(s.seeds) == [1, 2, 3]
    assert s.total_steps == 10_000_000
    assert s.reference == REFERENCE
    assert sorted(s.primary) == ["elo_vs_ref", "rush_e0"]
    assert s.init_checkpoint is None
    assert s.eval_checkpoints == 3


def test_scale14_bots_is_phase_1():
    s = shipped("scale14_bots")
    assert s.base_config == "python/configs/curriculum.json"
    assert s.init_from is None


def test_scale14_sp_is_phase_2():
    s = shipped("scale14_sp")
    assert s.base_config == "python/configs/selfplay.json"
    assert s.init_from == "scale14_bots"


@pytest.mark.parametrize("name", ["scale14_bots", "scale14_sp"])
def test_scale14_arms_give_the_networks_of_the_table(name):
    s = shipped(name)
    for arm, (width, blocks) in {"w32b4": (32, 4), "w64b4": (64, 4), "w128b4": (128, 4), "w64b8": (64, 8)}.items():
        overrides = [f"{k}={json.dumps(v)}" for k, v in s.arms[arm].items()]
        cfg = load_config(str(REPO / s.base_config), overrides)
        assert (cfg.width, cfg.blocks) == (width, blocks), arm


def test_old_shipped_specs_have_no_new_fields():
    for name in ("smoke", "selfplay12b", "selfplay12c"):
        s = shipped(name)
        assert s.reference is None and s.init_from is None, name
