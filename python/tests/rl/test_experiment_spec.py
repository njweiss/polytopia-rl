"""Milestone 12 tests 5 and part of 6: the experiment spec, `run_dir`, `eval_checkpoints`, the shipped
specs and the script ("Public API", "Experiments (12b)").

Expected values are the brief's: defaults `eval_checkpoints` 3, `h2h_seeds` 100, `total_steps` 10M
("Each experiment run is 10M steps unless its spec says otherwise"), `init_checkpoint` null; the
12b table (arms base / pbot40 / ent003, seeds 1 to 3, primary rush_e0 and elo_vs_init).
"""

import json
import os

import pytest
from sp_helpers import REPO

from polytopia_rl.experiment import Spec, eval_checkpoints, load_spec, run_dir
from polytopia_rl.suite import DEFAULT_SUITE

SMOKE = {
    "name": "smoke",
    "base_config": "python/configs/selfplay.json",
    "init_checkpoint": "runs/curriculum_seed1/latest.pt",
    "seeds": [1, 2],
    "total_steps": 10000000,
    "arms": {"base": {}, "pbot40": {"p_bot": 0.4}},
    "primary": ["rush_e0"],
    "suite_seeds": 200,
    "tournament_seeds": 100,
    "eval_checkpoints": 3,
    "h2h_seeds": 100,
}


def write(tmp_path, d, name="spec.json"):
    p = tmp_path / name
    p.write_text(json.dumps(d))
    return str(p)


def test_load_spec_reads_every_field(tmp_path):
    d = {**SMOKE, "arms": {"zeta": {"ent_coef": 0.003}, "alpha": {}, "mid": {"p_bot": 0.4, "lr": 5e-5}}, "eval_checkpoints": 2, "h2h_seeds": 7, "total_steps": 200000}
    s = load_spec(write(tmp_path, d))
    assert isinstance(s, Spec)
    assert s.name == "smoke"
    assert s.base_config == "python/configs/selfplay.json"
    assert s.init_checkpoint == "runs/curriculum_seed1/latest.pt"
    assert list(s.seeds) == [1, 2]
    assert s.total_steps == 200000
    assert list(s.arms) == ["zeta", "alpha", "mid"], "arms keep the spec order"
    assert s.arms["mid"] == {"p_bot": 0.4, "lr": 5e-5}
    assert list(s.primary) == ["rush_e0"]
    assert (s.suite_seeds, s.tournament_seeds, s.eval_checkpoints, s.h2h_seeds) == (200, 100, 2, 7)


def test_load_spec_defaults(tmp_path):
    d = {k: v for k, v in SMOKE.items() if k not in ("init_checkpoint", "total_steps", "eval_checkpoints", "h2h_seeds")}
    s = load_spec(write(tmp_path, d))
    assert s.init_checkpoint is None
    assert s.total_steps == 10_000_000
    assert s.eval_checkpoints == 3
    assert s.h2h_seeds == 100
    s = load_spec(write(tmp_path, {**SMOKE, "init_checkpoint": None}, "null.json"))
    assert s.init_checkpoint is None


def test_unknown_keys_raise(tmp_path):
    with pytest.raises(ValueError):
        load_spec(write(tmp_path, {**SMOKE, "parallel": 2}))


@pytest.mark.parametrize("key", ["seed", "total_steps"])
def test_launcher_keys_in_an_arm_are_rejected(tmp_path, key):
    with pytest.raises(ValueError):
        load_spec(write(tmp_path, {**SMOKE, "arms": {"base": {}, "bad": {key: 5}}}))


def test_run_dir_layout(tmp_path):
    s = load_spec(write(tmp_path, SMOKE))
    assert os.path.normpath(run_dir(s, "base", 1)) == os.path.join("runs", "exp", "smoke", "base", "seed1")
    assert os.path.normpath(run_dir(s, "pbot40", 2)) == os.path.join("runs", "exp", "smoke", "pbot40", "seed2")


def touch(d, names):
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        (d / n).write_bytes(b"")


def names_of(paths):
    for p in paths:
        assert os.path.exists(p), p
    return [os.path.basename(p) for p in paths]


def test_eval_checkpoints_picks_the_last_k_distinct_steps_in_numeric_order(tmp_path):
    # 2048 sorts before 512 as text; ckpt_final, latest and best are not ckpt_{step} files.
    touch(tmp_path, ["ckpt_256.pt", "ckpt_512.pt", "ckpt_1024.pt", "ckpt_2048.pt", "ckpt_final.pt", "latest.pt", "best.pt", "ckpt_x.pt", "notes.txt"])
    assert names_of(eval_checkpoints(str(tmp_path), 3)) == ["ckpt_512.pt", "ckpt_1024.pt", "ckpt_2048.pt"]
    assert names_of(eval_checkpoints(str(tmp_path), 1)) == ["ckpt_2048.pt"]
    assert names_of(eval_checkpoints(str(tmp_path), 10)) == ["ckpt_256.pt", "ckpt_512.pt", "ckpt_1024.pt", "ckpt_2048.pt"]


def test_eval_checkpoints_returns_fewer_when_the_run_has_fewer(tmp_path):
    touch(tmp_path, ["ckpt_100000.pt", "ckpt_final.pt", "latest.pt"])
    assert names_of(eval_checkpoints(str(tmp_path), 3)) == ["ckpt_100000.pt"]
    empty = tmp_path / "empty"
    empty.mkdir()
    assert eval_checkpoints(str(empty), 3) == []


# ---------------------------------------------------------------- shipped files


def test_smoke_spec_is_two_arms_by_two_seeds():
    s = load_spec(str(REPO / "python" / "experiments" / "smoke.json"))
    assert s.name == "smoke"
    assert s.base_config == "python/configs/selfplay.json"
    assert s.init_checkpoint == "runs/curriculum_seed1/latest.pt"
    assert list(s.seeds) == [1, 2]
    assert len(s.arms) == 2
    assert s.primary and all(m in [c.name for c in DEFAULT_SUITE] + ["elo_vs_init"] for m in s.primary)


def test_selfplay12b_spec_is_the_owners_experiment():
    s = load_spec(str(REPO / "python" / "experiments" / "selfplay12b.json"))
    assert s.base_config == "python/configs/selfplay.json"
    assert s.init_checkpoint == "runs/curriculum_seed1/latest.pt"
    assert list(s.seeds) == [1, 2, 3]
    assert s.total_steps == 10_000_000
    assert s.arms == {"base": {"p_bot": 0.2}, "pbot40": {"p_bot": 0.4}, "ent003": {"ent_coef": 0.003}}
    assert list(s.arms) == ["base", "pbot40", "ent003"]
    assert sorted(s.primary) == ["elo_vs_init", "rush_e0"]
    assert s.eval_checkpoints == 3


def test_experiment_script_exists():
    p = REPO / "scripts" / "experiment.sh"
    assert p.exists() and os.access(p, os.X_OK)
    assert "polytopia_rl.experiment" in p.read_text()
