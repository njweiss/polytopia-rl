"""Milestone 11 tests 2: checkpoints carry the spec ("Hashes and checkpoints").

The file stays a superset of the milestone 10 format, so `train.load_policy` still reads it.
"""

import warnings

import polytopia_sim as ps
import pytest
import torch

from polytopia_rl import train
from polytopia_rl.checkpoint import Checkpoint, SpecMismatch, SpecWarning, load_checkpoint, save_checkpoint
from polytopia_rl.config import PPOConfig
from polytopia_rl.model import PolicyNet

SIZE = 6


def tiny_cfg():
    return PPOConfig(size=SIZE, width=8, blocks=1, device="cpu")


def spec():
    return dict(ps.Env(size=SIZE).spec())


def trained_net_and_opt(seed=0):
    torch.manual_seed(seed)
    net = PolicyNet(spec(), width=8, blocks=1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(2):  # give Adam some state
        opt.zero_grad()
        sum(p.sum() for p in net.parameters()).backward()
        opt.step()
    return net, opt


def save(path, net, opt, sp, **kw):
    args = dict(
        model=net,
        optimizer=opt,
        cfg=tiny_cfg(),
        global_step=777,
        stage=2,
        seed_index={"next_k": [3, 4]},
        rng_state={"numpy": [1, 2, 3]},
        spec=sp,
        extra={"pool": ["snap_0"]},
    )
    args.update(kw)
    save_checkpoint(str(path), **args)


def load_quietly(path, sp):
    """`load_checkpoint` that fails on any `SpecWarning`."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", SpecWarning)
        return load_checkpoint(str(path), spec=sp)


def same_state(a, b):
    assert a.keys() == b.keys()
    return all(torch.equal(a[k], b[k]) for k in a)


def test_round_trip_keeps_every_field_and_the_weights_bit_exactly(tmp_path):
    net, opt = trained_net_and_opt()
    sp = spec()
    save(tmp_path / "c.pt", net, opt, sp)
    ck = load_quietly(tmp_path / "c.pt", sp)
    assert isinstance(ck, Checkpoint)
    assert ck.global_step == 777 and ck.stage == 2
    assert ck.seed_index == {"next_k": [3, 4]}
    assert ck.rng_state == {"numpy": [1, 2, 3]}
    assert ck.extra == {"pool": ["snap_0"]}
    assert ck.cfg == tiny_cfg()
    assert ck.spec["obs_hash"] == sp["obs_hash"] and ck.spec["rules_hash"] == sp["rules_hash"]
    assert same_state(ck.model, net.state_dict())
    net2 = PolicyNet(sp, width=8, blocks=1)
    net2.load_state_dict(ck.model)
    assert same_state(net2.state_dict(), net.state_dict())
    opt2 = torch.optim.Adam(net2.parameters(), lr=1e-3)
    opt2.load_state_dict(ck.optimizer)
    for s1, s2 in zip(opt.state_dict()["state"].values(), opt2.state_dict()["state"].values()):
        assert torch.equal(s1["exp_avg"], s2["exp_avg"]) and torch.equal(s1["exp_avg_sq"], s2["exp_avg_sq"])


def test_file_is_a_superset_of_the_milestone_10_format(tmp_path):
    net, opt = trained_net_and_opt()
    sp = spec()
    save(tmp_path / "c.pt", net, opt, sp)
    raw = torch.load(tmp_path / "c.pt", map_location="cpu", weights_only=False)
    for k in ["model", "optimizer", "config", "global_step", "stage", "spec"]:
        assert k in raw, k
    assert PPOConfig.from_json(raw["config"]) == tiny_cfg()
    assert raw["spec"]["obs_hash"] == sp["obs_hash"]
    net2, cfg2 = train.load_policy(str(tmp_path / "c.pt"))
    assert cfg2 == tiny_cfg()
    assert same_state(net2.state_dict(), net.state_dict())


def test_obs_hash_mismatch_raises(tmp_path):
    net, opt = trained_net_and_opt()
    other = {**spec(), "obs_hash": "0123456789abcdef"}
    save(tmp_path / "c.pt", net, opt, other)
    with pytest.raises(SpecMismatch):
        load_checkpoint(str(tmp_path / "c.pt"), spec=spec())
    with pytest.raises(SpecMismatch):
        load_checkpoint(str(tmp_path / "c.pt"), spec=spec(), strict_rules=True)


def test_obs_hash_mismatch_loads_with_a_warning_when_not_strict(tmp_path):
    net, opt = trained_net_and_opt()
    other = {**spec(), "obs_hash": "0123456789abcdef"}
    save(tmp_path / "c.pt", net, opt, other)
    with pytest.warns(SpecWarning):
        ck = load_checkpoint(str(tmp_path / "c.pt"), spec=spec(), strict_obs=False)
    assert ck.spec["obs_hash"] == "0123456789abcdef"
    assert same_state(ck.model, net.state_dict())


def test_rules_hash_mismatch_warns_or_raises_when_strict(tmp_path):
    net, opt = trained_net_and_opt()
    other = {**spec(), "rules_hash": "fedcba9876543210"}
    save(tmp_path / "c.pt", net, opt, other)
    with pytest.warns(SpecWarning):
        ck = load_checkpoint(str(tmp_path / "c.pt"), spec=spec())
    assert ck.global_step == 777
    with pytest.raises(SpecMismatch):
        load_checkpoint(str(tmp_path / "c.pt"), spec=spec(), strict_rules=True)


def test_old_format_loads_with_a_warning_and_is_stamped_on_resave(tmp_path):
    net, opt = trained_net_and_opt()
    old = {
        "model": net.state_dict(),
        "optimizer": opt.state_dict(),
        "config": tiny_cfg().to_json(),
        "global_step": 512,
        "update": 4,
        "stage": 1,
        "history": {"steps_in_stage": 0, "outcomes": []},
        "recent": [],
        "best_score": None,
        "vec_env": {"seed_base": 0},
        "rng": {},
    }
    torch.save(old, tmp_path / "m10.pt")
    with pytest.warns(SpecWarning):
        ck = load_checkpoint(str(tmp_path / "m10.pt"), spec=spec())
    assert ck.spec is None
    assert ck.global_step == 512 and ck.stage == 1
    assert ck.cfg == tiny_cfg()
    assert same_state(ck.model, net.state_dict())
    # Resaved with the current spec, it loads cleanly.
    net2 = PolicyNet(spec(), width=8, blocks=1)
    net2.load_state_dict(ck.model)
    save(tmp_path / "new.pt", net2, opt, spec(), global_step=ck.global_step)
    ck2 = load_quietly(tmp_path / "new.pt", spec())
    assert ck2.spec["obs_hash"] == spec()["obs_hash"] and ck2.global_step == 512

