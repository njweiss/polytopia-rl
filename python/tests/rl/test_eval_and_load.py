"""Milestone 16b tests 6: `train.load_policy` hash enforcement and batched evaluation in
`evaluate.main` and curriculum periodic evaluation.

Checks (docs/MILESTONE_16b.md, Rule 6):
- `train.load_policy` delegates to `checkpoint.load_net`: raises `SpecMismatch` on wrong `obs_hash`,
  warns (`SpecWarning`) on wrong `rules_hash` or unstamped M10 checkpoints, and loads valid
  checkpoints cleanly with bit-identical weights and config.
- `evaluate.main` passes a `batch_fn` (`net_batch_policy(..., greedy=True)`) to `evaluate(...)`
  and matches sequential greedy W-L-D game-for-game on CPU.
- Curriculum periodic eval in `train.run` passes `batch_fn` to `evaluate(...)` and produces the
  same greedy W-L-D on CPU as sequential eval.
"""

import json
import warnings

import polytopia_sim as ps
import pytest
import torch

from polytopia_rl import evaluate as eval_mod
from polytopia_rl import train
from polytopia_rl.checkpoint import SpecMismatch, SpecWarning, save_checkpoint
from polytopia_rl.config import PPOConfig
from polytopia_rl.evaluate import env_kwargs_of, evaluate, net_policy
from polytopia_rl.model import PolicyNet

SIZE = 6


def _tiny_cfg(**kw):
    base = dict(size=SIZE, turn_cap=3, width=8, blocks=1, device="cpu", seed=2)
    return PPOConfig(**{**base, **kw})


def _spec():
    return dict(ps.Env(size=SIZE).spec())


def _make_net(seed=0):
    torch.manual_seed(seed)
    return PolicyNet(_spec(), width=8, blocks=1)


def _same_state(a, b):
    assert a.keys() == b.keys()
    return all(torch.equal(a[k], b[k]) for k in a)


def _save(path, net, sp, cfg=None):
    save_checkpoint(
        str(path),
        model=net,
        optimizer=None,
        cfg=cfg or _tiny_cfg(),
        global_step=64,
        stage=0,
        seed_index=None,
        rng_state=None,
        spec=sp,
    )


def test_load_policy_raises_spec_mismatch_on_wrong_obs_hash(tmp_path):
    net = _make_net()
    bad_spec = {**_spec(), "obs_hash": "0123456789abcdef"}
    path = tmp_path / "bad_obs.pt"
    _save(path, net, bad_spec)
    with pytest.raises(SpecMismatch):
        train.load_policy(str(path))


def test_load_policy_warns_on_wrong_rules_hash_and_loads_weights(tmp_path):
    net = _make_net(seed=1)
    bad_rules = {**_spec(), "rules_hash": "fedcba9876543210"}
    path = tmp_path / "bad_rules.pt"
    _save(path, net, bad_rules)
    with pytest.warns(SpecWarning):
        net2, cfg2 = train.load_policy(str(path))
    assert cfg2 == _tiny_cfg()
    assert _same_state(net2.state_dict(), net.state_dict())


def test_load_policy_warns_on_unstamped_m10_checkpoint(tmp_path):
    net = _make_net(seed=2)
    path = tmp_path / "m10.pt"
    torch.save(
        {
            "model": net.state_dict(),
            "optimizer": None,
            "config": _tiny_cfg().to_json(),
            "global_step": 32,
            "stage": 0,
        },
        path,
    )
    with pytest.warns(SpecWarning):
        net2, cfg2 = train.load_policy(str(path))
    assert cfg2 == _tiny_cfg()
    assert _same_state(net2.state_dict(), net.state_dict())


def test_load_policy_valid_checkpoint_loads_quietly_and_identically(tmp_path):
    net = _make_net(seed=3)
    path = tmp_path / "ok.pt"
    _save(path, net, _spec())
    with warnings.catch_warnings():
        warnings.simplefilter("error", SpecWarning)
        net2, cfg2 = train.load_policy(str(path))
    assert cfg2 == _tiny_cfg()
    assert _same_state(net2.state_dict(), net.state_dict())


def test_evaluate_main_passes_batch_fn_and_matches_sequential_wld(tmp_path, monkeypatch):
    cfg = _tiny_cfg()
    net = _make_net(seed=4)
    path = tmp_path / "ck.pt"
    _save(path, net, _spec(), cfg=cfg)

    seen = []
    real_eval = eval_mod.evaluate

    def spy_eval(*args, **kwargs):
        res = real_eval(*args, **kwargs)
        seen.append((args[1], kwargs.get("batch_fn"), res))
        return res

    monkeypatch.setattr(eval_mod, "evaluate", spy_eval)
    monkeypatch.setattr(train, "resolve_device", lambda device: "cpu")

    eval_mod.main(["--checkpoint", str(path), "--opponents", "random,rush", "--seeds", "2"])

    assert [opp for opp, _, _ in seen] == ["random", "rush"]
    for opp, batch_fn, res in seen:
        assert callable(batch_fn), f"evaluate.main must pass batch_fn for {opp}"
        seq = real_eval(net_policy(net, device="cpu"), opp, 2, env_kwargs_of(cfg), greedy=True)
        assert (res.wins, res.losses, res.draws) == (seq.wins, seq.losses, seq.draws), opp
        assert res.per_game == seq.per_game, opp


def test_curriculum_periodic_eval_passes_batch_fn_and_matches_sequential_wld(tmp_path, monkeypatch):
    cfg = PPOConfig(
        size=SIZE,
        turn_cap=3,
        num_envs=2,
        num_steps=8,
        num_minibatches=2,
        update_epochs=1,
        width=8,
        blocks=1,
        total_steps=16,
        eval_every=16,
        eval_seeds=2,
        device="cpu",
        seed=5,
        selfplay=False,
        opponents=["random"],
        stages=[{"opponents": ["random"]}],
    )
    seen = []
    real_eval = train.evaluate

    def spy_eval(*args, **kwargs):
        res = real_eval(*args, **kwargs)
        seen.append((args[1], kwargs.get("batch_fn"), res))
        return res

    monkeypatch.setattr(train, "evaluate", spy_eval)
    run_dir = tmp_path / "run"
    train.run(cfg, str(run_dir))

    assert len(seen) == 1
    opp, batch_fn, res = seen[0]
    assert opp == "random" and callable(batch_fn), "curriculum periodic eval must pass batch_fn"

    net, _ = train.load_policy(str(run_dir / "ckpt_16.pt"), device="cpu")
    seq = evaluate(net_policy(net, device="cpu"), "random", 2, env_kwargs_of(cfg), greedy=True)
    assert (res.wins, res.losses, res.draws) == (seq.wins, seq.losses, seq.draws)
    assert res.per_game == seq.per_game

    eval_rows = [json.loads(line) for line in (run_dir / "eval.jsonl").read_text().splitlines()]
    assert len(eval_rows) == 1
    logged = eval_rows[0]["random"]
    assert (logged["wins"], logged["losses"], logged["draws"]) == (seq.wins, seq.losses, seq.draws)
    assert logged["win_rate"] == pytest.approx(seq.win_rate)
