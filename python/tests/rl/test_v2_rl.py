"""Milestone 20 Phase A: Python RL stack on `ruleset="v2"` (`PPOConfig`, `PolicyNet`,
`SyncVecEnv`, `SelfPlayVecEnv` interleaved rollout, `train`, `evaluate`, `suite`, `checkpoint`)."""

import json

import numpy as np
import polytopia_sim as ps
import pytest
import torch

from polytopia_rl import train
from polytopia_rl.checkpoint import SpecMismatch, load_checkpoint, load_net, save_checkpoint
from polytopia_rl.config import PPOConfig
from polytopia_rl.evaluate import evaluate, net_policy
from polytopia_rl.model import PolicyNet
from polytopia_rl.ppo import RolloutBuffer, collect_rollout, collect_rollout_interleaved, train_update
from polytopia_rl.selfplay import SnapshotPool, make_selfplay_vec_env, net_batch_policy
from polytopia_rl.suite import DEFAULT_SUITE, run_suite
from polytopia_rl.vec_env import env_kwargs_of, env_masks_fn, make_vec_env


def tiny_v2_cfg(**kw) -> PPOConfig:
    base = dict(
        size=6,
        turn_cap=3,
        ruleset="v2",
        num_envs=2,
        num_steps=8,
        num_minibatches=2,
        update_epochs=1,
        width=8,
        blocks=1,
        total_steps=32,
        eval_every=32,
        eval_seeds=1,
        device="cpu",
        seed=1,
        opponents=["random"],
        stages=[{"opponents": ["random"]}],
    )
    return PPOConfig(**{**base, **kw})


def test_ppo_config_ruleset_default_round_trip_and_validation():
    default = PPOConfig()
    assert default.ruleset == "v1"
    assert env_kwargs_of(default)["ruleset"] == "v1"

    # Legacy JSON without `ruleset` defaults to "v1".
    legacy = json.loads(default.to_json())
    legacy.pop("ruleset", None)
    assert PPOConfig.from_json(json.dumps(legacy)).ruleset == "v1"

    v2 = PPOConfig(ruleset="v2")
    assert v2.ruleset == "v2"
    assert env_kwargs_of(v2)["ruleset"] == "v2"
    assert PPOConfig.from_json(v2.to_json()) == v2

    for bad in ("v3", "V2", "", "v0"):
        data = json.loads(default.to_json())
        data["ruleset"] = bad
        with pytest.raises(ValueError):
            PPOConfig.from_json(json.dumps(data))


def test_policy_net_forward_act_evaluate_and_gradients_on_v2():
    env = ps.Env(size=6, opponent="random", ruleset="v2")
    spec = env.spec()
    env.reset(2)

    torch.manual_seed(0)
    net = PolicyNet(spec, width=16, blocks=1)
    board_np, glob_np = env.observation()
    board = torch.from_numpy(board_np)[None]
    glob = torch.from_numpy(glob_np)[None]

    feats = net.trunk(board, glob)
    assert feats.fmap.shape == (1, 16, 6, 6)
    assert net.actor_logits(feats).shape == (1, 3 * 36 + 1)
    gen = torch.Generator().manual_seed(0)
    out = net.act(feats, env_masks_fn([env], 36), gen)
    assert out.kind_mask.shape == (1, 15)
    assert out.arg_mask.shape == (1, 36)
    assert net.kind_logits(feats, out.actors).shape == (1, 15)
    assert net.arg_logits(feats, out.actors, out.kinds).shape == (1, 36)
    assert torch.isfinite(out.logp).all() and torch.isfinite(out.entropy).all() and torch.isfinite(out.value).all()

    ev = net.evaluate(board, glob, out.actors, out.kinds, out.args, out.actor_mask, out.kind_mask, out.arg_mask)
    assert torch.allclose(ev.logp, out.logp, atol=1e-5)
    assert torch.allclose(ev.entropy, out.entropy, atol=1e-5)
    assert torch.allclose(ev.value, out.value, atol=1e-5)

    loss = -(ev.logp.mean()) + ev.value.pow(2).mean()
    loss.backward()
    assert net.stem.weight.grad is not None and torch.isfinite(net.stem.weight.grad).all()
    assert net.stem.weight.grad.abs().sum() > 0


def test_sync_vec_env_and_interleaved_selfplay_on_v2():
    cfg = tiny_v2_cfg(opponents=["random", "rush"])
    spec = ps.Env(size=6, ruleset="v2").spec()
    net = PolicyNet(spec, width=8, blocks=1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(0)

    # SyncVecEnv + collect_rollout + train_update.
    vec = make_vec_env(cfg, seed_base=0)
    state = vec.reset_all()
    assert state.board.shape == (2, 64, 6, 6)
    assert state.globals.shape == (2, 33)
    actors = np.array([int(np.flatnonzero(m)[0]) for m in vec.actor_masks()])
    assert vec.kind_masks(actors).shape == (2, 15)
    buf = RolloutBuffer(cfg.num_steps, cfg.num_envs, spec, device="cpu")
    collect_rollout(net, vec, buf, gen)
    assert buf.kind_mask.shape == (cfg.num_steps, cfg.num_envs, 15)
    metrics = train_update(net, opt, buf, cfg, gen)
    assert np.isfinite(metrics["pg_loss"]) and np.isfinite(metrics["v_loss"])

    # SelfPlayVecEnv + collect_rollout_interleaved + train_update.
    sp_cfg = tiny_v2_cfg(selfplay=True, rollout="interleaved", p_bot=0.5)
    pool = SnapshotPool(pool_max=4, p_latest=0.5)
    pool.add(net.state_dict(), 0)
    opp_gen = torch.Generator().manual_seed(1)

    def make_policy(entry):
        snap = PolicyNet(spec, width=8, blocks=1)
        snap.load_state_dict(entry.state)
        return net_batch_policy(snap, opp_gen, greedy=False, device="cpu")

    sp_vec = make_selfplay_vec_env(
        sp_cfg,
        seed_base=10,
        pool=pool,
        make_policy=make_policy,
        rng=np.random.default_rng(7),
    )
    sp_state = sp_vec.reset_all()
    assert sp_state.board.shape == (2, 64, 6, 6)
    assert sp_state.globals.shape == (2, 33)
    sp_buf = RolloutBuffer(sp_cfg.num_steps, sp_cfg.num_envs, spec, device="cpu")
    collect_rollout_interleaved(net, sp_vec, sp_buf, gen)
    assert sp_buf.kind_mask.shape == (sp_cfg.num_steps, sp_cfg.num_envs, 15)
    sp_metrics = train_update(net, opt, sp_buf, sp_cfg, gen)
    assert np.isfinite(sp_metrics["pg_loss"]) and np.isfinite(sp_metrics["v_loss"])


def test_end_to_end_v2_train_evaluate_suite_and_checkpoint_round_trip(tmp_path):
    cfg = tiny_v2_cfg()
    v2_spec = ps.Env(size=6, ruleset="v2").spec()
    run_dir = tmp_path / "v2_run"

    net = train.run(cfg, str(run_dir))
    assert (run_dir / "latest.pt").exists()
    assert (run_dir / "ckpt_final.pt").exists()

    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert set(rows[-1]["kind_freq"]) == set(v2_spec["action_types"])
    assert len(rows[-1]["kind_freq"]) == 15

    # load_net must fetch the v2 spec using cfg.ruleset (not the v1 spec!).
    net2, ck = load_net(str(run_dir / "latest.pt"), device="cpu")
    assert ck.cfg.ruleset == "v2"
    assert ck.spec["ruleset"] == "v2"
    assert ck.spec["obs_hash"] == v2_spec["obs_hash"]
    assert ck.spec["rules_hash"] == v2_spec["rules_hash"]
    for k, v in net.state_dict().items():
        assert torch.equal(v, net2.state_dict()[k]), f"weight mismatch {k}"

    # evaluate() and run_suite() carry ruleset="v2" and record v2 hashes.
    ev = evaluate(net_policy(net2), "random", seeds=1, env_kwargs=env_kwargs_of(cfg), greedy=True)
    assert ev.games == 2
    suite_res = run_suite(net_policy(net2), [DEFAULT_SUITE[0]], seeds=1, env_kwargs=env_kwargs_of(cfg), step=32)
    assert suite_res.obs_hash == v2_spec["obs_hash"]
    assert suite_res.rules_hash == v2_spec["rules_hash"]


def test_checkpoint_spec_mismatch_between_v1_and_v2(tmp_path):
    v1_spec = ps.Env(size=6, ruleset="v1").spec()
    v2_spec = ps.Env(size=6, ruleset="v2").spec()
    assert v1_spec["obs_hash"] != v2_spec["obs_hash"]

    cfg_v1 = tiny_v2_cfg(ruleset="v1")
    cfg_v2 = tiny_v2_cfg(ruleset="v2")
    net_v1 = PolicyNet(v1_spec, width=8, blocks=1)
    net_v2 = PolicyNet(v2_spec, width=8, blocks=1)

    p_v1 = tmp_path / "v1.pt"
    p_v2 = tmp_path / "v2.pt"
    save_checkpoint(
        str(p_v1),
        model=net_v1,
        optimizer=None,
        cfg=cfg_v1,
        global_step=16,
        stage=0,
        seed_index={},
        rng_state={},
        spec=v1_spec,
    )
    save_checkpoint(
        str(p_v2),
        model=net_v2,
        optimizer=None,
        cfg=cfg_v2,
        global_step=16,
        stage=0,
        seed_index={},
        rng_state={},
        spec=v2_spec,
    )

    with pytest.raises(SpecMismatch):
        load_checkpoint(str(p_v1), spec=v2_spec)
    with pytest.raises(SpecMismatch):
        load_checkpoint(str(p_v2), spec=v1_spec)

    # Matching specs load cleanly (and load_net picks the right spec per checkpoint).
    loaded_v1, ck_v1 = load_net(str(p_v1), device="cpu")
    loaded_v2, ck_v2 = load_net(str(p_v2), device="cpu")
    assert ck_v1.spec["obs_hash"] == v1_spec["obs_hash"]
    assert ck_v2.spec["obs_hash"] == v2_spec["obs_hash"]
    assert loaded_v1.stem.in_channels == 37
    assert loaded_v2.stem.in_channels == 64
