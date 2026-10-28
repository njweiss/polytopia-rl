"""Milestone 13b tests 6 and 7: `SelfPlayVecEnv.tick` with a stacked policy ("In the vec env"), the
`opponent_forward` config key and the trainer's wiring.

Fake traces (sp_helpers.FakeTwoSeatEnv with per-episode turn lengths, snapshot opponents playing
`fake_scorer`): a vec env given a mock stacked policy is run tick by tick next to a twin without one
(the M13 per-snapshot calls, the reference). Per tick the stacked vec must make exactly one
`sync(pool.entries)` and one stacked call whose groups are the twin's calls (names, slots), no
per-snapshot call, and end in the same state with the same transitions.

Real env (size 6, turn cap 3): greedy snapshot opponents through the real `stacked_batch_policy`
against the per-snapshot `net_batch_policy` calls; the trajectories must be identical, and every
greedy decision met on the reference side has a top-two gap above 1e-3 (tie-free).
"""

import json

import numpy as np
import polytopia_sim as ps
import pytest
import torch
from il_helpers import FakeBotEnv, KeyedVec, keyed_pick, learner_path, pool_of, reward_cfg
from sp_helpers import FakeTwoSeatEnv, fake_scorer
from stk_helpers import greedy_gaps, random_net

from polytopia_rl import ppo, selfplay, stacked, train
from polytopia_rl.config import PPOConfig, load_config
from polytopia_rl.selfplay import SnapshotPool, net_batch_policy
from polytopia_rl.stacked import StackedSnapshots, stacked_batch_policy

FAKE_REWARD = reward_cfg(shaping=1.0, gamma=0.9, scale=10.0)
N = 6


def slots_of(vec, envs):
    return [next(i for i, x in enumerate(vec.envs) if x is e) for e in envs]


class VarKTwoSeat(FakeTwoSeatEnv):
    def reset(self, seed):
        self.k = 2 + int(seed) % 3
        super().reset(seed)


class VarKBot(FakeBotEnv):
    def reset(self, seed):
        self.k = 2 + int(seed) % 3
        super().reset(seed)


class PerSnapshotCalls:
    """make_policy whose batch policies play `fake_scorer` and record `(entry name, slots)`."""

    def __init__(self):
        self.vec = None
        self.calls = []

    def __call__(self, entry):
        def policy(envs):
            self.calls.append((entry.name, slots_of(self.vec, envs)))
            return fake_scorer(envs)

        return policy


class MockStacked:
    """A stacked policy playing `fake_scorer`; logs `("sync", names)` and `("call", [(name, slots)])`."""

    def __init__(self):
        self.vec = None
        self.log = []

    def sync(self, entries):
        self.log.append(("sync", [e.name for e in entries]))

    def __call__(self, groups):
        groups = list(groups)
        self.log.append(("call", [(name, slots_of(self.vec, envs)) for name, envs in groups]))
        return [fake_scorer(envs) for _, envs in groups]


CHOICES = [0, 1, 2, "economy"]
PICK = keyed_pick(len(CHOICES), salt=7)


def fake_vec(stacked_policy=None):
    pool = pool_of(3)
    calls = PerSnapshotCalls()
    vec = KeyedVec(
        lambda opp: VarKTwoSeat(turns=2) if opp == "none" else VarKBot(turns=2),
        N,
        0,
        1,
        pool=pool,
        make_policy=calls,
        rng=np.random.default_rng(0),
        p_bot=0.3,
        reward=FAKE_REWARD,
        choices=[c if isinstance(c, str) else pool.entries[c] for c in CHOICES],
        pick=PICK,
        stacked_policy=stacked_policy,
    )
    calls.vec = vec
    if stacked_policy is not None:
        stacked_policy.vec = vec
    return vec, calls


def transitions(out):
    return [
        (t.slot, t.reward, bool(t.terminated), bool(t.truncated), None if t.final_globals is None else np.asarray(t.final_globals).tolist())
        for t in out
    ]


def test_one_stacked_call_per_tick_with_the_m13_groups_and_the_same_trajectories_and_none_on_the_lockstep_path():
    ref, ref_calls = fake_vec()
    mock = MockStacked()
    stk, stk_calls = fake_vec(mock)
    ref.reset_all()
    stk.reset_all()
    assert mock.log == [], "reset_all plays the opening moves on the lockstep path (per-snapshot calls)"
    assert stk_calls.calls == ref_calls.calls
    rng = np.random.default_rng(5)
    seen = dict(no_moves=0, one_group=0, two_groups=0)
    names = [e.name for e in stk.pool.entries]
    for t in range(250):
        assert stk.learner_slots() == ref.learner_slots(), t
        acts = {i: learner_path(ref.envs[i]) for i in ref.learner_slots() if rng.random() < 0.7}
        ref_calls.calls.clear()
        stk_calls.calls.clear()
        mock.log.clear()
        out_ref = ref.tick(acts)
        out_stk = stk.tick(acts)
        assert transitions(out_stk) == transitions(out_ref), t
        assert stk.state_dict() == ref.state_dict(), t
        assert stk_calls.calls == [], "no per-snapshot call in a tick when a stacked policy is given"
        if ref_calls.calls:
            assert mock.log == [("sync", names), ("call", ref_calls.calls)], (t, mock.log, ref_calls.calls)
            seen["one_group" if len(ref_calls.calls) == 1 else "two_groups"] += 1
        else:
            assert all(kind != "call" for kind, _ in mock.log), (t, mock.log)
            seen["no_moves"] += 1
    assert all(v > 0 for v in seen.values()), seen
    assert len(ref.episode_stats()) == len(stk.episode_stats()) > 0
    # The lockstep path (`step`, `_advance`) never uses the stacked policy.
    mock.log.clear()
    stk_calls.calls.clear()
    stk.reset_all()
    for _ in range(40):
        stk.step(*zip(*[learner_path(env) for env in stk.envs]))
    assert mock.log == []
    assert stk_calls.calls, "the lockstep path played its snapshot opponents through make_policy"


# ---------------------------------------------------------------- real env, greedy snapshots

SIZE, CAP = 6, 3
REAL_N = 8


class Counting:
    """Wraps a stacked policy; counts calls and syncs."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0
        self.syncs = 0

    def sync(self, entries):
        self.syncs += 1
        self.inner.sync(entries)

    def __call__(self, groups):
        self.calls += 1
        return self.inner(groups)


def test_greedy_stacked_trajectories_equal_the_per_snapshot_ones_on_the_real_env():
    nets = {f"snap_{100 * i}": random_net(90 + i) for i in range(3)}
    rw = reward_cfg(1.0)
    gaps = []
    ref_calls_per_tick = []

    def build(stacked_policy, make_policy):
        pool = SnapshotPool()
        for i, net in enumerate(nets.values()):
            pool.add(net.state_dict(), 100 * i)
        return KeyedVec(
            lambda opp: ps.Env(size=SIZE, turn_cap=CAP, agent_seat="seed_parity", opponent=opp, **rw),
            REAL_N,
            40,
            1,
            pool=pool,
            make_policy=make_policy,
            rng=np.random.default_rng(0),
            p_bot=0.3,
            reward=rw,
            choices=[pool.entries[0], pool.entries[1], pool.entries[2], "economy"],
            pick=keyed_pick(4, salt=5),
            stacked_policy=stacked_policy,
        )

    def ref_policy(entry):
        net = nets[entry.name]
        inner = net_batch_policy(net, torch.Generator().manual_seed(0), greedy=True)

        def policy(envs):
            gaps.extend(greedy_gaps(net, envs))
            ref_calls_per_tick[-1] += 1
            return inner(envs)

        return policy

    stk_per_snapshot = []

    def stk_make_policy(entry):
        inner = net_batch_policy(nets[entry.name], torch.Generator().manual_seed(0), greedy=True)

        def policy(envs):
            stk_per_snapshot.append(entry.name)
            return inner(envs)

        return policy

    counting = Counting(stacked_batch_policy(StackedSnapshots(random_net(99), "cpu"), torch.Generator().manual_seed(0), greedy=True))
    ref = build(None, ref_policy)
    stk = build(counting, stk_make_policy)
    ref_calls_per_tick.append(0)
    ref.reset_all()
    stk.reset_all()
    assert counting.calls == 0
    stk_per_snapshot.clear()
    ticks_with_moves = 0
    multi = 0
    for t in range(60):
        assert stk.learner_slots() == ref.learner_slots(), t
        acts_ref = {i: ref.envs[i].bot_action("rush" if ref.envs[i].info()["seed"] % 3 else "economy") for i in ref.learner_slots()}
        acts_stk = {i: stk.envs[i].bot_action("rush" if stk.envs[i].info()["seed"] % 3 else "economy") for i in stk.learner_slots()}
        assert acts_ref == acts_stk, t
        ref_calls_per_tick.append(0)
        before = counting.calls
        out_ref = ref.tick(acts_ref)
        out_stk = stk.tick(acts_stk)
        assert transitions(out_stk) == transitions(out_ref), t
        assert stk.state_dict() == ref.state_dict(), t
        calls = ref_calls_per_tick[-1]
        assert counting.calls - before == (1 if calls else 0), (t, calls)
        ticks_with_moves += calls > 0
        multi += calls > 1
    assert stk_per_snapshot == [], "no per-snapshot call in a tick"
    assert ticks_with_moves >= 20 and multi >= 5, (ticks_with_moves, multi)
    assert len(gaps) >= 100
    assert min(gaps) > 1e-3, "the fixture must be tie-free (pick other seeds)"
    assert len(stk.episode_stats()) > 0


# ---------------------------------------------------------------- config and trainer


def test_opponent_forward_defaults_to_stacked_and_rejects_unknown_values(tmp_path):
    assert PPOConfig().opponent_forward == "stacked"
    assert PPOConfig.from_json("{}").opponent_forward == "stacked"
    assert PPOConfig.from_json('{"opponent_forward": "per_snapshot"}').opponent_forward == "per_snapshot"
    assert PPOConfig.from_json('{"opponent_forward": "stacked"}').opponent_forward == "stacked"
    for bad in ("Stacked", "per-snapshot", "vmap", "", None, 1):
        with pytest.raises(ValueError):
            PPOConfig.from_json(json.dumps({"opponent_forward": bad}))
    path = tmp_path / "c.json"
    path.write_text("{}")
    assert load_config(str(path), ["opponent_forward=per_snapshot"]).opponent_forward == "per_snapshot"
    with pytest.raises(ValueError):
        load_config(str(path), ["opponent_forward=batched"])


BATCH = 4 * 16


def sp_cfg(**kw):
    base = dict(
        size=6,
        turn_cap=3,
        num_envs=4,
        num_steps=16,
        num_minibatches=2,
        update_epochs=1,
        width=8,
        blocks=1,
        total_steps=2 * BATCH,
        eval_every=100 * BATCH,
        eval_seeds=1,
        device="cpu",
        seed=3,
        opponents=["economy", "rush"],
        stages=[{"opponents": ["economy", "rush"]}],
        selfplay=True,
        snapshot_every=BATCH,
        rating_every=100 * BATCH,
        rating_seeds=1,
        tournament_seeds=1,
        p_bot=0.25,
    )
    return PPOConfig(**{**base, **kw})


class Stop(Exception):
    pass


def wiring_spy(mp, stop: bool):
    """Records the trainer's `make_selfplay_vec_env` keywords, its `StackedSnapshots(base_net, device)`
    and `stacked_batch_policy(stack, generator, greedy)` calls; the stacked policies it gets are
    wrapped in `Counting`. With `stop`, the first rollout raises `Stop`."""
    got = {"vec_kw": [], "bases": [], "policies": []}
    orig_make = selfplay.make_selfplay_vec_env

    def make(*a, **kw):
        got["vec_kw"].append(kw)
        return orig_make(*a, **kw)

    mp.setattr(selfplay, "make_selfplay_vec_env", make)
    mp.setattr(train, "make_selfplay_vec_env", make)
    orig_cls = stacked.StackedSnapshots

    class SpyStack(orig_cls):
        def __init__(self, base_net, device):
            got["bases"].append((base_net, device))
            super().__init__(base_net, device)

    mp.setattr(stacked, "StackedSnapshots", SpyStack)
    mp.setattr(train, "StackedSnapshots", SpyStack, raising=False)
    orig_sbp = stacked.stacked_batch_policy

    def sbp(stack, generator, greedy=False):
        policy = Counting(orig_sbp(stack, generator, greedy))
        got["policies"].append(dict(stack=stack, generator=generator, greedy=greedy, policy=policy))
        return policy

    mp.setattr(stacked, "stacked_batch_policy", sbp)
    mp.setattr(train, "stacked_batch_policy", sbp, raising=False)
    if stop:
        for name in ("collect_rollout", "collect_rollout_interleaved"):

            def boom(*a, **kw):
                raise Stop

            mp.setattr(ppo, name, boom)
            mp.setattr(train, name, boom, raising=False)
    return got


def test_the_trainer_passes_a_stacked_policy_by_default_and_none_for_per_snapshot(monkeypatch, tmp_path):
    for greedy in (False, True):
        with monkeypatch.context() as mp:
            got = wiring_spy(mp, stop=True)
            with pytest.raises(Stop):
                train.run(sp_cfg(opponent_greedy=greedy), str(tmp_path / f"greedy_{greedy}"))
        (kw,) = got["vec_kw"]
        assert len(got["policies"]) == 1, "one stacked_batch_policy per vec env"
        (p,) = got["policies"]
        assert kw.get("stacked_policy") is p["policy"]
        assert p["greedy"] is greedy
        assert isinstance(p["generator"], torch.Generator)
        assert isinstance(p["stack"], stacked.StackedSnapshots)
        ((_, device),) = got["bases"]
        assert device == "cpu"
    with monkeypatch.context() as mp:
        got = wiring_spy(mp, stop=True)
        with pytest.raises(Stop):
            train.run(sp_cfg(opponent_forward="per_snapshot"), str(tmp_path / "per_snapshot"))
    (kw,) = got["vec_kw"]
    assert kw.get("stacked_policy") is None


def test_a_stacked_run_uses_the_stacked_policy_on_a_copy_of_the_learner_and_lockstep_or_per_snapshot_runs_never_do(monkeypatch, tmp_path):
    with monkeypatch.context() as mp:
        got = wiring_spy(mp, stop=False)
        net = train.run(sp_cfg(), str(tmp_path / "stacked"))
    assert len(got["policies"]) == 1, "one stacked_batch_policy per vec env"
    (p,) = got["policies"]
    assert p["policy"].calls > 0 and p["policy"].syncs > 0
    ((base, _),) = got["bases"]
    assert base is not net, "the stack's base net is a copy of the learner's architecture"
    ptrs = {q.data_ptr() for q in net.parameters()}
    assert all(q.data_ptr() not in ptrs for q in base.parameters())
    for name, cfg in (("lockstep", sp_cfg(rollout="lockstep")), ("per_snapshot", sp_cfg(opponent_forward="per_snapshot"))):
        with monkeypatch.context() as mp:
            got = wiring_spy(mp, stop=False)
            train.run(cfg, str(tmp_path / name))
        assert all(p["policy"].calls == 0 for p in got["policies"]), name
