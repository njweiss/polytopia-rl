"""Milestone 16b tests 2: batched buffer writes and action readback in `collect_rollout_interleaved`.

Checks (docs/MILESTONE_16b.md, Rule 2):
- Stale `buffer.next_values` are overwritten with 0.0 on every non-truncated transition.
- Truncated transitions write their bootstrap `next_values` at the pre-increment row `t[slot]`,
  both mid-rollout and on the final row `T - 1`, even when truncated and non-truncated slots finish
  in the same tick.
- Transition fields (`rewards`, `terminated`, `truncated`, `next_values`) are written in one
  batched scatter per field per tick (plus one bootstrap scatter for `cut`), and learner actions
  (`out.actors`, `out.kinds`, `out.args`) are transferred in batch without per-element indexing.
- Bit-identical equivalence to `collect_rollout` on the real env across mixed bots + snapshots,
  shaping on/off, and truncations.
"""

import numpy as np
import polytopia_sim as ps
import pytest
import torch
from il_helpers import BotLearner, StubNet, keyed_pick, keyed_vec, pool_of, reward_cfg, stub_value
from sp_helpers import FakeTwoSeatEnv, bot_batch, fake_scorer

from polytopia_rl.model import ActOut, PolicyNet
from polytopia_rl.ppo import RolloutBuffer, collect_rollout, collect_rollout_interleaved
from polytopia_rl.selfplay import SelfPlayVecEnv, net_batch_policy

FAKE_REWARD = reward_cfg(shaping=1.0, gamma=0.9, scale=10.0)
FIELDS = (
    "board",
    "globals",
    "actors",
    "kinds",
    "args",
    "actor_mask",
    "kind_mask",
    "arg_mask",
    "logp",
    "values",
    "rewards",
    "terminated",
    "truncated",
    "next_values",
    "last_value",
)


class _WriteCounter:
    """Proxy around a `Tensor` that counts `__setitem__` / `index_put_` scatter calls."""

    def __init__(self, tensor: torch.Tensor):
        self._t = tensor
        self.writes = 0

    def __setitem__(self, key, value):
        self.writes += 1
        self._t[key] = value

    def index_put_(self, indices, values, accumulate=False):
        self.writes += 1
        return self._t.index_put_(indices, values, accumulate=accumulate)

    def __getattr__(self, name):
        return getattr(self._t, name)


class _NoScalarIndexTensor(torch.Tensor):
    """`Tensor` subclass that records element-wise `tensor[r]` indexing on the host loop."""

    @staticmethod
    def __new__(cls, data: torch.Tensor, counter: list):
        t = torch.Tensor._make_subclass(cls, data.detach().clone(), require_grad=False)
        t._counter = counter
        return t

    def __getitem__(self, idx):
        self._counter.append(idx)
        return super().__getitem__(idx)


class _CountingStubNet(StubNet):
    def __init__(self):
        super().__init__()
        self.scalar_reads = []

    def act(self, feats, masks_fn, generator, greedy=False, stats=True):
        out = super().act(feats, masks_fn, generator, greedy=greedy, stats=stats)
        return ActOut(
            actors=_NoScalarIndexTensor(out.actors, self.scalar_reads),
            kinds=_NoScalarIndexTensor(out.kinds, self.scalar_reads),
            args=_NoScalarIndexTensor(out.args, self.scalar_reads),
            logp=out.logp,
            entropy=out.entropy,
            value=out.value,
            actor_mask=out.actor_mask,
            kind_mask=out.kind_mask,
            arg_mask=out.arg_mask,
        )


def _fake_obs(observer, t, parity):
    board = np.zeros((2, 4, 4), dtype=np.float32)
    board[0], board[1] = observer, t
    return board, np.array([observer, t, parity], dtype=np.float32)


class _SlotTruncEnv(FakeTwoSeatEnv):
    """`FakeTwoSeatEnv` whose truncation limit varies by seed so some slots truncate mid-rollout,
    some on the final row, and others never truncate in the same tick."""

    def reset(self, seed):
        # Even seeds (learner first) truncate after 2, 4, or never; odd seeds (opponent first)
        # truncate at action 4 on their first episode (s // 4 even) and never on the next (s // 4 odd).
        s = int(seed)
        if s % 2 == 1:
            self.truncate_at = 4 if (s // 4) % 2 == 0 else None
        else:
            self.truncate_at = [2, 4, None][(s // 2) % 3]
        super().reset(seed)


def test_batched_collector_zeros_stale_next_values_and_places_truncations_at_pre_increment_row():
    n, T = 4, 4

    def make_vec():
        vec = SelfPlayVecEnv(
            lambda opp: _SlotTruncEnv(k=3, turns=5),
            n,
            0,
            pool=pool_of(1),
            make_policy=lambda entry: fake_scorer,
            rng=np.random.default_rng(0),
            p_bot=0.0,
            reward=FAKE_REWARD,
        )
        vec.reset_all()
        return vec

    vec_ref, vec_got = make_vec(), make_vec()
    spec = FakeTwoSeatEnv().spec()
    buf_ref, buf_got = RolloutBuffer(T, n, spec), RolloutBuffer(T, n, spec)
    net_ref, net_got = StubNet(), StubNet()

    for rollout_idx, sentinel in enumerate((99.5, -42.25)):
        # Poison next_values before each rollout so forgetting to zero non-truncated rows shows up.
        buf_got.next_values.fill_(sentinel)
        s_ref = collect_rollout(net_ref, vec_ref, buf_ref, torch.Generator().manual_seed(rollout_idx))
        s_got = collect_rollout_interleaved(net_got, vec_got, buf_got, torch.Generator().manual_seed(rollout_idx))

        for k in FIELDS:
            assert torch.equal(getattr(buf_got, k), getattr(buf_ref, k)), f"rollout {rollout_idx} field {k}"
        assert sorted((s.env_index, s.seed, s.length, s.ret) for s in s_got) == sorted(
            (s.env_index, s.seed, s.length, s.ret) for s in s_ref
        )
        assert vec_got.state_dict() == vec_ref.state_dict()

        # Verify both truncated and non-truncated rows occurred and hold exact values.
        assert bool(buf_got.truncated.any()) and bool((~buf_got.truncated).any())
        assert torch.all(buf_got.next_values[~buf_got.truncated] == 0.0)
        assert torch.all(buf_got.next_values[buf_got.truncated] != sentinel)

    # Check exact hand-derived placement on a fresh rollout 1: slot 0 (seed 0, truncate_at=2)
    # truncates at t=1 (action 2); slot 1 (seed 1, truncate_at=4) truncates at t=0 (action 4).
    vec_h = make_vec()
    buf_h = RolloutBuffer(T, n, spec)
    buf_h.next_values.fill_(77.0)
    collect_rollout_interleaved(StubNet(), vec_h, buf_h, torch.Generator().manual_seed(0))
    assert bool(buf_h.truncated[0, 0]) is False and buf_h.next_values[0, 0].item() == 0.0
    assert bool(buf_h.truncated[1, 0]) is True
    assert buf_h.next_values[1, 0].item() == stub_value(*_fake_obs(0, 2, 0))
    assert bool(buf_h.truncated[0, 1]) is True
    assert buf_h.next_values[0, 1].item() == stub_value(*_fake_obs(1, 4, 1))
    assert bool(buf_h.truncated[1, 1]) is False and buf_h.next_values[1, 1].item() == 0.0


def test_batched_collector_scatters_done_and_cut_once_per_tick_and_reads_actions_in_batch():
    n, T = 6, 4
    vec = SelfPlayVecEnv(
        lambda opp: FakeTwoSeatEnv(k=3, turns=5, truncate_at=4),
        n,
        0,
        pool=pool_of(1),
        make_policy=lambda entry: fake_scorer,
        rng=np.random.default_rng(0),
        p_bot=0.0,
        reward=FAKE_REWARD,
    )
    vec.reset_all()
    spec = FakeTwoSeatEnv().spec()
    buf = RolloutBuffer(T, n, spec)

    ticks_with_done = 0
    ticks_with_cut = 0
    total_done = 0
    total_cut = 0
    inner_tick = vec.tick

    def counted_tick(actions):
        nonlocal ticks_with_done, ticks_with_cut, total_done, total_cut
        done = inner_tick(actions)
        if done:
            ticks_with_done += 1
            total_done += len(done)
        cut = [tr for tr in done if tr.truncated]
        if cut:
            ticks_with_cut += 1
            total_cut += len(cut)
        return done

    vec.tick = counted_tick

    rw_c = _WriteCounter(buf.rewards)
    term_c = _WriteCounter(buf.terminated)
    trunc_c = _WriteCounter(buf.truncated)
    nv_c = _WriteCounter(buf.next_values)
    buf.rewards, buf.terminated, buf.truncated, buf.next_values = rw_c, term_c, trunc_c, nv_c

    net = _CountingStubNet()
    collect_rollout_interleaved(net, vec, buf, torch.Generator().manual_seed(0))

    assert total_done == T * n and ticks_with_done < total_done, "fixture must batch multiple completions per tick"
    assert total_cut > ticks_with_cut > 0, "fixture must batch multiple truncations per tick"
    assert rw_c.writes == ticks_with_done, f"expected 1 rewards scatter/tick ({ticks_with_done}), got {rw_c.writes}"
    assert term_c.writes == ticks_with_done, f"expected 1 terminated scatter/tick ({ticks_with_done}), got {term_c.writes}"
    assert trunc_c.writes == ticks_with_done, f"expected 1 truncated scatter/tick ({ticks_with_done}), got {trunc_c.writes}"
    assert nv_c.writes == ticks_with_done + ticks_with_cut, (
        f"expected 1 zero-scatter/done-tick + 1 bootstrap-scatter/cut-tick ({ticks_with_done + ticks_with_cut}), got {nv_c.writes}"
    )
    assert net.scalar_reads == [], f"expected batched action readback (no per-slot indexing), got {len(net.scalar_reads)} reads"


@pytest.mark.parametrize("shaping", [0.0, 1.0])
@pytest.mark.parametrize("max_actions", [5000, 9])
def test_batched_collector_matches_lockstep_on_real_env(shaping, max_actions):
    size, cap, n, T = 6, 3, 6, 12
    snap_bot = {"snap_0": "economy", "snap_100": "rush"}
    choices = ["economy", "rush", 0, 1]
    rw = reward_cfg(shaping)

    def run_one(collector):
        vec = keyed_vec(
            lambda opp: ps.Env(size=size, turn_cap=cap, agent_seat="seed_parity", opponent=opp, max_agent_actions=max_actions, **rw),
            n,
            pool=pool_of(2),
            make_policy=lambda entry: bot_batch(snap_bot[entry.name]),
            choices=choices,
            pick=keyed_pick(len(choices)),
            seed_base=100,
            reward=rw,
        )
        vec.reset_all()
        net = StubNet(BotLearner(vec, lambda s: "rush" if s % 2 else "economy"))
        buf = RolloutBuffer(T, n, ps.Env(size=size).spec())
        snaps = []
        for _ in range(2):
            buf.next_values.fill_(123.0)
            stats = collector(net, vec, buf, torch.Generator().manual_seed(0))
            snaps.append(
                (
                    {k: getattr(buf, k).clone() for k in FIELDS},
                    sorted((s.env_index, s.seed, s.opponent, s.agent_seat, s.outcome, s.length, s.ret) for s in stats),
                    vec.state_dict(),
                )
            )
        return snaps

    ref = run_one(collect_rollout)
    got = run_one(collect_rollout_interleaved)
    for r, (a, b) in enumerate(zip(ref, got)):
        for k in FIELDS:
            assert torch.equal(a[0][k], b[0][k]), f"rollout {r} field {k}"
        assert a[1] == b[1], f"rollout {r} stats"
        assert a[2] == b[2], f"rollout {r} vec state"


def test_batched_collector_with_real_policynet_is_deterministic_on_cpu():
    size, cap = 6, 3
    spec = ps.Env(size=size).spec()

    def run_once():
        torch.manual_seed(11)
        net = PolicyNet(spec, width=8, blocks=1)
        torch.manual_seed(12)
        snap = PolicyNet(spec, width=8, blocks=1)
        opp_gen = torch.Generator().manual_seed(5)
        rw = reward_cfg(1.0)
        vec = SelfPlayVecEnv(
            lambda opp: ps.Env(size=size, turn_cap=cap, agent_seat="seed_parity", opponent=opp, max_agent_actions=10, **rw),
            4,
            0,
            pool=pool_of(1),
            make_policy=lambda entry: net_batch_policy(snap, opp_gen),
            rng=np.random.default_rng(2),
            p_bot=0.3,
            reward=rw,
        )
        vec.reset_all()
        buf = RolloutBuffer(10, 4, spec)
        buf.next_values.fill_(55.0)
        stats = collect_rollout_interleaved(net, vec, buf, torch.Generator().manual_seed(9))
        return {k: getattr(buf, k).clone() for k in FIELDS}, stats, vec.state_dict()

    a, b = run_once(), run_once()
    for k in FIELDS:
        assert torch.equal(a[0][k], b[0][k]), k
    assert a[1] == b[1]
    assert a[2] == b[2]
