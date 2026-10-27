"""Milestone 13 tests 1, 3, 4 and 6: `collect_rollout_interleaved` ("Collecting a rollout",
"Equivalence").

Fake traces (sp_helpers.FakeTwoSeatEnv, k = 3 actions per turn, the opponent plays `fake_scorer`, the
stub learner plays Move 1 while legal, else EndTurn): rows, rewards (M9 in float32, gamma 0.9,
potential_scale 10), bootstrap values and truncation values are written out by hand from the fake's
rules. The stub's log-probs and values are fixed functions of the observation bytes
(il_helpers.stub_logp / stub_value), so the expected numbers are those functions of the hand-derived
observations.

Equivalence (the core test) on the real env: the learner plays a Rust bot per slot (`BotLearner`),
snapshot opponents are batch policies on `Env.bot_action`, and every episode's opponent is a fixed
function of its seed (`KeyedVec`), so per slot the decisions do not depend on the slot order. Then
today's `collect_rollout` and `collect_rollout_interleaved` must fill identical buffers ([t, i] is
slot i's t-th transition in both), return the same episode stats as multisets, and leave the vec env in
the same state, over three consecutive rollouts with one reused buffer per collector.
"""

import copy

import numpy as np
import polytopia_sim as ps
import pytest
import torch
from il_helpers import BotLearner, StubNet, keyed_pick, keyed_vec, pool_of, reward_cfg, stub_logp, stub_value
from rl_helpers import END_TURN, MOVE
from sp_helpers import FakeTwoSeatEnv, bot_batch, fake_scorer

from polytopia_rl.model import PolicyNet
from polytopia_rl.ppo import RolloutBuffer, collect_rollout, collect_rollout_interleaved
from polytopia_rl.selfplay import SelfPlayVecEnv, net_batch_policy

G = 3 * 16
f32 = np.float32
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


def m9(before, after, terminal=0.0, done=False):
    phi = lambda s: (f32(s[0]) - f32(s[1])) / f32(10.0)  # noqa: E731
    after_phi = f32(0.0) if done else phi(after)
    return f32(1.0) * f32(terminal) + f32(1.0) * (f32(0.9) * after_phi - phi(before))


def fake_obs(observer, t, parity):
    board = np.zeros((2, 4, 4), dtype=np.float32)
    board[0], board[1] = observer, t
    return board, np.array([observer, t, parity], dtype=np.float32)


def fake_spec():
    return FakeTwoSeatEnv().spec()


def fake_vec(num_envs, seed_stride=1, make_policy=None, pool=None, **fake_kw):
    return SelfPlayVecEnv(
        lambda opp: FakeTwoSeatEnv(**fake_kw),
        num_envs,
        0,
        seed_stride,
        pool=pool or pool_of(1),
        make_policy=make_policy or (lambda entry: fake_scorer),
        rng=np.random.default_rng(0),
        p_bot=0.0,
        reward=FAKE_REWARD,
    )


def snapshot(buffer, stats, vec):
    return (
        {k: getattr(buffer, k).clone() for k in FIELDS},
        sorted((s.env_index, s.seed, s.opponent, s.agent_seat, s.outcome, s.length, s.ret) for s in stats),
        vec.state_dict(),
    )


def assert_same(a, b, what=""):
    for k in FIELDS:
        assert torch.equal(a[0][k], b[0][k]), f"{what}: buffer field {k} differs"
    assert a[1] == b[1], f"{what}: episode stats differ"
    assert a[2] == b[2], f"{what}: vec env state differs"


class TickCount:
    """Wraps `vec.tick` to count ticks and the learner actions each slot receives."""

    def __init__(self, vec):
        self.n = 0
        self.actions = [0] * vec.num_envs
        inner = vec.tick

        def tick(learner_actions):
            self.n += 1
            for i in learner_actions:
                self.actions[i] += 1
            return inner(learner_actions)

        vec.tick = tick


# ---------------------------------------------------------------- fake traces


MOVE_ROW = (0, MOVE, 1)
END_ROW = (G, END_TURN, 0)


def check_rows(buffer, i, obs, actions, rewards, terminated, truncated=None, next_values=None):
    T = buffer.num_steps
    for t in range(T):
        board, glob = obs[t]
        assert buffer.globals[t, i].tolist() == glob.tolist(), (i, t)
        assert torch.equal(buffer.board[t, i], torch.from_numpy(board)), (i, t)
        a, k, x = actions[t]
        assert (buffer.actors[t, i].item(), buffer.kinds[t, i].item(), buffer.args[t, i].item()) == (a, k, x), (i, t)
        assert np.flatnonzero(buffer.actor_mask[t, i].numpy()).tolist() == [a], (i, t)
        assert np.flatnonzero(buffer.kind_mask[t, i].numpy()).tolist() == [k], (i, t)
        assert np.flatnonzero(buffer.arg_mask[t, i].numpy()).tolist() == ([0, 1] if k == MOVE else []), (i, t)
        assert buffer.logp[t, i].item() == stub_logp(board, glob), (i, t)
        assert buffer.values[t, i].item() == stub_value(board, glob), (i, t)
        assert buffer.rewards[t, i].item() == rewards[t], (i, t, buffer.rewards[t, i].item(), rewards[t])
        assert bool(buffer.terminated[t, i]) == terminated[t], (i, t)
        assert bool(buffer.truncated[t, i]) == (truncated or [False] * T)[t], (i, t)
        assert buffer.next_values[t, i].item() == (next_values or [0.0] * T)[t], (i, t)


def test_rows_rewards_and_bootstrap_by_hand_across_an_episode_end():
    # Seeds 0 and 1 (learner seats 0 and 1), turns=1: each episode is one turn per seat, so the
    # learner's third action ends it (a 2-2 draw); next seeds 2 (learner first) and 3 (opponent first).
    vec = fake_vec(2, k=3, turns=1)
    vec.reset_all()
    net = StubNet()
    buffer = RolloutBuffer(4, 2, fake_spec())
    stats = collect_rollout_interleaved(net, vec, buffer, torch.Generator().manual_seed(0))
    acts = [MOVE_ROW, MOVE_ROW, END_ROW, MOVE_ROW]
    check_rows(
        buffer,
        0,
        [fake_obs(0, 0, 0), fake_obs(0, 1, 0), fake_obs(0, 2, 0), fake_obs(0, 0, 0)],
        acts,
        [m9((0, 0), (1, 0)), m9((1, 0), (2, 0)), m9((2, 0), (2, 2), 0.0, True), m9((0, 0), (1, 0))],
        [False, False, True, False],
    )
    check_rows(
        buffer,
        1,
        [fake_obs(1, 3, 1), fake_obs(1, 4, 1), fake_obs(1, 5, 1), fake_obs(1, 3, 1)],
        acts,
        [m9((0, 2), (1, 2)), m9((1, 2), (2, 2)), m9((2, 2), (2, 2), 0.0, True), m9((0, 2), (1, 2))],
        [False, False, True, False],
    )
    assert buffer.last_value.tolist() == [stub_value(*fake_obs(0, 1, 0)), stub_value(*fake_obs(1, 4, 1))]
    got = sorted((s.env_index, s.seed, s.agent_seat, s.outcome, s.length, s.opponent) for s in stats)
    assert got == [(0, 0, 0, "draw", 3, "snap_0"), (1, 1, 1, "draw", 3, "snap_0")]
    assert sum(net.act_sizes) == 8, "exactly T learner rows per slot"


def test_rollout_ends_only_when_every_slot_is_on_the_learner_move_and_finished_slots_idle():
    # T=3: both episodes end at the third transition. Slot 0 restarts learner-first (seed 2) and must
    # not act again; slot 1 restarts opponent-first (seed 3) and the rollout must run on until that
    # opponent has played its whole turn.
    vec = fake_vec(2, k=3, turns=1)
    vec.reset_all()
    ticks = TickCount(vec)
    net = StubNet()
    buffer = RolloutBuffer(3, 2, fake_spec())
    collect_rollout_interleaved(net, vec, buffer, torch.Generator().manual_seed(0))
    assert ticks.actions == [3, 3]
    assert sum(net.act_sizes) == 6
    assert vec.learner_slots() == [0, 1]
    assert vec.observations()[1].tolist() == [[0.0, 0.0, 0.0], [1.0, 3.0, 1.0]]
    assert buffer.last_value.tolist() == [stub_value(*fake_obs(0, 0, 0)), stub_value(*fake_obs(1, 3, 1))]
    # Ticks: 3 with learner actions, then slot 0's opponent needs 2 more and slot 1's new opponent 3.
    assert ticks.n == 6


def test_truncation_values_come_from_the_final_observation_and_are_reset_between_rollouts():
    vec = fake_vec(1, k=3, turns=5, truncate_at=4)
    vec.reset_all()
    net = StubNet()
    buffer = RolloutBuffer(3, 1, fake_spec())
    gen = torch.Generator().manual_seed(0)
    # Rollout 1: Move, Move, EndTurn; the opponent's first move (scoring) is action 4 and truncates.
    s1 = collect_rollout_interleaved(net, vec, buffer, gen)
    check_rows(
        buffer,
        0,
        [fake_obs(0, 0, 0), fake_obs(0, 1, 0), fake_obs(0, 2, 0)],
        [MOVE_ROW, MOVE_ROW, END_ROW],
        [m9((0, 0), (1, 0)), m9((1, 0), (2, 0)), m9((2, 0), (2, 1))],
        [False, False, False],
        [False, False, True],
        [0.0, 0.0, stub_value(*fake_obs(1, 4, 0))],
    )
    assert buffer.last_value.tolist() == [stub_value(*fake_obs(1, 3, 1))], "seed 1 after its opponent's turn"
    assert [(s.seed, s.length) for s in s1] == [(0, 3)]
    # Rollout 2 (same buffer): the learner's first move is action 4 of seed 1 and truncates; then
    # seed 2 runs untruncated, so its rows must not keep rollout 1's truncation value.
    s2 = collect_rollout_interleaved(net, vec, buffer, gen)
    check_rows(
        buffer,
        0,
        [fake_obs(1, 3, 1), fake_obs(0, 0, 0), fake_obs(0, 1, 0)],
        [MOVE_ROW, MOVE_ROW, MOVE_ROW],
        [m9((0, 2), (1, 2)), m9((0, 0), (1, 0)), m9((1, 0), (2, 0))],
        [False, False, False],
        [True, False, False],
        [stub_value(*fake_obs(1, 4, 1)), 0.0, 0.0],
    )
    assert [(s.seed, s.length) for s in s2] == [(1, 1)]


# ---------------------------------------------------------------- equivalence on the real env

SIZE, CAP = 6, 3
N, T = 8, 16
SNAP_BOT = {"snap_0": "economy", "snap_100": "rush"}
CHOICES = ["economy", "rush", 0, 1]


def learner_bot(seed):
    return "rush" if seed % 3 else "economy"


class Recorded:
    """Delegates to a real env and logs resets and episode ends."""

    def __init__(self, env, kind, log):
        self._env, self._kind, self._log = env, kind, log

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, seed):
        self._log.append(("reset", int(seed), self._kind))
        self._env.reset(seed)

    def step_factored(self, actor, kind, arg):
        out = self._env.step_factored(actor, kind, arg)
        if out[2] or out[3]:
            info = self._env.info()
            self._log.append(("end", info["seed"], self._kind, bool(out[2]), bool(out[3]), info["turn"]))
        return out


def real_vec(shaping, max_actions, log, make_policy=None, num_envs=N):
    rw = reward_cfg(shaping)

    def env_fn(opp):
        return Recorded(ps.Env(size=SIZE, turn_cap=CAP, agent_seat="seed_parity", opponent=opp, max_agent_actions=max_actions, **rw), opp, log)

    return keyed_vec(
        env_fn,
        num_envs,
        pool=pool_of(2),
        make_policy=make_policy or (lambda entry: bot_batch(SNAP_BOT[entry.name])),
        choices=CHOICES,
        pick=keyed_pick(len(CHOICES)),
        seed_base=40,
        reward=rw,
    )


def run_path(collector, shaping, max_actions, rollouts=3):
    log = []
    vec = real_vec(shaping, max_actions, log)
    vec.reset_all()
    log.append(("rollouts start",))
    net = StubNet(BotLearner(vec, learner_bot))
    buffer = RolloutBuffer(T, N, ps.Env(size=SIZE).spec())
    out = []
    for _ in range(rollouts):
        stats = collector(net, vec, buffer, torch.Generator().manual_seed(0))
        out.append(snapshot(buffer, stats, vec))
    return out, log


@pytest.mark.parametrize("shaping", [0.0, 1.0])
@pytest.mark.parametrize("max_actions", [5000, 9])
def test_interleaved_rollout_equals_the_lockstep_rollout_per_slot(shaping, max_actions):
    ref, log = run_path(collect_rollout, shaping, max_actions)
    got, _ = run_path(collect_rollout_interleaved, shaping, max_actions)
    for r, (a, b) in enumerate(zip(ref, got)):
        assert_same(a, b, f"rollout {r}")
    # The fixture covers what the brief asks for (checked on the reference run, which the
    # interleaved run equals).
    after = log[log.index(("rollouts start",)) :]
    ends = [e for e in after if e[0] == "end"]
    two_seat_ends = [e for e in ends if e[2] == "none"]
    assert any(e[0] == "reset" and e[1] % 2 == 1 and e[2] == "none" for e in after), "an opponent-first reset"
    stats = [s for _, st, _ in ref for s in st]
    pairs = {(s[3], s[2] in SNAP_BOT) for s in stats}
    assert pairs == {(0, True), (0, False), (1, True), (1, False)}, pairs
    assert {s[2] for s in stats} == {"economy", "rush", "snap_0", "snap_100"}
    if max_actions == 5000:
        assert any(e[3] and e[5] < CAP for e in two_seat_ends), "a capital capture"
        assert any(e[3] and e[5] >= CAP for e in two_seat_ends), "an end at the turn cap"
        assert any(e[3] and e[5] < CAP for e in ends if e[2] != "none")
    else:
        assert any(e[4] for e in two_seat_ends), "a truncation in a snapshot slot"
        assert any(bool(st[0]["truncated"].any()) for st in ref)
        assert any(bool((st[0]["next_values"] != 0).any()) for st in ref)


# ---------------------------------------------------------------- call counts


class VarK(FakeTwoSeatEnv):
    """Turn length `k = 2 + seed % 4`, so slots fall out of phase."""

    def reset(self, seed):
        self.k = 2 + int(seed) % 4
        super().reset(seed)


class Counted:
    """make_policy recording `(tick, entry name, batch size)` per call."""

    def __init__(self, clock):
        self.clock = clock
        self.calls = []

    def __call__(self, entry):
        def policy(envs):
            self.calls.append((self.clock(), entry.name, len(envs)))
            return fake_scorer(envs)

        return policy


def test_interleaved_calls_are_bounded_per_tick_and_fewer_than_lockstep():
    n, steps = 8, 12

    def build(clock):
        counted = Counted(clock)
        vec = keyed_vec(lambda opp: VarK(turns=3), n, pool=pool_of(2), make_policy=counted, choices=[0, 1], pick=keyed_pick(2), reward=FAKE_REWARD)
        vec.reset_all()
        counted.calls.clear()
        return vec, counted

    # Lockstep reference: opponent rounds per learner step.
    vec_l, counted_l = build(lambda: 0)
    net_l = StubNet()
    buf_l = RolloutBuffer(steps, n, fake_spec())
    collect_rollout(net_l, vec_l, buf_l, torch.Generator())
    # Interleaved: count per tick.
    clock = {"tick": 0}
    vec_i, counted_i = build(lambda: clock["tick"])
    inner = vec_i.tick

    def tick(actions):
        clock["tick"] += 1
        return inner(actions)

    vec_i.tick = tick
    learner_ticks = []

    class Net(StubNet):
        def act(self, *a, **kw):
            learner_ticks.append(clock["tick"])
            return super().act(*a, **kw)

    buf_i = RolloutBuffer(steps, n, fake_spec())
    collect_rollout_interleaved(Net(), vec_i, buf_i, torch.Generator())
    for k in FIELDS:
        assert torch.equal(getattr(buf_l, k), getattr(buf_i, k)), k
    # At most one learner call before each tick, at most one call per snapshot per tick.
    assert len(learner_ticks) == len(set(learner_ticks))
    per_tick = {}
    for t, name, _ in counted_i.calls:
        per_tick.setdefault(t, []).append(name)
    assert all(len(v) == len(set(v)) <= 2 for v in per_tick.values()), per_tick
    opp_l, opp_i = len(counted_l.calls), len(counted_i.calls)
    rounds = opp_l / steps
    assert rounds > 2, f"the fixture needs several opponent calls per learner step (got {rounds})"
    assert opp_i < opp_l, (opp_i, opp_l)
    assert opp_i + len(learner_ticks) < opp_l + steps, "fewer calls per rollout in total"
    mean_l = np.mean([b for _, _, b in counted_l.calls])
    mean_i = np.mean([b for _, _, b in counted_i.calls])
    assert mean_i > mean_l, (mean_i, mean_l)


# ---------------------------------------------------------------- determinism and resume


def det_setup(seed_base=0):
    spec = ps.Env(size=SIZE).spec()
    nets = {}
    for s in (0, 100, 1):
        torch.manual_seed(s + 17)
        nets[s] = PolicyNet(spec, width=8, blocks=1)
    opp_gen = torch.Generator().manual_seed(7)
    rw = reward_cfg(1.0)
    vec = SelfPlayVecEnv(
        lambda opp: ps.Env(size=SIZE, turn_cap=CAP, agent_seat="seed_parity", opponent=opp, **rw),
        6,
        seed_base,
        pool=pool_of(2),
        make_policy=lambda entry: net_batch_policy(nets[entry.step], opp_gen),
        rng=np.random.default_rng(3),
        p_bot=0.4,
        reward=rw,
    )
    return vec, nets[1], opp_gen, RolloutBuffer(16, 6, spec)


def test_two_identical_interleaved_runs_are_identical_and_a_resume_continues_exactly():
    def run():
        vec, net, opp_gen, buffer = det_setup()
        vec.reset_all()
        gen = torch.Generator().manual_seed(4)
        first = snapshot(buffer, collect_rollout_interleaved(net, vec, buffer, gen), vec)
        saved = (vec.state_dict(), copy.deepcopy(vec.rng.bit_generator.state), gen.get_state(), opp_gen.get_state())
        second = snapshot(buffer, collect_rollout_interleaved(net, vec, buffer, gen), vec)
        return first, second, saved

    a1, a2, saved = run()
    b1, b2, _ = run()
    assert_same(a1, b1, "run 1 vs run 2, rollout 1")
    assert_same(a2, b2, "run 1 vs run 2, rollout 2")
    assert not torch.equal(a1[0]["board"], a2[0]["board"])
    # Resume from the rollout boundary into fresh objects.
    state, rng_state, gen_state, opp_state = saved
    vec, net, opp_gen, buffer = det_setup(seed_base=state["seed_base"])
    vec.load_state_dict(state)
    vec.rng.bit_generator.state = rng_state
    opp_gen.set_state(opp_state)
    gen = torch.Generator()
    gen.set_state(gen_state)
    c2 = snapshot(buffer, collect_rollout_interleaved(net, vec, buffer, gen), vec)
    assert_same(a2, c2, "resumed rollout 2")
