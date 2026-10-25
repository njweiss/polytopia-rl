"""Milestone 11 tests 7: `SelfPlayVecEnv` ("Self-play": execution, reward consistency).

Parity: with every opponent a snapshot whose policy plays exactly the Rust economy bot (through
`Env.bot_action`, docs/QUESTIONS.md Q23), the two-seat path must give the same observations, masks,
rewards, dones and episode stats as the milestone 10 bot-mode `SyncVecEnv` on the same seeds, with
shaping off and on (gamma < 1, so the reward must span the whole opponent turn exactly as the M9
formula does).

Hand-computed rewards on `FakeTwoSeatEnv` (k = 3 actions per turn, potential_scale 10, gamma 0.9):
with phi = (mine - theirs) / 10 (0 when terminal) and r = terminal + gamma * phi' - phi, in float32
as the Rust env computes it.
"""

import functools

import numpy as np
import polytopia_sim as ps
import pytest
import torch
from rl_helpers import END_TURN, MOVE
from sp_helpers import FakeTwoSeatEnv, bot_batch, fake_scorer

from polytopia_rl.model import PolicyNet
from polytopia_rl.selfplay import SelfPlayVecEnv, SnapshotPool, net_batch_policy
from polytopia_rl.vec_env import TRAIN_SEED_LIMIT, SyncVecEnv

SIZE = 6
CAP = 4


def reward_cfg(shaping):
    return dict(terminal_weight=1.0, shaping_weight=shaping, gamma=0.999, potential_scale=1000.0)


def env_kw(shaping):
    return dict(size=SIZE, turn_cap=CAP, agent_seat="seed_parity", **reward_cfg(shaping))


def pool_of(n):
    pool = SnapshotPool()
    for s in range(n):
        pool.add({"w": torch.zeros(1)}, s * 100)
    return pool


def sp_env(num_envs, shaping, pool, make_policy, p_bot=0.0, seed_base=10, rng_seed=0):
    kw = env_kw(shaping)
    return SelfPlayVecEnv(
        lambda opp: ps.Env(**kw, opponent=opp),
        num_envs,
        seed_base,
        pool=pool,
        make_policy=make_policy,
        rng=np.random.default_rng(rng_seed),
        p_bot=p_bot,
        reward=reward_cfg(shaping),
    )


def choose(vec, rng):
    """A random legal path per env, through the vec env's mask helpers."""
    am = vec.actor_masks()
    actors = np.array([rng.choice(np.flatnonzero(m)) for m in am], dtype=np.int64)
    km = vec.kind_masks(actors)
    kinds = np.array([rng.choice(np.flatnonzero(m)) for m in km], dtype=np.int64)
    xm = vec.arg_masks(actors, kinds)
    args = np.array([rng.choice(np.flatnonzero(m)) if m.any() else 0 for m in xm], dtype=np.int64)
    return actors, kinds, args


def stat_key(s):
    return (s.env_index, s.seed, s.agent_seat, s.outcome, s.length, s.ret)


@pytest.mark.parametrize("shaping", [0.0, 1.0])
def test_scripted_snapshot_matches_the_bot_mode_env(shaping):
    e = 4  # seeds 10..13: learner seats 0, 1, 0, 1
    ref = SyncVecEnv([functools.partial(ps.Env, **env_kw(shaping), opponent="economy") for _ in range(e)], 10, opponents=["economy"] * e)
    sp = sp_env(e, shaping, pool_of(1), lambda entry: bot_batch("economy"))
    for a, b in zip(ref.reset_all().__dict__.values(), sp.reset_all().__dict__.values()):
        np.testing.assert_array_equal(a, b)
    rng = np.random.default_rng(7)
    episodes = 0
    for t in range(250):
        np.testing.assert_array_equal(sp.actor_masks(), ref.actor_masks(), err_msg=f"t={t}")
        actors, kinds, args = choose(ref, rng)
        np.testing.assert_array_equal(sp.kind_masks(actors), ref.kind_masks(actors))
        np.testing.assert_array_equal(sp.arg_masks(actors, kinds), ref.arg_masks(actors, kinds))
        a, b = ref.step(actors, kinds, args), sp.step(actors, kinds, args)
        np.testing.assert_array_equal(b.reward, a.reward, err_msg=f"t={t}")
        np.testing.assert_array_equal(b.terminated, a.terminated)
        np.testing.assert_array_equal(b.truncated, a.truncated)
        for x, y in zip(ref.observations(), sp.observations()):
            np.testing.assert_array_equal(y, x, err_msg=f"t={t}")
        sa, sb = ref.episode_stats(), sp.episode_stats()
        assert [stat_key(s) for s in sb] == [stat_key(s) for s in sa], f"t={t}"
        assert all(s.opponent == "snap_0" for s in sb)
        episodes += len(sa)
    assert episodes >= 8, "several episodes on both seats"


def test_bot_opponents_use_the_bot_mode_env_and_are_logged():
    e = 3
    kw = env_kw(1.0)
    sp = sp_env(e, 1.0, pool_of(1), lambda entry: bot_batch("economy"), p_bot=1.0)
    sp.reset_all()
    rng = np.random.default_rng(3)
    actions = [[] for _ in range(e)]
    finished = {i: [] for i in range(e)}  # per slot, the learner actions of finished episodes
    seen = 0
    for _ in range(200):
        acts = choose(sp, rng)
        out = sp.step(*acts)
        for i in range(e):
            actions[i].append(tuple(int(x[i]) for x in acts))
            if out.terminated[i] or out.truncated[i]:
                finished[i].append(actions[i])
                actions[i] = []
        for s in sp.episode_stats():
            acts_i = finished[s.env_index].pop(0)
            seen += 1
            assert s.opponent in ("economy", "rush")
            env = ps.Env(**kw, opponent=s.opponent)
            env.reset(s.seed)
            ret = 0.0
            for a in acts_i:
                r, _, term, trunc = env.step_factored(*a)
                ret += float(r)
            assert term or trunc
            assert s.length == len(acts_i) and s.ret == ret
            assert s.agent_seat == s.seed % 2
            assert s.outcome == {f"win_{s.agent_seat}": "win", f"win_{1 - s.agent_seat}": "loss"}.get(env.info()["outcome"], "draw")
    assert all(not v for v in finished.values()) and seen >= 5


class Counter:
    """make_policy whose batch policies record every call: (phase, entry step, envs)."""

    def __init__(self):
        self.phase = 0
        self.calls = []

    def __call__(self, entry):
        inner = bot_batch("economy" if entry.step == 0 else "rush")

        def policy(envs):
            for env in envs:
                info = env.info()
                assert info["outcome"] is None, "only running games are asked to move"
                assert info["current_player"] == 1 - info["seed"] % 2, "only the opponent's move"
            assert len({id(env) for env in envs}) == len(envs)
            self.calls.append((self.phase, entry.step, [id(env) for env in envs]))
            return inner(envs)

        return policy


def check_batched(calls, phase):
    by_entry = {}
    for ph, step, ids in calls:
        if ph == phase:
            by_entry.setdefault(step, []).append(ids)
    for step, lists in by_entry.items():
        appearances = {}
        for ids in lists:
            for i in ids:
                appearances[i] = appearances.get(i, 0) + 1
        # One call per round, and a slot that still has to move is in every round's call.
        assert len(lists) == max(appearances.values()), (phase, step, [len(x) for x in lists])


def test_opponent_moves_are_batched_per_snapshot_and_seat_follows_seed_parity():
    e = 8
    counter = Counter()
    sp = sp_env(e, 0.0, pool_of(2), counter, seed_base=20)
    sp.reset_all()
    # Learner seat is seed % 2; the opponent has already moved where the learner is seat 1.
    odd = set()
    for i in range(e):
        info = sp.envs[i].info()
        seed = info["seed"]
        assert seed == 20 + i
        assert info["current_player"] == seed % 2 and info["turn"] == 0
        assert (info["agent_actions"] > 0) == (seed % 2 == 1)
        if seed % 2:
            odd.add(id(sp.envs[i]))
    check_batched(counter.calls, 0)
    first = {}
    for ph, step, ids in counter.calls:
        if ph == 0:
            first.setdefault(step, set(ids))
    assert set().union(*first.values()) == odd, "the first round moves every seat-1 slot at once"
    rng = np.random.default_rng(0)
    checked = 0
    for t in range(1, 120):
        counter.phase = t
        out = sp.step(*choose(sp, rng))
        if not (out.terminated.any() or out.truncated.any()):
            check_batched(counter.calls, t)
            checked += 1
        for i in range(e):
            info = sp.envs[i].info()
            assert info["outcome"] is None and info["current_player"] == info["seed"] % 2
    assert checked > 20
    assert len({step for _, step, _ in counter.calls}) == 2, "both snapshots played"


def test_two_identical_runs_are_identical():
    spec = ps.Env(size=SIZE).spec()
    nets = {}
    for s in (0, 100):
        torch.manual_seed(s)
        nets[s] = PolicyNet(spec, width=8, blocks=1)

    def run():
        gen = torch.Generator().manual_seed(4)
        sp = sp_env(4, 1.0, pool_of(2), lambda entry: net_batch_policy(nets[entry.step], gen), p_bot=0.3, rng_seed=5)
        sp.reset_all()
        rng = np.random.default_rng(1)
        out = []
        for _ in range(60):
            st = sp.step(*choose(sp, rng))
            out.append((st.reward.tolist(), st.terminated.tolist(), sp.observations()[1].tolist()))
            out.append([(s.seed, s.opponent, s.outcome, s.length, s.ret) for s in sp.episode_stats()])
        return out

    assert run() == run()


def test_training_seeds_only():
    with pytest.raises(ValueError):
        sp_env(2, 0.0, pool_of(1), lambda entry: bot_batch("economy"), seed_base=TRAIN_SEED_LIMIT - 1)


# ---------------------------------------------------------------- FakeTwoSeatEnv


def fake_vec(seed_stride=1, **fake_kw):
    return SelfPlayVecEnv(
        lambda opp: FakeTwoSeatEnv(**fake_kw),
        1,
        0,
        seed_stride,
        pool=pool_of(1),
        make_policy=lambda entry: fake_scorer,
        rng=np.random.default_rng(0),
        p_bot=0.0,
        reward=dict(terminal_weight=1.0, shaping_weight=1.0, gamma=0.9, potential_scale=10.0),
    )


G = 3 * 16  # the fake's global actor (size 4)
f32 = np.float32


def m9(before, after, terminal=0.0, done=False):
    """The M9 reward in float32: before/after are (mine, theirs) scores."""
    phi = lambda s: (f32(s[0]) - f32(s[1])) / f32(10.0)  # noqa: E731
    after_phi = f32(0.0) if done else phi(after)
    return f32(1.0) * f32(terminal) + f32(1.0) * (f32(0.9) * after_phi - phi(before))


def step1(vec, a):
    return vec.step(np.array([a[0]]), np.array([a[1]]), np.array([a[2]]))


def test_rewards_span_the_whole_opponent_turn():
    vec = fake_vec(k=3, turns=1)
    vec.reset_all()  # seed 0: learner is seat 0 and moves first
    board, glob = vec.observations()
    assert glob[0].tolist() == [0.0, 0.0, 0.0]
    out = [step1(vec, a) for a in [(0, MOVE, 1), (0, MOVE, 0), (G, END_TURN, 0)]]
    want = [
        m9((0, 0), (1, 0)),
        m9((1, 0), (1, 0)),
        # The opponent scores twice and ends the game (turns=1): loss, 1 point against 2.
        m9((1, 0), (1, 2), terminal=-1.0, done=True),
    ]
    assert [o.reward[0] for o in out] == want
    assert [bool(o.terminated[0]) for o in out] == [False, False, True]
    (s,) = vec.episode_stats()
    assert (s.seed, s.agent_seat, s.outcome, s.length, s.opponent) == (0, 0, "loss", 3, "snap_0")
    assert s.ret == pytest.approx(float(sum(want)))
    # Auto-reset to seed 1: learner is seat 1, so the opponent's three actions come first.
    board, glob = vec.observations()
    assert glob[0].tolist() == [1.0, 3.0, 1.0]
    out = [step1(vec, a) for a in [(0, MOVE, 1), (0, MOVE, 1), (G, END_TURN, 0)]]
    want = [
        m9((0, 2), (1, 2)),
        m9((1, 2), (2, 2)),
        m9((2, 2), (2, 2), terminal=0.0, done=True),  # seat 1 ends turn 1: draw
    ]
    assert [o.reward[0] for o in out] == want
    (s,) = vec.episode_stats()
    assert (s.seed, s.agent_seat, s.outcome, s.length) == (1, 1, "draw", 3)


def test_truncation_in_the_middle_of_the_opponent_turn_finishes():
    vec = fake_vec(k=3, turns=5, truncate_at=4)
    vec.reset_all()
    out = [step1(vec, a) for a in [(0, MOVE, 0), (0, MOVE, 0), (G, END_TURN, 0)]]
    assert [bool(o.truncated[0]) for o in out] == [False, False, True]
    assert not any(bool(o.terminated[0]) for o in out)
    # The opponent scored once before the cut.
    assert out[2].reward[0] == m9((0, 0), (0, 1))
    (s,) = vec.episode_stats()
    assert (s.seed, s.length) == (0, 3)
    _, glob = vec.observations()
    assert glob[0].tolist() == [1.0, 3.0, 1.0], "next episode (seed 1) after the opponent's first turn"


def test_truncation_on_the_learner_move_keeps_its_final_observation():
    vec = fake_vec(seed_stride=2, k=3, turns=5, truncate_at=2)  # seeds 0, 2, ...: learner seat 0
    vec.reset_all()
    a = step1(vec, (0, MOVE, 1))
    b = step1(vec, (0, MOVE, 1))
    assert not a.truncated[0] and b.truncated[0] and not b.terminated[0]
    assert b.final_globals[0].tolist() == [0.0, 2.0, 0.0]
    assert np.all(b.final_board[0, 1] == 2.0) and np.all(b.final_board[0, 0] == 0.0)
    assert np.all(a.final_board == 0)
    assert b.reward[0] == m9((1, 0), (2, 0))
    (s,) = vec.episode_stats()
    assert (s.seed, s.length) == (0, 2)
    _, glob = vec.observations()
    assert glob[0].tolist() == [0.0, 0.0, 0.0], "next episode, seed 2"
