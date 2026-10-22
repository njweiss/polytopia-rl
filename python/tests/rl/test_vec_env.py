"""Milestone 10 tests 2: SyncVecEnv on real `polytopia_sim.Env`s.

Every check replays the same actions on independent "twin" `Env`s and compares; the seed sequence
is the brief's formula `seed_base + i * stride + k * num_envs * stride` ("Environment wiring").
"""

import functools

import numpy as np
import polytopia_sim as ps
import pytest
from rl_helpers import END_TURN, TRAIN

from polytopia_rl.config import PPOConfig
from polytopia_rl.vec_env import TRAIN_SEED_LIMIT, SyncVecEnv, make_vec_env

SIZE = 6
N = SIZE * SIZE
KW = dict(size=SIZE, opponent="random", agent_seat="seed_parity", shaping_weight=1.0, gamma=0.999)


def env_fns(n, **kw):
    return [functools.partial(ps.Env, **{**KW, **kw}) for _ in range(n)]


class Twins:
    """One independent Env per vec slot, reset and stepped by the test with the expected seeds."""

    def __init__(self, n, seed_base, stride, **kw):
        self.envs = [ps.Env(**{**KW, **kw}) for _ in range(n)]
        self.n, self.base, self.stride = n, seed_base, stride
        self.k = [0] * n
        for i in range(n):
            self.envs[i].reset(self.seed(i))

    def seed(self, i):
        return self.base + i * self.stride + self.k[i] * self.n * self.stride


def vec_choice(vec, choose):
    """Pick one path per env with `choose(mask getter)`, through the vec env's mask helpers."""
    am = vec.actor_masks()
    actors = np.array([choose("actor", i, am[i], None, None) for i in range(vec.num_envs)], dtype=np.int64)
    km = vec.kind_masks(actors)
    kinds = np.array([choose("kind", i, km[i], None, None) for i in range(vec.num_envs)], dtype=np.int64)
    xm = vec.arg_masks(actors, kinds)
    args = np.array([choose("arg", i, xm[i], actors[i], kinds[i]) for i in range(vec.num_envs)], dtype=np.int64)
    return actors, kinds, args


def end_turn_chooser(stage, i, mask, actor, kind):
    if stage == "actor":
        return len(mask) - 1 if mask[-1] else int(np.flatnonzero(mask)[0])
    if stage == "kind":
        return END_TURN if mask[END_TURN] else int(np.flatnonzero(mask)[0])
    nz = np.flatnonzero(mask)
    return int(nz[0]) if nz.size else 0


def random_chooser(rng):
    def choose(stage, i, mask, actor, kind):
        nz = np.flatnonzero(mask)
        return int(rng.choice(nz)) if nz.size else 0

    return choose


def run_against_twins(vec, twins, choose, steps, opponents=None):
    """Step `vec` and the twins in lockstep; check observations, masks, step results, final
    observations, seeds and EpisodeStats. Returns the EpisodeStats seen."""
    stats_seen = []
    expected_stats = []
    ret = [0.0] * vec.num_envs
    length = [0] * vec.num_envs
    for _ in range(steps):
        board, glob = vec.observations()
        am = vec.actor_masks()
        for i, tw in enumerate(twins.envs):
            b, g = tw.observation()
            np.testing.assert_array_equal(board[i], b)
            np.testing.assert_array_equal(glob[i], g)
            np.testing.assert_array_equal(am[i], tw.actor_mask())
        actors, kinds, args = vec_choice(vec, choose)
        km = vec.kind_masks(actors)
        xm = vec.arg_masks(actors, kinds)
        for i, tw in enumerate(twins.envs):
            np.testing.assert_array_equal(km[i], tw.kind_mask(int(actors[i])))
            ref = tw.arg_mask(int(actors[i]), int(kinds[i]))
            np.testing.assert_array_equal(xm[i, : ref.size], ref)
            assert not xm[i, ref.size :].any()
        out = vec.step(actors, kinds, args)
        for i, tw in enumerate(twins.envs):
            r, _, term, trunc = tw.step_factored(int(actors[i]), int(kinds[i]), int(args[i]))
            assert out.reward[i] == np.float32(r)
            assert bool(out.terminated[i]) == term and bool(out.truncated[i]) == trunc
            ret[i] += r
            length[i] += 1
            if term or trunc:
                b, g = tw.observation()
                np.testing.assert_array_equal(out.final_board[i], b)
                np.testing.assert_array_equal(out.final_globals[i], g)
                info = tw.info()
                seat = info["agent_seat"]
                seed = twins.seed(i)
                assert seat == seed % 2
                if info["outcome"] == f"win_{seat}":
                    outcome = "win"
                elif info["outcome"] == f"win_{1 - seat}":
                    outcome = "loss"
                else:
                    outcome = "draw"
                expected_stats.append((i, seed, seat, outcome, length[i], ret[i]))
                ret[i], length[i] = 0.0, 0
                twins.k[i] += 1
                tw.reset(twins.seed(i))
        stats_seen += vec.episode_stats()
    assert len(stats_seen) == len(expected_stats)
    for st, (i, seed, seat, outcome, ln, rt) in zip(stats_seen, expected_stats):
        assert (st.env_index, st.seed, st.agent_seat, st.outcome, st.length) == (i, seed, seat, outcome, ln)
        assert st.ret == pytest.approx(rt, abs=1e-4)
        if opponents is not None:
            assert st.opponent == opponents[i]
    return stats_seen


def test_shapes_and_dtypes():
    vec = SyncVecEnv(env_fns(3), seed_base=5)
    assert vec.num_envs == 3
    st = vec.reset_all()
    spec = ps.Env(**KW).spec()
    c, g = spec["channels"], spec["globals"]
    assert st.board.shape == (3, c, SIZE, SIZE) and st.board.dtype == np.float32
    assert st.globals.shape == (3, g) and st.globals.dtype == np.float32
    board, glob = vec.observations()
    assert board.shape == (3, c, SIZE, SIZE) and board.dtype == np.float32
    assert glob.shape == (3, g) and glob.dtype == np.float32
    np.testing.assert_array_equal(board, st.board)
    am = vec.actor_masks()
    assert am.shape == (3, 3 * N + 1) and am.dtype == np.bool_
    actors = np.full(3, 3 * N, dtype=np.int64)
    km = vec.kind_masks(actors)
    assert km.shape == (3, 10) and km.dtype == np.bool_
    kinds = np.full(3, END_TURN, dtype=np.int64)
    xm = vec.arg_masks(actors, kinds)
    assert xm.shape == (3, N) and xm.dtype == np.bool_
    assert not xm.any()  # EndTurn has no argument
    out = vec.step(actors, kinds, np.zeros(3, dtype=np.int64))
    assert out.reward.shape == (3,) and out.reward.dtype == np.float32
    assert out.terminated.shape == (3,) and out.terminated.dtype == np.bool_
    assert out.truncated.shape == (3,) and out.truncated.dtype == np.bool_
    assert out.final_board.shape == (3, c, SIZE, SIZE)
    assert out.final_globals.shape == (3, g)


def test_seed_sequence_auto_reset_and_episode_stats():
    # turn_cap 2 and EndTurn-first play: every episode is a few steps, so many resets happen.
    base, stride, n = 100, 7, 3
    vec = SyncVecEnv(env_fns(n, turn_cap=2), seed_base=base, seed_stride=stride)
    vec.reset_all()
    twins = Twins(n, base, stride, turn_cap=2)
    stats = run_against_twins(vec, twins, end_turn_chooser, steps=40)
    for i in range(n):
        seeds = [s.seed for s in stats if s.env_index == i]
        assert len(seeds) >= 5
        assert seeds == [base + i * stride + k * n * stride for k in range(len(seeds))]
    all_seeds = [s.seed for s in stats]
    assert len(set(all_seeds)) == len(all_seeds)
    assert {s.agent_seat for s in stats} == {0, 1}  # both seed parities seen


def test_random_play_against_twins_with_labels():
    base, n = 0, 4
    vec = SyncVecEnv(env_fns(n, turn_cap=3), seed_base=base, opponents=["a", "b", "c", "d"])
    vec.reset_all()
    twins = Twins(n, base, 1, turn_cap=3)
    stats = run_against_twins(vec, twins, random_chooser(np.random.default_rng(0)), steps=80, opponents=["a", "b", "c", "d"])
    assert len(stats) >= 4
    assert {s.outcome for s in stats} <= {"win", "loss", "draw"}


def test_truncated_vs_terminated():
    # max_agent_actions 3 with a 30-turn cap: every episode is truncated at its 3rd agent action.
    vec = SyncVecEnv(env_fns(2, turn_cap=30, max_agent_actions=3), seed_base=11)
    vec.reset_all()
    twins = Twins(2, 11, 1, turn_cap=30, max_agent_actions=3)
    for step in range(1, 10):
        actors, kinds, args = vec_choice(vec, end_turn_chooser)
        out = vec.step(actors, kinds, args)
        for i, tw in enumerate(twins.envs):
            r, _, term, trunc = tw.step_factored(int(actors[i]), int(kinds[i]), int(args[i]))
            if trunc:
                np.testing.assert_array_equal(out.final_board[i], tw.observation()[0])
                twins.k[i] += 1
                tw.reset(twins.seed(i))
        assert not out.terminated.any()
        assert out.truncated.tolist() == [step % 3 == 0] * 2
        np.testing.assert_array_equal(vec.observations()[0], np.stack([tw.observation()[0] for tw in twins.envs]))
    # A short turn cap ends episodes with `terminated`.
    vec = SyncVecEnv(env_fns(2, turn_cap=1), seed_base=11)
    vec.reset_all()
    seen_term = False
    for _ in range(10):
        out = vec.step(*vec_choice(vec, end_turn_chooser))
        assert not out.truncated.any()
        seen_term |= bool(out.terminated.any())
    assert seen_term


def test_arg_masks_padded_beyond_arg_len():
    # Train has 5 unit types: entries 5..N-1 are False; the first 5 are the env's arg mask.
    vec = SyncVecEnv(env_fns(1), seed_base=3)
    vec.reset_all()
    twins = Twins(1, 3, 1)
    tw = twins.envs[0]
    checked = 0
    for _ in range(30):
        am = vec.actor_masks()[0]
        for a in [a for a in np.flatnonzero(am) if N <= a < 2 * N]:
            if vec.kind_masks(np.array([a]))[0, TRAIN]:
                xm = vec.arg_masks(np.array([a]), np.array([TRAIN]))[0]
                np.testing.assert_array_equal(xm[:5], tw.arg_mask(int(a), TRAIN))
                assert xm[:5].any() and not xm[5:].any()
                checked += 1
        actors, kinds, args = vec_choice(vec, end_turn_chooser)
        vec.step(actors, kinds, args)
        _, _, term, trunc = tw.step_factored(int(actors[0]), int(kinds[0]), int(args[0]))
        if term or trunc:
            twins.k[0] += 1
            tw.reset(twins.seed(0))
    assert checked > 0, "no city could train within 30 steps"


def test_two_vec_envs_with_same_seeds_evolve_identically():
    rng_a, rng_b = np.random.default_rng(9), np.random.default_rng(9)
    va = SyncVecEnv(env_fns(3, turn_cap=3), seed_base=50)
    vb = SyncVecEnv(env_fns(3, turn_cap=3), seed_base=50)
    va.reset_all()
    vb.reset_all()
    for _ in range(60):
        pa = vec_choice(va, random_chooser(rng_a))
        pb = vec_choice(vb, random_chooser(rng_b))
        for x, y in zip(pa, pb):
            np.testing.assert_array_equal(x, y)
        oa, ob = va.step(*pa), vb.step(*pb)
        np.testing.assert_array_equal(oa.reward, ob.reward)
        np.testing.assert_array_equal(oa.terminated, ob.terminated)
        for x, y in zip(va.observations(), vb.observations()):
            np.testing.assert_array_equal(x, y)
    assert [(s.seed, s.outcome, s.length) for s in va.episode_stats()] == [
        (s.seed, s.outcome, s.length) for s in vb.episode_stats()
    ]


def test_reset_all_uses_the_next_seeds():
    vec = SyncVecEnv(env_fns(2), seed_base=20, seed_stride=3)
    vec.reset_all()
    vec.reset_all()  # second reset: k = 1 for both envs
    for i in range(2):
        tw = ps.Env(**KW)
        tw.reset(20 + i * 3 + 1 * 2 * 3)
        np.testing.assert_array_equal(vec.observations()[0][i], tw.observation()[0])


def test_training_seeds_must_stay_below_eval_range():
    assert TRAIN_SEED_LIMIT == 1_000_000_000
    with pytest.raises(ValueError):
        SyncVecEnv(env_fns(2), seed_base=TRAIN_SEED_LIMIT - 1).reset_all()


def test_make_vec_env_wires_config_and_round_robin_opponents():
    cfg = PPOConfig(
        size=SIZE,
        turn_cap=3,
        num_envs=4,
        opponents=["economy", "rush"],
        opponent_epsilon=0.3,
        shaping_weight=0.5,
        terminal_weight=2.0,
        gamma=0.99,
        potential_scale=500.0,
    )
    vec = make_vec_env(cfg, seed_base=10)
    assert vec.num_envs == 4
    vec.reset_all()
    opponents = ["economy", "rush", "economy", "rush"]
    twins = Twins(4, 10, 1)
    twins.envs = []
    for i in range(4):
        tw = ps.Env(
            size=SIZE,
            turn_cap=3,
            full_observability=False,
            opponent=opponents[i],
            opponent_epsilon=0.3,
            agent_seat="seed_parity",
            terminal_weight=2.0,
            shaping_weight=0.5,
            gamma=0.99,
            potential_scale=500.0,
        )
        tw.reset(twins.seed(i))
        twins.envs.append(tw)
    stats = run_against_twins(vec, twins, random_chooser(np.random.default_rng(4)), steps=100, opponents=opponents)
    assert {s.opponent for s in stats} == {"economy", "rush"}

