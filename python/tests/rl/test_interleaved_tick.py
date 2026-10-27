"""Milestone 13 tests 2: the tick interface of `SelfPlayVecEnv` ("Ticks").

Hand traces on `FakeTwoSeatEnv` (sp_helpers): seat 0 moves first, each turn is `k - 1` Moves then
EndTurn, a Move with arg 1 scores a point, the game ends when seat 1 ends turn `turns`; observation
globals are `[observer, actions played, seed % 2]`. The snapshot opponent plays `fake_scorer` (Move 1
while legal, else EndTurn). Expected rewards are the M9 formula in float32 with gamma 0.9 and
potential_scale 10 (`m9`, as in test_selfplay_env.py), over the learner's action and everything up to
its next move or the episode end. Seeds follow `seed_base + i * stride + k * num_envs * stride`; the
learner is seat `seed % 2`.

The invariant test drives a mixed vec env (two snapshots and a fake bot, per-episode turn lengths
`k = 2 + seed % 3`, random subsets of learner slots acting) and checks every tick against an
independent model built from the envs' own move log: who was to move, who moved, which transitions
complete, and the batch-policy calls.
"""

import numpy as np
import pytest
from il_helpers import FakeBotEnv, keyed_pick, keyed_vec, learner_path, pool_of, reward_cfg
from rl_helpers import END_TURN, MOVE
from sp_helpers import FakeTwoSeatEnv, fake_scorer

from polytopia_rl.selfplay import SelfPlayVecEnv, Transition

G = 3 * 16  # the fake's global actor (size 4)
f32 = np.float32
FAKE_REWARD = reward_cfg(shaping=1.0, gamma=0.9, scale=10.0)


def m9(before, after, terminal=0.0, done=False):
    """The M9 reward in float32: before/after are the learner's (mine, theirs) scores."""
    phi = lambda s: (f32(s[0]) - f32(s[1])) / f32(10.0)  # noqa: E731
    after_phi = f32(0.0) if done else phi(after)
    return f32(1.0) * f32(terminal) + f32(1.0) * (f32(0.9) * after_phi - phi(before))


def fake_vec(num_envs=1, seed_stride=1, env_cls=FakeTwoSeatEnv, p_bot=0.0, **fake_kw):
    return SelfPlayVecEnv(
        lambda opp: (FakeTwoSeatEnv if opp == "none" else env_cls)(**fake_kw),
        num_envs,
        0,
        seed_stride,
        pool=pool_of(1),
        make_policy=lambda entry: fake_scorer,
        rng=np.random.default_rng(0),
        p_bot=p_bot,
        reward=FAKE_REWARD,
    )


def globs(vec):
    return vec.observations()[1].tolist()


def one(out):
    assert len(out) == 1 and isinstance(out[0], Transition)
    return out[0]


def check(tr, slot, reward, terminated=False, truncated=False, final_globals=None):
    assert tr.slot == slot
    assert tr.reward == reward, (tr.reward, reward)
    assert (bool(tr.terminated), bool(tr.truncated)) == (terminated, truncated)
    if final_globals is None:
        assert tr.final_board is None and tr.final_globals is None
    else:
        assert np.asarray(tr.final_globals).tolist() == final_globals
        fb = np.asarray(tr.final_board)
        assert np.all(fb[0] == final_globals[0]) and np.all(fb[1] == final_globals[1])


def test_snapshot_slot_trace_waits_one_opponent_move_per_tick_and_resets_without_moving():
    vec = fake_vec(k=3, turns=1)
    vec.reset_all()  # seed 0: learner seat 0 moves first
    assert vec.learner_slots() == [0]
    assert globs(vec) == [[0.0, 0.0, 0.0]]
    assert vec.tick({}) == [] and globs(vec) == [[0.0, 0.0, 0.0]], "no action, no opponent to move: nothing happens"
    check(one(vec.tick({0: (0, MOVE, 1)})), 0, m9((0, 0), (1, 0)))
    check(one(vec.tick({0: (0, MOVE, 0)})), 0, m9((1, 0), (1, 0)))
    assert vec.learner_slots() == [0] and globs(vec) == [[0.0, 2.0, 0.0]]
    # EndTurn: the opponent makes exactly one move in the same tick, then the slot waits.
    assert vec.tick({0: (G, END_TURN, 0)}) == []
    assert globs(vec) == [[1.0, 4.0, 0.0]] and vec.learner_slots() == []
    assert vec.tick({}) == []
    assert globs(vec) == [[1.0, 5.0, 0.0]] and vec.learner_slots() == []
    # The opponent's EndTurn ends the game (turns=1): loss, 1 point against 2. The transition spans
    # the learner's EndTurn and the whole opponent turn.
    want = m9((1, 0), (1, 2), terminal=-1.0, done=True)
    check(one(vec.tick({})), 0, want, terminated=True, final_globals=[1.0, 6.0, 0.0])
    (s,) = vec.episode_stats()
    assert (s.env_index, s.seed, s.agent_seat, s.outcome, s.length, s.opponent) == (0, 0, 0, "loss", 3, "snap_0")
    assert s.ret == pytest.approx(float(m9((0, 0), (1, 0)) + m9((1, 0), (1, 0)) + want))
    # Reset to seed 1 (learner seat 1): the opponent is to move but has not moved inside the reset.
    assert globs(vec) == [[0.0, 0.0, 1.0]] and vec.learner_slots() == []
    assert vec.tick({}) == [] and globs(vec) == [[0.0, 1.0, 1.0]]
    assert vec.tick({}) == [] and globs(vec) == [[0.0, 2.0, 1.0]]
    assert vec.tick({}) == [] and globs(vec) == [[1.0, 3.0, 1.0]]
    assert vec.learner_slots() == [0]
    check(one(vec.tick({0: (0, MOVE, 1)})), 0, m9((0, 2), (1, 2)))
    assert vec.episode_stats() == []


def test_truncation_by_an_opponent_move_completes_the_transition():
    vec = fake_vec(k=3, turns=5, truncate_at=4)
    vec.reset_all()
    vec.tick({0: (0, MOVE, 0)})
    vec.tick({0: (0, MOVE, 0)})
    # EndTurn is action 3; the opponent's first move (scoring) is action 4 and truncates.
    tr = one(vec.tick({0: (G, END_TURN, 0)}))
    check(tr, 0, m9((0, 0), (0, 1)), truncated=True, final_globals=[1.0, 4.0, 0.0])
    (s,) = vec.episode_stats()
    assert (s.seed, s.length) == (0, 3)
    assert globs(vec) == [[0.0, 0.0, 1.0]], "seed 1, opponent to move, not advanced in the reset"
    assert vec.learner_slots() == []


def test_an_episode_ended_by_the_learner_completes_at_once_and_the_next_opponent_waits_for_the_next_tick():
    vec = fake_vec(k=3, turns=5, truncate_at=2)
    vec.reset_all()
    check(one(vec.tick({0: (0, MOVE, 1)})), 0, m9((0, 0), (1, 0)))
    tr = one(vec.tick({0: (0, MOVE, 1)}))
    check(tr, 0, m9((1, 0), (2, 0)), truncated=True, final_globals=[0.0, 2.0, 0.0])
    assert globs(vec) == [[0.0, 0.0, 1.0]], "the new episode's opponent has not moved in this tick"
    assert vec.learner_slots() == []
    assert vec.tick({}) == [] and globs(vec) == [[0.0, 1.0, 1.0]]


def test_bot_slots_complete_in_the_same_tick_with_the_env_reward():
    vec = fake_vec(env_cls=FakeBotEnv, p_bot=1.0, k=3, turns=1)
    vec.reset_all()  # seed 0: learner seat 0
    assert vec.learner_slots() == [0]
    check(one(vec.tick({0: (0, MOVE, 1)})), 0, 0.125)
    check(one(vec.tick({0: (0, MOVE, 1)})), 0, 0.25)
    # The bot's whole turn is played inside the env step; it ends the game (2 points each: draw).
    tr = one(vec.tick({0: (G, END_TURN, 0)}))
    check(tr, 0, 0.375, terminated=True, final_globals=[1.0, 6.0, 0.0])
    (s,) = vec.episode_stats()
    assert s.opponent in ("economy", "rush") and (s.seed, s.outcome, s.length) == (0, "draw", 3)
    # Seed 1: the bot env played its opening turn inside its own reset; the learner moves at once.
    assert vec.learner_slots() == [0] and globs(vec) == [[1.0, 3.0, 1.0]]
    check(one(vec.tick({0: (0, MOVE, 1)})), 0, 0.125)


def test_tick_rejects_actions_for_slots_not_on_the_learner_move():
    vec = fake_vec(num_envs=2, k=3, turns=1)
    vec.reset_all()
    for i in (0, 1):
        vec.tick({i: (0, MOVE, 1)})
        vec.tick({i: (0, MOVE, 1)})
    vec.tick({0: (G, END_TURN, 0)})  # slot 0 waits for its opponent
    assert vec.learner_slots() == [1]
    with pytest.raises(ValueError):
        vec.tick({0: (0, MOVE, 1)})
    with pytest.raises(ValueError):
        vec.tick({1: (G, END_TURN, 0), 2: (0, MOVE, 1)})  # no slot 2


def test_an_opponent_first_episode_rejects_learner_actions_until_its_opponent_is_done():
    vec = fake_vec(num_envs=1, k=2, turns=1)
    vec.reset_all()
    vec.tick({0: (0, MOVE, 1)})
    vec.tick({0: (G, END_TURN, 0)})  # opponent Move
    tr = one(vec.tick({}))  # opponent EndTurn ends the game; reset to seed 1 (opponent first)
    assert tr.terminated
    assert vec.learner_slots() == []
    with pytest.raises(ValueError):
        vec.tick({0: (0, MOVE, 1)})


# ---------------------------------------------------------------- invariants on a mixed vec env

N = 6
LOG = []  # (seed, mover seat, done after the move, current player after the move), externally driven moves only


class _VarK:
    def reset(self, seed):
        self.k = 2 + int(seed) % 3
        super().reset(seed)


class LoggedTwoSeat(_VarK, FakeTwoSeatEnv):
    def step_factored(self, actor, kind, arg):
        mover = self.current
        out = super().step_factored(actor, kind, arg)
        LOG.append((self.seed, mover, self.done, self.current))
        return out


class LoggedBot(_VarK, FakeBotEnv):
    def step_factored(self, actor, kind, arg):
        out = super().step_factored(actor, kind, arg)
        LOG.append((self.seed, self.agent, self.done, self.current))
        return out


CHOICES = [0, 1, "economy"]  # snap_0, snap_100, a fake bot
PICK = keyed_pick(len(CHOICES))


class Calls:
    """make_policy whose batch policies record `(entry name, slots)` per call."""

    def __init__(self):
        self.calls = []

    def __call__(self, entry):
        def policy(envs):
            slots = [env.seed % N for env in envs]
            for env in envs:
                assert not env.done and env.current != env.seed % 2, "only the opponent's move"
            self.calls.append((entry.name, slots))
            return fake_scorer(envs)

        return policy


def entry_name_of(seed):
    c = CHOICES[PICK(seed)]
    return f"snap_{c * 100}" if isinstance(c, int) else c


def test_every_tick_matches_an_independent_model_of_the_rules():
    calls = Calls()
    vec = keyed_vec(
        lambda opp: LoggedTwoSeat(turns=2) if opp == "none" else LoggedBot(turns=2),
        N,
        pool=pool_of(2),
        make_policy=calls,
        choices=CHOICES,
        pick=PICK,
        reward=FAKE_REWARD,
    )
    vec.reset_all()
    rng = np.random.default_rng(5)
    pending: set[int] = set()  # snapshot slots whose learner action has not completed yet
    seen = dict(waits=0, opp_first_resets=0, ends=0, bot_moves=0, two_groups=0)
    for t in range(300):
        envs = list(vec.envs)
        seeds = [e.seed for e in envs]  # the same env object may be reset during the tick
        kinds = [type(e) for e in envs]
        running = [not e.done for e in envs]
        learner_turn = [running[i] and envs[i].current == envs[i].seed % 2 for i in range(N)]
        opp_turn = {i for i in range(N) if running[i] and not learner_turn[i]}
        assert vec.learner_slots() == [i for i in range(N) if learner_turn[i]], f"t={t}"
        assert pending <= opp_turn, "a waiting slot is not on the learner's move"
        acts = {i: learner_path(envs[i]) for i in range(N) if learner_turn[i] and rng.random() < 0.7}
        LOG.clear()
        calls.calls.clear()
        out = vec.tick(acts)
        moves = {i: [r for r in LOG if r[0] % N == i] for i in range(N)}
        movers = set()
        for i in range(N):
            seat = seeds[i] % 2
            mine = [r for r in moves[i] if r[0] == seeds[i] and r[1] == seat]
            theirs = [r for r in moves[i] if r[0] == seeds[i] and r[1] != seat]
            later = [r for r in moves[i] if r[0] != seeds[i]]
            assert len(mine) == (1 if i in acts else 0), (t, i, moves[i])
            assert not later, (t, i, "the next episode moved in the tick its predecessor ended")
            handed_over = i in acts and not mine[0][2] and mine[0][3] != seat
            expect_opp = kinds[i] is LoggedTwoSeat and (i in opp_turn or handed_over)
            assert len(theirs) == (1 if expect_opp else 0), (t, i, moves[i])
            if theirs:
                movers.add(i)
            if kinds[i] is LoggedBot and i in acts:
                seen["bot_moves"] += 1
        # Transitions: every slot with an outstanding learner action whose episode ended, or that is
        # back on its learner's move; ascending.
        waiting = pending | set(acts)
        ended = {i for i in range(N) if any(r[2] for r in moves[i])}
        now_learner = {i for i in range(N) if not vec.envs[i].done and vec.envs[i].current == vec.envs[i].seed % 2}
        done = {i for i in waiting if i in ended or i in now_learner}
        assert [tr.slot for tr in out] == sorted(done), (t, [tr.slot for tr in out], sorted(done))
        for tr in out:
            assert (tr.final_board is not None) == (tr.slot in ended) == bool(tr.terminated or tr.truncated)
        pending = waiting - done
        seen["waits"] += len(pending)
        seen["ends"] += len(ended)
        seen["opp_first_resets"] += sum(1 for i in ended if vec.envs[i].seed % 2 == 1 and isinstance(vec.envs[i], LoggedTwoSeat))
        # One batch-policy call per snapshot group, groups in order of their lowest slot.
        names = [name for name, _ in calls.calls]
        assert len(names) == len(set(names)), (t, calls.calls)
        for name, slots in calls.calls:
            assert len(slots) == len(set(slots))
            assert all(entry_name_of(seeds[i]) == name for i in slots), (t, calls.calls)
        assert sorted(i for _, slots in calls.calls for i in slots) == sorted(movers), (t, calls.calls, movers)
        assert [min(s) for _, s in calls.calls] == sorted(min(s) for _, s in calls.calls)
        assert all(s == sorted(s) for _, s in calls.calls), "slots in ascending order within a group"
        seen["two_groups"] += len(calls.calls) == 2
    assert all(v > 0 for v in seen.values()), seen
