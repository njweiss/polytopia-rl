"""Milestone 16b tests 1: `PyEnv.current_player()` and `SelfPlayVecEnv` polling.

Expected behavior (docs/MILESTONE_16b.md, Rule 1):
- `PyEnv.current_player()` returns `self.env()?.view().current()` (0 or 1), raising
  `RuntimeError("NotReset")` before the first `reset`, identical to `info()`.
- `SelfPlayVecEnv` uses `env.current_player()` in both `learner_slots()` and `_opponent_to_move()`
  when the env exposes it (without calling `env.info()`), and falls back to
  `env.info()["current_player"]` for fakes/mocks without `current_player()`.
"""

import numpy as np
import polytopia_sim as ps
import pytest
from il_helpers import pool_of, reward_cfg
from rl_helpers import END_TURN, MOVE, end_turn_path
from sp_helpers import FakeTwoSeatEnv, fake_scorer

from polytopia_rl.selfplay import SelfPlayVecEnv

FAKE_REWARD = reward_cfg(shaping=1.0, gamma=0.9, scale=10.0)


@pytest.mark.parametrize("opponent", ["rush", "none"])
def test_current_player_before_reset_raises_not_reset(opponent):
    env = ps.Env(size=6, opponent=opponent)
    with pytest.raises(RuntimeError, match="NotReset") as exc_info:
        env.current_player()
    assert not isinstance(exc_info.value, NotImplementedError)


@pytest.mark.parametrize("opponent", ["random", "economy", "rush"])
def test_current_player_matches_info_in_bot_mode(opponent):
    env = ps.Env(size=6, turn_cap=4, opponent=opponent, agent_seat="seed_parity")
    rng = np.random.default_rng(7)
    for seed in range(4):
        env.reset(seed)
        cp = env.current_player()
        assert type(cp) is int
        assert cp == env.info()["current_player"] == seed % 2
        for _ in range(25):
            if env.legal_count() == 0:
                break
            _, _, term, trunc = env.step_legal(int(rng.integers(env.legal_count())))
            cp = env.current_player()
            assert type(cp) is int
            assert cp == env.info()["current_player"]
            if term or trunc:
                break


def test_current_player_matches_info_in_two_seat_mode():
    env = ps.Env(size=6, turn_cap=3, opponent="none")
    for seed in (0, 1):
        env.reset(seed)
        assert env.current_player() == 0 == env.info()["current_player"]
        saw_seat_1 = False
        saw_back_to_0 = False
        while env.legal_count() > 0:
            a, k, x = end_turn_path(env)
            _, _, term, trunc = env.step_factored(a, k, x)
            cp = env.current_player()
            assert type(cp) is int
            assert cp == env.info()["current_player"]
            if cp == 1:
                saw_seat_1 = True
            elif saw_seat_1 and cp == 0:
                saw_back_to_0 = True
            if term or trunc:
                break
        assert saw_seat_1 and saw_back_to_0


class _SpyEnv:
    """Wraps an env, exposing `current_player()` and counting `info()` vs `current_player()`."""

    def __init__(self, inner):
        self._inner = inner
        self.info_calls = 0
        self.cp_calls = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def info(self):
        self.info_calls += 1
        return self._inner.info()

    def current_player(self):
        self.cp_calls += 1
        if hasattr(self._inner, "current_player"):
            return self._inner.current_player()
        return self._inner.info()["current_player"]

    def reset_counts(self):
        self.info_calls = 0
        self.cp_calls = 0


def test_selfplay_vec_env_polling_uses_current_player_not_info():
    spies = []

    def env_fn(opp):
        spy = _SpyEnv(FakeTwoSeatEnv(k=3, turns=2))
        spies.append(spy)
        return spy

    vec = SelfPlayVecEnv(
        env_fn,
        num_envs=2,
        seed_base=0,
        pool=pool_of(1),
        make_policy=lambda entry: fake_scorer,
        rng=np.random.default_rng(0),
        p_bot=0.0,
        reward=FAKE_REWARD,
    )
    vec.reset_all()
    for s in spies:
        s.reset_counts()

    # learner_slots() must poll current_player(), never info().
    assert vec.learner_slots() == [0, 1]
    assert sum(s.info_calls for s in spies) == 0
    assert sum(s.cp_calls for s in spies) == 2

    # Move slot 0 to its opponent's turn, then check _opponent_to_move() and an opponent-only tick.
    vec.tick({0: (0, MOVE, 1)})
    vec.tick({0: (0, MOVE, 1)})
    vec.tick({0: (3 * 16, END_TURN, 0)})
    for s in spies:
        s.reset_counts()

    assert vec._opponent_to_move(0) is True
    assert vec._opponent_to_move(1) is False
    assert vec.learner_slots() == [1]
    assert sum(s.info_calls for s in spies) == 0
    assert sum(s.cp_calls for s in spies) >= 4

    # An opponent-only tick with no learner action and no episode end must not call info() at all.
    for s in spies:
        s.reset_counts()
    assert vec.tick({}) == []
    assert sum(s.info_calls for s in spies) == 0
    assert sum(s.cp_calls for s in spies) > 0


def test_selfplay_vec_env_polling_with_real_env_avoids_info():
    spies = []

    def env_fn(opp):
        spy = _SpyEnv(ps.Env(size=6, turn_cap=3, opponent=opp, agent_seat="seed_parity"))
        spies.append(spy)
        return spy

    vec = SelfPlayVecEnv(
        env_fn,
        num_envs=2,
        seed_base=0,
        pool=pool_of(1),
        make_policy=lambda entry: (lambda envs: [end_turn_path(e) for e in envs]),
        rng=np.random.default_rng(0),
        p_bot=0.0,
        reward=FAKE_REWARD,
    )
    vec.reset_all()
    for s in spies:
        s.reset_counts()

    slots = vec.learner_slots()
    assert slots == [0, 1]
    assert sum(s.info_calls for s in spies) == 0
    assert sum(s.cp_calls for s in spies) == 2


def test_selfplay_vec_env_falls_back_to_info_for_fakes_without_current_player():
    fake = FakeTwoSeatEnv(k=2, turns=1)
    assert not hasattr(fake, "current_player")
    vec = SelfPlayVecEnv(
        lambda opp: FakeTwoSeatEnv(k=2, turns=1),
        num_envs=2,
        seed_base=0,
        pool=pool_of(1),
        make_policy=lambda entry: fake_scorer,
        rng=np.random.default_rng(0),
        p_bot=0.0,
        reward=FAKE_REWARD,
    )
    vec.reset_all()
    assert vec.learner_slots() == [0, 1]
    assert not vec._opponent_to_move(0)
    out = vec.tick({0: (0, MOVE, 1), 1: (0, MOVE, 1)})
    assert [tr.slot for tr in out] == [0, 1]
