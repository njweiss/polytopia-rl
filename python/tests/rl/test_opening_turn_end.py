"""Milestone 16b tests 4: opening-turn truncation/termination safeguard in `SelfPlayVecEnv.tick`.

Checks (docs/MILESTONE_16b.md, Rule 4 & Clarification B3):
- When a snapshot episode ends (truncates or terminates) during the opponent's opening turn before
  the learner ever acts (`self._before[i] is None` and `self._ended[i] is not None`):
  - `tick` emits NO `Transition` for slot `i`.
  - `tick` appends an `EpisodeStat` with `length=0`, `ret=0.0`, and `outcome` from
    `agent_outcome(info()["outcome"], seat)`.
  - `tick` resets slot `i` to its next seed without advancing the new episode's opponent in the
    same tick (preserving M13's reset rule).
- `collect_rollout_interleaved` completes all `T` rows per slot without hanging.
"""

import numpy as np
import pytest
import torch
from il_helpers import StubNet, pool_of, reward_cfg
from rl_helpers import MOVE
from sp_helpers import FakeTwoSeatEnv, fake_scorer

from polytopia_rl.ppo import RolloutBuffer, collect_rollout_interleaved
from polytopia_rl.selfplay import SelfPlayVecEnv

FAKE_REWARD = reward_cfg(shaping=1.0, gamma=0.9, scale=10.0)


class _OpeningTruncOnceEnv(FakeTwoSeatEnv):
    """Seed 0 (learner first): truncates at action 2; seed 1 (opponent first): opponent truncates
    at action 2 before learner acts; seed 2+: normal games (`truncate_at=None`)."""

    def reset(self, seed):
        self.truncate_at = 2 if int(seed) in (0, 1) else None
        super().reset(seed)


class _OpeningTermEnv(FakeTwoSeatEnv):
    """Seed 0: truncates at action 2; seed 1 (opponent first): terminates with `win_0` on the
    opponent's 2nd move (before learner seat 1 ever acts); seed 2+: normal games."""

    def reset(self, seed):
        self.truncate_at = 2 if int(seed) == 0 else None
        super().reset(seed)

    def step_factored(self, actor, kind, arg):
        reward, rewards, terminated, truncated = super().step_factored(actor, kind, arg)
        if self.seed == 1 and self.t == 2:
            terminated, truncated = True, False
            self.outcome = "win_0"
            self.done = True
        return reward, rewards, terminated, truncated


def _make_vec(env_cls, num_envs=1, seed_stride=1):
    vec = SelfPlayVecEnv(
        lambda opp: env_cls(k=3, turns=2),
        num_envs=num_envs,
        seed_base=0,
        seed_stride=seed_stride,
        pool=pool_of(1),
        make_policy=lambda entry: fake_scorer,
        rng=np.random.default_rng(0),
        p_bot=0.0,
        reward=FAKE_REWARD,
    )
    vec.reset_all()
    return vec


def test_opening_turn_truncation_emits_no_transition_records_stat_and_resets():
    vec = _make_vec(_OpeningTruncOnceEnv, num_envs=1)
    # Seed 0 (learner seat 0): 2 moves truncate seed 0 and reset to seed 1 (opponent seat 0 first).
    assert len(vec.tick({0: (0, MOVE, 1)})) == 1
    tr = vec.tick({0: (0, MOVE, 1)})
    assert len(tr) == 1 and tr[0].truncated
    assert [s.seed for s in vec.episode_stats()] == [0]
    assert vec.learner_slots() == [] and vec.envs[0].seed == 1

    # Opponent move 1 on seed 1: not done yet.
    assert vec.tick({}) == []
    assert vec.episode_stats() == []
    assert vec.envs[0].seed == 1

    # Opponent move 2 on seed 1: truncates during opening turn before learner ever acted.
    out = vec.tick({})
    assert out == [], "must not emit a Transition when the learner never acted"
    stats = vec.episode_stats()
    assert len(stats) == 1, "must record an EpisodeStat for the opening-turn truncation"
    s = stats[0]
    assert (s.env_index, s.seed, s.opponent, s.agent_seat, s.outcome, s.length, s.ret) == (
        0,
        1,
        "snap_0",
        1,
        "draw",
        0,
        0.0,
    )
    # Slot must have reset to seed 2 (learner seat 0 to move, t == 0).
    assert vec.envs[0].seed == 2 and vec.envs[0].t == 0
    assert vec.learner_slots() == [0]


def test_opening_turn_termination_records_loss_outcome_and_resets_without_advancing():
    # Use seed_stride=1, num_envs=1: seed 0 -> seed 1 (opening-turn win_0 for opponent) -> seed 2.
    vec = _make_vec(_OpeningTermEnv, num_envs=1)
    vec.tick({0: (0, MOVE, 1)})
    vec.tick({0: (0, MOVE, 1)})
    vec.episode_stats()
    assert vec.envs[0].seed == 1 and vec.learner_slots() == []

    assert vec.tick({}) == []
    assert vec.tick({}) == [], "must not emit a Transition on opening-turn termination"
    stats = vec.episode_stats()
    assert len(stats) == 1
    s = stats[0]
    assert (s.seed, s.agent_seat, s.outcome, s.length, s.ret) == (1, 1, "loss", 0, 0.0)
    assert vec.envs[0].seed == 2 and vec.envs[0].t == 0
    assert vec.learner_slots() == [0]


def test_opening_turn_reset_to_another_opponent_first_seed_does_not_advance_in_same_tick():
    # Slot 0 gets seeds 0, 1, 3? Use custom seed sequence via num_envs=2:
    # slot 0 gets seeds 0, 2, 4...; use seed_base=0, num_envs=1 with an env where seed 1 truncates
    # on action 2 and seed 2 is skipped? Better: force opponent-first on seed 3 after seed 1 with
    # num_envs=2, seed_base=1? Wait: seed_base=1 -> reset_all advances seed 1. Instead use
    # seed_base=0, num_envs=1, seed_stride=1, and make seed 2 also have an opponent-first check by
    # setting seed_base=0, num_envs=1, seed_stride=3 -> seeds 0 (even), 3 (odd), 6 (even) - wait
    # stride 1 gives 0, 1, 2; to go odd -> odd, seed_stride=2 from seed 0 gives 0, 2, 4.
    # For odd -> odd, start slot 0 on seed 0 -> 1 -> 3 using a tiny subclass that overrides _episode_seed.
    class _SeqVec(SelfPlayVecEnv):
        def _episode_seed(self, i, k):
            return [0, 1, 3, 4][k]

    vec = _SeqVec(
        lambda opp: _OpeningTruncOnceEnv(k=3, turns=2),
        num_envs=1,
        seed_base=0,
        pool=pool_of(1),
        make_policy=lambda entry: fake_scorer,
        rng=np.random.default_rng(0),
        p_bot=0.0,
        reward=FAKE_REWARD,
    )
    vec.reset_all()
    vec.tick({0: (0, MOVE, 1)})
    vec.tick({0: (0, MOVE, 1)})
    vec.episode_stats()
    assert vec.envs[0].seed == 1

    assert vec.tick({}) == []
    assert vec.tick({}) == []
    assert [s.seed for s in vec.episode_stats()] == [1]
    # Reset to seed 3 (opponent first): opponent must NOT have moved inside the reset tick.
    assert vec.envs[0].seed == 3 and vec.envs[0].t == 0
    assert vec.learner_slots() == []
    # Next tick plays seed 3's opponent move 1.
    assert vec.tick({}) == [] and vec.envs[0].t == 1


def test_collect_rollout_interleaved_completes_without_hanging_on_opening_turn_truncation():
    vec = _make_vec(_OpeningTruncOnceEnv, num_envs=1)
    ticks = 0
    inner = vec.tick

    def guarded_tick(actions):
        nonlocal ticks
        ticks += 1
        if ticks > 40:
            raise AssertionError("collect_rollout_interleaved hung after opening-turn truncation")
        return inner(actions)

    vec.tick = guarded_tick
    buf = RolloutBuffer(5, 1, FakeTwoSeatEnv().spec())
    stats = collect_rollout_interleaved(StubNet(), vec, buf, torch.Generator().manual_seed(0))

    # Row 0-1: seed 0 (truncated at row 1); seed 1 truncates on opening turn (no buffer row, length 0);
    # rows 2-4: seed 2 (learner seat 0 moves).
    assert [s.seed for s in stats[:2]] == [0, 1]
    assert stats[0].length == 2
    assert (stats[1].length, stats[1].ret, stats[1].outcome) == (0, 0.0, "draw")
    assert bool(buf.truncated[1, 0]) is True
    assert bool(buf.truncated[2, 0]) is False
    # Rows 2, 3, 4 are seed 2 (parity 0, actions 0, 1, 2).
    assert buf.globals[2, 0].tolist() == [0.0, 0.0, 0.0]
    assert buf.globals[3, 0].tolist() == [0.0, 1.0, 0.0]
    assert buf.globals[4, 0].tolist() == [0.0, 2.0, 0.0]
    assert vec.learner_slots() == [0]
