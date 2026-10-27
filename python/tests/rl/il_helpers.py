"""Shared helpers for the milestone 13 tests (interleaved self-play rollout).

- `StubNet`: a stand-in for `PolicyNet` with the methods the collectors use (`trunk`, `act`,
  `value`). Its decisions never depend on the batch a row is in nor on the generator, so the lockstep
  and the interleaved collectors see the same learner decisions per slot. Log-probs and values are
  fixed functions of the observation bytes (crc32), so a row written to the wrong place, or a
  bootstrap value taken from the wrong observation, shows up as a different number.
- `BotLearner`: a `StubNet` decision rule that plays a Rust bot (`Env.bot_action`) for the learner
  seat of the slot whose current observation the row is. `bot_action` draws from the env's own
  per-seat rng, advanced only by that call, so each slot's moves do not depend on the slot order.
- `FakeBotEnv`: the bot-mode counterpart of `sp_helpers.FakeTwoSeatEnv`.
- `keyed_vec`: a `SelfPlayVecEnv` whose episode opponents are a fixed function of the episode seed
  (the shared opponent-draw stream is consumed in reset order, which the two collectors are allowed
  to differ in), through a test subclass that overrides `_start`.

Nothing here computes an expected value with the code under test.
"""

import zlib

import numpy as np
import torch
from rl_helpers import END_TURN, MOVE
from sp_helpers import FakeTwoSeatEnv, fake_scorer

from polytopia_rl.model import ActOut
from polytopia_rl.selfplay import SelfPlayVecEnv, SnapshotPool

# ---------------------------------------------------------------- stub learner


def obs_key(board, glob) -> int:
    b = np.ascontiguousarray(np.asarray(board, dtype=np.float32)).tobytes()
    g = np.ascontiguousarray(np.asarray(glob, dtype=np.float32)).tobytes()
    return zlib.crc32(b + b"|" + g)


def stub_value(board, glob) -> np.float32:
    return np.float32((obs_key(board, glob) % 9973) / 9973.0 - 0.5)


def stub_logp(board, glob) -> np.float32:
    return np.float32(-(obs_key(board, glob) % 997) / 100.0)


def _np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class StubFeatures:
    def __init__(self, board, glob):
        self.board = _np(board)
        self.glob = _np(glob)


class StubNet:
    """`act` fetches the masks in the `PolicyNet.act` order (actor, kind with actors, arg with actors
    and kinds) and checks the chosen path against them. With `decide=None` the path is the first legal
    actor, its first legal kind and its last legal argument (0 when the kind has none); otherwise
    `decide(boards, globs)` returns one `(actor, kind, arg)` per row. `act_sizes` records the batch
    size of every `act` call; `rows` every (board, glob) the learner acted on, in call order."""

    def __init__(self, decide=None):
        self.decide = decide
        self.act_sizes = []
        self.rows = []

    def trunk(self, board, glob):
        return StubFeatures(board, glob)

    def value(self, feats):
        return torch.tensor([stub_value(b, g) for b, g in zip(feats.board, feats.glob)], dtype=torch.float32)

    def act(self, feats, masks_fn, generator, greedy=False, stats=True):
        n = len(feats.board)
        self.act_sizes.append(n)
        self.rows += [(b.copy(), g.copy()) for b, g in zip(feats.board, feats.glob)]
        fixed = self.decide(feats.board, feats.glob) if self.decide is not None else None
        actor_mask = np.asarray(_np(masks_fn("actor")), dtype=bool)
        if fixed is not None:
            actors = np.array([a for a, _, _ in fixed], dtype=np.int64)
        else:
            actors = np.array([int(np.flatnonzero(m)[0]) for m in actor_mask], dtype=np.int64)
        assert all(actor_mask[r, actors[r]] for r in range(n)), "learner actor not in its mask"
        kind_mask = np.asarray(_np(masks_fn("kind", actors)), dtype=bool)
        if fixed is not None:
            kinds = np.array([k for _, k, _ in fixed], dtype=np.int64)
        else:
            kinds = np.array([int(np.flatnonzero(m)[0]) for m in kind_mask], dtype=np.int64)
        assert all(kind_mask[r, kinds[r]] for r in range(n)), "learner kind not in its mask"
        arg_mask = np.asarray(_np(masks_fn("arg", actors, kinds)), dtype=bool)
        if fixed is not None:
            args = np.array([x for _, _, x in fixed], dtype=np.int64)
        else:
            args = np.array([int(np.flatnonzero(m)[-1]) if m.any() else 0 for m in arg_mask], dtype=np.int64)
        assert all(arg_mask[r, args[r]] or not arg_mask[r].any() for r in range(n)), "learner arg not in its mask"
        logp = torch.tensor([stub_logp(b, g) for b, g in zip(feats.board, feats.glob)], dtype=torch.float32)
        value = self.value(feats)
        t = torch.from_numpy
        return ActOut(
            actors=t(actors),
            kinds=t(kinds),
            args=t(args),
            logp=logp if stats else None,
            entropy=torch.zeros(n) if stats else None,
            value=value if stats else None,
            actor_mask=t(actor_mask),
            kind_mask=t(kind_mask),
            arg_mask=t(arg_mask),
        )


class BotLearner:
    """`decide` for `StubNet`: for each row, the slot of `vec` whose env is on its learner's move
    (`current_player == seed % 2`) with exactly that observation, and that env's
    `bot_action(name_of(seed))`."""

    def __init__(self, vec, name_of):
        self.vec = vec
        self.name_of = name_of

    def __call__(self, boards, globs):
        index = {}
        for i, env in enumerate(self.vec.envs):
            info = env.info()
            if info["outcome"] is None and info["current_player"] == info["seed"] % 2:
                key = obs_key(*env.observation())
                assert key not in index, "two slots with the same observation: the fixture cannot tell them apart"
                index[key] = i
        out = []
        for b, g in zip(boards, globs):
            i = index.pop(obs_key(b, g))  # KeyError: a row that is no slot's current learner observation
            env = self.vec.envs[i]
            out.append(tuple(int(x) for x in env.bot_action(self.name_of(env.info()["seed"]))))
        return out


# ---------------------------------------------------------------- vec envs


def reward_cfg(shaping=1.0, gamma=0.999, scale=1000.0):
    return dict(terminal_weight=1.0, shaping_weight=shaping, gamma=gamma, potential_scale=scale)


def pool_of(n):
    """Entries `snap_0`, `snap_100`, ... with dummy weights."""
    pool = SnapshotPool()
    for s in range(n):
        pool.add({"w": torch.zeros(1)}, s * 100)
    return pool


class KeyedVec(SelfPlayVecEnv):
    """Every episode's opponent is `choices[pick(seed)]` (bot names or `Entry`s), whatever the
    shared draw returned."""

    def __init__(self, *args, choices, pick, **kw):
        self.choices = choices
        self.pick = pick
        super().__init__(*args, **kw)

    def _start(self, i, seed, opp):
        super()._start(i, seed, self.choices[self.pick(seed)])


def keyed_pick(n, salt=99):
    return lambda seed: int(np.random.default_rng([seed, salt]).integers(n))


def keyed_vec(env_fn, num_envs, *, pool, make_policy, choices, pick, seed_base=0, seed_stride=1, reward=None, rng_seed=0, p_bot=0.4):
    """A `KeyedVec`; `choices` items are bot names or indices into `pool.entries`."""
    return KeyedVec(
        env_fn,
        num_envs,
        seed_base,
        seed_stride,
        pool=pool,
        make_policy=make_policy,
        rng=np.random.default_rng(rng_seed),
        p_bot=p_bot,
        reward=reward or reward_cfg(),
        choices=[c if isinstance(c, str) else pool.entries[c] for c in choices],
        pick=pick,
    )


# ---------------------------------------------------------------- fake envs


class FakeBotEnv(FakeTwoSeatEnv):
    """Bot mode on the fake game: the learner is seat `seed % 2`; the opponent plays `fake_scorer`
    inside `reset` and `step_factored` until the learner is to move or the game is over. The reward
    of a learner step is `0.125 * (learner actions so far)` (exact in float32, and never the
    self-play formula's value)."""

    def reset(self, seed):
        super().reset(seed)
        self.agent = int(seed) % 2
        self.agent_t = 0
        self._run_opponent()

    def _run_opponent(self):
        while not self.done and self.current != self.agent:
            FakeTwoSeatEnv.step_factored(self, *fake_scorer([self])[0])

    def step_factored(self, actor, kind, arg):
        FakeTwoSeatEnv.step_factored(self, actor, kind, arg)
        self.agent_t += 1
        self._run_opponent()
        terminated = self.outcome is not None
        truncated = self.done and not terminated
        return 0.125 * self.agent_t, np.zeros(2, dtype=np.float32), terminated, truncated

    def info(self):
        d = super().info()
        d["agent_seat"] = self.agent
        return d


def learner_path(env):
    """The fake learner's scripted move: Move with arg 1 while a Move is legal, else EndTurn."""
    if env.kind_mask(0)[MOVE]:
        return (0, MOVE, 1)
    return (env.global_actor, END_TURN, 0)

