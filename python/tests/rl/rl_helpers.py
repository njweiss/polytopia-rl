"""Shared helpers for the milestone 10 tests: a scripted `FakeEnv`, NumPy reference math and
simple scripted policies. Nothing here imports the code under test."""

import numpy as np

# Action-type indices (docs/MILESTONE_9 "Factored action space", `ActionType` declaration order).
MOVE, ATTACK, CAPTURE, RECOVER, PROMOTE, TRAIN, CHOOSE_REWARD, SITE, RESEARCH, END_TURN = range(10)
ACTION_TYPES = [
    "move",
    "attack",
    "capture",
    "recover",
    "promote",
    "train",
    "choose_reward",
    "site",
    "research",
    "end_turn",
]
NO_ARG_KINDS = (CAPTURE, RECOVER, PROMOTE, END_TURN)
# `arg_len` for the enum kinds: 5 unit types, 8 rewards, 6 site actions, 9 techs (docs/MILESTONE_9).
ENUM_ARG_LENS = {TRAIN: 5, CHOOSE_REWARD: 8, SITE: 6, RESEARCH: 9}


def arg_lens(n_tiles):
    return [n_tiles, n_tiles, 0, 0, 0, 5, 8, 6, 9, 0]


# ---------------------------------------------------------------- NumPy references


def np_masked_logp(logits, mask):
    """Log-probabilities of the masked softmax, float64; masked entries are -inf."""
    x = np.where(mask, np.asarray(logits, dtype=np.float64), -np.inf)
    m = x.max(axis=-1, keepdims=True)
    return x - (m + np.log(np.exp(x - m).sum(axis=-1, keepdims=True)))


def np_masked_entropy(logits, mask):
    lp = np_masked_logp(logits, mask)
    p = np.exp(lp)
    return -np.where(mask, p * np.where(mask, lp, 0.0), 0.0).sum(axis=-1)


# ---------------------------------------------------------------- FakeEnv


class FakeEnv:
    """A tiny scripted env with the `polytopia_sim.Env` methods the vector env uses.

    An episode is `pre_steps` forced steps (only the global actor's EndTurn is legal) followed by one
    choice step whose legal paths are `legal`: {(actor, kind): [args]}. The choice step pays +1 iff
    the chosen path is `target`, and terminates. If `truncate_at` is set, the episode is truncated
    after that many steps instead (when it comes first).

    Observation: board channel 0 is the step counter `t` everywhere, channel 1 is `seed % 2`
    everywhere; globals are `[t, seed % 2, 1]`.
    """

    def __init__(self, legal=None, target=None, size=4, pre_steps=1, truncate_at=None):
        self.size = size
        self.n = size * size
        self.global_actor = 3 * self.n
        self.legal = legal or {}
        self.target = target
        self.pre_steps = pre_steps
        self.truncate_at = truncate_at
        self.seed = None
        self.done = True

    def spec(self):
        return {
            "height": self.size,
            "width": self.size,
            "channels": 2,
            "globals": 3,
            "num_actors": 3 * self.n + 1,
            "arg_lens": arg_lens(self.n),
            "action_types": list(ACTION_TYPES),
        }

    def reset(self, seed):
        self.seed = int(seed)
        self.t = 0
        self.done = False
        self.outcome = None

    def _paths(self):
        if self.done:
            return {}
        if self.t < self.pre_steps:
            return {(self.global_actor, END_TURN): [0]}
        return self.legal

    def observation(self):
        board = np.zeros((2, self.size, self.size), dtype=np.float32)
        board[0] = self.t
        board[1] = self.seed % 2
        return board, np.array([self.t, self.seed % 2, 1.0], dtype=np.float32)

    def actor_mask(self):
        m = np.zeros(3 * self.n + 1, dtype=bool)
        for a, _ in self._paths():
            m[a] = True
        return m

    def kind_mask(self, actor):
        m = np.zeros(10, dtype=bool)
        for a, k in self._paths():
            if a == actor:
                m[k] = True
        return m

    def arg_mask(self, actor, kind):
        m = np.zeros(arg_lens(self.n)[kind], dtype=bool)
        for x in self._paths().get((actor, kind), []):
            if m.size:
                m[x] = True
        return m

    def step_factored(self, actor, kind, arg):
        if self.done:
            raise RuntimeError("Finished")
        actor, kind, arg = int(actor), int(kind), int(arg)
        if arg not in self._paths().get((actor, kind), []):
            raise ValueError(f"Illegal {(actor, kind, arg)}")
        reward = 0.0
        terminated = False
        if self.t >= self.pre_steps:
            terminated = True
            if (actor, kind, arg) == self.target:
                reward = 1.0
                self.outcome = "win_0"
            else:
                self.outcome = "win_1"
        self.t += 1
        truncated = (not terminated) and self.truncate_at is not None and self.t >= self.truncate_at
        self.done = terminated or truncated
        return reward, np.array([reward, -reward], dtype=np.float32), terminated, truncated

    def info(self):
        return {"seed": self.seed, "agent_seat": 0, "outcome": self.outcome, "agent_actions": self.t}


# ---------------------------------------------------------------- scripted choices on a real Env


def end_turn_path(env):
    """EndTurn when it is legal, else the first legal (actor, kind, arg) in index order."""
    am = env.actor_mask()
    g = len(am) - 1
    if am[g] and env.kind_mask(g)[END_TURN]:
        return g, END_TURN, 0
    return first_path(env)


def first_path(env):
    a = int(np.flatnonzero(env.actor_mask())[0])
    k = int(np.flatnonzero(env.kind_mask(a))[0])
    xm = env.arg_mask(a, k)
    return a, k, (int(np.flatnonzero(xm)[0]) if xm.size else 0)


def random_path(env, rng):
    """A uniformly random actor, then kind, then argument among the true mask entries."""
    a = int(rng.choice(np.flatnonzero(env.actor_mask())))
    k = int(rng.choice(np.flatnonzero(env.kind_mask(a))))
    xm = env.arg_mask(a, k)
    return a, k, (int(rng.choice(np.flatnonzero(xm))) if xm.size else 0)


def all_paths(env):
    out = []
    for a in np.flatnonzero(env.actor_mask()):
        for k in np.flatnonzero(env.kind_mask(int(a))):
            xm = env.arg_mask(int(a), int(k))
            if xm.size:
                out += [(int(a), int(k), int(x)) for x in np.flatnonzero(xm)]
            else:
                out.append((int(a), int(k), 0))
    return out


def random_bot_path(env, rng):
    """Imitates the milestone 8 random bot (crates/bots/src/random.rs): EndTurn with probability
    0.15 when other actions exist, else uniform over the other legal actions."""
    paths = all_paths(env)
    others = [p for p in paths if p[1] != END_TURN]
    if len(others) < len(paths):
        if not others or rng.random() < 0.15:
            return next(p for p in paths if p[1] == END_TURN)
    return others[int(rng.integers(len(others)))]
