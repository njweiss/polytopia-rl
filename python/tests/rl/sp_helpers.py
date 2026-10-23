"""Shared helpers for the milestone 11 tests. Nothing here imports the code under test.

- `fnv1a64`, `rules_bytes`: independent re-implementations of the brief's hash definitions.
- `bt_reference`: an independent Bradley-Terry fit (gradient ascent on the log-likelihood in Elo
  units, with the brief's 0.5-0.5 prior per played pair), used to check fitted ratings.
- `FakeTwoSeatEnv`: a scripted two-seat game for the self-play env.
- Scripted policies: the Rust bots through `Env.bot_action`, end-turn, and a pseudo-random policy
  that is a pure function of `(seed, seat, agent actions so far)` so games can be replayed.
"""

import pathlib

import numpy as np
from rl_helpers import END_TURN, MOVE, arg_lens, end_turn_path, random_bot_path

REPO = pathlib.Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------- hashes


def fnv1a64(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def hex16(data: bytes) -> str:
    return f"{fnv1a64(data):016x}"


def obs_canonical_from_spec(spec, size):
    """The brief's canonical string, built from the spec's lists (docs/QUESTIONS.md Q22)."""
    items = [str(size)]
    items += list(spec["channel_names"]) + list(spec["global_names"]) + list(spec["action_types"])
    items += [str(n) for n in spec["arg_lens"]]
    for key in ("unit_types", "rewards", "techs", "site_actions"):
        items += list(spec[key])
    return "\n".join(items)


RULES_VERSION = 1  # docs/MILESTONE_11 "Notes for the implementer": starts at 1


def rules_bytes(version=RULES_VERSION) -> bytes:
    out = b""
    for p in sorted((REPO / "data").glob("*.json"), key=lambda p: p.name):
        out += p.name.encode() + b"\0" + p.read_bytes()
    return out + b"RULES_VERSION\0" + str(version).encode()


# ---------------------------------------------------------------- ratings


def bt_reference(wins, games, anchor, anchor_rating=0.0, iters=200):
    """Maximise sum_ij w'_ij * log P(i beats j) with P = 1 / (1 + 10^((r_j - r_i) / 400)) and
    w' = wins + 0.5, n' = games + 1 on played pairs, by full Newton steps with r[anchor] fixed."""
    wins = np.asarray(wins, dtype=np.float64)
    games = np.asarray(games, dtype=np.float64)
    played = games > 0
    w = np.where(played, wins + 0.5, 0.0)
    n = np.where(played, games + 1.0, 0.0)
    k = np.log(10) / 400
    free = [i for i in range(len(w)) if i != anchor]
    r = np.zeros(len(w))
    for _ in range(iters):
        p = 1 / (1 + np.exp(-k * (r[:, None] - r[None, :])))
        grad = k * (w - n * p).sum(axis=1)
        c = k * k * n * p * (1 - p)  # c[i, j] = -d2L / dr_i dr_j for i != j (negated)
        hess = np.diag(c.sum(axis=1)) - c  # -Hessian, a weighted graph Laplacian
        step = np.linalg.solve(hess[np.ix_(free, free)], grad[free])
        r[free] += step
        if np.abs(step).max() < 1e-12:
            break
    return r + anchor_rating


# ---------------------------------------------------------------- scripted policies


def bot_policy(name):
    """`policy_fn(env, greedy)` that plays exactly what Rust bot `name` would play."""

    def policy(env, greedy):
        return env.bot_action(name)

    return policy


def bot_batch(name):
    """A batch policy (list of envs -> list of actions) playing Rust bot `name`."""

    def policy(envs):
        return [env.bot_action(name) for env in envs]

    return policy


def end_turn_policy(env, greedy):
    return end_turn_path(env)


def keyed_random_policy(salt=0):
    """Random-bot imitation whose choice depends only on (seed, seat, agent actions so far, salt):
    replaying the same game reproduces it exactly."""

    def policy(env, greedy):
        info = env.info()
        rng = np.random.default_rng([info["seed"], info["agent_seat"], info["agent_actions"], salt])
        return random_bot_path(env, rng)

    return policy


def my_city_tiles(env):
    """Tiles (row, col) that show `city.mine` in the env's current observation."""
    board, _ = env.observation()
    c = list(env.spec()["channel_names"]).index("city.mine")
    return {(int(r), int(q)) for r, q in zip(*np.nonzero(board[c] > 0.5))}


def has_own_capital(env):
    board, _ = env.observation()
    names = list(env.spec()["channel_names"])
    mine, cap = board[names.index("city.mine")], board[names.index("city.capital")]
    return bool(((mine > 0.5) & (cap > 0.5)).any())


def play_out(env, policy, greedy=True):
    """Play `policy` (agent seat, bot mode) to the end; returns (length, first_city_lost_turn), with
    the brief's detection rule written independently (docs/QUESTIONS.md Q24)."""
    length = 0
    prev = my_city_tiles(env)
    first_lost = None
    while True:
        a = policy(env, greedy)
        _, _, term, trunc = env.step_factored(*a)
        length += 1
        cur = my_city_tiles(env)
        if first_lost is None and prev - cur:
            first_lost = env.info()["turn"]
        prev = cur
        if term or trunc:
            return length, first_lost


# ---------------------------------------------------------------- fake two-seat game


class FakeTwoSeatEnv:
    """A scripted two-seat game. Seat 0 moves first; every turn of either seat is `k` actions:
    `k - 1` Moves (unit actor 0, kind MOVE, arg 0 or 1; arg 1 adds 1 to the mover's score) and then
    the global actor's EndTurn. `turn` increases when seat 1 ends its turn; when it reaches `turns`
    the game ends by score (`win_0`, `win_1` or `draw`). With `truncate_at`, the episode is truncated
    once that many actions (both seats) have been played, if not terminal.

    `step_factored` returns a reward of 0 (the self-play env must compute its own); stepping a
    finished game raises RuntimeError. Observation (the current player's, or the last actor's once
    finished): board channel 0 = observer seat, channel 1 = actions played; globals
    `[observer, actions played, seed % 2]`.
    """

    def __init__(self, k=3, turns=2, truncate_at=None, size=4):
        self.k, self.turns, self.truncate_at = k, turns, truncate_at
        self.size = size
        self.n = size * size
        self.global_actor = 3 * self.n
        self.seed = None

    def spec(self):
        return {
            "height": self.size,
            "width": self.size,
            "channels": 2,
            "globals": 3,
            "num_actors": 3 * self.n + 1,
            "arg_lens": arg_lens(self.n),
            "action_types": ["move", "attack", "capture", "recover", "promote", "train", "choose_reward", "site", "research", "end_turn"],
        }

    def reset(self, seed):
        self.seed = int(seed)
        self.t = 0
        self.in_turn = 0
        self.current = 0
        self.last_actor = 0
        self.turn = 0
        self.scores = [0, 0]
        self.outcome = None
        self.done = False
        self.truncated = False

    def _paths(self):
        if self.done:
            return {}
        if self.in_turn < self.k - 1:
            return {(0, MOVE): [0, 1]}
        return {(self.global_actor, END_TURN): [0]}

    def observer(self):
        return self.last_actor if self.done else self.current

    def observation(self):
        board = np.zeros((2, self.size, self.size), dtype=np.float32)
        board[0] = self.observer()
        board[1] = self.t
        return board, np.array([self.observer(), self.t, self.seed % 2], dtype=np.float32)

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
        self.last_actor = self.current
        self.t += 1
        terminated = False
        if kind == MOVE:
            self.scores[self.current] += arg
            self.in_turn += 1
        else:
            self.in_turn = 0
            if self.current == 1:
                self.turn += 1
                if self.turn >= self.turns:
                    terminated = True
                    s0, s1 = self.scores
                    self.outcome = "win_0" if s0 > s1 else "win_1" if s1 > s0 else "draw"
            self.current = 1 - self.current
        truncated = (not terminated) and self.truncate_at is not None and self.t >= self.truncate_at
        self.done = terminated or truncated
        return 0.0, np.zeros(2, dtype=np.float32), terminated, truncated

    def info(self):
        return {
            "seed": self.seed,
            "turn": self.turn,
            "current_player": self.current,
            "agent_seat": 0,
            "scores": tuple(self.scores),
            "outcome": self.outcome,
            "agent_actions": self.t,
        }


def fake_scorer(envs):
    """Batch policy for FakeTwoSeatEnv: Move with arg 1 while a Move is legal, else EndTurn."""
    out = []
    for env in envs:
        if env.kind_mask(0)[MOVE]:
            out.append((0, MOVE, 1))
        else:
            out.append((env.global_actor, END_TURN, 0))
    return out
