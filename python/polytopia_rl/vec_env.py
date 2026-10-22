"""A synchronous vector of environments in one process (docs/MILESTONE_10 "Environment wiring").

An env is anything with the `polytopia_sim.Env` methods used here: `reset(seed)`, `observation()`,
`actor_mask()`, `kind_mask(actor)`, `arg_mask(actor, kind)`, `step_factored(actor, kind, arg)`,
`info()` (keys `agent_seat` and `outcome`) and `spec()`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polytopia_sim

TRAIN_SEED_LIMIT = 1_000_000_000  # training seeds are < this; evaluation seeds are >= it


@dataclass
class VecState:
    board: np.ndarray  # (E, C, H, W) float32
    globals: np.ndarray  # (E, G) float32


@dataclass
class VecStep:
    reward: np.ndarray  # (E,) float32
    terminated: np.ndarray  # (E,) bool
    truncated: np.ndarray  # (E,) bool
    final_board: np.ndarray  # (E, C, H, W) float32: the observation before the auto-reset (zeros where not finished)
    final_globals: np.ndarray  # (E, G) float32: likewise


@dataclass
class EpisodeStat:
    env_index: int
    seed: int
    opponent: str | None
    agent_seat: int
    outcome: str  # "win" | "loss" | "draw", from the agent seat's perspective (no outcome: "draw")
    length: int  # agent actions in the episode
    ret: float  # undiscounted sum of the agent's rewards


class SyncVecEnv:
    """Env `i`'s k-th episode uses seed `seed_base + i * seed_stride + k * num_envs * seed_stride`.

    `opponents[i]` labels env `i` in its `EpisodeStat`s. A seed `>= TRAIN_SEED_LIMIT` raises
    `ValueError` (in `__init__` or `reset_all`).
    """

    num_envs: int

    def __init__(
        self,
        env_fns: list[Callable[[], object]],
        seed_base: int,
        seed_stride: int = 1,
        opponents: list[str] | None = None,
    ):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.seed_base = seed_base
        self.seed_stride = seed_stride
        self.opponents = opponents
        spec = self.envs[0].spec()
        self.num_tiles = spec["height"] * spec["width"]
        self.board_shape = (spec["channels"], spec["height"], spec["width"])
        self.num_globals = spec["globals"]
        self._next_k = [0] * self.num_envs  # episodes started per env
        self._seed = [None] * self.num_envs  # seed of the episode in progress
        self._actions = [[] for _ in range(self.num_envs)]  # agent actions of that episode
        self._ret = [0.0] * self.num_envs
        self._stats: list[EpisodeStat] = []
        for i in range(self.num_envs):
            self._check_seed(self._episode_seed(i, 0))

    def _episode_seed(self, i: int, k: int) -> int:
        return self.seed_base + i * self.seed_stride + k * self.num_envs * self.seed_stride

    def _check_seed(self, seed: int) -> None:
        if seed >= TRAIN_SEED_LIMIT:
            raise ValueError(f"training seed {seed} is in the evaluation range (>= {TRAIN_SEED_LIMIT})")

    def _reset(self, i: int) -> None:
        seed = self._episode_seed(i, self._next_k[i])
        self._check_seed(seed)
        self.envs[i].reset(seed)
        self._next_k[i] += 1
        self._seed[i] = seed
        self._actions[i] = []
        self._ret[i] = 0.0

    def reset_all(self) -> VecState:
        """Reset every env with its next seed."""
        for i in range(self.num_envs):
            self._reset(i)
        return VecState(*self.observations())

    def step(self, actors, kinds, args) -> VecStep:
        """Step env `i` with `(actors[i], kinds[i], args[i])`; finished envs record their final
        observation and `EpisodeStat`, then reset with their next seed."""
        e = self.num_envs
        out = VecStep(
            reward=np.zeros(e, dtype=np.float32),
            terminated=np.zeros(e, dtype=bool),
            truncated=np.zeros(e, dtype=bool),
            final_board=np.zeros((e, *self.board_shape), dtype=np.float32),
            final_globals=np.zeros((e, self.num_globals), dtype=np.float32),
        )
        for i, env in enumerate(self.envs):
            action = (int(actors[i]), int(kinds[i]), int(args[i]))
            reward, _, terminated, truncated = env.step_factored(*action)
            self._actions[i].append(action)
            self._ret[i] += float(reward)
            out.reward[i] = reward
            out.terminated[i] = terminated
            out.truncated[i] = truncated
            if terminated or truncated:
                out.final_board[i], out.final_globals[i] = env.observation()
                self._stats.append(self._episode_stat(i))
                self._reset(i)
        return out

    def _episode_stat(self, i: int) -> EpisodeStat:
        info = self.envs[i].info()
        seat = info["agent_seat"]
        if info["outcome"] == f"win_{seat}":
            outcome = "win"
        elif info["outcome"] == f"win_{1 - seat}":
            outcome = "loss"
        else:
            outcome = "draw"
        return EpisodeStat(
            env_index=i,
            seed=self._seed[i],
            opponent=None if self.opponents is None else self.opponents[i],
            agent_seat=seat,
            outcome=outcome,
            length=len(self._actions[i]),
            ret=self._ret[i],
        )

    def observations(self) -> tuple[np.ndarray, np.ndarray]:
        obs = [env.observation() for env in self.envs]
        return np.stack([b for b, _ in obs]), np.stack([g for _, g in obs])

    def actor_masks(self) -> np.ndarray:
        """(E, 3N+1) bool."""
        return np.stack([env.actor_mask() for env in self.envs])

    def kind_masks(self, actors) -> np.ndarray:
        """(E, 10) bool."""
        return np.stack([env.kind_mask(int(a)) for env, a in zip(self.envs, actors)])

    def arg_masks(self, actors, kinds) -> np.ndarray:
        """(E, N) bool, False beyond `arg_len(kind)`; all False for kinds without an argument."""
        out = np.zeros((self.num_envs, self.num_tiles), dtype=bool)
        for i, env in enumerate(self.envs):
            m = env.arg_mask(int(actors[i]), int(kinds[i]))
            out[i, : m.size] = m
        return out

    def episode_stats(self) -> list[EpisodeStat]:
        """Episodes finished since the last call, in the order they finished."""
        stats, self._stats = self._stats, []
        return stats

    def state_dict(self) -> dict:
        """Everything needed to restore the envs exactly: per-env episode index and seed, and the
        agent actions of each episode in progress (restored by reset and replay)."""
        return {
            "seed_base": self.seed_base,
            "seed_stride": self.seed_stride,
            "next_k": list(self._next_k),
            "seeds": list(self._seed),
            "actions": [[list(a) for a in acts] for acts in self._actions],
        }

    def load_state_dict(self, state: dict) -> None:
        self.seed_base = state["seed_base"]
        self.seed_stride = state["seed_stride"]
        for i, env in enumerate(self.envs):
            self._next_k[i] = state["next_k"][i]
            self._seed[i] = state["seeds"][i]
            self._actions[i] = []
            self._ret[i] = 0.0
            env.reset(self._seed[i])
            for action in state["actions"][i]:
                reward, _, terminated, truncated = env.step_factored(*action)
                assert not (terminated or truncated), "a replayed episode ended early"
                self._actions[i].append(tuple(action))
                self._ret[i] += float(reward)
        self._stats = []


def make_vec_env(cfg, seed_base: int, seed_stride: int = 1) -> SyncVecEnv:
    """`cfg.num_envs` envs `polytopia_sim.Env(size, turn_cap, full_observability,
    opponent=cfg.opponents[i % len(cfg.opponents)], opponent_epsilon, agent_seat="seed_parity",
    terminal_weight, shaping_weight, gamma, potential_scale)` from `cfg`, labelled with their opponents."""
    opponents = [cfg.opponents[i % len(cfg.opponents)] for i in range(cfg.num_envs)]

    def env_fn(opponent):
        return lambda: polytopia_sim.Env(
            size=cfg.size,
            turn_cap=cfg.turn_cap,
            full_observability=cfg.full_observability,
            opponent=opponent,
            opponent_epsilon=cfg.opponent_epsilon,
            agent_seat="seed_parity",
            terminal_weight=cfg.terminal_weight,
            shaping_weight=cfg.shaping_weight,
            gamma=cfg.gamma,
            potential_scale=cfg.potential_scale,
        )

    return SyncVecEnv([env_fn(o) for o in opponents], seed_base, seed_stride, opponents=opponents)
