"""A synchronous vector of environments in one process (docs/MILESTONE_10 "Environment wiring").

An env is anything with the `polytopia_sim.Env` methods used here: `reset(seed)`, `observation()`,
`actor_mask()`, `kind_mask(actor)`, `arg_mask(actor, kind)`, `step_factored(actor, kind, arg)`,
`info()` (keys `agent_seat` and `outcome`) and `spec()`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

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
        raise NotImplementedError

    def reset_all(self) -> VecState:
        """Reset every env with its next seed."""
        raise NotImplementedError

    def step(self, actors, kinds, args) -> VecStep:
        """Step env `i` with `(actors[i], kinds[i], args[i])`; finished envs record their final
        observation and `EpisodeStat`, then reset with their next seed."""
        raise NotImplementedError

    def observations(self) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def actor_masks(self) -> np.ndarray:
        """(E, 3N+1) bool."""
        raise NotImplementedError

    def kind_masks(self, actors) -> np.ndarray:
        """(E, 10) bool."""
        raise NotImplementedError

    def arg_masks(self, actors, kinds) -> np.ndarray:
        """(E, N) bool, False beyond `arg_len(kind)`; all False for kinds without an argument."""
        raise NotImplementedError

    def episode_stats(self) -> list[EpisodeStat]:
        """Episodes finished since the last call, in the order they finished."""
        raise NotImplementedError

    def state_dict(self) -> dict:
        """Everything needed to restore the envs exactly: per-env episode index and seed, and the
        agent actions of each episode in progress (restored by reset and replay)."""
        raise NotImplementedError

    def load_state_dict(self, state: dict) -> None:
        raise NotImplementedError


def make_vec_env(cfg, seed_base: int, seed_stride: int = 1) -> SyncVecEnv:
    """`cfg.num_envs` envs `polytopia_sim.Env(size, turn_cap, full_observability,
    opponent=cfg.opponents[i % len(cfg.opponents)], opponent_epsilon, agent_seat="seed_parity",
    terminal_weight, shaping_weight, gamma, potential_scale)` from `cfg`, labelled with their opponents."""
    raise NotImplementedError
