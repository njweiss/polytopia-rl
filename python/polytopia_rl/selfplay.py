"""Snapshot pool, matchmaking and the self-play vector env (docs/MILESTONE_11 "Self-play").

An env is anything with the `polytopia_sim.Env` methods `reset(seed)`, `observation()`,
`actor_mask()`, `kind_mask(actor)`, `arg_mask(actor, kind)`, `step_factored(actor, kind, arg)`,
`info()` (keys `seed`, `turn`, `current_player`, `agent_seat`, `scores`, `outcome`,
`agent_actions`) and `spec()`.

A batch policy is a callable `policy(envs) -> list[(actor, kind, arg)]` that chooses one action for
each env (all on the move of the seat it plays) in a single batched forward pass.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

BOT_OPPONENTS = ("economy", "rush")


@dataclass
class Entry:
    step: int
    state: dict  # a frozen copy of the learner's state dict at `step` (CPU tensors)
    name: str  # f"snap_{step}"


class SnapshotPool:
    """Snapshots in the order added. `add` appends, then thins while there are more than
    `pool_max` entries."""

    def __init__(self, pool_max: int = 40, p_latest: float = 0.5):
        self.pool_max = pool_max
        self.p_latest = p_latest
        self.entries: list[Entry] = []

    def add(self, model_state: dict, step: int) -> Entry:
        """Append a deep copy of `model_state` (later changes to the learner do not reach it)."""
        raise NotImplementedError

    def thin(self) -> None:
        """Keep the entries at even positions (0, 2, 4, ...) and the newest; drop the rest."""
        raise NotImplementedError

    def sample(self, rng: np.random.Generator) -> Entry:
        """The newest entry with probability `p_latest`, else an entry drawn uniformly from all of
        them (the newest included)."""
        raise NotImplementedError


def draw_opponent(pool: SnapshotPool, rng: np.random.Generator, p_bot: float):
    """With probability `p_bot` a bot name drawn uniformly from `BOT_OPPONENTS`, else
    `pool.sample(rng)` (an `Entry`)."""
    raise NotImplementedError


def net_batch_policy(net, generator, greedy: bool = False, device: str = "cpu") -> Callable:
    """A batch policy for a `PolicyNet`: one `trunk` + `act` call for all the envs, masks from
    each env's own observation and masks; sampled with `generator`, or argmax when `greedy`."""
    raise NotImplementedError


def regression(win_rates: dict, floor: float) -> bool:
    """True when any win rate is below `floor`."""
    raise NotImplementedError


class SelfPlayVecEnv:
    """The `SyncVecEnv` interface (`num_envs`, `reset_all`, `step`, `observations`, `actor_masks`,
    `kind_masks`, `arg_masks`, `episode_stats`, `state_dict`, `load_state_dict`) for a learner
    playing against bots and pool snapshots.

    `env_fn(opponent)` builds an env: `opponent` is a bot name (bot mode, agent seat from seed
    parity) or `"none"` (two-seat, for snapshot opponents). Seeds follow `SyncVecEnv`; the learner's
    seat is `seed % 2`. At each episode start the slot's opponent is `draw_opponent(pool, rng,
    p_bot)`. Snapshot slots play `make_policy(entry)(envs)` for their opponent moves, grouped by
    snapshot, one call per group per round, until every slot is terminal or on the learner's move.
    The learner's reward in a snapshot slot is the M9 formula over the learner's action and the whole
    opponent turn, with `reward` = {terminal_weight, shaping_weight, gamma, potential_scale}.
    `EpisodeStat.opponent` is the bot name or the entry's name. `envs[i]` is the env slot `i` is
    currently playing in.
    """

    def __init__(
        self,
        env_fn: Callable,
        num_envs: int,
        seed_base: int,
        seed_stride: int = 1,
        *,
        pool: SnapshotPool,
        make_policy: Callable,
        rng: np.random.Generator,
        p_bot: float = 0.2,
        reward: dict,
    ):
        raise NotImplementedError
