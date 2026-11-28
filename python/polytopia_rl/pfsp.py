"""PFSP matchmaking, bot specs, win-rate tracker and shaping schedule (docs/MILESTONE_23)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BotSpec:
    name: str
    bot: str
    epsilon: float = 0.0
    weight: float = 1.0


def parse_bot_specs(raw: list[dict] | None, default_epsilon: float = 0.0) -> list[BotSpec]:
    raise NotImplementedError


def pfsp_weights(
    win_rates: Sequence[float],
    mode: str,
    power: float = 2.0,
    eps: float = 1e-3,
    base_weights: Sequence[float] | None = None,
) -> np.ndarray:
    raise NotImplementedError


class WinRateTracker:
    def __init__(self, alpha: float = 0.05, prior: float = 0.5):
        raise NotImplementedError

    def record(self, opponent: str, outcome: str) -> float:
        raise NotImplementedError

    def win_rate(self, opponent: str) -> float:
        raise NotImplementedError

    def games(self, opponent: str) -> int:
        raise NotImplementedError

    def state_dict(self) -> dict:
        raise NotImplementedError

    def load_state_dict(self, state: dict) -> None:
        raise NotImplementedError


def shaping_schedule(
    update: int,
    start: float,
    end: float,
    anneal_updates: int | None,
) -> float:
    raise NotImplementedError
