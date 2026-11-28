"""PFSP matchmaking, bot specs, win-rate tracker and shaping schedule (docs/MILESTONE_23)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

_VALID_BOTS = ("random", "economy", "rush")
_BOT_SPEC_KEYS = frozenset({"name", "bot", "epsilon", "weight"})
_OUTCOME_SCORES = {"win": 1.0, "draw": 0.5, "loss": 0.0}


@dataclass(frozen=True)
class BotSpec:
    name: str
    bot: str
    epsilon: float = 0.0
    weight: float = 1.0


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def parse_bot_specs(raw: list[dict] | None, default_epsilon: float = 0.0) -> list[BotSpec]:
    if not _is_number(default_epsilon) or not (0.0 <= float(default_epsilon) <= 1.0):
        raise ValueError(f"default_epsilon {default_epsilon!r} must be in [0, 1]")
    if raw is None:
        eps = float(default_epsilon)
        return [
            BotSpec("economy", "economy", eps, 1.0),
            BotSpec("rush", "rush", eps, 1.0),
        ]
    if not isinstance(raw, list) or len(raw) == 0:
        raise ValueError("selfplay_bots must be a non-empty list of bot spec dicts")
    seen: set[str] = set()
    specs: list[BotSpec] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError(f"bot spec {item!r} must be a dict")
        extra = set(item.keys()) - _BOT_SPEC_KEYS
        if extra:
            raise ValueError(f"unknown bot spec keys: {sorted(extra)}")
        if "name" not in item or "bot" not in item:
            raise ValueError(f"bot spec {item!r} must include 'name' and 'bot'")
        name = item["name"]
        if not isinstance(name, str) or not name:
            raise ValueError(f"bot spec name {name!r} must be a non-empty str")
        if name == "self" or name.startswith("snap_") or name.startswith("anchor_"):
            raise ValueError(f"bot spec name {name!r} uses a reserved prefix/name")
        if name in seen:
            raise ValueError(f"duplicate bot spec name {name!r}")
        bot = item["bot"]
        if not isinstance(bot, str) or bot not in _VALID_BOTS:
            raise ValueError(f"bot {bot!r} must be one of {_VALID_BOTS}")
        epsilon = item.get("epsilon", 0.0)
        if not _is_number(epsilon) or not (0.0 <= float(epsilon) <= 1.0):
            raise ValueError(f"epsilon {epsilon!r} must be in [0, 1]")
        weight = item.get("weight", 1.0)
        if not _is_number(weight) or float(weight) <= 0.0:
            raise ValueError(f"weight {weight!r} must be > 0")
        seen.add(name)
        specs.append(BotSpec(name=name, bot=bot, epsilon=float(epsilon), weight=float(weight)))
    return specs


def pfsp_weights(
    win_rates: Sequence[float],
    mode: str,
    power: float = 2.0,
    eps: float = 1e-3,
    base_weights: Sequence[float] | None = None,
) -> np.ndarray:
    if len(win_rates) == 0:
        raise ValueError("win_rates must be non-empty")
    if mode not in ("off", "hard", "variance"):
        raise ValueError(f"pfsp mode {mode!r} must be 'off', 'hard', or 'variance'")
    if not _is_number(power) or float(power) <= 0.0:
        raise ValueError(f"power {power!r} must be > 0")
    if not _is_number(eps) or float(eps) < 0.0:
        raise ValueError(f"eps {eps!r} must be >= 0")

    raw_w = np.asarray(win_rates, dtype=np.float64)
    if raw_w.ndim != 1 or np.any(~np.isfinite(raw_w)):
        raise ValueError("win_rates must be a 1D sequence of finite numbers")
    w = np.clip(raw_w, 0.0, 1.0)
    if base_weights is None:
        b = np.ones(len(w), dtype=np.float64)
    else:
        b = np.asarray(base_weights, dtype=np.float64)
        if b.ndim != 1 or len(b) != len(w) or np.any(~np.isfinite(b)) or np.any(b <= 0.0):
            raise ValueError("base_weights must match win_rates length with strictly positive finite values")

    p = float(power)
    e = float(eps)
    if mode == "off":
        raw = b.copy()
    elif mode == "hard":
        raw = b * ((1.0 - w) ** p + e)
    else:
        raw = b * ((w * (1.0 - w)) ** p + e)

    total = float(raw.sum())
    if total <= 0.0 or not math.isfinite(total):
        return b / float(b.sum())
    return raw / total


class WinRateTracker:
    def __init__(self, alpha: float = 0.05, prior: float = 0.5):
        if not _is_number(alpha) or not (0.0 < float(alpha) <= 1.0):
            raise ValueError(f"alpha {alpha!r} must be in (0, 1]")
        if not _is_number(prior) or not (0.0 <= float(prior) <= 1.0):
            raise ValueError(f"prior {prior!r} must be in [0, 1]")
        self.alpha = float(alpha)
        self.prior = float(prior)
        self._stats: dict[str, tuple[float, int]] = {}

    def record(self, opponent: str, outcome: str) -> float:
        if outcome not in _OUTCOME_SCORES:
            raise ValueError(f"outcome {outcome!r} must be 'win', 'draw', or 'loss'")
        score = _OUTCOME_SCORES[outcome]
        prev, count = self._stats.get(opponent, (self.prior, 0))
        n = count + 1
        eta = max(self.alpha, 1.0 / n)
        new = prev + eta * (score - prev)
        self._stats[opponent] = (new, n)
        return new

    def win_rate(self, opponent: str) -> float:
        entry = self._stats.get(opponent)
        return self.prior if entry is None else entry[0]

    def games(self, opponent: str) -> int:
        entry = self._stats.get(opponent)
        return 0 if entry is None else entry[1]

    def state_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "prior": self.prior,
            "stats": {name: [ema, count] for name, (ema, count) in sorted(self._stats.items())},
        }

    def load_state_dict(self, state: dict) -> None:
        self.alpha = float(state["alpha"])
        self.prior = float(state["prior"])
        self._stats = {str(name): (float(vals[0]), int(vals[1])) for name, vals in state["stats"].items()}


def shaping_schedule(
    update: int,
    start: float,
    end: float,
    anneal_updates: int | None,
) -> float:
    if anneal_updates is None:
        return float(start)
    if anneal_updates <= 0:
        raise ValueError(f"anneal_updates {anneal_updates!r} must be > 0")
    frac = min(1.0, max(0.0, (update - 1) / anneal_updates))
    return float(start + frac * (end - start))
