"""Opponent curriculum (docs/MILESTONE_10 "Curriculum")."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StageHistory:
    steps_in_stage: int = 0  # agent steps trained in the current stage
    outcomes: list[str] = field(default_factory=list)  # finished episodes in this stage, oldest first


def next_stage(stage: int, history: StageHistory, cfg) -> int:
    """`stage + 1` when `history.steps_in_stage >= cfg.min_stage_steps` and the win rate of the last
    `cfg.stage_window` outcomes (the window must be full) is `>= cfg.stage_win_rate`; the last stage
    (`len(cfg.stages) - 1`) never advances. Otherwise `stage`."""
    raise NotImplementedError
