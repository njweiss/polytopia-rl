"""Milestone 10 tests 8: `next_stage` truth table ("Curriculum"). The env rebuild (policy and
optimizer persist) is covered by test_train.py."""

import pytest

from polytopia_rl.config import PPOConfig
from polytopia_rl.curriculum import StageHistory, next_stage

CFG = PPOConfig(
    stages=[{"opponents": ["random"]}, {"opponents": ["economy"]}, {"opponents": ["economy", "rush"]}],
    min_stage_steps=100,
    stage_window=4,
    stage_win_rate=0.75,
)
W, L, D = "win", "loss", "draw"


@pytest.mark.parametrize(
    "stage,steps,outcomes,expected",
    [
        (0, 99, [W, W, W, W], 0),  # not enough steps
        (0, 100, [W, W, W], 0),  # window not full
        (0, 100, [W, L, L, W], 0),  # below the threshold (0.5)
        (0, 100, [W, W, L, W], 1),  # at the threshold (0.75)
        (0, 100, [W, W, W, W], 1),  # above
        (1, 5000, [W, W, W, W], 2),
        (0, 100, [L, L, W, W, W, W], 1),  # only the last 4 count
        (0, 100, [W, W, W, W, L, L], 0),
        (0, 100, [W, W, D, W], 1),  # a draw is not a win: 3/4
        (0, 100, [W, D, D, W], 0),
        (2, 10**9, [W] * 10, 2),  # the last stage never advances
    ],
)
def test_next_stage_truth_table(stage, steps, outcomes, expected):
    hist = StageHistory(steps_in_stage=steps, outcomes=list(outcomes))
    assert next_stage(stage, hist, CFG) == expected
    assert hist.outcomes == list(outcomes) and hist.steps_in_stage == steps  # pure


def test_two_stage_config_last_stage_is_one():
    cfg = PPOConfig(stages=[{"opponents": ["random"]}, {"opponents": ["economy"]}], min_stage_steps=0, stage_window=1, stage_win_rate=0.0)
    assert next_stage(0, StageHistory(0, [L]), cfg) == 1
    assert next_stage(1, StageHistory(0, [W]), cfg) == 1
