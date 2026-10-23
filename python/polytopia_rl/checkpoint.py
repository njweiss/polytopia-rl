"""Checkpoint save/load with spec hashes (docs/MILESTONE_11 "Hashes and checkpoints").

A checkpoint file is a `torch.save`d dict, a superset of the milestone 10 format: at least `model`
(state dict), `optimizer` (state dict), `config` (`PPOConfig.to_json()`), `global_step`, `stage`
and `spec` (the `Env.spec()` dict, which holds `obs_hash` and `rules_hash`). A milestone 10 file
has no `spec`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class SpecMismatch(Exception):
    """The checkpoint was trained for a different observation layout (`obs_hash`), or for a
    different rules build (`rules_hash`) when that is strict."""


class SpecWarning(UserWarning):
    """Emitted (`warnings.warn`) for a non-strict mismatch and for a file without `spec`."""


@dataclass
class Checkpoint:
    model: dict  # the model's state dict
    optimizer: dict | None  # the optimizer's state dict
    cfg: object  # PPOConfig
    global_step: int
    stage: int
    seed_index: object  # whatever the trainer passed (its vec env state)
    rng_state: object
    spec: dict | None  # None for a milestone 10 file
    extra: dict = field(default_factory=dict)


def save_checkpoint(path, *, model, optimizer, cfg, global_step, stage, seed_index, rng_state, spec, extra=None) -> None:
    """`model` and `optimizer` are a module and an optimizer (their state dicts are saved), `cfg` a
    `PPOConfig`, `spec` the `Env.spec()` dict; `extra` holds any further trainer state and comes
    back as `Checkpoint.extra`."""
    raise NotImplementedError


def load_checkpoint(path, *, spec, strict_obs=True, strict_rules=False) -> Checkpoint:
    """Load `path` and check its `spec` against `spec` (the current build's `Env.spec()`).

    Different `obs_hash`: `SpecMismatch`, or a `SpecWarning` when `strict_obs=False`. Different
    `rules_hash`: a `SpecWarning`, or `SpecMismatch` when `strict_rules=True`. No `spec` (milestone
    10 format): a `SpecWarning`, and `Checkpoint.spec` is None."""
    raise NotImplementedError
