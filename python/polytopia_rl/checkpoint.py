"""Checkpoint save/load with spec hashes (docs/MILESTONE_11 "Hashes and checkpoints").

A checkpoint file is a `torch.save`d dict, a superset of the milestone 10 format: at least `model`
(state dict), `optimizer` (state dict), `config` (`PPOConfig.to_json()`), `global_step`, `stage`
and `spec` (the `Env.spec()` dict, which holds `obs_hash` and `rules_hash`). A milestone 10 file
has no `spec`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import torch

from polytopia_rl.config import PPOConfig

# Top-level keys of the file; every other key is `extra`. `vec_env` and `rng` are the milestone 10
# names of `seed_index` and `rng_state`.
_CORE = ("model", "optimizer", "config", "global_step", "stage", "vec_env", "rng", "spec")


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
    back as `Checkpoint.extra` (stored as top-level keys)."""
    ck = dict(extra or {})
    ck.update(
        model=model.state_dict(),
        optimizer=None if optimizer is None else optimizer.state_dict(),
        config=cfg.to_json(),
        global_step=global_step,
        stage=stage,
        vec_env=seed_index,
        rng=rng_state,
        spec=dict(spec),
    )
    torch.save(ck, path)


def load_checkpoint(path, *, spec, strict_obs=True, strict_rules=False) -> Checkpoint:
    """Load `path` and check its `spec` against `spec` (the current build's `Env.spec()`).

    Different `obs_hash`: `SpecMismatch`, or a `SpecWarning` when `strict_obs=False`. Different
    `rules_hash`: a `SpecWarning`, or `SpecMismatch` when `strict_rules=True`. No `spec` (milestone
    10 format): a `SpecWarning`, and `Checkpoint.spec` is None."""
    return checked(torch.load(path, map_location="cpu", weights_only=False), path, spec, strict_obs, strict_rules)


def checked(raw: dict, path, spec, strict_obs=True, strict_rules=False) -> Checkpoint:
    """`load_checkpoint` on a dict already read from `path`."""
    saved = raw.get("spec")
    if saved is None:
        warnings.warn(f"{path}: no spec (milestone 10 format); assuming the current build", SpecWarning, stacklevel=2)
    else:
        for key, strict in (("obs_hash", strict_obs), ("rules_hash", strict_rules)):
            if saved[key] != spec[key]:
                msg = f"{path}: {key} {saved[key]} differs from the current build's {spec[key]}"
                if strict:
                    raise SpecMismatch(msg)
                warnings.warn(msg, SpecWarning, stacklevel=2)
    return Checkpoint(
        model=raw["model"],
        optimizer=raw.get("optimizer"),
        cfg=PPOConfig.from_json(raw["config"]),
        global_step=raw["global_step"],
        stage=raw["stage"],
        seed_index=raw.get("vec_env"),
        rng_state=raw.get("rng"),
        spec=saved,
        extra={k: v for k, v in raw.items() if k not in _CORE},
    )


def load_net(path, device="cpu", strict_obs=True):
    """`(PolicyNet, Checkpoint)` for a checkpoint, checked against the current build's spec for the
    checkpoint's map size. Prints a loud warning when `strict_obs=False` lets a mismatch through."""
    import polytopia_sim

    from polytopia_rl.model import PolicyNet

    raw = torch.load(path, map_location=device, weights_only=False)
    cfg = PPOConfig.from_json(raw["config"])
    spec = polytopia_sim.Env(size=cfg.size).spec()
    if not strict_obs and raw.get("spec") and raw["spec"]["obs_hash"] != spec["obs_hash"]:
        print(f"WARNING: {path} was trained for a different observation (obs_hash {raw['spec']['obs_hash']}); results are not comparable", flush=True)
    ck = checked(raw, path, spec, strict_obs=strict_obs)
    net = PolicyNet(spec, width=cfg.width, blocks=cfg.blocks).to(device)
    net.load_state_dict(ck.model)
    return net, ck


def describe(path, ck: Checkpoint) -> str:
    """The first output line of every CLI that takes a checkpoint: its hashes and training step."""
    if ck.spec is None:
        return f"checkpoint {path}: obs_hash none rules_hash none (unstamped milestone 10 file) step {ck.global_step}"
    return f"checkpoint {path}: obs_hash {ck.spec['obs_hash']} rules_hash {ck.spec['rules_hash']} step {ck.global_step}"
