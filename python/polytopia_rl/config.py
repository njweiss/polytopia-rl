"""Training configuration (docs/MILESTONE_10 "Default hyperparameters")."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field


def _default_stages() -> list[dict]:
    return [{"opponents": ["random"]}, {"opponents": ["economy"]}, {"opponents": ["economy", "rush"]}]


@dataclass
class PPOConfig:
    # Environment (passed to `polytopia_sim.Env`).
    size: int = 11
    turn_cap: int = 30
    full_observability: bool = False
    opponents: list[str] = field(default_factory=lambda: ["random"])
    opponent_epsilon: float = 0.0
    shaping_weight: float = 1.0
    terminal_weight: float = 1.0
    potential_scale: float = 1000.0
    # PPO.
    num_envs: int = 32
    num_steps: int = 128
    num_minibatches: int = 8
    update_epochs: int = 4
    lr: float = 3e-4
    # LR schedule (docs/MILESTONE_12c): `anneal_lr` false keeps `lr` for the whole run;
    # `lr_anneal_steps` None anneals over `total_steps`.
    anneal_lr: bool = True
    lr_anneal_steps: int | None = None
    max_grad_norm: float = 0.5
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    target_kl: float | None = None
    norm_adv: bool = True
    clip_vloss: bool = True
    # Self-play rollout collector (docs/MILESTONE_13): "interleaved" | "lockstep".
    rollout: str = "interleaved"
    # Model.
    width: int = 64
    blocks: int = 4
    # Run.
    total_steps: int = 20_000_000
    seed: int = 1
    device: str = "auto"
    # Curriculum: each stage dict overrides config fields (only "opponents" is used).
    stages: list[dict] = field(default_factory=_default_stages)
    min_stage_steps: int = 1_000_000
    stage_window: int = 200
    stage_win_rate: float = 0.8
    # Evaluation and checkpoints during training.
    eval_every: int = 500_000
    eval_seeds: int = 50
    # Self-play (docs/MILESTONE_11 "Self-play"); `selfplay` switches the trainer to it.
    selfplay: bool = False
    snapshot_every: int = 1_000_000
    pool_max: int = 40
    p_bot: float = 0.2
    p_latest: float = 0.5
    opponent_greedy: bool = False
    rating_every: int = 2_000_000
    rating_seeds: int = 50
    regression_floor: float = 0.9
    tournament_seeds: int = 100

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> PPOConfig:
        """Fields missing from `text` keep their defaults; an unknown key, or `lr_anneal_steps` below
        `total_steps`, raises `ValueError`."""
        data = json.loads(text)
        _check_keys(data)
        cfg = cls(**data)
        if cfg.lr_anneal_steps is not None and cfg.lr_anneal_steps < cfg.total_steps:
            raise ValueError(
                f"lr_anneal_steps {cfg.lr_anneal_steps} is below total_steps {cfg.total_steps}: the LR would go negative"
            )
        return cfg


def _check_keys(keys) -> None:
    known = {f.name for f in dataclasses.fields(PPOConfig)}
    unknown = sorted(set(keys) - known)
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}")


def load_config(path: str, overrides: list[str] = ()) -> PPOConfig:
    """Load a JSON config, then apply `key=value` overrides in order.

    A value is parsed as JSON when it parses (`3`, `0.5`, `true`, `null`, `["economy","rush"]`),
    otherwise it is the raw string (`device=cuda`). An unknown key, or an override without `=`,
    raises `ValueError`.
    """
    with open(path) as f:
        data = json.load(f)
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise ValueError(f"override {item!r} is not key=value")
        _check_keys([key])
        try:
            data[key] = json.loads(raw)
        except json.JSONDecodeError:
            data[key] = raw
    return PPOConfig.from_json(json.dumps(data))
