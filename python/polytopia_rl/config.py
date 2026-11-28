"""Training configuration (docs/MILESTONE_10 "Default hyperparameters")."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field

from polytopia_rl.pfsp import parse_bot_specs


def _default_stages() -> list[dict]:
    return [{"opponents": ["random"]}, {"opponents": ["economy"]}, {"opponents": ["economy", "rush"]}]


@dataclass
class PPOConfig:
    # Environment (passed to `polytopia_sim.Env`).
    size: int = 11
    turn_cap: int = 30
    full_observability: bool = False
    ruleset: str = "v1"
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
    # Self-play rollout collector (docs/MILESTONE_13, MILESTONE_23): "interleaved" | "lockstep" | "stream".
    rollout: str = "interleaved"
    # Snapshot opponents' forward in tick-based collectors (docs/MILESTONE_13b, MILESTONE_23): "stacked" | "per_snapshot" | "auto".
    opponent_forward: str = "stacked"
    # Model.
    width: int = 64
    blocks: int = 4
    arch: str = "v1"
    # Transfer (docs/MILESTONE_22): prefix warm-start and policy distillation.
    warm_start_checkpoint: str | None = None
    teacher_checkpoint: str | None = None
    distill_coef: float = 0.0
    distill_steps: int | None = None
    distill_temp: float = 1.0
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
    # Revised self-play (docs/MILESTONE_23): live self-play, PFSP, flexible bots, anchors, gating,
    # shaping annealing and paired seeds.
    p_self: float = 0.0
    pfsp_mode: str = "off"
    pfsp_power: float = 2.0
    pfsp_ema_alpha: float = 0.05
    selfplay_bots: list[dict] | None = None
    anchor_checkpoints: list[str] = field(default_factory=list)
    snapshot_win_rate: float | None = None
    snapshot_min_games: int = 20
    snapshot_max_steps: int | None = None
    shaping_anneal_steps: int | None = None
    shaping_end: float = 0.0
    paired_seeds: bool = False

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> PPOConfig:
        """Fields missing from `text` keep their defaults; an unknown key, `lr_anneal_steps` below
        `total_steps`, a `ruleset` other than "v1" / "v2", an `arch` other than "v1" / "v2", a
        `rollout` other than "interleaved" / "lockstep" / "stream", an `opponent_forward` other
        than "stacked" / "per_snapshot" / "auto", invalid self-play probabilities (`p_bot`,
        `p_self`, `p_latest`, `p_bot + p_self > 1`, or `p_self > 0` without `rollout="stream"`),
        invalid PFSP / bot-spec / anchor / snapshot-gating / shaping-annealing settings, negative
        `distill_coef` or `distill_steps`, `distill_coef > 0` without `teacher_checkpoint`, or
        non-positive `distill_temp` raises `ValueError`."""
        data = json.loads(text)
        _check_keys(data)
        cfg = cls(**data)
        if cfg.ruleset not in ("v1", "v2"):
            raise ValueError(f"ruleset {cfg.ruleset!r} is not 'v1' or 'v2'")
        if cfg.arch not in ("v1", "v2"):
            raise ValueError(f"arch {cfg.arch!r} is not 'v1' or 'v2'")
        if cfg.lr_anneal_steps is not None and cfg.lr_anneal_steps < cfg.total_steps:
            raise ValueError(
                f"lr_anneal_steps {cfg.lr_anneal_steps} is below total_steps {cfg.total_steps}: the LR would go negative"
            )
        if cfg.rollout not in ("interleaved", "lockstep", "stream"):
            raise ValueError(f"rollout {cfg.rollout!r} is not 'interleaved', 'lockstep', or 'stream'")
        if cfg.opponent_forward not in ("stacked", "per_snapshot", "auto"):
            raise ValueError(f"opponent_forward {cfg.opponent_forward!r} is not 'stacked', 'per_snapshot', or 'auto'")
        if not (0.0 <= cfg.p_bot <= 1.0):
            raise ValueError(f"p_bot {cfg.p_bot} must be in [0, 1]")
        if not (0.0 <= cfg.p_self <= 1.0):
            raise ValueError(f"p_self {cfg.p_self} must be in [0, 1]")
        if cfg.p_bot + cfg.p_self > 1.0 + 1e-12:
            raise ValueError(f"p_bot + p_self ({cfg.p_bot + cfg.p_self}) must be <= 1")
        if not (0.0 <= cfg.p_latest <= 1.0):
            raise ValueError(f"p_latest {cfg.p_latest} must be in [0, 1]")
        if cfg.p_self > 0.0 and cfg.rollout != "stream":
            raise ValueError("p_self > 0 requires rollout='stream'")
        if cfg.pfsp_mode not in ("off", "hard", "variance"):
            raise ValueError(f"pfsp_mode {cfg.pfsp_mode!r} is not 'off', 'hard', or 'variance'")
        if cfg.pfsp_power <= 0.0:
            raise ValueError(f"pfsp_power {cfg.pfsp_power} must be > 0")
        if not (0.0 < cfg.pfsp_ema_alpha <= 1.0):
            raise ValueError(f"pfsp_ema_alpha {cfg.pfsp_ema_alpha} must be in (0, 1]")
        if cfg.selfplay_bots is not None:
            parse_bot_specs(cfg.selfplay_bots)
        if not isinstance(cfg.anchor_checkpoints, list) or any(not isinstance(p, str) or not p for p in cfg.anchor_checkpoints):
            raise ValueError("anchor_checkpoints must be a list of non-empty strings")
        if cfg.snapshot_win_rate is not None and not (0.0 <= cfg.snapshot_win_rate <= 1.0):
            raise ValueError(f"snapshot_win_rate {cfg.snapshot_win_rate} must be in [0, 1]")
        if cfg.snapshot_min_games < 1:
            raise ValueError(f"snapshot_min_games {cfg.snapshot_min_games} must be >= 1")
        if cfg.snapshot_max_steps is not None and cfg.snapshot_max_steps < cfg.snapshot_every:
            raise ValueError(f"snapshot_max_steps {cfg.snapshot_max_steps} must be >= snapshot_every {cfg.snapshot_every}")
        if cfg.shaping_anneal_steps is not None and cfg.shaping_anneal_steps <= 0:
            raise ValueError(f"shaping_anneal_steps {cfg.shaping_anneal_steps} must be > 0")
        if cfg.shaping_end < 0.0:
            raise ValueError(f"shaping_end {cfg.shaping_end} must be >= 0")
        if cfg.distill_coef < 0.0:
            raise ValueError(f"distill_coef {cfg.distill_coef} must be >= 0")
        if cfg.distill_coef > 0.0 and cfg.teacher_checkpoint is None:
            raise ValueError("distill_coef > 0 requires teacher_checkpoint")
        if cfg.distill_steps is not None and cfg.distill_steps < 0:
            raise ValueError(f"distill_steps {cfg.distill_steps} must be >= 0 or None")
        if cfg.distill_temp <= 0.0:
            raise ValueError(f"distill_temp {cfg.distill_temp} must be > 0")
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
