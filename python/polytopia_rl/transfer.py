"""v1 -> v2 policy transfer: prefix warm-start and policy distillation (docs/MILESTONE_22)."""

from __future__ import annotations

import polytopia_sim
import torch
import torch.nn.functional as F
from torch import Tensor

from polytopia_rl import categorical
from polytopia_rl.checkpoint import Checkpoint, checked
from polytopia_rl.config import PPOConfig
from polytopia_rl.model import TILE_KINDS, PolicyNet

_WARM_START_SPECIAL_KEYS = frozenset(
    {
        "stem.weight",
        "global_mlp.0.0.weight",
        "kind_head.mlp.2.weight",
        "kind_head.mlp.2.bias",
        "arg_head.kind_embed.weight",
        "arg_head.tile_bias",
    }
)


def _validate_prefix_specs(src_spec: dict, dst_spec: dict) -> tuple[int, int, int]:
    """Validate that `src_spec` and `dst_spec` share map dimensions and `src_spec` is a prefix of
    (or equal to) `dst_spec` across channels, globals, action types, and per-kind arg lengths."""
    if src_spec["height"] != dst_spec["height"] or src_spec["width"] != dst_spec["width"]:
        raise ValueError(
            f"map size mismatch: src=({src_spec['height']}, {src_spec['width']}), "
            f"dst=({dst_spec['height']}, {dst_spec['width']})"
        )
    c1, g1, k1 = src_spec["channels"], src_spec["globals"], len(src_spec["action_types"])
    c2, g2, k2 = dst_spec["channels"], dst_spec["globals"], len(dst_spec["action_types"])
    if c1 > c2 or g1 > g2 or k1 > k2:
        raise ValueError(
            f"src spec ({c1} ch, {g1} gl, {k1} kinds) is not a prefix of dst spec ({c2} ch, {g2} gl, {k2} kinds)"
        )
    if dst_spec["channel_names"][:c1] != src_spec["channel_names"]:
        raise ValueError("src channel_names is not a prefix of dst channel_names")
    if dst_spec["global_names"][:g1] != src_spec["global_names"]:
        raise ValueError("src global_names is not a prefix of dst global_names")
    if dst_spec["action_types"][:k1] != src_spec["action_types"]:
        raise ValueError("src action_types is not a prefix of dst action_types")
    for k in range(k1):
        if src_spec["arg_lens"][k] > dst_spec["arg_lens"][k]:
            raise ValueError(
                f"arg_lens[{k}] ({src_spec['action_types'][k]}) shrank from "
                f"{src_spec['arg_lens'][k]} to {dst_spec['arg_lens'][k]}"
            )
    return c1, g1, k1


def warm_start_v1_to_v2(
    dst_net: PolicyNet,
    v1_state: dict[str, Tensor],
    v1_spec: dict,
    v2_spec: dict,
) -> None:
    """Copy weights from a v1 `arch="v1"` state dict into a v2 `arch="v1"` `PolicyNet`.

    New v2 input columns (`stem.weight[:, c1:]`, `global_mlp.0.0.weight[:, g1:]`) are zeroed so
    step-0 outputs on v1-prefix features are preserved exactly, while new output rows/heads keep
    `dst_net`'s fresh orthogonal initialization. Enum heads are matched by kind name.
    """
    if dst_net.arch != "v1":
        raise ValueError(f"warm_start_v1_to_v2 requires dst_net.arch == 'v1', got {dst_net.arch!r}")
    c1, g1, k1 = _validate_prefix_specs(v1_spec, v2_spec)
    if dst_net.num_tiles != v2_spec["height"] * v2_spec["width"]:
        raise ValueError("dst_net map size does not match v2_spec")
    if "stem.weight" not in v1_state or "global_mlp.0.0.weight" not in v1_state:
        raise ValueError("v1_state is missing required arch='v1' keys")
    if v1_state["stem.weight"].shape[0] != dst_net.stem.weight.shape[0]:
        raise ValueError(
            f"width mismatch: v1={v1_state['stem.weight'].shape[0]}, dst={dst_net.stem.weight.shape[0]}"
        )
    if v1_state["stem.weight"].shape[1] != c1 or dst_net.stem.weight.shape[1] != v2_spec["channels"]:
        raise ValueError("stem input channels do not match v1_spec / v2_spec")
    if (
        v1_state["global_mlp.0.0.weight"].shape[1] != g1
        or dst_net.global_mlp[0][0].weight.shape[1] != v2_spec["globals"]
    ):
        raise ValueError("global_mlp input width does not match v1_spec / v2_spec")

    v1_tile_kinds = {k for k, name in enumerate(v1_spec["action_types"]) if name in TILE_KINDS}
    v1_enum_kinds = [k for k, n in enumerate(v1_spec["arg_lens"]) if n > 0 and k not in v1_tile_kinds]
    v1_enum_by_name = {v1_spec["action_types"][k]: i for i, k in enumerate(v1_enum_kinds)}

    dst_state = dst_net.state_dict()
    exact_keys = {
        k for k in dst_state if k not in _WARM_START_SPECIAL_KEYS and not k.startswith("arg_head.enum.")
    }
    v1_enum_keys = {f"arg_head.enum.{i}.{s}" for i in range(len(v1_enum_kinds)) for s in ("weight", "bias")}
    expected_v1_keys = exact_keys | _WARM_START_SPECIAL_KEYS | v1_enum_keys
    if set(v1_state.keys()) != expected_v1_keys:
        missing = sorted(expected_v1_keys - set(v1_state.keys()))
        extra = sorted(set(v1_state.keys()) - expected_v1_keys)
        raise ValueError(f"v1_state key mismatch (blocks/arch incompatibility): missing={missing}, extra={extra}")

    for k in exact_keys:
        if v1_state[k].shape != dst_state[k].shape:
            raise ValueError(f"shape mismatch at {k}: v1={tuple(v1_state[k].shape)}, dst={tuple(dst_state[k].shape)}")
    if v1_state["kind_head.mlp.2.weight"].shape != (k1, dst_net.kind_head.mlp[2].weight.shape[1]):
        raise ValueError("kind_head.mlp.2.weight shape mismatch")
    if v1_state["kind_head.mlp.2.bias"].shape != (k1,):
        raise ValueError("kind_head.mlp.2.bias shape mismatch")
    if v1_state["arg_head.kind_embed.weight"].shape != (k1, dst_net.arg_head.kind_embed.weight.shape[1]):
        raise ValueError("arg_head.kind_embed.weight shape mismatch")
    if v1_state["arg_head.tile_bias"].shape != (k1,):
        raise ValueError("arg_head.tile_bias shape mismatch")

    for dst_idx, dst_k in enumerate(dst_net.arg_head.enum_kinds):
        name = v2_spec["action_types"][dst_k]
        if name in v1_enum_by_name:
            src_idx = v1_enum_by_name[name]
            src_w = v1_state[f"arg_head.enum.{src_idx}.weight"]
            src_b = v1_state[f"arg_head.enum.{src_idx}.bias"]
            dst_m = dst_net.arg_head.enum[dst_idx]
            if src_w.shape[1] != dst_m.weight.shape[1] or src_w.shape[0] > dst_m.weight.shape[0]:
                raise ValueError(f"enum head {name!r} shape mismatch: v1={tuple(src_w.shape)}, dst={tuple(dst_m.weight.shape)}")
            if src_b.shape != (src_w.shape[0],):
                raise ValueError(f"enum head {name!r} bias shape mismatch")

    with torch.no_grad():
        for k in exact_keys:
            dst_state[k].copy_(v1_state[k])

        dst_net.stem.weight[:, :c1].copy_(v1_state["stem.weight"])
        dst_net.stem.weight[:, c1:].zero_()

        dst_net.global_mlp[0][0].weight[:, :g1].copy_(v1_state["global_mlp.0.0.weight"])
        dst_net.global_mlp[0][0].weight[:, g1:].zero_()

        dst_net.kind_head.mlp[2].weight[:k1].copy_(v1_state["kind_head.mlp.2.weight"])
        dst_net.kind_head.mlp[2].bias[:k1].copy_(v1_state["kind_head.mlp.2.bias"])

        dst_net.arg_head.kind_embed.weight[:k1].copy_(v1_state["arg_head.kind_embed.weight"])
        dst_net.arg_head.tile_bias[:k1].copy_(v1_state["arg_head.tile_bias"])

        for dst_idx, dst_k in enumerate(dst_net.arg_head.enum_kinds):
            name = v2_spec["action_types"][dst_k]
            if name in v1_enum_by_name:
                src_idx = v1_enum_by_name[name]
                src_w = v1_state[f"arg_head.enum.{src_idx}.weight"]
                src_b = v1_state[f"arg_head.enum.{src_idx}.bias"]
                n = src_w.shape[0]
                dst_m = dst_net.arg_head.enum[dst_idx]
                dst_m.weight[:n].copy_(src_w)
                dst_m.bias[:n].copy_(src_b)


def load_warm_start(dst_net: PolicyNet, ckpt_path: str, v2_spec: dict) -> Checkpoint:
    """Load a v1 `arch="v1"` checkpoint, validate its obs_hash, and warm-start `dst_net`."""
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = PPOConfig.from_json(raw["config"])
    if cfg.ruleset != "v1":
        raise ValueError(f"{ckpt_path}: warm-start source must have ruleset 'v1', got {cfg.ruleset!r}")
    if cfg.arch != "v1":
        raise ValueError(f"{ckpt_path}: warm-start source must have arch 'v1', got {cfg.arch!r}")
    v1_spec = polytopia_sim.Env(size=cfg.size, ruleset="v1").spec()
    ck = checked(raw, ckpt_path, v1_spec, strict_obs=True)
    warm_start_v1_to_v2(dst_net, ck.model, v1_spec, v2_spec)
    return ck


def masked_kl(
    teacher_logits: Tensor,
    student_logits: Tensor,
    mask: Tensor,
    temp: float = 1.0,
) -> Tensor:
    """Temperature-scaled `T^2 * KL(p_teacher || p_student)` over `mask`, detached on teacher."""
    if temp <= 0.0:
        raise ValueError(f"temp {temp} must be > 0")
    t = categorical.masked_logits(teacher_logits.detach() / temp, mask, check=False)
    s = categorical.masked_logits(student_logits / temp, mask, check=False)
    p_t = F.softmax(t, dim=-1)
    log_p_t = F.log_softmax(t, dim=-1)
    log_p_s = F.log_softmax(s, dim=-1)
    kl = torch.where(mask, p_t * (log_p_t - log_p_s), torch.zeros_like(p_t)).sum(dim=-1).clamp_min(0.0)
    return kl * (temp * temp)


def distill_schedule(
    update: int,
    base_coef: float,
    total_updates: int,
    distill_updates: int | None,
) -> float:
    """Linear annealing schedule for `distill_coef` matching `lr` annealing in `train.py`."""
    if base_coef == 0.0:
        return 0.0
    if distill_updates == 0:
        return float(base_coef)
    horizon = total_updates if distill_updates is None else distill_updates
    if horizon <= 0:
        return 0.0
    return float(base_coef * max(0.0, 1.0 - (update - 1) / horizon))


def project_v2_to_v1_obs(board: Tensor, globals: Tensor, v1_spec: dict) -> tuple[Tensor, Tensor]:
    """Slice board channels and globals to the `v1_spec` prefix."""
    return board[:, : v1_spec["channels"]], globals[:, : v1_spec["globals"]]


def compute_teacher_logits(
    teacher: PolicyNet,
    teacher_spec: dict,
    student_spec: dict,
    board: Tensor,
    globals: Tensor,
    actors: Tensor,
    kinds: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run frozen `teacher` once on projected observations, clamping `kinds` to teacher's range."""
    _validate_prefix_specs(teacher_spec, student_spec)
    k1 = len(teacher_spec["action_types"])
    t_board, t_globals = project_v2_to_v1_obs(board, globals, teacher_spec)
    safe_kinds = kinds.clamp(max=k1 - 1)
    with torch.no_grad():
        feats = teacher.trunk(t_board, t_globals)
        a_logits = teacher.actor_logits(feats)
        k_logits = teacher.kind_logits(feats, actors)
        arg_logits = teacher.arg_logits(feats, actors, safe_kinds)
    return a_logits, k_logits, arg_logits


def distill_kl(
    teacher_logits: tuple[Tensor, Tensor, Tensor],
    student_logits: tuple[Tensor, Tensor, Tensor],
    kinds: Tensor,
    actor_mask: Tensor,
    kind_mask: Tensor,
    arg_mask: Tensor,
    teacher_spec: dict,
    student_spec: dict,
    temp: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Three-stage factored masked KL from `teacher_logits` to `student_logits`."""
    _validate_prefix_specs(teacher_spec, student_spec)
    t_actor, t_kind, t_arg = teacher_logits
    s_actor, s_kind, s_arg = student_logits
    k1 = len(teacher_spec["action_types"])
    k2 = len(student_spec["action_types"])
    device = s_actor.device
    zero = torch.zeros(s_actor.shape[0], dtype=s_actor.dtype, device=device)

    kl_actor = masked_kl(t_actor, s_actor, actor_mask, temp=temp)

    km_t = kind_mask[:, :k1]
    has_k = km_t.any(dim=1)
    safe_km = km_t | ~has_k.unsqueeze(1)
    kl_kind = torch.where(has_k, masked_kl(t_kind, s_kind[:, :k1], safe_km, temp=temp), zero)

    v1_arg_lens = torch.tensor(
        list(teacher_spec["arg_lens"]) + [0] * (k2 - k1),
        dtype=torch.int64,
        device=device,
    )
    len_b = v1_arg_lens[kinds]
    n_args = arg_mask.shape[1]
    am_t = arg_mask & (torch.arange(n_args, device=device).unsqueeze(0) < len_b.unsqueeze(1))
    has_a = am_t.any(dim=1)
    safe_am = am_t | ~has_a.unsqueeze(1)
    kl_arg = torch.where(has_a, masked_kl(t_arg, s_arg, safe_am, temp=temp), zero)

    total = kl_actor + kl_kind + kl_arg
    return total, {"actor": kl_actor, "kind": kl_kind, "arg": kl_arg}
