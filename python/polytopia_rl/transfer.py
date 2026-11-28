"""v1 -> v2 policy transfer: prefix warm-start and policy distillation (docs/MILESTONE_22)."""

from __future__ import annotations

from torch import Tensor

from polytopia_rl.checkpoint import Checkpoint
from polytopia_rl.model import PolicyNet


def warm_start_v1_to_v2(
    dst_net: PolicyNet,
    v1_state: dict[str, Tensor],
    v1_spec: dict,
    v2_spec: dict,
) -> None:
    raise NotImplementedError("milestone 22")


def load_warm_start(dst_net: PolicyNet, ckpt_path: str, v2_spec: dict) -> Checkpoint:
    raise NotImplementedError("milestone 22")


def masked_kl(
    teacher_logits: Tensor,
    student_logits: Tensor,
    mask: Tensor,
    temp: float = 1.0,
) -> Tensor:
    raise NotImplementedError("milestone 22")


def distill_schedule(
    update: int,
    base_coef: float,
    total_updates: int,
    distill_updates: int | None,
) -> float:
    raise NotImplementedError("milestone 22")


def compute_teacher_logits(
    teacher: PolicyNet,
    teacher_spec: dict,
    student_spec: dict,
    board: Tensor,
    globals: Tensor,
    actors: Tensor,
    kinds: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    raise NotImplementedError("milestone 22")


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
    raise NotImplementedError("milestone 22")
