"""Masked categorical distributions (docs/MILESTONE_10 "Policy").

All functions take `logits` of shape (B, K) and a bool `mask` of the same shape; a masked-out entry
has probability exactly 0.
"""

from __future__ import annotations

import torch
from torch import Tensor


def masked_logits(logits: Tensor, mask: Tensor) -> Tensor:
    """`logits` where `mask`, a large negative constant elsewhere. Raises `ValueError` if a row of
    `mask` has no true entry."""
    raise NotImplementedError


def sample(logits: Tensor, mask: Tensor, generator: torch.Generator, greedy: bool = False) -> Tensor:
    """(B,) int64 indices: drawn from the masked softmax with `generator`, or its argmax if `greedy`."""
    raise NotImplementedError


def log_prob(logits: Tensor, mask: Tensor, index: Tensor) -> Tensor:
    """(B,) log-probability of `index` under the masked distribution."""
    raise NotImplementedError


def entropy(logits: Tensor, mask: Tensor) -> Tensor:
    """(B,) exact entropy of the masked distribution (0 when one option is valid)."""
    raise NotImplementedError
