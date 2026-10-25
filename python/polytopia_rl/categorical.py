"""Masked categorical distributions (docs/MILESTONE_10 "Policy").

All functions take `logits` of shape (B, K) and a bool `mask` of the same shape; a masked-out entry
has probability exactly 0.
"""

from __future__ import annotations

import torch
from torch import Tensor

# A finite stand-in for -inf: exp() of it underflows to exactly 0, but it never produces NaN
# (inf - inf) in the softmax or in gradients.
MASKED = -1e9


def masked_logits(logits: Tensor, mask: Tensor, check: bool = True) -> Tensor:
    """`logits` where `mask`, a large negative constant elsewhere. Raises `ValueError` if a row of
    `mask` has no true entry (`check=False` skips that test, and the GPU sync it costs)."""
    if check and not bool(mask.any(dim=-1).all()):
        raise ValueError("a mask row has no valid option")
    return torch.where(mask, logits, torch.full_like(logits, MASKED))


def sample(logits: Tensor, mask: Tensor, generator: torch.Generator, greedy: bool = False, check: bool = True) -> Tensor:
    """(B,) int64 indices: drawn from the masked softmax with `generator`, or its argmax if `greedy`.
    `check` as in `masked_logits`."""
    x = masked_logits(logits, mask, check)
    if greedy:
        return x.argmax(dim=-1)
    # Sample on the generator's device (a CPU generator also serves a model on the GPU).
    probs = torch.softmax(x, dim=-1).to(generator.device)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1).to(logits.device)


def log_prob(logits: Tensor, mask: Tensor, index: Tensor) -> Tensor:
    """(B,) log-probability of `index` under the masked distribution."""
    logp = torch.log_softmax(masked_logits(logits, mask), dim=-1)
    return logp.gather(-1, index.long().unsqueeze(-1)).squeeze(-1)


def entropy(logits: Tensor, mask: Tensor) -> Tensor:
    """(B,) exact entropy of the masked distribution (0 when one option is valid)."""
    logp = torch.log_softmax(masked_logits(logits, mask), dim=-1)
    # Masked entries have p = 0; drop them explicitly so 0 * (-1e9) never enters the sum.
    return -torch.where(mask, logp.exp() * logp, torch.zeros_like(logp)).sum(dim=-1)
