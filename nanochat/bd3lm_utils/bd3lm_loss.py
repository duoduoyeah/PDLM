"""
Loss computation for BD3LM.

This module handles computing the weighted loss from model logits and targets.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple


def compute_token_nll(
    logits: Tensor,
    targets: Tensor,
) -> Tensor:
    """
    Compute per-token negative log-likelihood.

    Args:
        logits: Model output logits, shape (B, L, vocab_size)
        targets: Target token ids, shape (B, L)

    Returns:
        nll: Per-token NLL, shape (B, L)
             nll[b, i] = -log P(targets[b, i] | logits[b, i])
    """
    # Log softmax to get log probabilities
    log_probs = F.log_softmax(logits, dim=-1)

    # Gather the log prob of the target token at each position
    # targets: (B, L) -> (B, L, 1) for gather
    target_log_probs = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1))

    # Remove last dim and negate to get NLL
    # (B, L, 1) -> (B, L)
    nll = -target_log_probs.squeeze(-1)

    return nll


def apply_loss_scale(
    nll: Tensor,
    loss_scale: Tensor,
) -> Tensor:
    """
    Apply loss_scale weighting to per-token NLL.

    Args:
        nll: Per-token NLL, shape (B, L)
        loss_scale: Weight from noise schedule, shape (B, L)
                    For LogLinear schedule: loss_scale = -1/t

    Returns:
        weighted_nll: Weighted NLL, shape (B, L)
                      weighted_nll = loss_scale * nll
    """
    # loss_scale is negative (e.g., -1/t), so loss_scale * nll gives negative value
    # We want positive loss, so we use: -loss_scale * nll = (1/t) * nll
    # But the original code does: loss = loss_scale * log_p_theta
    # where log_p_theta is negative, so loss_scale * log_p_theta is positive
    # Here nll = -log_p_theta, so we need: loss_scale * (-nll) = -loss_scale * nll
    return -loss_scale * nll


def aggregate_loss(
    weighted_nll: Tensor,
    attention_mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Aggregate weighted NLL into a scalar loss.

    Args:
        weighted_nll: Weighted per-token NLL, shape (B, L)
        attention_mask: Mask for valid tokens, shape (B, L)
                        1 = valid token, 0 = padding
                        If None, all tokens are considered valid.

    Returns:
        loss: Scalar loss value (mean over valid tokens)
    """
    if attention_mask is None:
        return weighted_nll.mean()

    # Mask out invalid tokens and compute mean over valid ones
    masked_nll = weighted_nll * attention_mask
    loss = masked_nll.sum() / attention_mask.sum()

    return loss


def compute_bd3lm_loss(
    logits: Tensor,
    targets: Tensor,
    loss_scale: Tensor,
    attention_mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Full BD3LM loss computation pipeline.

    This combines:
    1. compute_token_nll: Get per-token NLL from logits
    2. apply_loss_scale: Apply noise schedule weighting
    3. aggregate_loss: Reduce to scalar

    Args:
        logits: Model output logits, shape (B, L, vocab_size)
        targets: Target token ids, shape (B, L)
        loss_scale: Weight from noise schedule, shape (B, L)
        attention_mask: Mask for valid tokens, shape (B, L), optional

    Returns:
        loss: Scalar loss value
        per_token_nll: Per-token NLL (unweighted), shape (B, L)
                       (useful for computing perplexity)
    """
    # Step 1: Compute per-token NLL
    per_token_nll = compute_token_nll(logits, targets)

    # Step 2: Apply loss_scale weighting
    weighted_nll = apply_loss_scale(per_token_nll, loss_scale)

    # Step 3: Aggregate to scalar
    loss = aggregate_loss(weighted_nll, attention_mask)

    return loss, per_token_nll
