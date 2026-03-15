"""
Sub-token encoding utilities for BD3-LM-Prime.

Implements base-b decomposition of token IDs into sub-token sequences,
following the Prime scheme from "Beyond Masked and Unmasked" (Chao et al.).

For vocab_size=4096, target_length=2: base=64, each token becomes 2 digits in base-64.
"""

import math
import torch


def get_prime_params(pure_vocab_size: int, target_length: int):
    """Compute base from vocab size and target length.

    Args:
        pure_vocab_size: Number of pure tokens (e.g., 4096).
        target_length: Number of sub-tokens per token (ℓ).

    Returns:
        base: Numerical base for encoding.
        mask_sub_token: Sub-token ID used for masking (= base).
        sub_token_vocab_size: Total sub-token vocabulary size (= base + 1).
    """
    base = math.ceil(pure_vocab_size ** (1.0 / target_length))
    assert base ** target_length >= pure_vocab_size, \
        f"base^l = {base}^{target_length} = {base**target_length} < {pure_vocab_size}"
    mask_sub_token = base  # mask sub-token is one past the valid range
    sub_token_vocab_size = base + 1
    return base, mask_sub_token, sub_token_vocab_size


def encode(x: torch.Tensor, base: int, target_length: int,
           mask_token_id: int = -1, mask_sub_token: int = -1) -> torch.Tensor:
    """Encode token IDs to sub-token sequences via base-b decomposition.

    Args:
        x: (B, T) token IDs in [0, vocab_size) or mask_token_id.
        base: Numerical base for decomposition.
        target_length: Number of sub-tokens per token (ℓ).
        mask_token_id: Token ID representing mask in original space.
            Positions with this ID are encoded as ℓ copies of mask_sub_token.
        mask_sub_token: Sub-token ID for masked positions (typically = base).

    Returns:
        (B, T * target_length) sub-token IDs.
    """
    if target_length == 1:
        return x

    B, T = x.shape
    is_mask = (x == mask_token_id) if mask_token_id >= 0 else torch.zeros_like(x, dtype=torch.bool)

    # Replace mask positions with 0 temporarily for clean encoding
    x_clean = x.clone()
    x_clean[is_mask] = 0

    # Base-b decomposition: most significant digit first
    # powers = [base^(l-1), base^(l-2), ..., base^0]
    powers = base ** torch.arange(target_length - 1, -1, -1, device=x.device, dtype=x.dtype)
    digits = (x_clean.unsqueeze(-1) // powers) % base  # (B, T, l)
    result = digits.long().reshape(B, T * target_length)  # (B, T*l)

    # Fill mask positions: each sub-token becomes mask_sub_token
    if mask_sub_token >= 0 and is_mask.any():
        mask_expanded = is_mask.unsqueeze(-1).expand(-1, -1, target_length)  # (B, T, l)
        mask_expanded = mask_expanded.reshape(B, T * target_length)
        result[mask_expanded] = mask_sub_token

    return result


def decode(y: torch.Tensor, base: int, target_length: int,
           seq_length: int = -1) -> torch.Tensor:
    """Decode sub-token sequences back to token IDs.

    Args:
        y: (B, T * target_length) sub-token IDs.
        base: Numerical base.
        target_length: Number of sub-tokens per token (ℓ).
        seq_length: Original sequence length T. If -1, inferred from y.

    Returns:
        (B, T) token IDs.
    """
    if target_length == 1:
        return y

    B = y.shape[0]
    if seq_length < 0:
        seq_length = y.shape[1] // target_length

    y = y.reshape(B, seq_length, target_length)
    powers = base ** torch.arange(target_length - 1, -1, -1, device=y.device, dtype=y.dtype)
    return (y * powers).sum(dim=-1)
