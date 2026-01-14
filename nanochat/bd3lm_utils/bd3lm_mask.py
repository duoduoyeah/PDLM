"""
Masking/Noising for BD3LM.

This module handles converting clean sequences to noisy (masked) sequences
based on noise level t (temperature).

Uses LogLinear noise schedule only:
- p(t) = t  (masking probability equals t)
- loss_scale = -1/t
"""

import torch
from torch import Tensor
from typing import Tuple


def sample_t(
    batch_size: int,
    num_blocks: int,
    sampling_eps_min: float = 1e-3,
    sampling_eps_max: float = 1.0,
    device: torch.device = None,
    antithetic_sampling: bool = True,
) -> Tensor:
    """
    Sample noise level t for each block.

    Args:
        batch_size: Batch size (B)
        num_blocks: Number of blocks per sequence
        sampling_eps_min: Minimum t value
        sampling_eps_max: Maximum t value
        device: Device to create tensor on
        antithetic_sampling: If True, use stratified sampling for variance reduction

    Returns:
        t: Sampled noise levels, shape (B, num_blocks)
           Each value in [sampling_eps_min, sampling_eps_max]
    """
    # Sample uniform random values
    t = torch.rand((batch_size, num_blocks), device=device)

    # Antithetic sampling: stratified sampling for variance reduction
    if antithetic_sampling:
        total_samples = batch_size * num_blocks
        offset = torch.arange(total_samples, device=device) / total_samples
        offset = offset.view(batch_size, num_blocks)
        t = (t / total_samples + offset) % 1

    # Scale to [sampling_eps_min, sampling_eps_max]
    t = t * (sampling_eps_max - sampling_eps_min) + sampling_eps_min

    return t


def get_mask_probability(t: Tensor) -> Tensor:
    """
    Convert noise level t to masking probability p.

    For LogLinear schedule: p = t

    Args:
        t: Noise level, shape (B, num_blocks) or (B, L)

    Returns:
        p: Masking probability, same shape as t
    """
    # LogLinear: p = t
    return t


def get_loss_scale(t: Tensor) -> Tensor:
    """
    Get loss_scale weight from noise level t.

    For LogLinear schedule: loss_scale = -1/t

    Args:
        t: Noise level, shape (B, num_blocks) or (B, L)

    Returns:
        loss_scale: Weight for loss computation, same shape as t
    """
    # LogLinear: loss_scale = -1/t
    return -1.0 / t


def expand_block_to_seq(
    block_values: Tensor,
    block_size: int,
) -> Tensor:
    """
    Expand per-block values to per-position values.

    Args:
        block_values: Values per block, shape (B, num_blocks)
        block_size: Size of each block

    Returns:
        seq_values: Values per position, shape (B, L)
                    where L = num_blocks * block_size
    """
    # Repeat each block value block_size times
    # (B, num_blocks) -> (B, L) where L = num_blocks * block_size
    return block_values.repeat_interleave(block_size, dim=-1)


def q_xt(
    x0: Tensor,
    t: Tensor,
    mask_token_id: int,
    block_size: int = 1,
    ignore_first_token: bool = True,
) -> Tuple[Tensor, Tensor]:
    """
    Full forward noising process: x0 -> xt.

    Steps:
    1. get_mask_probability: t -> p
    2. Expand p to sequence length if needed
    3. Sample mask indices based on p
    4. Apply mask to create xt
    5. (Optional) Keep first token unmasked

    Args:
        x0: Clean input token ids, shape (B, L)
        t: Noise level, shape (B, num_blocks)
        mask_token_id: Token id to use for masked positions
        block_size: Block size for block-wise diffusion
        ignore_first_token: If True, the first token (position 0) will NOT be masked

    Returns:
        xt: Noisy (masked) sequence, shape (B, L)
        mask: Boolean mask indicating masked positions, shape (B, L)
    """
    # Step 1: t -> p (for loglinear, p = t)
    p = get_mask_probability(t)

    # Step 2: Expand p from (B, num_blocks) to (B, L) if needed
    if p.shape[-1] != x0.shape[-1]:
        p = expand_block_to_seq(p, block_size)

    # Step 3: Sample which positions to mask
    # Each position is masked independently with probability p
    rand = torch.rand_like(x0, dtype=p.dtype)
    mask = rand < p  # True = will be masked

    # Step 4: Keep first token unmasked
    if ignore_first_token:
        mask[:, 0] = False

    # Step 5: Apply mask
    xt = torch.where(mask, mask_token_id, x0)

    return xt, mask
