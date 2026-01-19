"""
Masking/Noising for BD3LM.

This module handles converting clean sequences to noisy (masked) sequences
based on noise level t (temperature).

Uses LogLinear noise schedule only:
- p(t) = t  (masking probability equals t)
- loss_scale = -1/t

Key design:
- Every block is guaranteed to have at least 1 mask
- First, one position per block is forced to be masked (random or fixed for target_shift)
- Then, remaining positions are masked with adjusted probability p'
- t is sampled from [1/block_size, 1] to ensure p' >= 0
"""

import torch
from torch import Tensor
from typing import Tuple, Optional


def sample_t(
    batch_size: int,
    num_blocks: int,
    block_size: int,
    sampling_eps_max: float = 1.0,
    device: torch.device = None,
    antithetic_sampling: bool = True,
) -> Tensor:
    """
    Sample noise level t for each block.

    The minimum t is automatically set to 1/block_size to ensure that the
    adjusted probability p' for non-forced positions is >= 0.

    Args:
        batch_size: Batch size (B)
        num_blocks: Number of blocks per sequence
        block_size: Size of each block (used to compute t_min = 1/block_size)
        sampling_eps_max: Maximum t value (default 1.0)
        device: Device to create tensor on
        antithetic_sampling: If True, use stratified sampling for variance reduction

    Returns:
        t: Sampled noise levels, shape (B, num_blocks)
           Each value in [1/block_size, sampling_eps_max]
    """
    # Minimum t is 1/block_size to ensure p' >= 0
    sampling_eps_min = 1.0 / block_size

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


def get_adjusted_mask_prob(t: Tensor, block_size: int) -> Tensor:
    """
    Compute adjusted masking probability for non-forced positions.

    Given that one position per block is forced to be masked, compute the
    probability p' for the remaining (block_size - 1) positions such that
    the expected total masks per block equals block_size * t.

    Formula: p' = (block_size * t - 1) / (block_size - 1)

    Args:
        t: Noise level per block, shape (B, num_blocks)
        block_size: Size of each block

    Returns:
        p_adjusted: Adjusted probability for non-forced positions, shape (B, num_blocks)
                    Values in [0, 1] (clamped)
    """
    p_adjusted = (block_size * t - 1) / (block_size - 1)
    # Clamp to [0, 1] for safety (should already be >= 0 if t >= 1/block_size)
    p_adjusted = p_adjusted.clamp(min=0.0, max=1.0)
    return p_adjusted


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
    prefix_pure_tokens: int = 0,
    prefix_sliding_tokens: int = 0,
    forced_mask_position: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Full forward noising process: x0 -> xt.

    New design ensures at least 1 mask per block:
    1. Force one position per block to be masked (random or fixed)
    2. Apply adjusted probability p' to remaining positions
    3. (Optional) Keep prefix tokens unmasked

    Args:
        x0: Clean input token ids, shape (B, L)
        t: Noise level, shape (B, num_blocks). num_blocks is inferred from t.shape[1].
        mask_token_id: Token id to use for masked positions
        block_size: Block size for block-wise diffusion
        prefix_pure_tokens: Number of prefix tokens to keep unmasked (AR prefix).
                           These positions will never be masked.
        prefix_sliding_tokens: Number of sliding prefix tokens (epoch % block_size).
                              Blocks start after this offset. These positions are not
                              in any block and won't be masked by block logic.
        forced_mask_position: If None, randomly select one position per block to force mask.
                             If int in [0, block_size-1], force that position in each block.
                             (This is for target_shift mode where we always mask a fixed position)

    Returns:
        xt: Noisy (masked) sequence, shape (B, L)
        mask: Boolean mask indicating masked positions, shape (B, L)
    """
    B, L = x0.shape
    num_blocks = t.shape[1]  # infer from t, source of truth
    device = x0.device

    # Step 1: Initialize mask as all False
    mask = torch.zeros_like(x0, dtype=torch.bool)

    # Step 2: Force one position per block to be masked
    if forced_mask_position is None:
        # Random position per block: shape (B, num_blocks)
        forced_pos_in_block = torch.randint(0, block_size, (B, num_blocks), device=device)
    else:
        # Fixed position for all blocks (target_shift mode)
        forced_pos_in_block = torch.full((B, num_blocks), forced_mask_position, device=device)

    # Convert block-relative positions to absolute positions
    # Blocks start at prefix_sliding_tokens offset
    # block_offsets: [prefix_sliding_tokens, prefix_sliding_tokens + block_size, ...]
    block_offsets = torch.arange(num_blocks, device=device) * block_size + prefix_sliding_tokens
    forced_abs_pos = forced_pos_in_block + block_offsets.unsqueeze(0)  # (B, num_blocks)

    # Set forced positions in mask
    batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(B, num_blocks)
    mask[batch_indices.flatten(), forced_abs_pos.flatten()] = True
    mask = mask.view(B, L)

    # Step 3: Apply adjusted probability to non-forced positions (only in block region)
    p_adjusted = get_adjusted_mask_prob(t, block_size)  # (B, num_blocks)
    p_adjusted_blocks = expand_block_to_seq(p_adjusted, block_size)  # (B, num_blocks * block_size)

    # Create full-length p_adjusted with zeros for prefix_sliding_tokens region
    p_adjusted_expanded = torch.zeros(B, L, dtype=p_adjusted_blocks.dtype, device=device)
    block_region_len = p_adjusted_blocks.shape[1]
    p_adjusted_expanded[:, prefix_sliding_tokens:prefix_sliding_tokens + block_region_len] = p_adjusted_blocks

    # Sample additional masks for non-forced positions
    rand = torch.rand_like(x0, dtype=p_adjusted_expanded.dtype)
    additional_mask = rand < p_adjusted_expanded

    # Combine: forced positions OR additional random masks
    mask = mask | additional_mask

    # Step 4: Keep prefix tokens unmasked (AR prefix positions)
    # This is applied last and overrides everything
    if prefix_pure_tokens > 0:
        mask[:, :prefix_pure_tokens] = False

    # Step 5: Apply mask
    xt = torch.where(mask, mask_token_id, x0)

    return xt, mask
