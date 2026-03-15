"""
Dataloader for BD3-LM-Prime (Block Diffusion with Prime sub-token encoding).

Like BD3-LM but masks individual sub-tokens independently within each token,
creating intermediate states between fully masked and fully revealed.
"""

import torch

from nanochat.dataloader_common import create_token_buffer
from nanochat.bd3lm_prime import BD3LMPrimeConfig
from nanochat.bd3lm_utils.bd3lm_mask import sample_t, q_xt, get_loss_scale, expand_block_to_seq
from nanochat.bd3lm_utils.prime_encoding import encode


def bd3lm_prime_data_loader(
    B,
    T,
    split,
    device="cuda",
    model_config: BD3LMPrimeConfig = None,
    resume_state_dict=None,
    tokenizer_threads=4,
    tokenizer_batch_size=128,
):
    """
    Dataloader for BD3-LM-Prime models.

    Masks at sub-token level: each token's ℓ sub-tokens are independently masked.

    Args:
        B: Batch size
        T: Sequence length (in tokens, not sub-tokens)
        split: "train" or "val"
        device: Target device
        model_config: BD3LMPrimeConfig instance
        resume_state_dict: Optional state dict for resuming

    Yields:
        inputs_sub: (B, T * ℓ) sub-token ids (with mask sub-tokens)
        targets: (B, T) target token ids (pure tokens)
        loss_extras: {"loss_scale": (B, T), "loss_mask": (B, T)}
        state_dict: dict for resuming training
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    assert model_config is not None, "model_config is required"
    assert isinstance(model_config, BD3LMPrimeConfig), f"Expected BD3LMPrimeConfig, got {type(model_config)}"

    # Extract config
    block_size = model_config.bucket_size
    mask_token_id = model_config.mask_token_id
    mask_sub_token = model_config.mask_sub_token
    prefix_pure_tokens = model_config.prefix_pure_tokens
    target_shift = model_config.target_shift
    l = model_config.target_length
    base = model_config.base

    # Validate
    assert block_size >= 1, "block_size (bucket_size) must be >= 1"
    assert T % block_size == 0, f"T ({T}) must be divisible by block_size ({block_size})"
    assert mask_sub_token >= 0, "mask_sub_token must be set"

    # Sub-token block size: each token-level block has block_size * l sub-tokens
    sub_block_size = block_size * l

    needed_tokens = B * T
    token_buffer = create_token_buffer(
        split, resume_state_dict, tokenizer_threads, tokenizer_batch_size
    )
    use_cuda = device == "cuda"

    while True:
        tokens, pq_idx, rg_idx, epoch = token_buffer.get_tokens(needed_tokens)

        targets_cpu = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda)
        targets_cpu = targets_cpu.view(B, T)

        # Encode clean tokens to sub-tokens: (B, T) -> (B, T*l)
        targets_sub = encode(targets_cpu, base, l)

        # Block boundary shifting
        prefix_sliding_tokens = epoch % block_size
        num_blocks = (T - prefix_sliding_tokens) // block_size

        # Sample noise level t per block
        # Use sub_block_size so t_min = 1/(block_size * l),
        # allowing finer-grained noise at sub-token level
        t = sample_t(
            batch_size=B,
            num_blocks=num_blocks,
            block_size=sub_block_size,
            sampling_eps_max=1.0,
            device="cpu",
            antithetic_sampling=True,
        )

        # Determine forced mask position (sub-token level)
        if target_shift >= 1:
            # Map token-level forced position to first sub-token of that token
            forced_mask_position = (target_shift - 1) * l
        else:
            forced_mask_position = None

        # Apply masking at sub-token level using q_xt
        # Treats sub-token sequence as positions, blocks have sub_block_size positions
        inputs_sub, sub_mask = q_xt(
            x0=targets_sub,
            t=t,
            mask_token_id=mask_sub_token,
            block_size=sub_block_size,
            prefix_pure_tokens=prefix_pure_tokens * l,
            prefix_sliding_tokens=prefix_sliding_tokens * l,
            forced_mask_position=forced_mask_position,
        )

        # Compute loss_scale per block, expand to per-token
        loss_scale_per_block = get_loss_scale(t)
        loss_scale_blocks = expand_block_to_seq(loss_scale_per_block, block_size)
        loss_scale = torch.zeros(B, T, dtype=loss_scale_blocks.dtype)
        block_region_len = loss_scale_blocks.shape[1]
        loss_scale[:, prefix_sliding_tokens:prefix_sliding_tokens + block_region_len] = loss_scale_blocks

        # Create loss_mask at token level: True if ANY sub-token of that token is masked
        sub_mask_reshaped = sub_mask.view(B, T, l)
        loss_mask = sub_mask_reshaped.any(dim=-1)  # (B, T)

        # Exclude prefix_pure_tokens from loss
        if prefix_pure_tokens > 0:
            loss_mask[:, :prefix_pure_tokens] = False

        # Move to device
        inputs_sub = inputs_sub.to(device=device, non_blocking=use_cuda)
        targets = targets_cpu.to(device=device, non_blocking=use_cuda)
        loss_scale = loss_scale.to(device=device, non_blocking=use_cuda)
        loss_mask = loss_mask.to(device=device, non_blocking=use_cuda)

        loss_extras = {"loss_scale": loss_scale, "loss_mask": loss_mask}
        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}
        yield inputs_sub, targets, loss_extras, state_dict
