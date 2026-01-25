"""
Dataloader for BD3LM (Block Discrete Denoising Diffusion Language Model).
"""

import torch

from nanochat.dataloader_common import create_token_buffer
from nanochat.bd3lm import BDLMConfig
from nanochat.bd3lm_utils.bd3lm_mask import sample_t, q_xt, get_loss_scale, expand_block_to_seq


def bd3lm_data_loader(
    B,
    T,
    split,
    device="cuda",
    model_config: BDLMConfig = None,
    resume_state_dict=None,
    tokenizer_threads=4,
    tokenizer_batch_size=128,
):
    """
    Dataloader for BD3LM models.

    Args:
        B: Batch size
        T: Sequence length
        split: "train" or "val"
        device: Target device
        model_config: BDLMConfig instance
        resume_state_dict: Optional state dict for resuming

    Yields:
        inputs: (B, T) input token ids (with MASK tokens)
        targets: (B, T) target token ids (pure tokens)
        loss_extras: {"loss_scale": (B, T), "loss_mask": (B, T)}
        state_dict: dict for resuming training
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    assert model_config is not None, "model_config is required"
    assert isinstance(model_config, BDLMConfig), f"Expected BDLMConfig, got {type(model_config)}"

    # Extract config
    block_size = model_config.bucket_size
    mask_token_id = model_config.mask_token_id
    prefix_pure_tokens = model_config.prefix_pure_tokens
    target_shift = model_config.target_shift

    # Validate
    assert block_size >= 1, "block_size (bucket_size) must be >= 1"
    assert T % block_size == 0, f"T ({T}) must be divisible by block_size ({block_size})"
    assert mask_token_id is not None and mask_token_id != -1, "mask_token_id must be provided"
    if target_shift >= 1:
        assert 1 <= target_shift <= block_size, f"target_shift must be in [1, block_size]"

    needed_tokens = B * T

    token_buffer = create_token_buffer(
        split, resume_state_dict, tokenizer_threads, tokenizer_batch_size
    )

    use_cuda = device == "cuda"

    while True:
        tokens, pq_idx, rg_idx, epoch = token_buffer.get_tokens(needed_tokens)

        targets_cpu = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda)
        targets_cpu = targets_cpu.view(B, T)

        # Compute prefix_sliding_tokens from epoch for block boundary shifting
        prefix_sliding_tokens = epoch % block_size
        num_blocks = (T - prefix_sliding_tokens) // block_size

        # Sample noise level t per block
        t = sample_t(
            batch_size=B,
            num_blocks=num_blocks,
            block_size=block_size,
            sampling_eps_max=1.0,
            device="cpu",
            antithetic_sampling=True,
        )

        # Determine forced mask position
        forced_mask_position = (target_shift - 1) if target_shift >= 1 else None

        # Apply masking
        inputs_cpu, mask = q_xt(
            x0=targets_cpu,
            t=t,
            mask_token_id=mask_token_id,
            block_size=block_size,
            prefix_pure_tokens=prefix_pure_tokens,
            prefix_sliding_tokens=prefix_sliding_tokens,
            forced_mask_position=forced_mask_position,
        )

        # Compute loss_scale
        loss_scale_per_block = get_loss_scale(t)
        loss_scale_blocks = expand_block_to_seq(loss_scale_per_block, block_size)
        loss_scale = torch.zeros(B, T, dtype=loss_scale_blocks.dtype)
        block_region_len = loss_scale_blocks.shape[1]
        loss_scale[:, prefix_sliding_tokens:prefix_sliding_tokens + block_region_len] = loss_scale_blocks

        # Create loss_mask
        if forced_mask_position is not None:
            # Target_shift mode: only compute loss at forced position
            loss_mask = torch.zeros_like(targets_cpu, dtype=torch.bool)
            for block_idx in range(num_blocks):
                pos = prefix_sliding_tokens + block_idx * block_size + forced_mask_position
                loss_mask[:, pos] = True
        else:
            # Normal mode: compute loss on all masked positions
            loss_mask = mask

        # Exclude prefix_pure_tokens from loss
        if prefix_pure_tokens > 0:
            loss_mask[:, :prefix_pure_tokens] = False

        # Move to device
        inputs = inputs_cpu.to(device=device, non_blocking=use_cuda)
        targets = targets_cpu.to(device=device, non_blocking=use_cuda)
        loss_scale = loss_scale.to(device=device, non_blocking=use_cuda)
        loss_mask = loss_mask.to(device=device, non_blocking=use_cuda)

        loss_extras = {"loss_scale": loss_scale, "loss_mask": loss_mask}
        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}
        yield inputs, targets, loss_extras, state_dict
