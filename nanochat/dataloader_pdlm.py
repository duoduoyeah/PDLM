"""
Dataloader for PDLM (Parallel Denoising Language Model).
"""

import torch

from nanochat.dataloader_common import create_token_buffer
from nanochat.pdlm import PDLMConfig
from nanochat.group_tokenizer.token_map import get_token_map


def pdlm_data_loader(
    B,
    T,
    split,
    device="cuda",
    model_config: PDLMConfig = None,
    resume_state_dict=None,
    tokenizer_threads=4,
    tokenizer_batch_size=128,
):
    """
    Dataloader for PDLM models.

    For stage2 (Group → Pure denoising):
        - inputs (xt): group tokens at block positions, pure tokens at prefix
        - targets (x0): pure tokens everywhere
        - Loss computed on block positions in xt (first half after model concat)

    Args:
        B: Batch size
        T: Sequence length
        split: "train" or "val"
        device: Target device
        model_config: PDLMConfig instance
        resume_state_dict: Optional state dict for resuming

    Yields:
        inputs: (B, T) input token ids (xt: noised)
        targets: (B, T) target token ids (x0: pure)
        loss_extras: {"loss_mask": (B, T)} for stage2
        state_dict: dict for resuming training
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    assert model_config is not None, "model_config is required"
    assert isinstance(model_config, PDLMConfig), f"Expected PDLMConfig, got {type(model_config)}"

    # Extract config
    stage = model_config.stage
    block_size = model_config.bucket_size
    prefix_pure_tokens = model_config.prefix_pure_tokens

    # Validate
    assert block_size >= 1, "block_size (bucket_size) must be >= 1"
    assert T % block_size == 0, f"T ({T}) must be divisible by block_size ({block_size})"

    needed_tokens = B * T

    token_buffer = create_token_buffer(
        split, resume_state_dict, tokenizer_threads, tokenizer_batch_size
    )

    # Load token_map for group token conversion
    token_map = get_token_map(device="cpu")

    use_cuda = device == "cuda"

    while True:
        tokens, pq_idx, rg_idx, epoch = token_buffer.get_tokens(needed_tokens)

        targets_cpu = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda)
        targets_cpu = targets_cpu.view(B, T)

        if stage == "stage2":
            # Stage 2: Group → Pure denoising
            # xt (inputs): group tokens at block positions, pure at prefix
            # x0 (targets): pure tokens everywhere
            # Loss: only on block positions

            # PDLM doesn't use sliding prefix (unlike BD3LM which cycles based on epoch)
            prefix_sliding_tokens = 0
            num_blocks = (T - prefix_sliding_tokens) // block_size
            block_region_len = num_blocks * block_size

            # Create inputs: start with pure tokens
            inputs_cpu = targets_cpu.clone()

            # Get block region
            block_start = prefix_sliding_tokens
            block_end = block_start + block_region_len
            block_tokens = targets_cpu[:, block_start:block_end]  # (B, block_region_len)

            # Convert pure tokens to group tokens
            # pure_to_group: (pure_vocab_size, overlap_k)
            pure_to_group = token_map.pure_to_group  # (pure_vocab, k)
            overlap_k = token_map.overlap_k
            pure_vocab_size = token_map.pure_vocab_size

            if overlap_k == 1:
                # Each token belongs to exactly one group
                group_ids = pure_to_group[block_tokens, 0]  # (B, block_region_len)
            else:
                # Random selection: pick one of k groups for each position
                k_idx = torch.randint(0, overlap_k, block_tokens.shape)
                # Gather group ids
                flat_tokens = block_tokens.flatten()
                flat_k_idx = k_idx.flatten()
                group_ids = pure_to_group[flat_tokens, flat_k_idx]
                group_ids = group_ids.view(block_tokens.shape)

            # Convert group indices to group token ids (offset by pure_vocab_size)
            group_token_ids = pure_vocab_size + group_ids

            # Replace block positions with group tokens
            inputs_cpu[:, block_start:block_end] = group_token_ids

            # Create loss_mask: only block positions
            loss_mask = torch.zeros(B, T, dtype=torch.bool)
            loss_mask[:, block_start:block_end] = True

            # Exclude prefix_pure_tokens if set
            if prefix_pure_tokens > 0:
                loss_mask[:, :prefix_pure_tokens] = False

            # Move to device
            inputs = inputs_cpu.to(device=device, non_blocking=use_cuda)
            targets = targets_cpu.to(device=device, non_blocking=use_cuda)
            loss_mask = loss_mask.to(device=device, non_blocking=use_cuda)

            loss_extras = {"loss_mask": loss_mask}

        elif stage in ["stage1_mtp", "stage1_mask"]:
            # Stage 1: Pure/MASK → Group prediction
            raise NotImplementedError(f"PDLM {stage} not yet implemented in dataloader")

        elif stage == "both_mtp":
            # Both stages combined: Stage 1 (MTP) + Stage 2 (denoising)
            # Need extra K tokens for MTP future targets
            K = model_config.n_future_tokens
            needed_tokens_both = B * T + K

            # Re-fetch with extra tokens if needed
            if needed_tokens != needed_tokens_both:
                tokens, pq_idx, rg_idx, epoch = token_buffer.get_tokens(needed_tokens_both)

            # Use a scratch buffer for creating shifted views
            scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda)

            # targets (x0): pure tokens - shape (B, T)
            targets_cpu = scratch[:B * T].view(B, T)

            # Create inputs (xt): group tokens at block positions
            prefix_sliding_tokens = 0
            num_blocks = (T - prefix_sliding_tokens) // block_size
            block_region_len = num_blocks * block_size

            inputs_cpu = targets_cpu.clone()

            block_start = prefix_sliding_tokens
            block_end = block_start + block_region_len
            block_tokens = targets_cpu[:, block_start:block_end]  # (B, block_region_len)

            # Convert pure tokens to group tokens for inputs
            pure_to_group = token_map.pure_to_group  # (pure_vocab, overlap_k)
            overlap_k = token_map.overlap_k
            pure_vocab_size = token_map.pure_vocab_size

            if overlap_k == 1:
                group_ids = pure_to_group[block_tokens, 0]
            else:
                k_idx = torch.randint(0, overlap_k, block_tokens.shape)
                flat_tokens = block_tokens.flatten()
                flat_k_idx = k_idx.flatten()
                group_ids = pure_to_group[flat_tokens, flat_k_idx]
                group_ids = group_ids.view(block_tokens.shape)

            group_token_ids = pure_vocab_size + group_ids
            inputs_cpu[:, block_start:block_end] = group_token_ids

            # Create loss_mask for Stage 2: only block positions
            loss_mask = torch.zeros(B, T, dtype=torch.bool)
            loss_mask[:, block_start:block_end] = True

            if prefix_pure_tokens > 0:
                loss_mask[:, :prefix_pure_tokens] = False

            # Create MTP targets: (B, T, K, overlap_k) future group tokens for Stage 1
            # For each position t, we need group tokens at t+1, t+2, ..., t+K
            mtp_targets = torch.zeros(B, T, K, overlap_k, dtype=torch.long)

            for k in range(K):
                shift = k + 1
                # Future pure tokens at position t+shift
                # scratch has B*T + K tokens, so we can index [shift : B*T + shift]
                future_pure = scratch[shift : B * T + shift].view(B, T)
                # Convert to group tokens (all overlap_k options)
                mtp_targets[:, :, k, :] = pure_to_group[future_pure]

            # Move to device
            inputs = inputs_cpu.to(device=device, non_blocking=use_cuda)
            targets = targets_cpu.to(device=device, non_blocking=use_cuda)
            loss_mask = loss_mask.to(device=device, non_blocking=use_cuda)
            mtp_targets = mtp_targets.to(device=device, non_blocking=use_cuda)

            loss_extras = {
                "loss_mask": loss_mask,
                "mtp_targets": mtp_targets,
            }

        elif stage == "both_mask":
            # Both stages with MASK instead of MTP - placeholder
            raise NotImplementedError(f"PDLM {stage} not yet implemented in dataloader")

        else:
            raise ValueError(f"Unknown stage: {stage}")

        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}
        yield inputs, targets, loss_extras, state_dict
