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

    # Compute needed_tokens based on stage (fetch extra tokens for shifted targets)
    if stage in ("stage1_block", "both_block", "block_pdlm_inference"):
        needed_tokens = B * T + block_size
    elif stage == "both_mtp":
        needed_tokens = B * T + model_config.n_future_tokens
    else:
        needed_tokens = B * T

    token_buffer = create_token_buffer(
        split, resume_state_dict, tokenizer_threads, tokenizer_batch_size
    )

    # Load token_map for group token conversion
    token_map = get_token_map(device="cpu")

    use_cuda = device == "cuda"

    while True:
        tokens, pq_idx, rg_idx, epoch = token_buffer.get_tokens(needed_tokens)

        scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda)
        targets_cpu = scratch[:B * T].view(B, T)

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
            num_groups = token_map.num_groups
            soft_p_within = model_config.soft_p_within

            # Step 1: Get correct group (same as current hard mapping)
            if overlap_k == 1:
                correct_group_ids = pure_to_group[block_tokens, 0]  # (B, block_region_len)
            else:
                k_idx = torch.randint(0, overlap_k, block_tokens.shape)
                flat_tokens = block_tokens.flatten()
                flat_k_idx = k_idx.flatten()
                correct_group_ids = pure_to_group[flat_tokens, flat_k_idx]
                correct_group_ids = correct_group_ids.view(block_tokens.shape)

            # Step 2: Apply soft noise
            if soft_p_within >= 1.0:
                group_ids = correct_group_ids
            else:
                wrong_group_ids = torch.randint(0, num_groups, block_tokens.shape)
                within_mask = torch.rand(block_tokens.shape) < soft_p_within
                group_ids = torch.where(within_mask, correct_group_ids, wrong_group_ids)

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

        elif stage == "stage1_mask":
            # Stage 1 MASK: [prefix, MASK, MASK, ...] → group tokens
            # Input: pure tokens at prefix, MASK at block positions
            # Output: predict group tokens for each MASK position
            # Loss: any-correct over overlap_k valid groups

            mask_id = model_config.mask_token_id
            assert mask_id != -1, "stage1_mask requires mask_token_id to be set"

            # PDLM doesn't use sliding prefix
            prefix_sliding_tokens = 0
            num_blocks = (T - prefix_sliding_tokens) // block_size
            block_region_len = num_blocks * block_size

            # Block boundaries
            block_start = prefix_sliding_tokens
            block_end = block_start + block_region_len

            # Get token map info
            pure_to_group = token_map.pure_to_group  # (pure_vocab, overlap_k)
            overlap_k = token_map.overlap_k

            # inputs: replace block positions with MASK token
            inputs_cpu = targets_cpu.clone()
            inputs_cpu[:, block_start:block_end] = mask_id

            # group_targets: convert pure tokens to group IDs for loss
            # Shape: (B, T, overlap_k) - all valid group assignments
            group_targets = torch.full((B, T, overlap_k), -1, dtype=torch.long)
            block_pure = targets_cpu[:, block_start:block_end]  # (B, block_region_len)
            group_targets[:, block_start:block_end, :] = pure_to_group[block_pure]

            # loss_mask: True at MASK positions (block region), False elsewhere
            loss_mask = torch.zeros(B, T, dtype=torch.bool)
            loss_mask[:, block_start:block_end] = True

            # Exclude prefix_pure_tokens from loss if set
            if prefix_pure_tokens > 0:
                loss_mask[:, :prefix_pure_tokens] = False

            # Move to device
            inputs = inputs_cpu.to(device=device, non_blocking=use_cuda)
            targets = targets_cpu.to(device=device, non_blocking=use_cuda)
            loss_mask = loss_mask.to(device=device, non_blocking=use_cuda)
            group_targets = group_targets.to(device=device, non_blocking=use_cuda)

            loss_extras = {
                "loss_mask": loss_mask,
                "group_targets": group_targets,
            }

        elif stage == "stage1_block":
            # Stage 1 Block: pure tokens in, predict next block with block-causal mask.
            # No MASK tokens, no 2L structure — just L pure tokens.

            prefix_sliding_tokens = 0
            num_blocks = (T - prefix_sliding_tokens) // block_size
            block_region_len = num_blocks * block_size
            block_start = prefix_sliding_tokens

            # inputs: just pure tokens
            inputs_cpu = targets_cpu.clone()

            stage1_target_mode = getattr(model_config, "stage1_target_mode", "pure")

            if stage1_target_mode == "group":
                # Group-target mode: predict group token at same position in next block
                # Same as both_block Stage 1 side
                pure_to_group = token_map.pure_to_group  # (pure_vocab, overlap_k)

                # Shifted view: position i predicts group of token at i + block_size
                future_pure = scratch[block_size:B * T + block_size].view(B, T)
                block_targets = pure_to_group[future_pure]  # (B, T, overlap_k)

                # block_loss_mask: skip block 0 (no prior block context), include last block
                block_loss_mask = torch.zeros(B, T, dtype=torch.bool)
                for blk in range(1, num_blocks):
                    blk_start = block_start + blk * block_size
                    block_loss_mask[:, blk_start:blk_start + block_size] = True

                if prefix_pure_tokens > 0:
                    block_loss_mask[:, :prefix_pure_tokens] = False

                # Move to device
                inputs = inputs_cpu.to(device=device, non_blocking=use_cuda)
                targets = targets_cpu.to(device=device, non_blocking=use_cuda)
                block_targets = block_targets.to(device=device, non_blocking=use_cuda)
                block_loss_mask = block_loss_mask.to(device=device, non_blocking=use_cuda)

                loss_extras = {
                    "block_targets": block_targets,
                    "block_loss_mask": block_loss_mask,
                }
            else:
                # Pure-target mode (default): predict pure token at same position in next block
                # pure_targets[i] = pure token at position i + block_size
                # loss_mask: True for blocks 1 through N-2 (skip block 0 = no prior block context, skip last block = no future to predict)

                # pure_targets: shifted view — position i predicts token at i + block_size
                pure_targets = scratch[block_size:B * T + block_size].view(B, T)

                # loss_mask: skip block 0 (no prior block context), skip last block (no future to predict)
                loss_mask = torch.zeros(B, T, dtype=torch.bool)
                for blk in range(1, num_blocks - 1):
                    blk_start = block_start + blk * block_size
                    loss_mask[:, blk_start:blk_start + block_size] = True

                if prefix_pure_tokens > 0:
                    loss_mask[:, :prefix_pure_tokens] = False

                # Move to device
                inputs = inputs_cpu.to(device=device, non_blocking=use_cuda)
                targets = targets_cpu.to(device=device, non_blocking=use_cuda)  # kept for API consistency
                loss_mask = loss_mask.to(device=device, non_blocking=use_cuda)
                pure_targets = pure_targets.to(device=device, non_blocking=use_cuda)

                # Also include pure_to_group for eval accuracy computation
                loss_extras = {
                    "loss_mask": loss_mask,
                    "pure_targets": pure_targets,
                    "pure_to_group": token_map.pure_to_group.to(device=device, non_blocking=use_cuda),
                }

        elif stage == "stage1_mtp":
            # Stage 1 MTP: handled separately by MTP dataloader
            raise NotImplementedError(f"PDLM {stage} not yet implemented in dataloader")

        elif stage == "both_mtp":
            # Both stages combined: Stage 1 (MTP) + Stage 2 (denoising)
            K = model_config.n_future_tokens

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

        elif stage == "both_block":
            # Both stages combined: Stage 1 (block→block) + Stage 2 (denoising)
            # Same 2L structure as both_mtp, but Stage 1 uses block→block on x0 instead of MTP head.

            prefix_sliding_tokens = 0
            num_blocks = (T - prefix_sliding_tokens) // block_size
            block_region_len = num_blocks * block_size

            block_start = prefix_sliding_tokens
            block_end = block_start + block_region_len

            pure_to_group = token_map.pure_to_group  # (pure_vocab, overlap_k)
            overlap_k = token_map.overlap_k
            pure_vocab_size = token_map.pure_vocab_size

            # === Stage 2 side (inputs/xt): group tokens at block positions ===
            inputs_cpu = targets_cpu.clone()
            block_tokens = targets_cpu[:, block_start:block_end]  # (B, block_region_len)
            num_groups = token_map.num_groups
            soft_p_within = model_config.soft_p_within

            # Step 1: Get correct group
            if overlap_k == 1:
                correct_group_ids = pure_to_group[block_tokens, 0]
            else:
                k_idx = torch.randint(0, overlap_k, block_tokens.shape)
                flat_tokens = block_tokens.flatten()
                flat_k_idx = k_idx.flatten()
                correct_group_ids = pure_to_group[flat_tokens, flat_k_idx]
                correct_group_ids = correct_group_ids.view(block_tokens.shape)

            # Step 2: Apply soft noise (only on xt side; Stage 1 targets stay correct)
            if soft_p_within >= 1.0:
                group_ids = correct_group_ids
            else:
                wrong_group_ids = torch.randint(0, num_groups, block_tokens.shape)
                within_mask = torch.rand(block_tokens.shape) < soft_p_within
                group_ids = torch.where(within_mask, correct_group_ids, wrong_group_ids)

            group_token_ids = pure_vocab_size + group_ids
            inputs_cpu[:, block_start:block_end] = group_token_ids

            # Stage 2 loss_mask: True at block positions
            loss_mask = torch.zeros(B, T, dtype=torch.bool)
            loss_mask[:, block_start:block_end] = True
            if prefix_pure_tokens > 0:
                loss_mask[:, :prefix_pure_tokens] = False

            # === Stage 1 side: block→block targets on x0 half ===
            # Shifted view: position i predicts group of token at i + block_size
            future_pure = scratch[block_size:B * T + block_size].view(B, T)
            block_targets = pure_to_group[future_pure]  # (B, T, overlap_k)

            # block_loss_mask: skip block 0 (no prior block context)
            block_loss_mask = torch.zeros(B, T, dtype=torch.bool)
            for blk in range(1, num_blocks):
                blk_start = block_start + blk * block_size
                block_loss_mask[:, blk_start:blk_start + block_size] = True
            if prefix_pure_tokens > 0:
                block_loss_mask[:, :prefix_pure_tokens] = False

            # Move to device
            inputs = inputs_cpu.to(device=device, non_blocking=use_cuda)
            targets = targets_cpu.to(device=device, non_blocking=use_cuda)
            loss_mask = loss_mask.to(device=device, non_blocking=use_cuda)
            block_targets = block_targets.to(device=device, non_blocking=use_cuda)
            block_loss_mask = block_loss_mask.to(device=device, non_blocking=use_cuda)

            loss_extras = {
                "loss_mask": loss_mask,
                "block_targets": block_targets,
                "block_loss_mask": block_loss_mask,
            }

        elif stage == "block_pdlm_inference":
            # Block PDLM Inference: Stage 2 trains with mixed pure/group inputs.
            # For each block independently, sample r ~ Uniform{0, ..., block_size-1}
            # positions 0..r-1 keep pure tokens, positions r..block_size-1 get group tokens.
            # Stage 2 targets: pure-input positions -> next block pure token,
            #                  group-input positions -> current block pure token.
            # Stage 1 targets: shifted pure tokens (same as both_block).

            prefix_sliding_tokens = 0
            num_blocks = (T - prefix_sliding_tokens) // block_size
            block_region_len = num_blocks * block_size

            block_start = prefix_sliding_tokens
            block_end = block_start + block_region_len

            pure_to_group = token_map.pure_to_group  # (pure_vocab, overlap_k)
            overlap_k = token_map.overlap_k
            pure_vocab_size = token_map.pure_vocab_size

            # === Build inputs (xt): mixed pure/group tokens ===
            inputs_cpu = targets_cpu.clone()

            # For each block, sample r (number of revealed pure positions)
            # r_per_block: (B, num_blocks) each in [0, block_size)
            r_per_block = torch.randint(0, block_size, (B, num_blocks))

            # Build position mask: True = group token, False = keep pure (vectorized)
            pos_in_block = (torch.arange(block_region_len) % block_size).unsqueeze(0)  # (1, block_region_len)
            r_expanded = r_per_block.repeat_interleave(block_size, dim=1)  # (B, block_region_len)
            group_mask = torch.zeros(B, T, dtype=torch.bool)
            group_mask[:, block_start:block_end] = (pos_in_block >= r_expanded)

            # Convert pure tokens to group tokens at group positions
            block_tokens = targets_cpu[:, block_start:block_end]  # (B, block_region_len)
            num_groups = token_map.num_groups
            soft_p_within = model_config.soft_p_within

            if overlap_k == 1:
                correct_group_ids = pure_to_group[block_tokens, 0]
            else:
                k_idx = torch.randint(0, overlap_k, block_tokens.shape)
                flat_tokens = block_tokens.flatten()
                flat_k_idx = k_idx.flatten()
                correct_group_ids = pure_to_group[flat_tokens, flat_k_idx]
                correct_group_ids = correct_group_ids.view(block_tokens.shape)

            # Apply soft noise (only on xt side; targets stay correct)
            if soft_p_within >= 1.0:
                group_ids = correct_group_ids
            else:
                wrong_group_ids = torch.randint(0, num_groups, block_tokens.shape)
                within_mask = torch.rand(block_tokens.shape) < soft_p_within
                group_ids = torch.where(within_mask, correct_group_ids, wrong_group_ids)

            group_token_ids = pure_vocab_size + group_ids  # (B, block_region_len)

            # Expand to full sequence and apply mask
            full_group_token_ids = torch.zeros(B, T, dtype=torch.long)
            full_group_token_ids[:, block_start:block_end] = group_token_ids
            inputs_cpu = torch.where(group_mask, full_group_token_ids, inputs_cpu)

            # === Build stage2_targets: mixed targets ===
            # Pure-input positions (not masked): target = next block pure token
            # Group-input positions (masked): target = current block pure token (denoise)
            future_pure = scratch[block_size:B * T + block_size].view(B, T)
            stage2_targets = torch.where(group_mask, targets_cpu, future_pure)

            # === Build stage2_loss_mask: all block positions are valid ===
            stage2_loss_mask = torch.zeros(B, T, dtype=torch.bool)
            stage2_loss_mask[:, block_start:block_end] = True
            # Skip block 0 for pure-input positions (no prior block to predict from)
            # Actually: group-input positions in block 0 are still valid (denoise current),
            # but pure-input positions in block 0 predict next block which is valid.
            # However, to be consistent, skip block 0 entirely like both_block.
            stage2_loss_mask[:, block_start:block_start + block_size] = False

            if prefix_pure_tokens > 0:
                stage2_loss_mask[:, :prefix_pure_tokens] = False

            # === Build stage1_targets: shifted pure tokens (same as both_block) ===
            stage1_targets = future_pure.clone()

            # === Build stage1_loss_mask: skip block 0 ===
            stage1_loss_mask = torch.zeros(B, T, dtype=torch.bool)
            stage1_loss_mask[:, block_start:block_end] = True
            stage1_loss_mask[:, block_start:block_start + block_size] = False
            if prefix_pure_tokens > 0:
                stage1_loss_mask[:, :prefix_pure_tokens] = False

            # Move to device
            inputs = inputs_cpu.to(device=device, non_blocking=use_cuda)
            targets = targets_cpu.to(device=device, non_blocking=use_cuda)
            stage2_targets = stage2_targets.to(device=device, non_blocking=use_cuda)
            stage2_loss_mask = stage2_loss_mask.to(device=device, non_blocking=use_cuda)
            stage1_targets = stage1_targets.to(device=device, non_blocking=use_cuda)
            stage1_loss_mask = stage1_loss_mask.to(device=device, non_blocking=use_cuda)

            loss_extras = {
                "stage2_targets": stage2_targets,
                "stage2_loss_mask": stage2_loss_mask,
                "stage1_targets": stage1_targets,
                "stage1_loss_mask": stage1_loss_mask,
            }

        elif stage == "mask_pdlm":
            # Mask PDLM: every position predicts itself (not the next block's token).
            # For each block independently, sample r ~ Uniform{0, ..., block_size}
            # r=0: all positions → mask token
            # r>=1: positions 0..r-2 → pure (keep), positions r-1..block_size-1 → group token
            # loss_mask: True at mask + group positions, False at pure positions. Skip block 0.

            prefix_sliding_tokens = 0
            num_blocks = (T - prefix_sliding_tokens) // block_size
            block_region_len = num_blocks * block_size

            block_start = prefix_sliding_tokens
            block_end = block_start + block_region_len

            pure_to_group = token_map.pure_to_group  # (pure_vocab, overlap_k)
            overlap_k = token_map.overlap_k
            pure_vocab_size = token_map.pure_vocab_size
            num_groups = token_map.num_groups
            mask_token_id = pure_vocab_size + num_groups  # last token in wte
            soft_p_within = model_config.soft_p_within

            # === Build inputs (xt): mixed mask/group/pure tokens (vectorized) ===
            inputs_cpu = targets_cpu.clone()

            # For each block, sample r (number of revealed pure positions + 1 for mask state)
            # r_per_block: (B, num_blocks) each in [0, block_size + 1)
            # 5 states for block_size=4: r=0 (all mask), r=1..4 (progressive reveal)
            r_per_block = torch.randint(0, block_size + 1, (B, num_blocks))

            # Vectorized position and r tensors for the block region
            # pos_in_block: (1, block_region_len) — position within each block (0,1,...,K-1,0,1,...)
            pos_in_block = (torch.arange(block_region_len) % block_size).unsqueeze(0)  # (1, block_region_len)
            # r_expanded: (B, block_region_len) — each position knows its block's r value
            r_expanded = r_per_block.repeat_interleave(block_size, dim=1)  # (B, block_region_len)

            # Compute masks vectorized: (B, block_region_len)
            is_mask = (r_expanded == 0)
            is_group = (~is_mask) & (pos_in_block >= r_expanded - 1)
            # (is_pure = everything else — not needed explicitly)

            # Get block tokens and convert to group tokens (vectorized, same as both_block)
            block_tokens = targets_cpu[:, block_start:block_end]  # (B, block_region_len)
            if overlap_k == 1:
                correct_group_ids = pure_to_group[block_tokens, 0]
            else:
                k_idx = torch.randint(0, overlap_k, block_tokens.shape)
                correct_group_ids = pure_to_group[block_tokens.flatten(), k_idx.flatten()].view(block_tokens.shape)

            # Apply soft noise (vectorized)
            if soft_p_within >= 1.0:
                group_ids = correct_group_ids
            else:
                wrong_group_ids = torch.randint(0, num_groups, block_tokens.shape)
                within_mask_soft = torch.rand(block_tokens.shape) < soft_p_within
                group_ids = torch.where(within_mask_soft, correct_group_ids, wrong_group_ids)

            group_token_ids = pure_vocab_size + group_ids  # (B, block_region_len)

            # Build final block tokens: mask where is_mask, group where is_group, pure elsewhere
            block_result = block_tokens.clone()
            block_result[is_mask] = mask_token_id
            block_result[is_group] = group_token_ids[is_group]
            inputs_cpu[:, block_start:block_end] = block_result

            # Build loss_mask: True at mask + group positions, skip block 0
            block_loss = is_mask | is_group  # (B, block_region_len)
            block_loss[:, :block_size] = False  # skip block 0
            loss_mask = torch.zeros(B, T, dtype=torch.bool)
            loss_mask[:, block_start:block_end] = block_loss

            if prefix_pure_tokens > 0:
                loss_mask[:, :prefix_pure_tokens] = False

            # Move to device
            inputs = inputs_cpu.to(device=device, non_blocking=use_cuda)
            targets = targets_cpu.to(device=device, non_blocking=use_cuda)
            loss_mask = loss_mask.to(device=device, non_blocking=use_cuda)

            loss_extras = {"loss_mask": loss_mask}

        elif stage == "both_mask":
            # Both stages with MASK instead of MTP - placeholder
            raise NotImplementedError(f"PDLM {stage} not yet implemented in dataloader")

        else:
            raise ValueError(f"Unknown stage: {stage}")

        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}
        yield inputs, targets, loss_extras, state_dict
