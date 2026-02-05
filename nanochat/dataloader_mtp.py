"""
Dataloader for MTP Stage 1: pure tokens → K future tokens.

Input: pure tokens (B, T)
Target: Format depends on stage1_target_mode:
        - pure mode: (B, T, K) pure token ids
        - group mode: (B, T, K, overlap_k) group token ids with overlap support
"""

import torch

from nanochat.dataloader_common import create_token_buffer
from nanochat.gpt_mtp import GPTMTPConfig
from nanochat.group_tokenizer.token_map import get_token_map


def mtp_data_loader(
    B,
    T,
    split,
    device="cuda",
    model_config: GPTMTPConfig = None,
    resume_state_dict=None,
    tokenizer_threads=4,
    tokenizer_batch_size=128,
):
    """
    Dataloader for MTP Stage 1: pure tokens → K future tokens.

    Args:
        B: Batch size
        T: Sequence length
        split: "train" or "val"
        device: Target device
        model_config: GPTMTPConfig instance
        resume_state_dict: Optional state dict for resuming

    Yields:
        inputs: (B, T) pure token ids
        targets: Format depends on model_config.stage1_target_mode:
                 - pure mode: (B, T, K) pure token ids
                 - group mode: (B, T, K, overlap_k) group token ids with overlap
        loss_extras: dict containing pure_to_group for pure mode, None for group mode
        state_dict: dict for resuming training
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    assert model_config is not None, "model_config is required"
    assert isinstance(model_config, GPTMTPConfig), f"Expected GPTMTPConfig, got {type(model_config)}"

    K = model_config.n_future_tokens
    assert K >= 1, "n_future_tokens must be >= 1"

    stage1_target_mode = getattr(model_config, "stage1_target_mode", "pure")

    # Load token map for pure -> group conversion (needed for both modes)
    token_map = get_token_map(device="cpu")
    pure_to_group = token_map.pure_to_group  # (pure_vocab_size, overlap_k)
    overlap_k = pure_to_group.shape[1]

    # We need B*T input tokens + K extra tokens for the future targets
    needed_tokens = B * T + K

    token_buffer = create_token_buffer(
        split, resume_state_dict, tokenizer_threads, tokenizer_batch_size
    )

    use_cuda = device == "cuda"

    # Pre-move pure_to_group to device for pure mode
    if stage1_target_mode == "pure":
        pure_to_group_device = pure_to_group.to(device=device, non_blocking=use_cuda)

    while True:
        tokens, pq_idx, rg_idx, epoch = token_buffer.get_tokens(needed_tokens)

        # Convert to tensor
        scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda)

        # Input tokens: first B*T tokens
        inputs_cpu = scratch[:B * T].view(B, T)

        if stage1_target_mode == "pure":
            # Pure mode: targets are pure token IDs (B, T, K)
            pure_targets = torch.zeros((B, T, K), dtype=torch.long)

            for k in range(K):
                shift = k + 1  # predict token at position t+shift
                future_start = shift
                future_end = B * T + shift
                future_pure = scratch[future_start:future_end].view(B, T)
                pure_targets[:, :, k] = future_pure

            # Move to device
            inputs = inputs_cpu.to(device=device, non_blocking=use_cuda)
            targets = pure_targets.to(device=device, non_blocking=use_cuda)

            # Pass pure_to_group mapping in loss_extras for teacher forcing
            loss_extras = {"pure_to_group": pure_to_group_device}

            state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}
            yield inputs, targets, loss_extras, state_dict

        else:
            # Group mode (legacy): targets are group token IDs (B, T, K, overlap_k)
            # Build targets: K future group tokens for each position
            # targets[b, t, k, :] = all valid group_tokens for pure_token[b, t + k + 1]
            group_targets = torch.full((B, T, K, overlap_k), -1, dtype=torch.long)

            for k in range(K):
                shift = k + 1  # predict token at position t+shift
                future_start = shift
                future_end = B * T + shift
                future_pure = scratch[future_start:future_end].view(B, T)

                # Convert pure tokens to group tokens
                # Store ALL valid groups (not just index 0)
                group_ids = pure_to_group[future_pure]  # (B, T, overlap_k)
                group_targets[:, :, k, :] = group_ids

            # Move to device
            inputs = inputs_cpu.to(device=device, non_blocking=use_cuda)
            targets = group_targets.to(device=device, non_blocking=use_cuda)

            state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}
            yield inputs, targets, None, state_dict
