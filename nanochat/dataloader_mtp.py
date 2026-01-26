"""
Dataloader for MTP Stage 1: pure tokens → K future group tokens.

Input: pure tokens (B, T)
Target: K future group tokens per position (B, T, K, overlap_k)
        When overlap_k > 1, each position has multiple valid groups.
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
    Dataloader for MTP Stage 1: pure tokens → K future group tokens.

    Args:
        B: Batch size
        T: Sequence length
        split: "train" or "val"
        device: Target device
        model_config: GPTMTPConfig instance
        resume_state_dict: Optional state dict for resuming

    Yields:
        inputs: (B, T) pure token ids
        targets: (B, T, K, overlap_k) K future group tokens per position,
                 with overlap_k valid groups for each position
        loss_extras: None (MTP doesn't need extra loss info)
        state_dict: dict for resuming training
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    assert model_config is not None, "model_config is required"
    assert isinstance(model_config, GPTMTPConfig), f"Expected GPTMTPConfig, got {type(model_config)}"

    K = model_config.n_future_tokens
    assert K >= 1, "n_future_tokens must be >= 1"

    # Load token map for pure -> group conversion
    token_map = get_token_map(device="cpu")
    pure_to_group = token_map.pure_to_group  # (pure_vocab_size, overlap_k)
    overlap_k = pure_to_group.shape[1]

    # We need B*T input tokens + K extra tokens for the future targets
    needed_tokens = B * T + K

    token_buffer = create_token_buffer(
        split, resume_state_dict, tokenizer_threads, tokenizer_batch_size
    )

    use_cuda = device == "cuda"

    while True:
        tokens, pq_idx, rg_idx, epoch = token_buffer.get_tokens(needed_tokens)

        # Convert to tensor
        scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda)

        # Input tokens: first B*T tokens
        inputs_cpu = scratch[:B * T].view(B, T)

        # Build targets: K future group tokens for each position
        # targets[b, t, k, :] = all valid group_tokens for pure_token[b, t + k + 1]
        #
        # For position t, we predict:
        #   k=0: group of token at t+1
        #   k=1: group of token at t+2
        #   ...
        #   k=K-1: group of token at t+K
        #
        # When overlap_k > 1, each pure token maps to multiple valid groups.
        group_targets = torch.full((B, T, K, overlap_k), -1, dtype=torch.long)

        for k in range(K):
            shift = k + 1  # predict token at position t+shift
            # Future pure tokens (shifted by shift positions)
            # We have tokens[0..B*T+K-1], inputs are tokens[0..B*T-1]
            # For shift=1, future tokens are tokens[1..B*T]
            # For shift=K, future tokens are tokens[K..B*T+K-1]
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
