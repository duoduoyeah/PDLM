"""
Dataloader for autoregressive (GPT) models.
"""

import torch

from nanochat.dataloader_common import create_token_buffer
from nanochat.gpt import GPTConfig


def ar_data_loader(
    B,
    T,
    split,
    device="cuda",
    model_config: GPTConfig = None,
    resume_state_dict=None,
    tokenizer_threads=4,
    tokenizer_batch_size=128,
):
    """
    Dataloader for autoregressive models (GPT).

    Args:
        B: Batch size
        T: Sequence length
        split: "train" or "val"
        device: Target device
        model_config: GPTConfig instance
        resume_state_dict: Optional state dict for resuming

    Yields:
        inputs: (B, T) input token ids
        targets: (B, T) target token ids (shifted by target_shift)
        loss_extras: None (AR doesn't need extra loss info)
        state_dict: dict for resuming training
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    assert model_config is not None, "model_config is required"
    assert isinstance(model_config, GPTConfig), f"Expected GPTConfig, got {type(model_config)}"

    target_shift = model_config.target_shift
    assert target_shift >= 1, "target_shift must be >= 1 for next-token prediction"

    # We need extra tokens for the shift
    needed_tokens = B * T + target_shift

    token_buffer = create_token_buffer(
        split, resume_state_dict, tokenizer_threads, tokenizer_batch_size
    )

    use_cuda = device == "cuda"
    loss_mask_block_size = getattr(model_config, 'loss_mask_block_size', 0)

    while True:
        tokens, pq_idx, rg_idx, epoch = token_buffer.get_tokens(needed_tokens)

        scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda)
        inputs_cpu = scratch[:-target_shift]
        targets_cpu = scratch[target_shift:]

        inputs = inputs_cpu.view(B, T).to(device=device, non_blocking=use_cuda)
        targets = targets_cpu.view(B, T).to(device=device, non_blocking=use_cuda)

        loss_extras = None
        if loss_mask_block_size > 0:
            # 4-state block loss mask (same pattern as mask_pdlm 4-state)
            # For block_size=4: expected 10/16 tokens compute loss
            K = loss_mask_block_size
            num_blocks = T // K
            block_region_len = num_blocks * K

            # Sample r ∈ {0, ..., K-1} per block per batch element
            r_per_block = torch.randint(0, K, (B, num_blocks))
            r_expanded = r_per_block.repeat_interleave(K, dim=1)  # (B, block_region_len)
            pos_in_block = (torch.arange(block_region_len) % K).unsqueeze(0)  # (1, block_region_len)

            # r=0: all compute loss; r>0: positions < r are pure (no loss)
            loss_mask = torch.ones(B, T, dtype=torch.bool)
            pure = (r_expanded > 0) & (pos_in_block < r_expanded)
            loss_mask[:, :block_region_len] = ~pure

            # Set targets to -1 at pure positions (cross_entropy ignore_index=-1)
            targets[loss_mask == False] = -1

            loss_mask = loss_mask.to(device=device, non_blocking=use_cuda)
            loss_extras = {"loss_mask": loss_mask}

        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}
        yield inputs, targets, loss_extras, state_dict
