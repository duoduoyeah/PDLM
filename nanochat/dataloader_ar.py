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

    while True:
        tokens, pq_idx, rg_idx, epoch = token_buffer.get_tokens(needed_tokens)

        scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda)
        inputs_cpu = scratch[:-target_shift]
        targets_cpu = scratch[target_shift:]

        inputs = inputs_cpu.view(B, T).to(device=device, non_blocking=use_cuda)
        targets = targets_cpu.view(B, T).to(device=device, non_blocking=use_cuda)

        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}
        yield inputs, targets, None, state_dict
