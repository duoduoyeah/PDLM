"""
Unified dataloader interface that dispatches to model-specific dataloaders.

For direct access, import from:
- nanochat.dataloader_ar (GPT/AR models)
- nanochat.dataloader_bd3lm (BD3LM models)
- nanochat.dataloader_pdlm (PDLM models)
"""

from nanochat.dataloader_ar import ar_data_loader
from nanochat.dataloader_bd3lm import bd3lm_data_loader
from nanochat.dataloader_pdlm import pdlm_data_loader
from nanochat.dataloader_mtp import mtp_data_loader
from nanochat.dataloader_bd3lm_prime import bd3lm_prime_data_loader

from nanochat.gpt import GPTConfig
from nanochat.bd3lm import BDLMConfig
from nanochat.pdlm import PDLMConfig
from nanochat.gpt_mtp import GPTMTPConfig
from nanochat.bd3lm_prime import BD3LMPrimeConfig


def get_data_loader(
    B,
    T,
    split,
    device="cuda",
    model_config=None,
    resume_state_dict=None,
    tokenizer_threads=4,
    tokenizer_batch_size=128,
):
    """
    Get the appropriate dataloader based on model config type.

    Args:
        B: Batch size
        T: Sequence length
        split: "train" or "val"
        device: Target device
        model_config: GPTConfig, BDLMConfig, or PDLMConfig
        resume_state_dict: Optional state dict for resuming

    Returns:
        Generator yielding (inputs, targets, loss_extras, state_dict)
    """
    assert model_config is not None, "model_config is required"

    if isinstance(model_config, GPTConfig):
        return ar_data_loader(
            B, T, split, device, model_config, resume_state_dict,
            tokenizer_threads, tokenizer_batch_size
        )
    elif isinstance(model_config, BDLMConfig):
        return bd3lm_data_loader(
            B, T, split, device, model_config, resume_state_dict,
            tokenizer_threads, tokenizer_batch_size
        )
    elif isinstance(model_config, PDLMConfig):
        return pdlm_data_loader(
            B, T, split, device, model_config, resume_state_dict,
            tokenizer_threads, tokenizer_batch_size
        )
    elif isinstance(model_config, BD3LMPrimeConfig):
        return bd3lm_prime_data_loader(
            B, T, split, device, model_config, resume_state_dict,
            tokenizer_threads, tokenizer_batch_size
        )
    elif isinstance(model_config, GPTMTPConfig):
        return mtp_data_loader(
            B, T, split, device, model_config, resume_state_dict,
            tokenizer_threads, tokenizer_batch_size
        )
    else:
        raise ValueError(f"Unknown model_config type: {type(model_config)}")


# Backward compatibility alias
def tokenizing_distributed_data_loader_with_state(
    B,
    T,
    split,
    tokenizer_threads=4,
    tokenizer_batch_size=128,
    device="cuda",
    resume_state_dict=None,
    model_config=None,
    # Legacy args (ignored if model_config provided)
    noise_total_steps=0,
    prefix_pure_tokens=0,
    model_type="pdlm",
    target_shift=1,
    bd3lm_block_size=1,
    bd3lm_mask_token_id=None,
):
    """
    Backward compatible interface. Prefer using get_data_loader() with model_config.
    """
    if model_config is not None:
        return get_data_loader(
            B, T, split, device, model_config, resume_state_dict,
            tokenizer_threads, tokenizer_batch_size
        )

    # Legacy mode: create a minimal config from args
    raise NotImplementedError(
        "Legacy mode without model_config is deprecated. "
        "Please pass model_config instead."
    )


def tokenizing_distributed_data_loader(*args, **kwargs):
    """Helper that only emits inputs/targets (no state_dict or loss_extras)."""
    for inputs, targets, loss_extras, state_dict in tokenizing_distributed_data_loader_with_state(*args, **kwargs):
        yield inputs, targets
