"""
Standalone BD3LM evaluation script.

Loads a BD3LM model from checkpoint and runs evaluation on validation data.

Usage:
    uv run -m scripts.bd3lm_eval --model_tag=d8 --step=1000
    uv run -m scripts.bd3lm_eval --model_tag=d8  # uses last step
    uv run -m scripts.bd3lm_eval  # uses largest model, last step
"""

import os
import json
import argparse
from contextlib import nullcontext

import torch

from nanochat.common import compute_init, autodetect_device_type, get_base_dir, print0
from nanochat.checkpoint_manager import load_checkpoint, find_last_step, find_largest_model
from nanochat.bd3lm import BDLM, BDLMConfig
from nanochat.bd3lm_eval import eval_bd3lm
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state
from nanochat.attn_masks import gen_mask
from nanochat.tokenizer import get_tokenizer


def load_bd3lm_model(model_tag=None, step=None, device_type="auto", ckpt_dir=None):
    """
    Load a BD3LM model from checkpoint.

    Args:
        model_tag: Model directory name (e.g., "d8"). If None, uses largest model.
        step: Checkpoint step. If None, uses last step.
        device_type: "cuda", "cpu", "mps", or "auto"
        ckpt_dir: Direct path to checkpoint directory. If provided, overrides model_tag.

    Returns:
        model: BD3LM model in eval mode
        meta_data: Metadata dict from checkpoint
        device: Device the model is on
        autocast_ctx: Autocast context for inference
    """
    device_type = autodetect_device_type() if device_type == "auto" else device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    # Determine checkpoint directory
    if ckpt_dir is not None:
        # Direct path provided - use it directly
        print0(f"Using direct checkpoint path: {ckpt_dir}")
    else:
        # Use base_dir/base_checkpoints/model_tag structure
        base_dir = get_base_dir()
        checkpoint_dir = os.path.join(base_dir, "base_checkpoints")

        if model_tag is None:
            model_tag = find_largest_model(checkpoint_dir)
            print0(f"No model_tag provided, using largest: {model_tag}")

        ckpt_dir = os.path.join(checkpoint_dir, model_tag)

    if step is None:
        step = find_last_step(ckpt_dir)
        print0(f"No step provided, using last: {step}")

    print0(f"Loading model: {model_tag} at step {step} on {device}")

    model_data, _, meta_data = load_checkpoint(ckpt_dir, step, device, load_optimizer=False)

    # Handle float32 conversion for CPU/MPS
    if device.type in {"cpu", "mps"}:
        model_data = {k: v.float() if v.dtype == torch.bfloat16 else v for k, v in model_data.items()}

    # Fix torch compile prefix
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}

    # Build model
    model_config_kwargs = meta_data["model_config"]
    model_config = BDLMConfig(**model_config_kwargs)

    with torch.device("meta"):
        model = BDLM(model_config)

    model.to_empty(device=device)
    model.init_weights()
    model.load_state_dict(model_data, strict=True, assign=True)
    model.eval()

    # Prepare autocast
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    return model, meta_data, device, autocast_ctx


def run_eval(
    model_tag=None,
    step=None,
    target_shift=None,
    num_batches=20,
    device_type="auto",
    ckpt_dir=None,
):
    """
    Run BD3LM evaluation.

    Args:
        model_tag: Model directory name
        step: Checkpoint step
        target_shift: None for auto-detect from checkpoint, -1 for normal mode, >= 1 for target_shift mode
        num_batches: Number of validation batches to evaluate
        device_type: Device type
        ckpt_dir: Direct path to checkpoint directory. If provided, overrides model_tag.

    Returns:
        eval_result: Dict with evaluation metrics
    """
    # Load model
    model, meta_data, device, autocast_ctx = load_bd3lm_model(model_tag, step, device_type, ckpt_dir=ckpt_dir)

    # Extract config from metadata
    user_config = meta_data.get("user_config", {})
    model_config = meta_data["model_config"]

    # Auto-detect target_shift from checkpoint if not provided
    if target_shift is None:
        # First try model_config (newer checkpoints), then user_config (older checkpoints)
        target_shift = model_config.get("target_shift", user_config.get("target_shift", -1))
        print0(f"Auto-detected target_shift={target_shift} from checkpoint")

    max_seq_len = model_config["sequence_len"]
    block_size = model_config.get("bucket_size", user_config.get("block_size", 4))
    mask_token_id = model_config.get("mask_token_id", -1)
    is_causal = model_config.get("is_causal", True)
    prefix_pure_tokens = model_config.get("prefix_pure_tokens", 1)

    # Get mask token id from tokenizer if not in config
    if mask_token_id == -1:
        tokenizer = get_tokenizer()
        try:
            mask_token_id = tokenizer.encode_special("<|MASK|>")
        except KeyError:
            raise ValueError("Could not find MASK token id")

    print0(f"Config: max_seq_len={max_seq_len}, block_size={block_size}, mask_token_id={mask_token_id}")
    print0(f"Eval mode: target_shift={target_shift}")

    # Create validation dataloader
    device_batch_size = user_config.get("device_batch_size", 32)
    val_loader = tokenizing_distributed_data_loader_with_state(
        device_batch_size,
        max_seq_len,
        split="val",
        device=device,
        resume_state_dict=None,
        noise_total_steps=0,
        prefix_pure_tokens=prefix_pure_tokens,
        model_type="bd3lm",
        target_shift=target_shift,
        bd3lm_block_size=block_size,
        bd3lm_mask_token_id=mask_token_id,
    )

    # Generate attention mask for eval (prefix_sliding_tokens=0 for eval)
    attn_mask = gen_mask(max_seq_len, block_size, attn_backend="sdpa", is_causal=is_causal, prefix_sliding_tokens=0).to(device=device)

    # Run evaluation - report actual data size
    total_sequences = num_batches * device_batch_size
    blocks_per_seq = max_seq_len // block_size
    eval_blocks_per_seq = blocks_per_seq - 1  # skip block 0
    total_eval_blocks = total_sequences * eval_blocks_per_seq
    print0(f"Running evaluation: {num_batches} batches × {device_batch_size} seqs = {total_sequences} sequences")
    print0(f"  {blocks_per_seq} blocks/seq, {eval_blocks_per_seq} evaluated (skip block 0) = {total_eval_blocks:,} total blocks")
    eval_result = eval_bd3lm(
        model=model,
        val_loader=val_loader,
        block_size=block_size,
        target_shift=target_shift,
        num_batches=num_batches,
        attn_mask=attn_mask,
        device=device,
        autocast_ctx=autocast_ctx,
        mask_token_id=mask_token_id,
    )

    return eval_result


def print_results(eval_result, target_shift, block_size):
    """Pretty print evaluation results."""
    print0("\n" + "=" * 60)
    print0("EVALUATION RESULTS")
    print0("=" * 60)

    if target_shift >= 1:
        # Target shift mode
        print0(f"\n[target_shift={target_shift}] Core metrics (all masked):")
        print0(f"  loss: {eval_result['loss']:.4f}")
        print0(f"  ppl:  {eval_result['ppl']:.2f}")

        # Suffix metrics
        max_suffix = block_size - target_shift
        if max_suffix > 0:
            print0(f"\nSuffix metrics:")
            for s in range(1, max_suffix + 1):
                key_loss = f"loss_{s}suffix"
                key_ppl = f"ppl_{s}suffix"
                if key_loss in eval_result:
                    print0(f"  {s} suffix clear: loss={eval_result[key_loss]:.4f}, ppl={eval_result[key_ppl]:.2f}")
    else:
        # Normal mode
        print0(f"\n[normal mode] overall_loss: {eval_result['overall_loss']:.4f}, overall_ppl: {eval_result['overall_ppl']:.2f}")

        # Position-centric printing: each position on one line with all suffix metrics
        print0(f"\nPer-position metrics (with suffix):")
        for pos in range(block_size):
            pos_data = eval_result["positions"][pos]
            line = f"  pos {pos}: loss={pos_data['loss']:.2f}, ppl={pos_data['ppl']:.1f}"
            # Append suffix metrics for this position
            max_suffix_for_pos = block_size - 1 - pos
            for s in range(1, max_suffix_for_pos + 1):
                if f"loss_{s}suffix" in pos_data:
                    line += f" | {s}s={pos_data[f'loss_{s}suffix']:.2f}/{pos_data[f'ppl_{s}suffix']:.1f}"
            print0(line)

    print0("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Standalone BD3LM evaluation")
    parser.add_argument("--model_tag", type=str, default=None, help="Model directory name (e.g., d8)")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="Direct path to checkpoint directory (overrides model_tag)")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step (default: last)")
    parser.add_argument("--target_shift", type=int, default=None, help="Target shift mode (default: auto-detect from checkpoint)")
    parser.add_argument("--num_batches", type=int, default=20, help="Number of validation batches")
    parser.add_argument("--device", type=str, default="auto", help="Device type (cuda/cpu/mps/auto)")
    parser.add_argument("--output_json", type=str, default=None, help="Optional: save results to JSON file")
    args = parser.parse_args()

    # Run evaluation
    eval_result = run_eval(
        model_tag=args.model_tag,
        step=args.step,
        target_shift=args.target_shift,
        num_batches=args.num_batches,
        device_type=args.device,
        ckpt_dir=args.ckpt_dir,
    )

    # Get block_size and target_shift for printing (re-load meta to get it)
    # Determine ckpt_dir for metadata loading
    if args.ckpt_dir is not None:
        ckpt_dir = args.ckpt_dir
    else:
        base_dir = get_base_dir()
        checkpoint_dir = os.path.join(base_dir, "base_checkpoints")
        model_tag = args.model_tag or find_largest_model(checkpoint_dir)
        ckpt_dir = os.path.join(checkpoint_dir, model_tag)
    step = args.step or find_last_step(ckpt_dir)
    meta_path = os.path.join(ckpt_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r") as f:
        meta_data = json.load(f)
    model_config = meta_data["model_config"]
    user_config = meta_data.get("user_config", {})
    block_size = model_config.get("bucket_size", user_config.get("block_size", 4))
    # Get target_shift: use CLI arg if provided, otherwise auto-detect from checkpoint
    target_shift = args.target_shift
    if target_shift is None:
        target_shift = model_config.get("target_shift", user_config.get("target_shift", -1))

    # Print results
    print_results(eval_result, target_shift, block_size)

    # Optionally save to JSON
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(eval_result, f, indent=2)
        print0(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
