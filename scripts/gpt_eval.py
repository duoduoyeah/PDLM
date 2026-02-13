"""
Standalone GPT evaluation script.

Loads a GPT model from checkpoint and runs evaluation on validation data.

Usage:
    uv run -m scripts.gpt_eval --ckpt_dir=/path/to/ckpt
    uv run -m scripts.gpt_eval --model_tag=d8 --step=1000
    uv run -m scripts.gpt_eval --model_tag=d8  # uses last step
"""

import os
import json
import argparse
from contextlib import nullcontext

import torch

from nanochat.common import compute_init, autodetect_device_type, get_base_dir, print0
from nanochat.checkpoint_manager import load_checkpoint, find_last_step, find_largest_model
from nanochat.gpt import GPT, GPTConfig
from nanochat.gpt_eval import eval_gpt
from nanochat.dataloader import get_data_loader


def load_gpt_model(model_tag=None, step=None, device_type="auto", ckpt_dir=None):
    """
    Load a GPT model from checkpoint.

    Args:
        model_tag: Model directory name (e.g., "d8"). If None, uses largest model.
        step: Checkpoint step. If None, uses last step.
        device_type: "cuda", "cpu", "mps", or "auto"
        ckpt_dir: Direct path to checkpoint directory. If provided, overrides model_tag.

    Returns:
        model: GPT model in eval mode
        meta_data: Metadata dict from checkpoint
        device: Device the model is on
        autocast_ctx: Autocast context for inference
        model_config: GPTConfig instance
    """
    device_type = autodetect_device_type() if device_type == "auto" else device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    # Determine checkpoint directory
    if ckpt_dir is not None:
        print0(f"Using direct checkpoint path: {ckpt_dir}")
    else:
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

    # Build model config
    model_config_kwargs = dict(meta_data["model_config"])

    # Critical: target_shift is NOT saved in model_config (see base_train.py:167-174),
    # it's in user_config. Inject it so the dataloader shifts correctly.
    user_config = meta_data.get("user_config", {})
    model_config_kwargs["target_shift"] = user_config.get("target_shift", 1)

    model_config = GPTConfig(**model_config_kwargs)

    with torch.device("meta"):
        model = GPT(model_config)

    model.to_empty(device=device)
    model.init_weights()
    model.load_state_dict(model_data, strict=True, assign=True)
    model.eval()

    # Prepare autocast
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    return model, meta_data, device, autocast_ctx, model_config


def run_eval(
    model_tag=None,
    step=None,
    num_batches=20,
    device_type="auto",
    ckpt_dir=None,
):
    """
    Run GPT evaluation.

    Args:
        model_tag: Model directory name
        step: Checkpoint step
        num_batches: Number of validation batches to evaluate
        device_type: Device type
        ckpt_dir: Direct path to checkpoint directory

    Returns:
        eval_result: Dict with evaluation metrics
        target_shift: The target_shift value used
    """
    # Load model
    model, meta_data, device, autocast_ctx, model_config = load_gpt_model(
        model_tag, step, device_type, ckpt_dir=ckpt_dir
    )

    # Extract config from metadata
    user_config = meta_data.get("user_config", {})
    model_config_dict = meta_data["model_config"]

    max_seq_len = model_config_dict["sequence_len"]
    target_shift = model_config.target_shift

    print0(f"Config: max_seq_len={max_seq_len}, target_shift={target_shift}")

    # Create validation dataloader
    device_batch_size = user_config.get("device_batch_size", 32)
    val_loader = get_data_loader(
        device_batch_size,
        max_seq_len,
        split="val",
        device=device,
        model_config=model_config,
        resume_state_dict=None,
    )

    # Report eval size
    total_sequences = num_batches * device_batch_size
    total_predictions = total_sequences * max_seq_len
    print0(f"Running evaluation: {num_batches} batches x {device_batch_size} seqs = {total_sequences} sequences")
    print0(f"  {max_seq_len} positions = {total_predictions:,} total predictions")

    # Run evaluation
    eval_result = eval_gpt(
        model=model,
        val_loader=val_loader,
        num_batches=num_batches,
        device=device,
        autocast_ctx=autocast_ctx,
    )

    return eval_result, target_shift


def print_results(eval_result, target_shift):
    """Pretty print evaluation results."""
    print0("\n" + "=" * 60)
    print0("GPT EVALUATION RESULTS")
    print0("=" * 60)

    print0(f"\ntarget_shift: {target_shift}")
    print0(f"\nMetrics:")
    print0(f"  loss:         {eval_result['overall_loss']:.4f}")
    print0(f"  ppl:          {eval_result['overall_ppl']:.2f}")
    print0(f"  entropy_ppl:  {eval_result['overall_entropy_ppl']:.2f}")
    print0(f"  accuracy:     {eval_result['overall_accuracy']:.2%}")
    print0(f"  tokens evaluated: {eval_result['num_tokens_evaluated']:,}")

    print0("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Standalone GPT evaluation")
    parser.add_argument("--model_tag", type=str, default=None, help="Model directory name (e.g., d8)")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="Direct path to checkpoint directory (overrides model_tag)")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step (default: last)")
    parser.add_argument("--num_batches", type=int, default=20, help="Number of validation batches")
    parser.add_argument("--device", type=str, default="auto", help="Device type (cuda/cpu/mps/auto)")
    parser.add_argument("--output_json", type=str, default=None, help="Optional: save results to JSON file")
    args = parser.parse_args()

    # Run evaluation
    eval_result, target_shift = run_eval(
        model_tag=args.model_tag,
        step=args.step,
        num_batches=args.num_batches,
        device_type=args.device,
        ckpt_dir=args.ckpt_dir,
    )

    # Print results
    print_results(eval_result, target_shift)

    # Optionally save to JSON
    if args.output_json:
        output = dict(eval_result)
        output["target_shift"] = target_shift
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print0(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
