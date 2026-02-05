"""
Standalone MTP Stage 1 evaluation script.

Loads an MTP model from checkpoint and runs evaluation on validation data.

Usage:
    uv run -m scripts.mtp_eval --model_tag=mtp_d8 --step=1000
    uv run -m scripts.mtp_eval --model_tag=mtp_d8  # uses last step
    uv run -m scripts.mtp_eval --ckpt_dir=/path/to/ckpt
"""

import os
import json
import argparse
from contextlib import nullcontext

import torch

from nanochat.common import compute_init, autodetect_device_type, get_base_dir, print0
from nanochat.checkpoint_manager import load_checkpoint, find_last_step, find_largest_model
from nanochat.gpt_mtp import GPTMTP, GPTMTPConfig
from nanochat.mtp_eval import eval_mtp, dump_mtp_batch
from nanochat.dataloader import get_data_loader
from nanochat.group_tokenizer.token_map import get_token_map


def load_mtp_model(model_tag=None, step=None, device_type="auto", ckpt_dir=None):
    """
    Load an MTP model from checkpoint.

    Args:
        model_tag: Model directory name (e.g., "mtp_d8"). If None, uses largest model.
        step: Checkpoint step. If None, uses last step.
        device_type: "cuda", "cpu", "mps", or "auto"
        ckpt_dir: Direct path to checkpoint directory. If provided, overrides model_tag.

    Returns:
        model: GPTMTP model in eval mode
        meta_data: Metadata dict from checkpoint
        device: Device the model is on
        autocast_ctx: Autocast context for inference
        model_config: GPTMTPConfig instance
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

    # Build model
    model_config_kwargs = meta_data["model_config"]
    # Old checkpoints don't have stage1_target_mode; they used group targets
    model_config_kwargs.setdefault("stage1_target_mode", "group")
    model_config = GPTMTPConfig(**model_config_kwargs)

    with torch.device("meta"):
        model = GPTMTP(model_config)

    model.to_empty(device=device)
    model.init_weights()
    model.load_state_dict(model_data, strict=True, assign=True)

    # Register group mask for pure mode (needed for inference collapse)
    stage1_target_mode = model_config_kwargs.get("stage1_target_mode", "group")
    if stage1_target_mode == "pure":
        token_map = get_token_map(device=device)
        model.register_group_mask(token_map.group_to_pure_mask)
        print0(f"Registered group_to_pure_mask for pure mode evaluation")

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
    Run MTP Stage 1 evaluation.

    Args:
        model_tag: Model directory name
        step: Checkpoint step
        num_batches: Number of validation batches to evaluate
        device_type: Device type
        ckpt_dir: Direct path to checkpoint directory

    Returns:
        eval_result: Dict with evaluation metrics
    """
    # Load model
    model, meta_data, device, autocast_ctx, model_config = load_mtp_model(
        model_tag, step, device_type, ckpt_dir=ckpt_dir
    )

    # Extract config from metadata
    user_config = meta_data.get("user_config", {})
    model_config_dict = meta_data["model_config"]

    max_seq_len = model_config_dict["sequence_len"]
    n_future_tokens = model_config_dict.get("n_future_tokens", 4)
    num_groups = model_config_dict["num_groups"]
    stage1_target_mode = model_config_dict.get("stage1_target_mode", "pure")

    print0(f"Config: max_seq_len={max_seq_len}, K={n_future_tokens}, num_groups={num_groups}, target_mode={stage1_target_mode}")

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
    total_predictions = total_sequences * max_seq_len * n_future_tokens
    print0(f"Running evaluation: {num_batches} batches x {device_batch_size} seqs = {total_sequences} sequences")
    print0(f"  {max_seq_len} positions x {n_future_tokens} predictions = {total_predictions:,} total predictions")

    # Run evaluation
    eval_result = eval_mtp(
        model=model,
        val_loader=val_loader,
        num_batches=num_batches,
        device=device,
        autocast_ctx=autocast_ctx,
    )

    return eval_result


def print_results(eval_result, K, num_groups):
    """Pretty print evaluation results."""
    print0("\n" + "=" * 60)
    print0("MTP STAGE 1 EVALUATION RESULTS")
    print0("=" * 60)

    # Overall metrics
    print0(f"\nOverall metrics:")
    print0(f"  loss:     {eval_result['overall_loss']:.4f}")
    print0(f"  ppl:      {eval_result['overall_ppl']:.2f}")
    print0(f"  accuracy: {eval_result['overall_accuracy']:.2%}")
    print0(f"  tokens evaluated: {eval_result['num_tokens_evaluated']:,}")

    # Per-position metrics
    print0(f"\nPer-position metrics (k=0..{K-1}):")
    for k in range(K):
        pos_data = eval_result["positions"][k]
        print0(f"  k={k}: loss={pos_data['loss']:.4f}, ppl={pos_data['ppl']:.2f}, accuracy={pos_data['accuracy']:.2%}")

    # Per-group accuracy summary
    print0(f"\nPer-group accuracy summary ({num_groups} groups):")
    group_accs = eval_result["per_group_accuracy"]

    # Find top and bottom groups
    sorted_groups = sorted(group_accs.items(), key=lambda x: x[1]["accuracy"], reverse=True)

    # Filter out groups with no samples
    sorted_groups = [(g, d) for g, d in sorted_groups if d["total"] > 0]

    if sorted_groups:
        print0("  Top 5 groups:")
        for g, d in sorted_groups[:5]:
            print0(f"    group {g:3d}: {d['accuracy']:.2%} ({d['correct']}/{d['total']})")

        print0("  Bottom 5 groups:")
        for g, d in sorted_groups[-5:]:
            print0(f"    group {g:3d}: {d['accuracy']:.2%} ({d['correct']}/{d['total']})")

        # Statistics
        accuracies = [d["accuracy"] for _, d in sorted_groups]
        print0(f"\n  Accuracy range: {min(accuracies):.2%} - {max(accuracies):.2%}")
        print0(f"  Mean accuracy:  {sum(accuracies) / len(accuracies):.2%}")

    print0("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Standalone MTP Stage 1 evaluation")
    parser.add_argument("--model_tag", type=str, default=None, help="Model directory name (e.g., mtp_d8)")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="Direct path to checkpoint directory (overrides model_tag)")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step (default: last)")
    parser.add_argument("--num_batches", type=int, default=20, help="Number of validation batches")
    parser.add_argument("--device", type=str, default="auto", help="Device type (cuda/cpu/mps/auto)")
    parser.add_argument("--output_json", type=str, default=None, help="Optional: save results to JSON file")
    parser.add_argument("--dump_mtp", type=str, default=None, help="Dump MTP predictions to file (path to output txt)")
    parser.add_argument("--dump_sequences", type=int, default=5, help="Number of sequences to dump (default: 5)")
    args = parser.parse_args()

    # Handle dump_mtp mode (separate from normal eval)
    if args.dump_mtp:
        model, meta_data, device, autocast_ctx, model_config = load_mtp_model(
            args.model_tag, args.step, args.device, ckpt_dir=args.ckpt_dir
        )
        model_config_dict = meta_data["model_config"]
        user_config = meta_data.get("user_config", {})
        max_seq_len = model_config_dict["sequence_len"]
        device_batch_size = user_config.get("device_batch_size", 32)

        val_loader = get_data_loader(
            device_batch_size, max_seq_len, split="val", device=device,
            model_config=model_config, resume_state_dict=None,
        )

        dump_mtp_batch(
            model=model,
            val_loader=val_loader,
            device=device,
            autocast_ctx=autocast_ctx,
            output_path=args.dump_mtp,
            num_sequences=args.dump_sequences,
        )
        return

    # Run evaluation
    eval_result = run_eval(
        model_tag=args.model_tag,
        step=args.step,
        num_batches=args.num_batches,
        device_type=args.device,
        ckpt_dir=args.ckpt_dir,
    )

    # Get K and num_groups for printing
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
    K = model_config.get("n_future_tokens", 4)
    num_groups = model_config["num_groups"]

    # Print results
    print_results(eval_result, K, num_groups)

    # Optionally save to JSON
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(eval_result, f, indent=2)
        print0(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
