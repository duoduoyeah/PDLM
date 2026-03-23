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
from nanochat.bd3lm_prime import BD3LMPrime, BD3LMPrimeConfig
from nanochat.bd3lm_eval import eval_bd3lm, eval_bd3lm_threshold, eval_bd3lm_ltr_lookahead, eval_bd3lm_ltr_lookahead_fresh
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state
from nanochat.attn_masks import gen_mask
from nanochat.tokenizer import get_tokenizer, get_tokenizer_from_dir


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

    # Build model — detect BD3-LM vs BD3-LM-Prime from metadata
    model_config_kwargs = meta_data["model_config"]
    model_type = meta_data.get("user_config", {}).get("model_type", "bd3lm")
    is_prime = model_type == "bd3lm_prime" or model_config_kwargs.get("target_length", 1) > 1

    if is_prime:
        model_config = BD3LMPrimeConfig(**model_config_kwargs)
        ModelClass = BD3LMPrime
        print0(f"Detected BD3-LM-Prime (target_length={model_config.target_length}, base={model_config.base})")
    else:
        model_config = BDLMConfig(**model_config_kwargs)
        ModelClass = BDLM

    with torch.device("meta"):
        model = ModelClass(model_config)

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
    total_sequences=None,
    device_type="auto",
    ckpt_dir=None,
    left_to_right=False,
    threshold_decode=False,
    two_tier=False,
    tau2_delta=0.2,
    ltr_sub_lookahead=False,
    ltr_sub_lookahead_fresh=False,
):
    """
    Run BD3LM evaluation.

    Args:
        model_tag: Model directory name
        step: Checkpoint step
        target_shift: None for auto-detect from checkpoint, -1 for normal mode, >= 1 for target_shift mode
        num_batches: Number of validation batches to evaluate (ignored if total_sequences is set)
        total_sequences: Total number of sequences to evaluate (overrides num_batches).
            num_batches is derived as total_sequences // device_batch_size.
        device_type: Device type
        ckpt_dir: Direct path to checkpoint directory. If provided, overrides model_tag.
        left_to_right: If True, run left-to-right teacher-forced eval (target_shift must be < 0).

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
        # When using direct ckpt_dir (e.g., HF downloaded model), tokenizer is at model_dir/tokenizer/
        # ckpt_dir structure: ${MODEL_DIR}/base_checkpoints/d8/ -> tokenizer at ${MODEL_DIR}/tokenizer/
        if ckpt_dir is not None:
            model_dir = os.path.dirname(os.path.dirname(ckpt_dir))
            tokenizer_dir = os.path.join(model_dir, "tokenizer")
            tokenizer = get_tokenizer_from_dir(tokenizer_dir)
        else:
            tokenizer = get_tokenizer()
        try:
            mask_token_id = tokenizer.encode_special("<|MASK|>")
        except KeyError:
            raise ValueError("Could not find MASK token id")

    # Resolve num_batches from total_sequences if provided
    device_batch_size = user_config.get("device_batch_size", 32)
    if total_sequences is not None:
        num_batches = total_sequences // device_batch_size
        if num_batches == 0:
            raise ValueError(f"total_sequences={total_sequences} < device_batch_size={device_batch_size}")
        actual_sequences = num_batches * device_batch_size
        if actual_sequences != total_sequences:
            print0(f"Warning: total_sequences={total_sequences} not divisible by device_batch_size={device_batch_size}, "
                   f"evaluating {actual_sequences} sequences ({num_batches} batches)")

    print0(f"Config: seq_len={max_seq_len}, block_size={block_size}, mask_token_id={mask_token_id}")
    print0(f"Eval mode: target_shift={target_shift}")
    print0(f"Batch: device_batch_size={device_batch_size}, num_batches={num_batches}, "
           f"total_sequences={num_batches * device_batch_size}")

    # Detect model type for val_loader config
    model_type = meta_data.get("user_config", {}).get("model_type", "bd3lm")
    is_prime = model_type == "bd3lm_prime" or model_config.get("target_length", 1) > 1
    ConfigClass = BD3LMPrimeConfig if is_prime else BDLMConfig
    bdlm_config = ConfigClass(**{**model_config, "target_shift": target_shift, "mask_token_id": mask_token_id})
    val_loader = tokenizing_distributed_data_loader_with_state(
        device_batch_size,
        max_seq_len,
        split="val",
        device=device,
        resume_state_dict=None,
        model_config=bdlm_config,
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
    if ltr_sub_lookahead_fresh:
        print0(f"Running BD3LM L2R sub-token lookahead (FRESH) evaluation...")
        eval_result = eval_bd3lm_ltr_lookahead_fresh(
            model=model,
            val_loader=val_loader,
            block_size=block_size,
            num_batches=num_batches,
            attn_mask=attn_mask,
            device=device,
            autocast_ctx=autocast_ctx,
            mask_token_id=mask_token_id,
        )
    elif ltr_sub_lookahead:
        print0(f"Running BD3LM L2R sub-token lookahead evaluation...")
        eval_result = eval_bd3lm_ltr_lookahead(
            model=model,
            val_loader=val_loader,
            block_size=block_size,
            num_batches=num_batches,
            attn_mask=attn_mask,
            device=device,
            autocast_ctx=autocast_ctx,
            mask_token_id=mask_token_id,
        )
    elif threshold_decode:
        mode_str = "two-tier " if two_tier else ""
        print0(f"Running BD3LM {mode_str}threshold decode evaluation...")
        eval_result = eval_bd3lm_threshold(
            model=model,
            val_loader=val_loader,
            block_size=block_size,
            num_batches=num_batches,
            attn_mask=attn_mask,
            device=device,
            autocast_ctx=autocast_ctx,
            mask_token_id=mask_token_id,
            two_tier=two_tier,
            tau2_delta=tau2_delta,
        )
    else:
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
            left_to_right=left_to_right,
        )

    return eval_result


def print_results(eval_result, target_shift, block_size):
    """Pretty print evaluation results."""
    print0("\n" + "=" * 60)
    print0("EVALUATION RESULTS")
    print0("=" * 60)

    # L2R sub-token lookahead result
    if "ltr_sub_lookahead" in eval_result:
        la = eval_result["ltr_sub_lookahead"]
        print0(f"\n[ltr_sub_lookahead] Sub-token lookahead L2R eval:")
        print0(f"  {'τ':>6s}  {'loss':>8s}  {'ppl':>8s}  {'acc':>8s}  {'argmax_p':>8s}")
        print0(f"  {'------':>6s}  {'--------':>8s}  {'--------':>8s}  {'--------':>8s}  {'--------':>8s}")
        for tau in sorted(la.keys(), key=lambda x: float(x)):
            r = la[tau]
            tau_str = "inf" if tau == "inf" or tau == float('inf') else f"{float(tau):.2f}"
            print0(f"  {tau_str:>6s}  {r['overall_loss']:8.4f}  {r['overall_ppl']:8.2f}  "
                   f"{r['overall_accuracy']:7.2%}  {r['overall_argmax_prob']:8.4f}")
        print0("\n" + "=" * 60)
        return

    # Threshold decode result
    if "threshold_decode" in eval_result:
        td = eval_result["threshold_decode"]
        print0(f"\n[threshold_decode] BD3-LM avg_steps by threshold τ:")
        print0(f"  {'τ':>6s}  {'avg_steps':>10s}")
        print0(f"  {'------':>6s}  {'----------':>10s}")
        for tau in sorted(td.keys(), key=float):
            print0(f"  {float(tau):6.2f}  {td[tau]['avg_steps']:10.4f}")
        print0("\n" + "=" * 60)
        return


    # Left-to-right mode result
    if "left_to_right" in eval_result:
        ltr = eval_result["left_to_right"]
        eppl = f", overall_entropy_ppl={ltr['overall_entropy_ppl']:.2f}" if "overall_entropy_ppl" in ltr else ""
        print0(f"\n[left_to_right] overall_loss={ltr['overall_loss']:.4f}, "
               f"overall_ppl={ltr['overall_ppl']:.2f}{eppl}, "
               f"overall_accuracy={ltr['overall_accuracy']:.2%}")
        print0(f"\nPer-position metrics (ppl / entropy_ppl / accuracy):")
        positions = ltr.get("positions", {})
        for k in range(block_size):
            pos_data = positions.get(k, positions.get(str(k), {}))
            ppl  = pos_data.get("ppl", float("nan"))
            eppl = pos_data.get("entropy_ppl", float("nan"))
            acc  = pos_data.get("accuracy", float("nan"))
            print0(f"  pos {k}: loss={pos_data.get('loss', float('nan')):.4f}, "
                   f"ppl={ppl:.2f}, entropy_ppl={eppl:.2f}, accuracy={acc:.2%}")
        print0("\n" + "=" * 60)
        return

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
    parser.add_argument("--num_batches", type=int, default=20, help="Number of validation batches (ignored if --total_sequences is set)")
    parser.add_argument("--total_sequences", type=int, default=None, help="Total sequences to evaluate (overrides --num_batches); num_batches = total_sequences // device_batch_size")
    parser.add_argument("--device", type=str, default="auto", help="Device type (cuda/cpu/mps/auto)")
    parser.add_argument("--output_json", type=str, default=None, help="Optional: save results to JSON file")
    parser.add_argument("--left_to_right", action="store_true",
                        help="Run left-to-right teacher-forced eval: at each position k, reveal "
                             "ground-truth prefix 0..k-1 within each block. Requires target_shift < 0.")
    parser.add_argument("--threshold_decode", action="store_true",
                        help="Run threshold-based parallel decoding: sweep τ from 0.0 to 1.0, "
                             "report avg_steps per threshold.")
    parser.add_argument("--two_tier", action="store_true",
                        help="Use two-tier threshold decode for BD3-LM-Prime: "
                             "high confidence → full decode, medium → half decode (reveal MSB).")
    parser.add_argument("--tau2_delta", type=float, default=0.2,
                        help="Delta between tau1 and tau2 for two-tier decode (tau2 = max(0, tau1 - delta)). Default: 0.2")
    parser.add_argument("--ltr_sub_lookahead", action="store_true",
                        help="L2R teacher-forced eval with sub-token lookahead half-decoding. "
                             "Measures whether partial sub-token reveals help prediction quality.")
    parser.add_argument("--ltr_sub_lookahead_fresh", action="store_true",
                        help="Fresh L2R sub-token lookahead: re-derive half-decodes at each step "
                             "from scratch (no inherited state). Removes trajectory inertia.")
    args = parser.parse_args()

    # Run evaluation
    eval_result = run_eval(
        model_tag=args.model_tag,
        step=args.step,
        target_shift=args.target_shift,
        num_batches=args.num_batches,
        total_sequences=args.total_sequences,
        device_type=args.device,
        ckpt_dir=args.ckpt_dir,
        left_to_right=args.left_to_right,
        threshold_decode=args.threshold_decode,
        two_tier=args.two_tier,
        tau2_delta=args.tau2_delta,
        ltr_sub_lookahead=args.ltr_sub_lookahead,
        ltr_sub_lookahead_fresh=args.ltr_sub_lookahead_fresh,
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
