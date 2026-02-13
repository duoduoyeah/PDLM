"""
Standalone PDLM evaluation script.

Supports both Stage 1 MASK and Stage 2 evaluation.
Loads a PDLM model from checkpoint and runs evaluation on validation data.

Usage:
    uv run -m scripts.pdlm_eval --model_tag=d8 --step=1000
    uv run -m scripts.pdlm_eval --model_tag=d8  # uses last step
    uv run -m scripts.pdlm_eval --ckpt_dir=/path/to/ckpt
    uv run -m scripts.pdlm_eval --ckpt_dir=/path/to/ckpt --run_compatibility  # also runs oracle accuracy
    uv run -m scripts.pdlm_eval --ckpt_dir=/path/to/ckpt --run_compatibility --no_oracle_accuracy
    uv run -m scripts.pdlm_eval --ckpt_dir=/path/to/ckpt --run_oracle_accuracy  # oracle only
"""

import os
import json
import argparse
from contextlib import nullcontext

import torch

from nanochat.common import compute_init, autodetect_device_type, get_base_dir, print0
from nanochat.checkpoint_manager import load_checkpoint, find_last_step, find_largest_model
from nanochat.pdlm import PDLM, PDLMConfig
from nanochat.pdlm_eval import eval_pdlm, eval_pdlm_stage1_mask, eval_pdlm_stage1_block, eval_pdlm_compatibility, eval_pdlm_full, eval_pdlm_both_block, eval_block_pdlm_inference, eval_mask_pdlm, dump_batch_to_file, dump_stage1_block_batch
from nanochat.dataloader import get_data_loader
from nanochat.attn_masks import gen_mask, gen_block_causal_mask
from nanochat.group_tokenizer.token_map import get_token_map


def load_pdlm_model(model_tag=None, step=None, device_type="auto", ckpt_dir=None):
    """
    Load a PDLM model from checkpoint.

    Args:
        model_tag: Model directory name (e.g., "d8"). If None, uses largest model.
        step: Checkpoint step. If None, uses last step.
        device_type: "cuda", "cpu", "mps", or "auto"
        ckpt_dir: Direct path to checkpoint directory. If provided, overrides model_tag.

    Returns:
        model: PDLM model in eval mode
        meta_data: Metadata dict from checkpoint
        device: Device the model is on
        autocast_ctx: Autocast context for inference

    Note:
        For HF downloaded models, set NANOCHAT_BASE_DIR to the model directory
        so that get_base_dir() returns the correct path for tokenizer/token_maps.pt
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
    model_config = PDLMConfig(**model_config_kwargs)

    with torch.device("meta"):
        model = PDLM(model_config)

    model.to_empty(device=device)
    model.init_weights()

    # Extract group_to_pure_mask before strict loading (it's a buffer, not a parameter)
    group_to_pure_mask = model_data.pop("group_to_pure_mask", None)

    model.load_state_dict(model_data, strict=True, assign=True)
    model.eval()

    # Register group mask for inference collapse (e.g. stage1_block eval)
    if group_to_pure_mask is not None:
        model.register_group_mask(group_to_pure_mask)

    # Prepare autocast
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    return model, meta_data, device, autocast_ctx, model_config


def run_eval(
    model_tag=None,
    step=None,
    num_batches=20,
    device_type="auto",
    ckpt_dir=None,
    run_compatibility=False,
    compatibility_batches=None,
    run_oracle_accuracy=False,
    oracle_accuracy_batches=None,
):
    """
    Run PDLM evaluation (Stage 1 MASK or Stage 2).

    Args:
        model_tag: Model directory name
        step: Checkpoint step
        num_batches: Number of validation batches to evaluate
        device_type: Device type
        ckpt_dir: Direct path to checkpoint directory. If provided, overrides model_tag.
        run_compatibility: Whether to run compatibility evaluation (Stage 2 only)
        compatibility_batches: Number of batches for compatibility (default: num_batches // 4)
        run_oracle_accuracy: Whether to run oracle accuracy evaluation
        oracle_accuracy_batches: Number of batches for oracle accuracy (default: num_batches // 4)

    Returns:
        eval_result: Dict with evaluation metrics
    """
    # Load model
    model, meta_data, device, autocast_ctx, model_config = load_pdlm_model(
        model_tag, step, device_type, ckpt_dir=ckpt_dir
    )

    # Extract config from metadata
    user_config = meta_data.get("user_config", {})
    model_config_dict = meta_data["model_config"]

    max_seq_len = model_config_dict["sequence_len"]
    block_size = model_config_dict.get("bucket_size", user_config.get("block_size", 4))
    is_causal = model_config_dict.get("is_causal", True)
    prefix_pure_tokens = model_config_dict.get("prefix_pure_tokens", 0)
    stage = model_config_dict.get("stage", "stage2")

    print0(f"Config: max_seq_len={max_seq_len}, block_size={block_size}, stage={stage}")
    print0(f"  prefix_pure_tokens={prefix_pure_tokens}, is_causal={is_causal}")

    # Create validation dataloader
    device_batch_size = user_config.get("device_batch_size", 32)
    val_loader = get_data_loader(
        device_batch_size,
        max_seq_len,
        split="val",  # Use validation set
        device=device,
        model_config=model_config,
        resume_state_dict=None,
    )

    # Generate attention mask for eval
    if stage == "stage1_block":
        # stage1_block uses L×L block-causal mask (no 2L structure)
        attn_mask = gen_block_causal_mask(
            max_seq_len, block_size, attn_backend="sdpa", is_causal=is_causal
        ).to(device=device)
    else:
        # Other stages use 2L×2L mask (prefix_sliding_tokens=0 for eval)
        attn_mask = gen_mask(
            max_seq_len, block_size, attn_backend="sdpa",
            is_causal=is_causal, prefix_sliding_tokens=0
        ).to(device=device)

    # Run evaluation - report actual data size
    total_sequences = num_batches * device_batch_size
    blocks_per_seq = max_seq_len // block_size
    eval_blocks_per_seq = blocks_per_seq - 1  # skip block 0
    total_eval_blocks = total_sequences * eval_blocks_per_seq
    print0(f"Running evaluation: {num_batches} batches × {device_batch_size} seqs = {total_sequences} sequences")
    print0(f"  {blocks_per_seq} blocks/seq, {eval_blocks_per_seq} evaluated (skip block 0) = {total_eval_blocks:,} total blocks")

    # Branch based on stage
    if stage == "stage1_mask":
        print0(f"Running Stage 1 MASK evaluation...")
        eval_result = eval_pdlm_stage1_mask(
            model=model,
            val_loader=val_loader,
            block_size=block_size,
            num_batches=num_batches,
            attn_mask=attn_mask,
            device=device,
            autocast_ctx=autocast_ctx,
            prefix_pure_tokens=prefix_pure_tokens,
        )
    elif stage == "stage1_block":
        print0(f"Running Stage 1 Block evaluation...")
        eval_result = eval_pdlm_stage1_block(
            model=model,
            val_loader=val_loader,
            block_size=block_size,
            num_batches=num_batches,
            attn_mask=attn_mask,
            device=device,
            autocast_ctx=autocast_ctx,
            prefix_pure_tokens=prefix_pure_tokens,
        )
    elif stage == "both_block":
        mtp_loss_weight = model_config_dict.get("mtp_loss_weight", 1.0)
        extras = []
        if run_compatibility:
            extras.append("compatibility")
        if run_oracle_accuracy:
            extras.append("oracle accuracy")
        extra_str = f" with {' + '.join(extras)}" if extras else ""
        print0(f"Running both_block evaluation{extra_str}...")
        eval_result = eval_pdlm_both_block(
            model=model,
            val_loader=val_loader,
            block_size=block_size,
            num_batches=num_batches,
            attn_mask=attn_mask,
            device=device,
            autocast_ctx=autocast_ctx,
            prefix_pure_tokens=prefix_pure_tokens,
            mtp_loss_weight=mtp_loss_weight,
            run_compatibility=run_compatibility,
            compatibility_batches=compatibility_batches,
            run_oracle_accuracy=run_oracle_accuracy,
            oracle_accuracy_batches=oracle_accuracy_batches,
        )
    elif stage == "block_pdlm_inference":
        mtp_loss_weight = model_config_dict.get("mtp_loss_weight", 1.0)
        print0(f"Running block_pdlm_inference evaluation...")
        eval_result = eval_block_pdlm_inference(
            model=model, val_loader=val_loader, block_size=block_size,
            num_batches=num_batches, attn_mask=attn_mask, device=device,
            autocast_ctx=autocast_ctx, prefix_pure_tokens=prefix_pure_tokens,
            mtp_loss_weight=mtp_loss_weight,
        )
    elif stage == "mask_pdlm":
        print0(f"Running mask_pdlm evaluation...")
        eval_result = eval_mask_pdlm(
            model=model,
            val_loader=val_loader,
            block_size=block_size,
            num_batches=num_batches,
            attn_mask=attn_mask,
            device=device,
            autocast_ctx=autocast_ctx,
            prefix_pure_tokens=prefix_pure_tokens,
        )
    elif run_compatibility or run_oracle_accuracy:
        extras = []
        if run_compatibility:
            extras.append("compatibility")
        if run_oracle_accuracy:
            extras.append("oracle accuracy")
        print0(f"Running full evaluation with {' + '.join(extras)}...")
        eval_result = eval_pdlm_full(
            model=model,
            val_loader=val_loader,
            block_size=block_size,
            num_batches=num_batches,
            attn_mask=attn_mask,
            device=device,
            autocast_ctx=autocast_ctx,
            prefix_pure_tokens=prefix_pure_tokens,
            run_compatibility=run_compatibility,
            compatibility_batches=compatibility_batches,
            run_oracle_accuracy=run_oracle_accuracy,
            oracle_accuracy_batches=oracle_accuracy_batches,
        )
    else:
        print0(f"Running Stage 2 evaluation...")
        eval_result = eval_pdlm(
            model=model,
            val_loader=val_loader,
            block_size=block_size,
            num_batches=num_batches,
            attn_mask=attn_mask,
            device=device,
            autocast_ctx=autocast_ctx,
            prefix_pure_tokens=prefix_pure_tokens,
        )

    # Add stage to result for print_results
    eval_result["stage"] = stage
    return eval_result


def print_results(eval_result, block_size):
    """Pretty print evaluation results."""
    stage = eval_result.get("stage", "stage2")

    print0("\n" + "=" * 60)
    if stage == "mask_pdlm":
        print0("PDLM MASK_PDLM EVALUATION RESULTS")
        print0("=" * 60)

        unified = eval_result["unified"]
        e2e = eval_result["end2end"]
        print0(f"\n[mask_pdlm] unified_loss: {eval_result['unified_loss']:.4f}, end2end_loss: {eval_result['end2end_loss']:.4f}")

        print0(f"\n  Unified: loss={unified['overall_loss']:.4f}, ppl={unified['overall_ppl']:.2f}")
        mask_bk = unified.get("mask_breakdown", {})
        group_bk = unified.get("group_breakdown", {})
        for pos in range(block_size):
            pos_data = unified["positions"][pos]
            parts = [f"pos {pos}: loss={pos_data['loss']:.4f}"]
            if pos in mask_bk:
                parts.append(f"mask={mask_bk[pos]['loss']:.4f}")
            g_parts = []
            for g in range(pos + 1):
                if pos in group_bk and g in group_bk[pos]:
                    g_parts.append(f"g{g}={group_bk[pos][g]['loss']:.4f}")
            if g_parts:
                parts.append("group[" + " ".join(g_parts) + "]")
            print0(f"    {' | '.join(parts)}")

        print0(f"\n  End-to-End: loss={e2e['overall_loss']:.4f}, ppl={e2e['overall_ppl']:.2f}")
        for pos in range(block_size):
            pos_data = e2e["positions"][pos]
            print0(f"    pos {pos}: loss={pos_data['loss']:.4f}, ppl={pos_data['ppl']:.2f}")

    elif stage == "both_block":
        print0("PDLM BOTH_BLOCK EVALUATION RESULTS")
        print0("=" * 60)

        s1 = eval_result["stage1"]
        s2 = eval_result["stage2"]

        print0(f"\n[block_pdlm] combined_loss: {eval_result['combined_loss']:.4f}, end2end_loss: {eval_result['end2end_loss']:.4f}")
        print0(f"  mtp_loss_weight: {eval_result['mtp_loss_weight']}, stage1_loss: {s1['overall_loss']:.4f}, stage2_loss: {s2['overall_loss']:.4f}")

        print0(f"\n  Stage 1 (Block→Block): loss={s1['overall_loss']:.4f}, ppl={s1['overall_ppl']:.2f}, accuracy={s1['overall_accuracy']:.2%}")
        for pos in range(block_size):
            pos_data = s1["positions"][pos]
            print0(f"    pos {pos}: loss={pos_data['loss']:.4f}, ppl={pos_data['ppl']:.2f}, accuracy={pos_data['accuracy']:.2%}")

        print0(f"\n  Stage 2 (Denoise): loss={s2['overall_loss']:.4f}, ppl={s2['overall_ppl']:.2f}")
        for pos in range(block_size):
            pos_data = s2["positions"][pos]
            print0(f"    pos {pos}: loss={pos_data['loss']:.4f}, ppl={pos_data['ppl']:.2f}")

        # End-to-end per-position metrics (if present)
        if "end2end" in eval_result:
            e2e = eval_result["end2end"]
            print0(f"\n  End-to-End (Stage1→Stage2): loss={e2e['overall_loss']:.4f}, ppl={e2e['overall_ppl']:.2f}")
            for pos in range(block_size):
                pos_data = e2e["positions"][pos]
                print0(f"    pos {pos}: loss={pos_data['loss']:.4f}, ppl={pos_data['ppl']:.2f}")

        # Compatibility metrics (if present, in stage2)
        if "compatibility" in s2:
            compat = s2["compatibility"]
            print0(f"\n  Compatibility: {compat['overall_compatibility']:.2%}")
            for pos in range(block_size):
                pos_data = compat["positions"][pos]
                print0(f"    pos {pos}: {pos_data['compatibility']:.2%} ({pos_data['matched']}/{pos_data['total']})")

        # Oracle accuracy metrics (if present, in stage2)
        if "oracle_accuracy" in s2:
            oracle = s2["oracle_accuracy"]
            print0(f"\n  Oracle accuracy: {oracle['overall_accuracy']:.2%}")
            for pos in range(block_size):
                pos_data = oracle["positions"][pos]
                print0(f"    pos {pos}: {pos_data['accuracy']:.2%} ({pos_data['matched']}/{pos_data['total']})")

    elif stage == "block_pdlm_inference":
        print0("BLOCK_PDLM_INFERENCE EVALUATION RESULTS")
        print0("=" * 60)

        s1 = eval_result["stage1"]
        s2 = eval_result["stage2"]

        print0(f"\n[block_pdlm_inference] combined_loss: {eval_result['combined_loss']:.4f}, end2end_loss: {eval_result['end2end_loss']:.4f}")
        print0(f"  mtp_loss_weight: {eval_result['mtp_loss_weight']}, stage1_loss: {s1['overall_loss']:.4f}, stage2_loss: {s2['overall_loss']:.4f}")

        print0(f"\n  Stage 1 (Pure→Pure): loss={s1['overall_loss']:.4f}, ppl={s1['overall_ppl']:.2f}")
        for pos in range(block_size):
            pos_data = s1["positions"][pos]
            print0(f"    pos {pos}: loss={pos_data['loss']:.4f}, ppl={pos_data['ppl']:.2f}")

        print0(f"\n  Stage 2 (Mixed→Pure): loss={s2['overall_loss']:.4f}, ppl={s2['overall_ppl']:.2f}")
        pure_bk = s2.get("pure_breakdown", {})
        group_bk = s2.get("group_breakdown", {})
        for pos in range(block_size):
            pos_data = s2["positions"][pos]
            parts = [f"pos {pos}: loss={pos_data['loss']:.4f}"]
            if pos in pure_bk:
                parts.append(f"pure={pure_bk[pos]['loss']:.4f}")
            g_parts = []
            for g in range(pos + 1):
                if pos in group_bk and g in group_bk[pos]:
                    g_parts.append(f"g{g}={group_bk[pos][g]['loss']:.4f}")
            if g_parts:
                parts.append("group[" + " ".join(g_parts) + "]")
            print0(f"    {' | '.join(parts)}")

        # End-to-end per-position metrics
        if "end2end" in eval_result:
            e2e = eval_result["end2end"]
            print0(f"\n  End-to-End (Iterative): loss={e2e['overall_loss']:.4f}, ppl={e2e['overall_ppl']:.2f}")
            for pos in range(block_size):
                pos_data = e2e["positions"][pos]
                print0(f"    pos {pos}: loss={pos_data['loss']:.4f}, ppl={pos_data['ppl']:.2f}")

    else:
        if stage == "stage1_mask":
            print0("PDLM STAGE 1 MASK EVALUATION RESULTS")
        elif stage == "stage1_block":
            print0("PDLM STAGE 1 BLOCK EVALUATION RESULTS")
        else:
            print0("PDLM STAGE 2 EVALUATION RESULTS")
        print0("=" * 60)

        # Overall metrics
        print0(f"\nOverall metrics:")
        print0(f"  loss: {eval_result['overall_loss']:.4f}")
        print0(f"  ppl:  {eval_result['overall_ppl']:.2f}")
        if "overall_accuracy" in eval_result:
            print0(f"  accuracy: {eval_result['overall_accuracy']:.2%}")
        print0(f"  tokens evaluated: {eval_result['num_tokens_evaluated']:,}")

        # Per-position metrics
        print0(f"\nPer-position metrics (within block):")
        for pos in range(block_size):
            pos_data = eval_result["positions"][pos]
            if "accuracy" in pos_data:
                print0(f"  pos {pos}: loss={pos_data['loss']:.4f}, ppl={pos_data['ppl']:.2f}, accuracy={pos_data['accuracy']:.2%}, tokens={pos_data['tokens']:,}")
            else:
                print0(f"  pos {pos}: loss={pos_data['loss']:.4f}, ppl={pos_data['ppl']:.2f}, tokens={pos_data['tokens']:,}")

        # Compatibility metrics (if present, Stage 2 only)
        if "compatibility" in eval_result:
            compat = eval_result["compatibility"]
            print0(f"\nCompatibility metrics (self-consistency):")
            print0(f"  overall compatibility: {compat['overall_compatibility']:.2%}")
            print0(f"  Per-position compatibility:")
            for pos in range(block_size):
                pos_data = compat["positions"][pos]
                print0(f"    pos {pos}: {pos_data['compatibility']:.2%} ({pos_data['matched']}/{pos_data['total']})")

        # Oracle accuracy metrics (if present)
        if "oracle_accuracy" in eval_result:
            oracle = eval_result["oracle_accuracy"]
            print0(f"\nOracle accuracy metrics (given ground truth context):")
            print0(f"  overall accuracy: {oracle['overall_accuracy']:.2%}")
            print0(f"  Per-position accuracy:")
            for pos in range(block_size):
                pos_data = oracle["positions"][pos]
                print0(f"    pos {pos}: {pos_data['accuracy']:.2%} ({pos_data['matched']}/{pos_data['total']})")

    print0("\n" + "=" * 60)


def run_generation(
    model_tag=None,
    step=None,
    device_type="auto",
    ckpt_dir=None,
    num_prompts=5,
    prompt_tokens=64,
    generate_tokens=128,
    temperature=0.0,
    topk=0,
):
    """
    Generate text using mask_pdlm iterative denoising.

    Loads model, grabs validation data as prompt source, generates block-by-block.
    """
    from nanochat.tokenizer import get_tokenizer

    # Load model
    model, meta_data, device, autocast_ctx, model_config = load_pdlm_model(
        model_tag, step, device_type, ckpt_dir=ckpt_dir
    )

    model_config_dict = meta_data["model_config"]
    user_config = meta_data.get("user_config", {})
    max_seq_len = model_config_dict["sequence_len"]
    block_size = model_config_dict.get("bucket_size", user_config.get("block_size", 4))
    stage = model_config_dict.get("stage", "stage2")

    assert stage == "mask_pdlm", f"Generation requires mask_pdlm model, got stage={stage}"

    # Create validation dataloader to get prompt tokens
    device_batch_size = user_config.get("device_batch_size", 32)
    val_loader = get_data_loader(
        device_batch_size, max_seq_len, split="val", device=device,
        model_config=model_config, resume_state_dict=None,
    )

    # Grab first batch targets as pure token source
    _, targets, _, _ = next(val_loader)

    # Load tokenizer and get bos token for stop condition
    tokenizer = get_tokenizer()
    bos_token_id = tokenizer.get_bos_token_id()

    print0(f"\nGenerating with mask_pdlm (block_size={block_size}, temperature={temperature}, topk={topk})")
    print0(f"Prompt tokens: {prompt_tokens}, Generate tokens: {generate_tokens}")
    print0("=" * 60)

    for i in range(min(num_prompts, targets.size(0))):
        prompt = targets[i, :prompt_tokens].tolist()
        prompt_text = tokenizer.decode(prompt)

        with autocast_ctx:
            generated, debug_blocks = model.generate_mask_pdlm(
                tokens=prompt,
                max_new_tokens=generate_tokens,
                block_size=block_size,
                temperature=temperature,
                topk=topk,
                stop_token=bos_token_id,
            )

        # Decode generated part (may be shorter than generate_tokens if stopped at bos)
        full_ids = generated[0].tolist()
        actual_generated = len(full_ids) - prompt_tokens
        generated_text = tokenizer.decode(full_ids[prompt_tokens:])

        print0(f"\n--- Prompt {i+1} ({prompt_tokens} tokens) ---")
        print0(prompt_text)
        print0(f"--- Generated ({actual_generated} tokens, {len(debug_blocks)} blocks) ---")
        print0(generated_text)
        print0("")

    print0("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Standalone PDLM evaluation (Stage 1 MASK or Stage 2)")
    parser.add_argument("--model_tag", type=str, default=None, help="Model directory name (e.g., d8)")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="Direct path to checkpoint directory (overrides model_tag)")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step (default: last)")
    parser.add_argument("--num_batches", type=int, default=20, help="Number of validation batches")
    parser.add_argument("--device", type=str, default="auto", help="Device type (cuda/cpu/mps/auto)")
    parser.add_argument("--output_json", type=str, default=None, help="Optional: save results to JSON file")
    parser.add_argument("--run_compatibility", action="store_true", help="Run compatibility evaluation (also enables oracle accuracy unless --no_oracle_accuracy)")
    parser.add_argument("--compatibility_batches", type=int, default=None, help="Number of batches for compatibility (default: num_batches // 4)")
    parser.add_argument("--run_oracle_accuracy", action="store_true", help="Run oracle accuracy evaluation (given ground truth context)")
    parser.add_argument("--no_oracle_accuracy", action="store_true", help="Disable oracle accuracy even when running compatibility")
    parser.add_argument("--oracle_accuracy_batches", type=int, default=None, help="Number of batches for oracle accuracy (default: num_batches // 4)")
    parser.add_argument("--generate", action="store_true", help="Run generation instead of loss evaluation (mask_pdlm only)")
    parser.add_argument("--num_prompts", type=int, default=5, help="Number of prompts for generation (default: 5)")
    parser.add_argument("--prompt_tokens", type=int, default=16, help="Number of prompt tokens (default: 16)")
    parser.add_argument("--generate_tokens", type=int, default=128, help="Number of tokens to generate (default: 128)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for generation (default: 0.0 = greedy)")
    parser.add_argument("--topk", type=int, default=0, help="Top-k sampling for generation (default: 0 = disabled)")
    parser.add_argument("--dump_batch", type=str, default=None, help="Dump one batch to file for debugging (path to output txt)")
    parser.add_argument("--dump_stage1_block", type=str, default=None, help="Dump stage1_block predictions to file (path to output txt)")
    parser.add_argument("--dump_sequences", type=int, default=5, help="Number of sequences to dump (default: 5)")
    args = parser.parse_args()

    # Handle generation mode (mask_pdlm only)
    if args.generate:
        run_generation(
            model_tag=args.model_tag,
            step=args.step,
            device_type=args.device,
            ckpt_dir=args.ckpt_dir,
            num_prompts=args.num_prompts,
            prompt_tokens=args.prompt_tokens,
            generate_tokens=args.generate_tokens,
            temperature=args.temperature,
            topk=args.topk,
        )
        return

    # Handle dump_batch mode (separate from normal eval)
    if args.dump_batch:
        model, meta_data, device, autocast_ctx, model_config = load_pdlm_model(
            args.model_tag, args.step, args.device, ckpt_dir=args.ckpt_dir
        )
        model_config_dict = meta_data["model_config"]
        user_config = meta_data.get("user_config", {})
        max_seq_len = model_config_dict["sequence_len"]
        block_size = model_config_dict.get("bucket_size", user_config.get("block_size", 4))
        is_causal = model_config_dict.get("is_causal", True)
        device_batch_size = user_config.get("device_batch_size", 32)

        val_loader = get_data_loader(
            device_batch_size, max_seq_len, split="val", device=device,
            model_config=model_config, resume_state_dict=None,
        )
        attn_mask = gen_mask(
            max_seq_len, block_size, attn_backend="sdpa",
            is_causal=is_causal, prefix_sliding_tokens=0
        ).to(device=device)

        dump_batch_to_file(
            model=model, val_loader=val_loader, block_size=block_size,
            attn_mask=attn_mask, device=device, autocast_ctx=autocast_ctx,
            output_path=args.dump_batch,
        )
        return

    # Handle dump_stage1_block mode (separate from normal eval)
    if args.dump_stage1_block:
        model, meta_data, device, autocast_ctx, model_config = load_pdlm_model(
            args.model_tag, args.step, args.device, ckpt_dir=args.ckpt_dir
        )
        model_config_dict = meta_data["model_config"]
        user_config = meta_data.get("user_config", {})
        max_seq_len = model_config_dict["sequence_len"]
        block_size = model_config_dict.get("bucket_size", user_config.get("block_size", 4))
        is_causal = model_config_dict.get("is_causal", True)
        device_batch_size = user_config.get("device_batch_size", 32)
        stage = model_config_dict.get("stage", "stage2")

        if stage != "stage1_block":
            print0(f"Warning: Model stage is {stage}, not stage1_block. Dump may not work correctly.")

        val_loader = get_data_loader(
            device_batch_size, max_seq_len, split="val", device=device,
            model_config=model_config, resume_state_dict=None,
        )
        # stage1_block uses L×L block-causal mask
        attn_mask = gen_block_causal_mask(
            max_seq_len, block_size, attn_backend="sdpa", is_causal=is_causal
        ).to(device=device)

        dump_stage1_block_batch(
            model=model, val_loader=val_loader, block_size=block_size,
            attn_mask=attn_mask, device=device, autocast_ctx=autocast_ctx,
            output_path=args.dump_stage1_block,
            num_sequences=args.dump_sequences,
        )
        return

    # Determine whether to run oracle accuracy:
    # - Enabled if --run_oracle_accuracy is passed
    # - Also enabled if --run_compatibility is passed (unless --no_oracle_accuracy)
    run_oracle = args.run_oracle_accuracy
    if args.run_compatibility and not args.no_oracle_accuracy:
        run_oracle = True

    # Run evaluation
    eval_result = run_eval(
        model_tag=args.model_tag,
        step=args.step,
        num_batches=args.num_batches,
        device_type=args.device,
        ckpt_dir=args.ckpt_dir,
        run_compatibility=args.run_compatibility,
        compatibility_batches=args.compatibility_batches,
        run_oracle_accuracy=run_oracle,
        oracle_accuracy_batches=args.oracle_accuracy_batches,
    )

    # Get block_size for printing
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

    # Print results
    print_results(eval_result, block_size)

    # Optionally save to JSON
    if args.output_json:
        # Backup existing file if it exists
        if os.path.exists(args.output_json):
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
            backup_path = args.output_json.replace(".json", f"_backup_{timestamp}.json")
            os.rename(args.output_json, backup_path)
            print0(f"Backed up existing results to {backup_path}")

        with open(args.output_json, "w") as f:
            json.dump(eval_result, f, indent=2)
        print0(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
