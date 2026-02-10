"""
Mask PDLM evaluation.

Unified loss (no stage1/stage2 split) + end-to-end iterative inference evaluation.
"""

import torch
import torch.nn.functional as F

from nanochat.pdlm_eval.common import model_eval_context, build_result_dict


def eval_mask_pdlm(
    model,
    val_loader,
    block_size,
    num_batches,
    attn_mask,
    device,
    autocast_ctx,
    prefix_pure_tokens=0,
):
    """
    Evaluate mask_pdlm model.

    Returns a dict with unified loss, end-to-end loss, and per-position breakdowns.
    """
    with model_eval_context(model):
        with torch.no_grad():
            # Cache batches
            cached_batches = []
            for _ in range(num_batches):
                batch = next(val_loader)
                cached_batches.append(batch)

            unified_result = _eval_unified(
                model, cached_batches, block_size, attn_mask, device, autocast_ctx,
            )
            end2end_result = _eval_end2end(
                model, cached_batches, block_size, attn_mask, device, autocast_ctx,
            )

    return {
        "stage": "mask_pdlm",
        "unified_loss": unified_result["overall_loss"],
        "end2end_loss": end2end_result["overall_loss"],
        "unified": unified_result,
        "end2end": end2end_result,
    }


def _eval_unified(model, cached_batches, block_size, attn_mask, device, autocast_ctx):
    """
    Evaluate unified loss: CE on non-pure xt positions (mask + group).
    Uses forward_for_eval_mask_pdlm() which returns xt-half logits.
    Per-position breakdown within blocks.
    """
    nll_by_pos = {p: {"nll": 0.0, "tokens": 0} for p in range(block_size)}

    for inputs, targets, loss_extras, _ in cached_batches:
        B, T = inputs.shape
        loss_mask = loss_extras["loss_mask"]  # (B, T)

        with autocast_ctx:
            logits = model.forward_for_eval_mask_pdlm(
                inputs, targets, attn_mask=attn_mask)
            # logits: (B, T, pure_vocab_size)

            log_probs = F.log_softmax(logits.float(), dim=-1)
            target_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

            num_blocks = T // block_size

            for pos in range(block_size):
                for block_idx in range(1, num_blocks):
                    pos_in_seq = block_idx * block_size + pos

                    mask_at_pos = loss_mask[:, pos_in_seq]
                    nll = -target_log_probs[:, pos_in_seq]
                    nll_masked = nll * mask_at_pos.float()
                    nll_by_pos[pos]["nll"] += nll_masked.sum().item()
                    nll_by_pos[pos]["tokens"] += mask_at_pos.sum().item()

    return build_result_dict(nll_by_pos, block_size, include_accuracy=False)


def _eval_end2end(model, cached_batches, block_size, attn_mask, device, autocast_ctx):
    """
    End-to-end iterative inference evaluation for mask_pdlm.

    For each block:
    1. Build xt with all mask tokens → forward → collapse to groups
    2. Build xt with all groups → forward → sample p0, refine groups
    3-5. Same pattern (iterative denoising)

    Evaluate CE at blocks 2..N-1 against ground truth pure tokens.
    """
    nll_by_pos = {p: {"nll": 0.0, "tokens": 0} for p in range(block_size)}
    pure_vocab_size = model.config.pure_vocab_size
    num_groups = model.config.num_groups
    mask_token_id = pure_vocab_size + num_groups
    group_offset = pure_vocab_size

    for inputs, targets, loss_extras, _ in cached_batches:
        B, T = inputs.shape
        num_blocks = T // block_size

        with autocast_ctx:
            # For each target block (2..N-1), simulate iterative inference
            for block_idx in range(2, num_blocks):
                blk_start = block_idx * block_size
                blk_end = blk_start + block_size
                ground_truth = targets[:, blk_start:blk_end]  # (B, K)

                # Build full input with mask tokens at target block
                # Use ground truth pure tokens for all preceding blocks (teacher forcing)
                eval_inputs = targets.clone()  # all pure tokens

                # Step 1: Set target block positions to mask tokens
                eval_inputs[:, blk_start:blk_end] = mask_token_id

                # Forward to get logits
                logits = model.forward_for_eval_mask_pdlm(
                    eval_inputs, targets, attn_mask=attn_mask)
                block_logits = logits[:, blk_start:blk_end, :]  # (B, K, pure_vocab_size)

                # Collapse to groups
                block_groups = model.collapse_pure_to_group(block_logits)  # (B, K)

                # Step 2: Replace mask tokens with group tokens, forward again
                eval_inputs[:, blk_start:blk_end] = block_groups + group_offset
                logits = model.forward_for_eval_mask_pdlm(
                    eval_inputs, targets, attn_mask=attn_mask)
                block_logits = logits[:, blk_start:blk_end, :]

                # Sample position 0
                block_pure = block_logits[:, 0, :].argmax(dim=-1)  # (B,)
                eval_inputs[:, blk_start] = block_pure

                # Update remaining groups
                for rp in range(1, block_size):
                    rp_logits = block_logits[:, rp:rp+1, :]
                    block_groups[:, rp] = model.collapse_pure_to_group(rp_logits).squeeze(1)
                    eval_inputs[:, blk_start + rp] = block_groups[:, rp] + group_offset

                # Steps 3-5: iterative denoising for remaining positions
                for denoise_step in range(1, block_size):
                    logits = model.forward_for_eval_mask_pdlm(
                        eval_inputs, targets, attn_mask=attn_mask)
                    block_logits = logits[:, blk_start:blk_end, :]

                    pos = denoise_step
                    if pos < block_size:
                        sampled = block_logits[:, pos, :].argmax(dim=-1)
                        eval_inputs[:, blk_start + pos] = sampled

                        # Update remaining groups
                        for rp in range(pos + 1, block_size):
                            rp_logits = block_logits[:, rp:rp+1, :]
                            new_group = model.collapse_pure_to_group(rp_logits).squeeze(1)
                            eval_inputs[:, blk_start + rp] = new_group + group_offset

                # Final logits after all denoising
                logits = model.forward_for_eval_mask_pdlm(
                    eval_inputs, targets, attn_mask=attn_mask)
                final_logits = logits[:, blk_start:blk_end, :]

                # Compute CE against ground truth
                log_probs = F.log_softmax(final_logits.float(), dim=-1)
                target_log_probs = log_probs.gather(-1, ground_truth.unsqueeze(-1)).squeeze(-1)

                for pos in range(block_size):
                    nll = -target_log_probs[:, pos]
                    nll_by_pos[pos]["nll"] += nll.sum().item()
                    nll_by_pos[pos]["tokens"] += B

    return build_result_dict(nll_by_pos, block_size, include_accuracy=False)
