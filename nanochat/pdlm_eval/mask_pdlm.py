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

    Processes ALL target blocks (2..N-1) simultaneously:
    1. Mask all target blocks → forward → collapse to groups
    2. Iterative denoising (block_size steps): sample one position per step, collapse rest
    3. Final forward → compute CE

    Total forward passes per batch: block_size + 2 (instead of per-block sequential).
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
            # Pre-build position masks for each within-block position across target blocks
            # pos_masks[p]: (B, T) bool — True at position p of each target block (2..N-1)
            pos_masks = []
            for p in range(block_size):
                mask = torch.zeros(B, T, dtype=torch.bool, device=device)
                for block_idx in range(2, num_blocks):
                    mask[:, block_idx * block_size + p] = True
                pos_masks.append(mask)
            # target_mask: union of all positions in target blocks
            target_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
            for p in range(block_size):
                target_mask |= pos_masks[p]

            # Teacher forcing: start with ground truth pure tokens, mask target blocks
            eval_inputs = targets.clone()
            eval_inputs[target_mask] = mask_token_id

            # Step 1: Forward with mask tokens → collapse to groups
            logits = model.forward_for_eval_mask_pdlm(
                eval_inputs, targets, attn_mask=attn_mask)
            all_groups = model.collapse_pure_to_group(logits)  # (B, T)
            eval_inputs[target_mask] = all_groups[target_mask] + group_offset

            # Steps 2..block_size+1: Iterative denoising (all blocks simultaneously)
            for denoise_step in range(block_size):
                logits = model.forward_for_eval_mask_pdlm(
                    eval_inputs, targets, attn_mask=attn_mask)

                # Sample pure token at denoise_step position of all target blocks
                sampled = logits.argmax(dim=-1)  # (B, T)
                eval_inputs[pos_masks[denoise_step]] = sampled[pos_masks[denoise_step]]

                # Collapse remaining positions back to groups
                if denoise_step < block_size - 1:
                    all_groups = model.collapse_pure_to_group(logits)
                    for rp in range(denoise_step + 1, block_size):
                        eval_inputs[pos_masks[rp]] = all_groups[pos_masks[rp]] + group_offset

            # Final forward for CE computation
            logits = model.forward_for_eval_mask_pdlm(
                eval_inputs, targets, attn_mask=attn_mask)

            # Compute CE at all target blocks against ground truth
            log_probs = F.log_softmax(logits.float(), dim=-1)
            target_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

            for pos in range(block_size):
                nll = -target_log_probs[pos_masks[pos]]
                nll_by_pos[pos]["nll"] += nll.sum().item()
                nll_by_pos[pos]["tokens"] += pos_masks[pos].sum().item()

    return build_result_dict(nll_by_pos, block_size, include_accuracy=False)
