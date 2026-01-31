"""
Stage 1 MASK evaluation for PDLM.

Stage 1 MASK (Pure/MASK -> Group):
- Input: [prefix, MASK, MASK, ...] at xt positions
- Output: predict group tokens for each MASK position
- Loss: any-correct over overlap_k valid groups
- Metrics: loss, PPL, accuracy per position
"""

import torch
import torch.nn.functional as F

from nanochat.pdlm_eval.common import model_eval_context, build_result_dict


def eval_pdlm_stage1_mask(
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
    Evaluate PDLM Stage 1 MASK model on validation set.

    Stage 1 MASK: [prefix, MASK, MASK, ...] -> group tokens
    Uses any-correct loss where predicting ANY valid group is correct.

    Args:
        model: PDLM model (stage1_mask)
        val_loader: validation data loader (yields inputs, targets, loss_extras, state_dict)
                    inputs: [prefix, MASK, MASK, ...]
                    targets: pure tokens everywhere
                    loss_extras: {"loss_mask", "group_targets"}
        block_size: block size for PDLM
        num_batches: number of batches to evaluate
        attn_mask: attention mask for the model
        device: device to run on
        autocast_ctx: autocast context for mixed precision
        prefix_pure_tokens: number of pure prefix tokens (no loss on these)

    Returns:
        dict with evaluation results:
        {
            "overall_loss": float,
            "overall_ppl": float,
            "overall_accuracy": float,  # any-correct accuracy
            "positions": {
                0: {"loss": X, "ppl": Y, "accuracy": Z, "tokens": N},
                1: {"loss": X, "ppl": Y, "accuracy": Z, "tokens": N},
                ...
            },
            "num_tokens_evaluated": int,
        }
    """
    with model_eval_context(model):
        with torch.no_grad():
            result = _eval_stage1_mask_per_position(
                model, val_loader, block_size, num_batches,
                attn_mask, device, autocast_ctx, prefix_pure_tokens
            )
    return result


def _eval_stage1_mask_per_position(
    model, val_loader, block_size, num_batches,
    attn_mask, device, autocast_ctx, prefix_pure_tokens,
):
    """
    Evaluate Stage 1 MASK loss and accuracy broken down by position within block.
    Skip block 0 (no context), compute from blocks 1 onwards.
    """
    # Accumulators: metrics_by_pos[pos] = {nll, correct, tokens}
    metrics_by_pos = {p: {"nll": 0.0, "correct": 0, "tokens": 0} for p in range(block_size)}

    for batch_idx in range(num_batches):
        inputs, targets, loss_extras, state = next(val_loader)
        # inputs: (B, T) - [prefix, MASK, MASK, ...]
        # targets: (B, T) - pure tokens everywhere
        # loss_extras: {"loss_mask": (B, T), "group_targets": (B, T, overlap_k)}

        B, T = inputs.shape
        loss_mask = loss_extras.get("loss_mask", None)
        group_targets = loss_extras.get("group_targets", None)

        if group_targets is None:
            raise ValueError("Stage 1 MASK eval requires group_targets in loss_extras")

        overlap_k = group_targets.size(-1)

        with autocast_ctx:
            # Forward pass: model concatenates [inputs | targets] internally
            # Returns logits for first T positions (xt half)
            logits = model.forward_for_eval(inputs, targets, attn_mask=attn_mask)
            # logits: (B, T, num_groups)

            # Compute log probabilities
            log_probs = F.log_softmax(logits.float(), dim=-1)

            # Get predictions
            preds = logits.argmax(dim=-1)  # (B, T)

            # Compute metrics for each position within blocks
            num_blocks = T // block_size

            for pos in range(block_size):
                # For each block (starting from block 1 to skip block 0)
                for block_idx in range(1, num_blocks):
                    pos_in_seq = block_idx * block_size + pos

                    # Check if this position should have loss (within loss_mask)
                    if loss_mask is not None:
                        mask_at_pos = loss_mask[:, pos_in_seq]  # (B,)
                    else:
                        mask_at_pos = torch.ones(B, dtype=torch.bool, device=device)

                    # Get valid group targets for this position
                    valid_groups = group_targets[:, pos_in_seq, :]  # (B, overlap_k)

                    # Compute any-correct NLL
                    # Gather log probs for valid targets
                    valid_mask = valid_groups >= 0  # (B, overlap_k)
                    safe_targets = valid_groups.clamp(min=0)
                    valid_log_probs = torch.gather(
                        log_probs[:, pos_in_seq, :], dim=-1, index=safe_targets
                    )  # (B, overlap_k)
                    valid_log_probs = valid_log_probs.masked_fill(~valid_mask, float('-inf'))
                    log_valid_prob = torch.logsumexp(valid_log_probs, dim=-1)  # (B,)
                    nll = -log_valid_prob

                    # Apply mask and accumulate
                    nll_masked = nll * mask_at_pos.float()
                    metrics_by_pos[pos]["nll"] += nll_masked.sum().item()
                    metrics_by_pos[pos]["tokens"] += mask_at_pos.sum().item()

                    # Compute any-correct accuracy
                    # Prediction is correct if it matches ANY valid group
                    pred_at_pos = preds[:, pos_in_seq]  # (B,)
                    # Check if pred matches any valid target
                    pred_expanded = pred_at_pos.unsqueeze(-1)  # (B, 1)
                    any_correct = ((valid_groups == pred_expanded) & valid_mask).any(dim=-1)  # (B,)
                    correct_masked = any_correct & mask_at_pos
                    metrics_by_pos[pos]["correct"] += correct_masked.sum().item()

    return build_result_dict(metrics_by_pos, block_size, include_accuracy=True)
