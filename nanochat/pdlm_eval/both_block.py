"""
Both Block evaluation for PDLM.

Combined Stage 1 (block->block) + Stage 2 (denoise) evaluation.
"""

import torch
import torch.nn.functional as F

from nanochat.pdlm_eval.common import model_eval_context, build_result_dict
from nanochat.pdlm_eval.stage2 import eval_pdlm_compatibility, eval_pdlm_oracle_accuracy


def eval_pdlm_both_block(
    model,
    val_loader,
    block_size,
    num_batches,
    attn_mask,
    device,
    autocast_ctx,
    prefix_pure_tokens=0,
    mtp_loss_weight=1.0,
    run_compatibility=False,
    compatibility_batches=None,
    run_oracle_accuracy=False,
    oracle_accuracy_batches=None,
):
    """
    Evaluate PDLM both_block model: combined Stage 1 (block->block) + Stage 2 (denoise).

    Returns a dict with three sections: Overall, Stage 1, Stage 2, plus optional
    compatibility and oracle accuracy.

    Args:
        model: PDLM model (both_block stage)
        val_loader: validation data loader
        block_size: block size for PDLM
        num_batches: number of batches to evaluate
        attn_mask: 2Lx2L attention mask
        device: device to run on
        autocast_ctx: autocast context for mixed precision
        prefix_pure_tokens: number of pure prefix tokens
        mtp_loss_weight: weight for stage1 loss in combined loss
        run_compatibility: whether to run compatibility eval (Stage 2)
        compatibility_batches: batches for compatibility (default: num_batches // 4)
        run_oracle_accuracy: whether to run oracle accuracy eval
        oracle_accuracy_batches: batches for oracle accuracy (default: num_batches // 4)

    Returns:
        dict with structure:
        {
            "stage": "both_block",
            "combined_loss": float,
            "end2end_loss": float,
            "mtp_loss_weight": float,
            "stage1": {"overall_loss", "overall_ppl", "overall_accuracy", "num_tokens_evaluated", "positions": {...}},
            "stage2": {"overall_loss", "overall_ppl", "num_tokens_evaluated", "positions": {...}, ...},
        }
    """
    with model_eval_context(model):
        with torch.no_grad():
            # Cache batches from val_loader
            cached_batches = []
            for _ in range(num_batches):
                batch = next(val_loader)
                cached_batches.append(batch)

            # Single-pass eval: get both stage1 and stage2 metrics
            stage1_result = _eval_both_block_stage1(
                model, cached_batches, block_size, attn_mask, device, autocast_ctx,
            )
            stage2_result = _eval_both_block_stage2(
                model, cached_batches, block_size, attn_mask, device, autocast_ctx,
            )

            # End-to-end eval (two-step inference)
            end2end_loss = _eval_both_block_end2end(
                model, cached_batches, block_size, attn_mask, device, autocast_ctx,
            )

            # Optional: compatibility and oracle accuracy (reuse existing functions)
            if run_compatibility:
                if compatibility_batches is None:
                    compatibility_batches = max(1, num_batches // 4)
                compat_batches = cached_batches[:compatibility_batches]
                compat_result = eval_pdlm_compatibility(
                    model, None, block_size, len(compat_batches),
                    attn_mask, device, autocast_ctx,
                    cached_batches=compat_batches,
                )
                stage2_result["compatibility"] = compat_result

            if run_oracle_accuracy:
                if oracle_accuracy_batches is None:
                    oracle_accuracy_batches = max(1, num_batches // 4)
                oracle_batches = cached_batches[:oracle_accuracy_batches]
                oracle_result = eval_pdlm_oracle_accuracy(
                    model, None, block_size, len(oracle_batches),
                    attn_mask, device, autocast_ctx,
                    cached_batches=oracle_batches,
                )
                stage2_result["oracle_accuracy"] = oracle_result

    # Combine
    stage1_loss = stage1_result["overall_loss"]
    stage2_loss = stage2_result["overall_loss"]
    combined_loss = mtp_loss_weight * stage1_loss + stage2_loss

    return {
        "stage": "both_block",
        "combined_loss": combined_loss,
        "end2end_loss": end2end_loss,
        "mtp_loss_weight": mtp_loss_weight,
        "stage1": stage1_result,
        "stage2": stage2_result,
    }


def _eval_both_block_stage1(
    model, cached_batches, block_size, attn_mask, device, autocast_ctx,
):
    """
    Evaluate Stage 1 (block->block group prediction) from both_block model.
    Uses forward_for_eval_both_block to get stage1_logits from x0 half.
    Skip block 0 and last block for loss computation.
    """
    metrics_by_pos = {p: {"nll": 0.0, "correct": 0, "tokens": 0} for p in range(block_size)}

    for inputs, targets, loss_extras, _ in cached_batches:
        B, T = inputs.shape
        block_targets = loss_extras.get("block_targets", None)
        block_loss_mask = loss_extras.get("block_loss_mask", None)

        if block_targets is None:
            raise ValueError("both_block eval requires block_targets in loss_extras")

        overlap_k = block_targets.size(-1)

        with autocast_ctx:
            _, stage1_logits = model.forward_for_eval_both_block(inputs, targets, attn_mask=attn_mask)
            # stage1_logits: (B, T, num_groups) - group logits from x0 half

            log_probs = F.log_softmax(stage1_logits.float(), dim=-1)
            preds = stage1_logits.argmax(dim=-1)  # (B, T)

            num_blocks = T // block_size

            for pos in range(block_size):
                # Skip block 0 and last block
                for block_idx in range(1, num_blocks - 1):
                    pos_in_seq = block_idx * block_size + pos

                    if block_loss_mask is not None:
                        mask_at_pos = block_loss_mask[:, pos_in_seq]
                    else:
                        mask_at_pos = torch.ones(B, dtype=torch.bool, device=device)

                    valid_groups = block_targets[:, pos_in_seq, :]  # (B, overlap_k)

                    # Compute any-correct NLL
                    valid_mask = valid_groups >= 0
                    safe_targets = valid_groups.clamp(min=0)
                    valid_log_probs = torch.gather(
                        log_probs[:, pos_in_seq, :], dim=-1, index=safe_targets
                    )
                    valid_log_probs = valid_log_probs.masked_fill(~valid_mask, float('-inf'))
                    log_valid_prob = torch.logsumexp(valid_log_probs, dim=-1)
                    nll = -log_valid_prob

                    nll_masked = nll * mask_at_pos.float()
                    metrics_by_pos[pos]["nll"] += nll_masked.sum().item()
                    metrics_by_pos[pos]["tokens"] += mask_at_pos.sum().item()

                    # Any-correct accuracy
                    pred_at_pos = preds[:, pos_in_seq]
                    pred_expanded = pred_at_pos.unsqueeze(-1)
                    any_correct = ((valid_groups == pred_expanded) & valid_mask).any(dim=-1)
                    correct_masked = any_correct & mask_at_pos
                    metrics_by_pos[pos]["correct"] += correct_masked.sum().item()

    return build_result_dict(metrics_by_pos, block_size, include_accuracy=True)


def _eval_both_block_stage2(
    model, cached_batches, block_size, attn_mask, device, autocast_ctx,
):
    """
    Evaluate Stage 2 (group->pure denoising) from both_block model.
    Uses forward_for_eval_both_block to get stage2_logits from xt half.
    Skip block 0 for loss computation.
    """
    nll_by_pos = {p: {"nll": 0.0, "tokens": 0} for p in range(block_size)}

    for inputs, targets, loss_extras, _ in cached_batches:
        B, T = inputs.shape
        loss_mask = loss_extras.get("loss_mask", None)

        with autocast_ctx:
            stage2_logits, _ = model.forward_for_eval_both_block(inputs, targets, attn_mask=attn_mask)
            # stage2_logits: (B, T, pure_vocab_size)

            log_probs = F.log_softmax(stage2_logits.float(), dim=-1)
            target_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

            num_blocks = T // block_size

            for pos in range(block_size):
                for block_idx in range(1, num_blocks):
                    pos_in_seq = block_idx * block_size + pos

                    if loss_mask is not None:
                        mask_at_pos = loss_mask[:, pos_in_seq]
                        nll = -target_log_probs[:, pos_in_seq]
                        nll_masked = nll * mask_at_pos.float()
                        nll_by_pos[pos]["nll"] += nll_masked.sum().item()
                        nll_by_pos[pos]["tokens"] += mask_at_pos.sum().item()
                    else:
                        nll = -target_log_probs[:, pos_in_seq]
                        nll_by_pos[pos]["nll"] += nll.sum().item()
                        nll_by_pos[pos]["tokens"] += B

    return build_result_dict(nll_by_pos, block_size, include_accuracy=False)


def _eval_both_block_end2end(
    model, cached_batches, block_size, attn_mask, device, autocast_ctx,
):
    """
    End-to-end two-step inference evaluation for both_block.

    1. Forward pass 1: get stage1_logits -> argmax predicted groups
    2. Build new_inputs with predicted group tokens at block positions
    3. Forward pass 2: forward_for_eval with new_inputs -> stage2 logits
    4. Compute CE loss at blocks 2..N-1 against targets

    Why blocks 2..N-1: Block 0 has no Stage 1 context. Block 1's groups come from
    block 0 (unreliable). Valid end-to-end blocks are 2 through N-1.
    """
    total_nll = 0.0
    total_tokens = 0

    for inputs, targets, loss_extras, _ in cached_batches:
        B, T = inputs.shape
        loss_mask = loss_extras.get("loss_mask", None)
        pure_vocab_size = model.config.pure_vocab_size

        with autocast_ctx:
            # Step 1: Get stage1 logits (group predictions from x0 half)
            _, stage1_logits = model.forward_for_eval_both_block(inputs, targets, attn_mask=attn_mask)
            predicted_groups = stage1_logits.argmax(dim=-1)  # (B, T)

            # Step 2: Build new inputs with predicted group tokens
            # For blocks 2..N-1, replace group tokens with predictions from previous block
            num_blocks = T // block_size
            new_inputs = inputs.clone()

            for block_idx in range(2, num_blocks):
                for pos in range(block_size):
                    src_pos = (block_idx - 1) * block_size + pos  # source: previous block's prediction
                    dst_pos = block_idx * block_size + pos
                    new_inputs[:, dst_pos] = pure_vocab_size + predicted_groups[:, src_pos]

            # Step 3: Forward pass 2 with predicted groups
            # forward_for_eval for both_block returns (B, T, pure_vocab_size) after slice fix
            logits = model.forward_for_eval(new_inputs, targets, attn_mask=attn_mask)
            # logits: (B, T, pure_vocab_size)

            log_probs = F.log_softmax(logits.float(), dim=-1)
            target_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

            # Step 4: Compute loss at blocks 2..N-1
            for block_idx in range(2, num_blocks):
                for pos in range(block_size):
                    pos_in_seq = block_idx * block_size + pos

                    if loss_mask is not None:
                        mask_at_pos = loss_mask[:, pos_in_seq]
                        nll = -target_log_probs[:, pos_in_seq]
                        total_nll += (nll * mask_at_pos.float()).sum().item()
                        total_tokens += mask_at_pos.sum().item()
                    else:
                        nll = -target_log_probs[:, pos_in_seq]
                        total_nll += nll.sum().item()
                        total_tokens += B

    return total_nll / total_tokens if total_tokens > 0 else 0.0
