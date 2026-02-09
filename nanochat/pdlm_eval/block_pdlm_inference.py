"""
Block PDLM Inference evaluation.

Combined Stage 1 (pure→pure block prediction) + Stage 2 (mixed→pure denoising) evaluation.
Includes end-to-end iterative inference evaluation.
"""

import torch
import torch.nn.functional as F

from nanochat.pdlm_eval.common import model_eval_context, build_result_dict


def eval_block_pdlm_inference(
    model,
    val_loader,
    block_size,
    num_batches,
    attn_mask,
    device,
    autocast_ctx,
    prefix_pure_tokens=0,
    mtp_loss_weight=1.0,
):
    """
    Evaluate block_pdlm_inference model.

    Returns a dict with Stage 1, Stage 2, End-to-End metrics, and combined loss.
    """
    with model_eval_context(model):
        with torch.no_grad():
            # Cache batches
            cached_batches = []
            for _ in range(num_batches):
                batch = next(val_loader)
                cached_batches.append(batch)

            stage1_result = _eval_stage1(
                model, cached_batches, block_size, attn_mask, device, autocast_ctx,
            )
            stage2_result = _eval_stage2(
                model, cached_batches, block_size, attn_mask, device, autocast_ctx,
            )
            end2end_result = _eval_end2end(
                model, cached_batches, block_size, attn_mask, device, autocast_ctx,
            )

    stage1_loss = stage1_result["overall_loss"]
    stage2_loss = stage2_result["overall_loss"]
    combined_loss = mtp_loss_weight * stage1_loss + stage2_loss

    return {
        "stage": "block_pdlm_inference",
        "combined_loss": combined_loss,
        "end2end_loss": end2end_result["overall_loss"],
        "mtp_loss_weight": mtp_loss_weight,
        "stage1": stage1_result,
        "stage2": stage2_result,
        "end2end": end2end_result,
    }


def _eval_stage1(model, cached_batches, block_size, attn_mask, device, autocast_ctx):
    """
    Evaluate Stage 1 (pure→pure prediction) from block_pdlm_inference model.
    Uses x0 half logits, targets are next-block pure tokens (shifted by block_size).
    Skip block 0.
    """
    nll_by_pos = {p: {"nll": 0.0, "tokens": 0} for p in range(block_size)}

    for inputs, targets, loss_extras, _ in cached_batches:
        B, T = inputs.shape
        stage1_targets = loss_extras["stage1_targets"]  # (B, T)
        stage1_loss_mask = loss_extras["stage1_loss_mask"]  # (B, T)

        with autocast_ctx:
            _, stage1_logits = model.forward_for_eval_block_pdlm_inference(
                inputs, targets, attn_mask=attn_mask)
            # stage1_logits: (B, T, pure_vocab_size)

            log_probs = F.log_softmax(stage1_logits.float(), dim=-1)
            target_log_probs = log_probs.gather(-1, stage1_targets.unsqueeze(-1)).squeeze(-1)

            num_blocks = T // block_size

            for pos in range(block_size):
                for block_idx in range(1, num_blocks):
                    pos_in_seq = block_idx * block_size + pos

                    mask_at_pos = stage1_loss_mask[:, pos_in_seq]
                    nll = -target_log_probs[:, pos_in_seq]
                    nll_masked = nll * mask_at_pos.float()
                    nll_by_pos[pos]["nll"] += nll_masked.sum().item()
                    nll_by_pos[pos]["tokens"] += mask_at_pos.sum().item()

    return build_result_dict(nll_by_pos, block_size, include_accuracy=False)


def _eval_stage2(model, cached_batches, block_size, attn_mask, device, autocast_ctx):
    """
    Evaluate Stage 2 (mixed→pure denoising) from block_pdlm_inference model.
    Uses xt half logits against stage2_targets.
    Only evaluate on group-input positions (r=0 case for clean comparison).
    """
    nll_by_pos = {p: {"nll": 0.0, "tokens": 0} for p in range(block_size)}

    for inputs, targets, loss_extras, _ in cached_batches:
        B, T = inputs.shape
        stage2_targets = loss_extras["stage2_targets"]  # (B, T)
        stage2_loss_mask = loss_extras["stage2_loss_mask"]  # (B, T)

        with autocast_ctx:
            stage2_logits, _ = model.forward_for_eval_block_pdlm_inference(
                inputs, targets, attn_mask=attn_mask)
            # stage2_logits: (B, T, pure_vocab_size)

            log_probs = F.log_softmax(stage2_logits.float(), dim=-1)
            target_log_probs = log_probs.gather(-1, stage2_targets.unsqueeze(-1)).squeeze(-1)

            num_blocks = T // block_size

            for pos in range(block_size):
                for block_idx in range(1, num_blocks):
                    pos_in_seq = block_idx * block_size + pos

                    mask_at_pos = stage2_loss_mask[:, pos_in_seq]
                    nll = -target_log_probs[:, pos_in_seq]
                    nll_masked = nll * mask_at_pos.float()
                    nll_by_pos[pos]["nll"] += nll_masked.sum().item()
                    nll_by_pos[pos]["tokens"] += mask_at_pos.sum().item()

    return build_result_dict(nll_by_pos, block_size, include_accuracy=False)


def _eval_end2end(model, cached_batches, block_size, attn_mask, device, autocast_ctx):
    """
    End-to-end iterative inference evaluation for block_pdlm_inference.

    For each block:
    1. Forward pure context → get stage1 logits → collapse to groups
    2. Build xt with all group tokens → forward 2L → denoise all at once

    Evaluate CE against ground truth pure tokens at blocks 2..N-1.
    """
    nll_by_pos = {p: {"nll": 0.0, "tokens": 0} for p in range(block_size)}
    pure_vocab_size = model.config.pure_vocab_size

    for inputs, targets, loss_extras, _ in cached_batches:
        B, T = inputs.shape

        with autocast_ctx:
            # Step 1: Get stage1 logits (pure→pure predictions from x0 half)
            _, stage1_logits = model.forward_for_eval_block_pdlm_inference(
                inputs, targets, attn_mask=attn_mask)
            # stage1_logits: (B, T, pure_vocab_size) - next block pure token predictions

            # Collapse pure logits to group tokens
            predicted_groups = model.collapse_pure_to_group(stage1_logits)  # (B, T)

            # Step 2: Build new inputs with predicted group tokens at all block positions
            num_blocks = T // block_size
            new_inputs = inputs.clone()

            for block_idx in range(2, num_blocks):
                for pos in range(block_size):
                    src_pos = (block_idx - 1) * block_size + pos
                    dst_pos = block_idx * block_size + pos
                    new_inputs[:, dst_pos] = pure_vocab_size + predicted_groups[:, src_pos]

            # Step 3: Forward with predicted groups → stage2 logits
            stage2_logits, _ = model.forward_for_eval_block_pdlm_inference(
                new_inputs, targets, attn_mask=attn_mask)

            log_probs = F.log_softmax(stage2_logits.float(), dim=-1)
            # For end-to-end, targets are the ground truth pure tokens
            target_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

            # Compute loss at blocks 2..N-1
            for pos in range(block_size):
                for block_idx in range(2, num_blocks):
                    pos_in_seq = block_idx * block_size + pos
                    nll = -target_log_probs[:, pos_in_seq]
                    nll_by_pos[pos]["nll"] += nll.sum().item()
                    nll_by_pos[pos]["tokens"] += B

    return build_result_dict(nll_by_pos, block_size, include_accuracy=False)
