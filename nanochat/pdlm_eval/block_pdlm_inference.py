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
    Includes per-position breakdown by pure vs group (by num_G_before).
    """
    pure_vocab_size = model.config.pure_vocab_size

    nll_by_pos = {p: {"nll": 0.0, "tokens": 0} for p in range(block_size)}
    nll_pure = {p: {"nll": 0.0, "tokens": 0} for p in range(block_size)}
    nll_group = {p: {g: {"nll": 0.0, "tokens": 0} for g in range(block_size)} for p in range(block_size)}

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

            # Detect token types from inputs
            is_group_token = (inputs >= pure_vocab_size)  # (B, T)
            is_pure_token = (inputs < pure_vocab_size)  # (B, T)

            for pos in range(block_size):
                for block_idx in range(1, num_blocks):
                    pos_in_seq = block_idx * block_size + pos
                    blk_start = block_idx * block_size

                    mask_at_pos = stage2_loss_mask[:, pos_in_seq]
                    nll = -target_log_probs[:, pos_in_seq]

                    # Overall
                    nll_by_pos[pos]["nll"] += (nll * mask_at_pos.float()).sum().item()
                    nll_by_pos[pos]["tokens"] += mask_at_pos.sum().item()

                    # Pure scenario
                    is_pure_here = is_pure_token[:, pos_in_seq] & mask_at_pos
                    nll_pure[pos]["nll"] += (nll * is_pure_here.float()).sum().item()
                    nll_pure[pos]["tokens"] += is_pure_here.sum().item()

                    # Group scenario — count preceding G's in same block
                    is_group_here = is_group_token[:, pos_in_seq] & mask_at_pos
                    if is_group_here.any():
                        if pos > 0:
                            num_g_before = is_group_token[:, blk_start:blk_start + pos].sum(dim=1)  # (B,)
                        else:
                            num_g_before = torch.zeros(B, dtype=torch.long, device=inputs.device)
                        for g in range(pos + 1):  # max G before pos is pos itself
                            g_match = is_group_here & (num_g_before == g)
                            nll_group[pos][g]["nll"] += (nll * g_match.float()).sum().item()
                            nll_group[pos][g]["tokens"] += g_match.sum().item()

    result = build_result_dict(nll_by_pos, block_size, include_accuracy=False)
    # Attach breakdowns
    result["pure_breakdown"] = _build_breakdown(nll_pure, block_size)
    result["group_breakdown"] = {}
    for pos in range(block_size):
        result["group_breakdown"][pos] = {}
        for g in range(pos + 1):
            m = nll_group[pos][g]
            if m["tokens"] > 0:
                loss = m["nll"] / m["tokens"]
                result["group_breakdown"][pos][g] = {
                    "loss": loss,
                    "ppl": torch.exp(torch.tensor(loss)).item(),
                    "tokens": m["tokens"],
                }
    return result


def _build_breakdown(nll_dict, block_size):
    """Build loss/ppl dict from nll tracker."""
    result = {}
    for pos in range(block_size):
        m = nll_dict[pos]
        if m["tokens"] > 0:
            loss = m["nll"] / m["tokens"]
            result[pos] = {"loss": loss, "ppl": torch.exp(torch.tensor(loss)).item(), "tokens": m["tokens"]}
    return result


def _eval_end2end(model, cached_batches, block_size, attn_mask, device, autocast_ctx):
    """
    End-to-end iterative inference evaluation for block_pdlm_inference.

    Processes ALL target blocks (2..N-1) simultaneously:
    1. Forward with training inputs → stage1 logits (x0 half) → collapse to groups
    2. Iterative denoising (block_size steps): sample one position per step,
       capture CE from that step's logits, collapse rest

    Total forward passes per batch: 1 + block_size.
    """
    nll_by_pos = {p: {"nll": 0.0, "tokens": 0} for p in range(block_size)}
    pure_vocab_size = model.config.pure_vocab_size

    for inputs, targets, loss_extras, _ in cached_batches:
        B, T = inputs.shape
        num_blocks = T // block_size

        with autocast_ctx:
            # Pre-build position masks for each within-block position across target blocks
            pos_masks = []
            for p in range(block_size):
                mask = torch.zeros(B, T, dtype=torch.bool, device=device)
                for block_idx in range(2, num_blocks):
                    mask[:, block_idx * block_size + p] = True
                pos_masks.append(mask)
            target_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
            for p in range(block_size):
                target_mask |= pos_masks[p]

            # Step 1: Forward with training inputs → stage1 logits (x0 half)
            _, stage1_logits = model.forward_for_eval_block_pdlm_inference(
                inputs, targets, attn_mask=attn_mask)
            predicted_groups = model.collapse_pure_to_group(stage1_logits)  # (B, T)

            # Build eval_inputs: teacher-force with ground truth, replace target blocks with groups
            eval_inputs = targets.clone()
            for block_idx in range(2, num_blocks):
                for pos in range(block_size):
                    src_pos = (block_idx - 1) * block_size + pos
                    dst_pos = block_idx * block_size + pos
                    eval_inputs[:, dst_pos] = pure_vocab_size + predicted_groups[:, src_pos]

            # Steps 2..block_size+1: Iterative denoising (all blocks simultaneously)
            for denoise_step in range(block_size):
                stage2_logits, _ = model.forward_for_eval_block_pdlm_inference(
                    eval_inputs, targets, attn_mask=attn_mask)

                # Capture CE for this position from current logits
                log_probs = F.log_softmax(stage2_logits.float(), dim=-1)
                target_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                nll = -target_log_probs[pos_masks[denoise_step]]
                nll_by_pos[denoise_step]["nll"] += nll.sum().item()
                nll_by_pos[denoise_step]["tokens"] += pos_masks[denoise_step].sum().item()

                # Sample pure token at denoise_step position of all target blocks
                sampled = stage2_logits.argmax(dim=-1)  # (B, T)
                eval_inputs[pos_masks[denoise_step]] = sampled[pos_masks[denoise_step]]

                # Collapse remaining positions back to groups
                if denoise_step < block_size - 1:
                    all_groups = model.collapse_pure_to_group(stage2_logits)
                    for rp in range(denoise_step + 1, block_size):
                        eval_inputs[pos_masks[rp]] = all_groups[pos_masks[rp]] + pure_vocab_size

    return build_result_dict(nll_by_pos, block_size, include_accuracy=False)
