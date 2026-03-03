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
    Per-position breakdown within blocks, split by mask vs group (by num_G_before).
    """
    pure_vocab_size = model.config.pure_vocab_size
    num_groups = model.config.num_groups
    mask_token_id = pure_vocab_size + num_groups

    # Overall per-position tracking
    nll_by_pos = {p: {"nll": 0.0, "entropy": 0.0, "tokens": 0} for p in range(block_size)}
    # Mask scenario (r=0): per position
    nll_mask = {p: {"nll": 0.0, "tokens": 0} for p in range(block_size)}
    # Group scenario: per (pos, num_G_before)
    nll_group = {p: {g: {"nll": 0.0, "tokens": 0} for g in range(block_size)} for p in range(block_size)}

    for inputs, targets, loss_extras, _ in cached_batches:
        B, T = inputs.shape
        loss_mask = loss_extras["loss_mask"]  # (B, T)

        with autocast_ctx:
            logits = model.forward_for_eval_mask_pdlm(
                inputs, targets, attn_mask=attn_mask)

            log_probs = F.log_softmax(logits.float(), dim=-1)
            target_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            entropy = -(log_probs.exp() * log_probs).sum(dim=-1)  # (B, T)

            num_blocks = T // block_size

            # Detect token types from inputs
            is_mask_token = (inputs == mask_token_id)  # (B, T)
            is_group_token = (inputs >= pure_vocab_size) & (~is_mask_token)  # (B, T)

            for pos in range(block_size):
                for block_idx in range(1, num_blocks):
                    pos_in_seq = block_idx * block_size + pos
                    blk_start = block_idx * block_size

                    has_loss = loss_mask[:, pos_in_seq]  # (B,)
                    nll = -target_log_probs[:, pos_in_seq]  # (B,)

                    # Overall
                    nll_by_pos[pos]["nll"] += (nll * has_loss.float()).sum().item()
                    nll_by_pos[pos]["entropy"] += (entropy[:, pos_in_seq] * has_loss.float()).sum().item()
                    nll_by_pos[pos]["tokens"] += has_loss.sum().item()

                    # Mask scenario
                    is_mask_here = is_mask_token[:, pos_in_seq] & has_loss
                    nll_mask[pos]["nll"] += (nll * is_mask_here.float()).sum().item()
                    nll_mask[pos]["tokens"] += is_mask_here.sum().item()

                    # Group scenario — count preceding G's in same block
                    is_group_here = is_group_token[:, pos_in_seq] & has_loss
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
    # Attach detailed breakdown
    result["mask_breakdown"] = _build_breakdown(nll_mask, block_size)
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
    End-to-end iterative inference evaluation for mask_pdlm.

    Processes ALL target blocks (2..N-1) simultaneously:
    5-state: 1 collapse pass + block_size denoising passes = block_size + 1 total.
    4-state: step 1 captures pos 0 CE + collapses rest, then block_size - 1 denoising = block_size total.
    """
    nll_by_pos = {p: {
        "nll": 0.0, "entropy": 0.0, "correct": 0, "tokens": 0,
        "argmax_prob": 0.0, "second_prob": 0.0, "third_prob": 0.0,
        "top3_sum": 0.0, "top5_sum": 0.0,
    } for p in range(block_size)}
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

            # Step 1: Forward with mask tokens → collapse (+ sample pos 0 for 4-state)
            logits = model.forward_for_eval_mask_pdlm(
                eval_inputs, targets, attn_mask=attn_mask)

            mask_pdlm_4state = getattr(model.config, 'mask_pdlm_4state', False)
            if mask_pdlm_4state:
                # 4-state: capture CE for pos 0 from mask logits, teacher-force, collapse rest
                log_probs_s1 = F.log_softmax(logits.float(), dim=-1)
                target_lp_s1 = log_probs_s1.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                entropy_s1 = -(log_probs_s1.exp() * log_probs_s1).sum(dim=-1)
                nll_p0 = -target_lp_s1[pos_masks[0]]
                nll_by_pos[0]["nll"] += nll_p0.sum().item()
                nll_by_pos[0]["entropy"] += entropy_s1[pos_masks[0]].sum().item()
                nll_by_pos[0]["tokens"] += pos_masks[0].sum().item()

                eval_inputs[pos_masks[0]] = targets[pos_masks[0]]
                all_groups = model.collapse_pure_to_group(logits)
                for rp in range(1, block_size):
                    eval_inputs[pos_masks[rp]] = all_groups[pos_masks[rp]] + group_offset

                denoise_start = 1
            else:
                # 5-state: collapse all to groups, no CE capture in step 1
                all_groups = model.collapse_pure_to_group(logits)
                eval_inputs[target_mask] = all_groups[target_mask] + group_offset
                denoise_start = 0

            # Iterative denoising (all blocks simultaneously)
            # Capture CE from each step's logits (when the position is actually predicted)
            for denoise_step in range(denoise_start, block_size):
                logits = model.forward_for_eval_mask_pdlm(
                    eval_inputs, targets, attn_mask=attn_mask)

                # Compute CE for this position from current logits (model sees group/mask input)
                log_probs = F.log_softmax(logits.float(), dim=-1)
                probs = log_probs.exp()
                target_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                entropy = -(probs * log_probs).sum(dim=-1)  # (B, T)
                nll = -target_log_probs[pos_masks[denoise_step]]
                nll_by_pos[denoise_step]["nll"] += nll.sum().item()
                nll_by_pos[denoise_step]["entropy"] += entropy[pos_masks[denoise_step]].sum().item()
                nll_by_pos[denoise_step]["tokens"] += pos_masks[denoise_step].sum().item()
                preds = logits[pos_masks[denoise_step]].argmax(dim=-1)
                nll_by_pos[denoise_step]["correct"] += (preds == targets[pos_masks[denoise_step]]).sum().item()

                # Top-5 confidence metrics
                top5_probs, _ = probs.topk(5, dim=-1)  # (B, T, 5)
                pos_top5 = top5_probs[pos_masks[denoise_step]]  # (N, 5)
                nll_by_pos[denoise_step]["argmax_prob"] += pos_top5[:, 0].sum().item()
                nll_by_pos[denoise_step]["second_prob"] += pos_top5[:, 1].sum().item()
                nll_by_pos[denoise_step]["third_prob"]  += pos_top5[:, 2].sum().item()
                nll_by_pos[denoise_step]["top3_sum"]    += pos_top5[:, :3].sum().item()
                nll_by_pos[denoise_step]["top5_sum"]    += pos_top5[:, :5].sum().item()

                # Use ground truth pure token at denoise_step position (teacher forcing on P)
                # Previous positions' errors must not propagate — same principle as AR eval
                eval_inputs[pos_masks[denoise_step]] = targets[pos_masks[denoise_step]]

                # Collapse remaining positions back to groups
                if denoise_step < block_size - 1:
                    all_groups = model.collapse_pure_to_group(logits)
                    for rp in range(denoise_step + 1, block_size):
                        eval_inputs[pos_masks[rp]] = all_groups[pos_masks[rp]] + group_offset

    return build_result_dict(nll_by_pos, block_size, include_accuracy=True)


def _make_parallel_schedules(block_size):
    """
    Generate parallel decode schedules for a given block_size.

    Returns (schedule_mid, schedule_end) where each is a list of position tuples.
    - schedule_mid: pair at positions (block_size//2 - 1, block_size//2)
    - schedule_end: pair at last two positions (block_size-2, block_size-1)

    For block_size=1: (None, None)
    For block_size=2: ([(0,1)], [(0,1)])  — only option
    For block_size=4: ([(0,),(1,2),(3,)], [(0,),(1,),(2,3)])
    For block_size=8: ([(0,),(1,),(2,),(3,4),(5,),(6,),(7,)], [(0,),...,(6,7)])
    """
    if block_size <= 1:
        return None, None

    if block_size == 2:
        both = [(0, 1)]
        return both, both

    mid = block_size // 2 - 1  # e.g. 1 for bs=4, 3 for bs=8

    # parallel_mid: pair at (mid, mid+1), all others solo
    schedule_mid = []
    i = 0
    while i < block_size:
        if i == mid:
            schedule_mid.append((mid, mid + 1))
            i += 2
        else:
            schedule_mid.append((i,))
            i += 1

    # parallel_end: pair at last two positions, all others solo
    schedule_end = [(p,) for p in range(block_size - 2)] + [(block_size - 2, block_size - 1)]

    return schedule_mid, schedule_end


def _eval_end2end_with_schedule(
    model, cached_batches, block_size, attn_mask, device, autocast_ctx, decode_schedule,
):
    """
    End-to-end iterative inference evaluation with a custom decode schedule.

    decode_schedule: list of tuples of positions to decode in each step.
      Each tuple is decoded in ONE forward pass — positions in the tuple share
      the same input context (no GT reveal between them within the step).

    Examples:
      [(0,),(1,),(2,),(3,)]    — sequential (same as _eval_end2end)
      [(0,),(1,2),(3,)]        — parallel_mid: pos 1+2 together
      [(0,),(1,),(2,3)]        — parallel_end: pos 2+3 together

    Total forward passes per batch: 1 (init) + len(decode_schedule).
    """
    nll_by_pos = {p: {"nll": 0.0, "entropy": 0.0, "correct": 0, "tokens": 0} for p in range(block_size)}
    pure_vocab_size = model.config.pure_vocab_size
    num_groups = model.config.num_groups
    mask_token_id = pure_vocab_size + num_groups
    group_offset = pure_vocab_size

    for inputs, targets, loss_extras, _ in cached_batches:
        B, T = inputs.shape
        num_blocks = T // block_size

        with autocast_ctx:
            # Pre-build position masks: pos_masks[p] is True at position p of each target block (2..N-1)
            pos_masks = []
            for p in range(block_size):
                mask = torch.zeros(B, T, dtype=torch.bool, device=device)
                for block_idx in range(2, num_blocks):
                    mask[:, block_idx * block_size + p] = True
                pos_masks.append(mask)
            target_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
            for p in range(block_size):
                target_mask |= pos_masks[p]

            # Init: MASK → groups (identical to _eval_end2end)
            eval_inputs = targets.clone()
            eval_inputs[target_mask] = mask_token_id
            logits = model.forward_for_eval_mask_pdlm(eval_inputs, targets, attn_mask=attn_mask)
            all_groups = model.collapse_pure_to_group(logits)
            eval_inputs[target_mask] = all_groups[target_mask] + group_offset

            # Execute decode schedule
            for step_idx, step_positions in enumerate(decode_schedule):
                # One forward pass for all positions in this step
                logits = model.forward_for_eval_mask_pdlm(eval_inputs, targets, attn_mask=attn_mask)

                log_probs = F.log_softmax(logits.float(), dim=-1)
                target_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                entropy = -(log_probs.exp() * log_probs).sum(dim=-1)

                # Capture CE + accuracy for every position in this step (same forward pass / same context)
                for p in step_positions:
                    nll = -target_log_probs[pos_masks[p]]
                    nll_by_pos[p]["nll"] += nll.sum().item()
                    nll_by_pos[p]["entropy"] += entropy[pos_masks[p]].sum().item()
                    nll_by_pos[p]["tokens"] += pos_masks[p].sum().item()
                    preds = logits[pos_masks[p]].argmax(dim=-1)
                    nll_by_pos[p]["correct"] += (preds == targets[pos_masks[p]]).sum().item()

                # Teacher-force all positions in this step to GT
                for p in step_positions:
                    eval_inputs[pos_masks[p]] = targets[pos_masks[p]]

                # Determine positions still pending after this step
                decoded_so_far = set()
                for prev_positions in decode_schedule[:step_idx + 1]:
                    decoded_so_far.update(prev_positions)
                remaining = [rp for rp in range(block_size) if rp not in decoded_so_far]

                # Collapse remaining positions back to groups
                if remaining:
                    all_groups = model.collapse_pure_to_group(logits)
                    for rp in remaining:
                        eval_inputs[pos_masks[rp]] = all_groups[pos_masks[rp]] + group_offset

    return build_result_dict(nll_by_pos, block_size, include_accuracy=True)


def _eval_end2end_refresh(model, cached_batches, block_size, attn_mask, device, autocast_ctx):
    """
    End-to-end iterative inference evaluation with group token refresh.

    Same as _eval_end2end but before recording loss at pos k, runs an extra
    "refresh" forward pass to update remaining group tokens with the current
    ground-truth context (P_gt at positions 0..k-1). This removes the bias
    against high-p (soft) models caused by stale group tokens.

    Forward passes per batch: 1 (init) + 2 * block_size (refresh + eval per step).
    """
    nll_by_pos = {p: {
        "nll": 0.0, "entropy": 0.0, "correct": 0, "tokens": 0,
        "argmax_prob": 0.0, "second_prob": 0.0, "third_prob": 0.0,
        "top3_sum": 0.0, "top5_sum": 0.0,
    } for p in range(block_size)}
    pure_vocab_size = model.config.pure_vocab_size
    num_groups = model.config.num_groups
    mask_token_id = pure_vocab_size + num_groups
    group_offset = pure_vocab_size

    for inputs, targets, loss_extras, _ in cached_batches:
        B, T = inputs.shape
        num_blocks = T // block_size

        with autocast_ctx:
            # Pre-build position masks: pos_masks[p] is True at position p of each target block (2..N-1)
            pos_masks = []
            for p in range(block_size):
                mask = torch.zeros(B, T, dtype=torch.bool, device=device)
                for block_idx in range(2, num_blocks):
                    mask[:, block_idx * block_size + p] = True
                pos_masks.append(mask)
            target_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
            for p in range(block_size):
                target_mask |= pos_masks[p]

            # Init: mask all target blocks → forward → collapse to groups
            eval_inputs = targets.clone()
            eval_inputs[target_mask] = mask_token_id
            logits = model.forward_for_eval_mask_pdlm(eval_inputs, targets, attn_mask=attn_mask)
            all_groups = model.collapse_pure_to_group(logits)
            eval_inputs[target_mask] = all_groups[target_mask] + group_offset

            for k in range(block_size):
                # eval_inputs: [P_gt_0..k-1, G_k, ..., G_{n-1}]
                # G comes from prior step's eval forward collapse

                # Step A: Refresh forward — update G for positions k..n-1
                # so they are aware of P_gt at positions 0..k-1
                logits_r = model.forward_for_eval_mask_pdlm(eval_inputs, targets, attn_mask=attn_mask)
                g_refresh = model.collapse_pure_to_group(logits_r)
                del logits_r  # free before step B forward
                for rp in range(k, block_size):
                    eval_inputs[pos_masks[rp]] = g_refresh[pos_masks[rp]] + group_offset
                del g_refresh

                # Step B: Eval forward with refreshed groups → record loss at pos k
                logits = model.forward_for_eval_mask_pdlm(eval_inputs, targets, attn_mask=attn_mask)
                log_probs = F.log_softmax(logits.float(), dim=-1)
                probs = log_probs.exp()
                target_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                entropy = -(probs * log_probs).sum(dim=-1)
                del log_probs  # free large tensor before collapse

                nll = -target_log_probs[pos_masks[k]]
                nll_by_pos[k]["nll"] += nll.sum().item()
                nll_by_pos[k]["entropy"] += entropy[pos_masks[k]].sum().item()
                nll_by_pos[k]["tokens"] += pos_masks[k].sum().item()
                preds = logits[pos_masks[k]].argmax(dim=-1)
                nll_by_pos[k]["correct"] += (preds == targets[pos_masks[k]]).sum().item()

                # Top-5 confidence metrics
                top5_probs, _ = probs.topk(5, dim=-1)  # (B, T, 5)
                pos_top5 = top5_probs[pos_masks[k]]    # (N, 5)
                nll_by_pos[k]["argmax_prob"] += pos_top5[:, 0].sum().item()
                nll_by_pos[k]["second_prob"] += pos_top5[:, 1].sum().item()
                nll_by_pos[k]["third_prob"]  += pos_top5[:, 2].sum().item()
                nll_by_pos[k]["top3_sum"]    += pos_top5[:, :3].sum().item()
                nll_by_pos[k]["top5_sum"]    += pos_top5[:, :5].sum().item()
                del probs

                # Teacher-force pos k to ground truth
                eval_inputs[pos_masks[k]] = targets[pos_masks[k]]

                # Collapse remaining positions from eval forward for next iter's starting G
                if k < block_size - 1:
                    g_next = model.collapse_pure_to_group(logits)
                    for rp in range(k + 1, block_size):
                        eval_inputs[pos_masks[rp]] = g_next[pos_masks[rp]] + group_offset
                del logits

    return build_result_dict(nll_by_pos, block_size, include_accuracy=True)


def eval_mask_pdlm_refresh(
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
    Evaluate mask_pdlm with group token refresh between denoising steps.

    Before recording loss at each position k, runs a refresh forward pass to
    update remaining group tokens with the current ground-truth context. This
    removes the staleness bias against high-p (soft) models in the standard
    end2end eval.

    Returns:
        {
            "stage": "mask_pdlm",
            "end2end_refresh": result_dict,
        }
    """
    with model_eval_context(model):
        with torch.no_grad():
            cached_batches = [next(val_loader) for _ in range(num_batches)]
            refresh_result = _eval_end2end_refresh(
                model, cached_batches, block_size, attn_mask, device, autocast_ctx,
            )

    return {
        "stage": "mask_pdlm",
        "end2end_refresh": refresh_result,
    }


def _eval_end2end_fresh_mask_g(model, cached_batches, block_size, attn_mask, device, autocast_ctx):
    """
    End-to-end eval with fresh mask-derived G tokens at each step.

    Like _eval_end2end_refresh, but Step A resets remaining positions to MASK
    tokens before collapsing to G (instead of using stale G as context).
    This removes trajectory inertia entirely: G is always derived from
    [GT_0..k-1, MASK, MASK, ...] rather than [GT_0..k-1, stale_G, ...].

    If fresh_acc ≈ p=0 acc, the inertia from stale G is the full explanation
    for the accuracy gap in normal end2end eval.

    Forward passes per batch: 1 (init) + 2 * block_size (mask-refresh + eval per step).
    """
    nll_by_pos = {p: {
        "nll": 0.0, "entropy": 0.0, "correct": 0, "tokens": 0,
        "argmax_prob": 0.0, "second_prob": 0.0, "third_prob": 0.0,
        "top3_sum": 0.0, "top5_sum": 0.0,
    } for p in range(block_size)}
    pure_vocab_size = model.config.pure_vocab_size
    num_groups = model.config.num_groups
    mask_token_id = pure_vocab_size + num_groups
    group_offset = pure_vocab_size

    for inputs, targets, loss_extras, _ in cached_batches:
        B, T = inputs.shape
        num_blocks = T // block_size

        with autocast_ctx:
            pos_masks = []
            for p in range(block_size):
                mask = torch.zeros(B, T, dtype=torch.bool, device=device)
                for block_idx in range(2, num_blocks):
                    mask[:, block_idx * block_size + p] = True
                pos_masks.append(mask)
            target_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
            for p in range(block_size):
                target_mask |= pos_masks[p]

            # Init: mask all target blocks → forward → collapse to groups (same as other evals)
            eval_inputs = targets.clone()
            eval_inputs[target_mask] = mask_token_id
            logits = model.forward_for_eval_mask_pdlm(eval_inputs, targets, attn_mask=attn_mask)
            all_groups = model.collapse_pure_to_group(logits)
            eval_inputs[target_mask] = all_groups[target_mask] + group_offset

            for k in range(block_size):
                # Step A: Build fresh_inputs with MASK for all remaining positions k..block_size-1.
                # This removes trajectory inertia: G is derived from [GT_0..k-1, MASK, ...]
                # rather than [GT_0..k-1, stale_G, ...].
                fresh_inputs = eval_inputs.clone()
                remaining_positions = torch.zeros(B, T, dtype=torch.bool, device=device)
                for rp in range(k, block_size):
                    remaining_positions |= pos_masks[rp]
                fresh_inputs[remaining_positions] = mask_token_id

                logits_r = model.forward_for_eval_mask_pdlm(fresh_inputs, targets, attn_mask=attn_mask)
                g_fresh = model.collapse_pure_to_group(logits_r)
                del logits_r, fresh_inputs, remaining_positions
                for rp in range(k, block_size):
                    eval_inputs[pos_masks[rp]] = g_fresh[pos_masks[rp]] + group_offset
                del g_fresh

                # Step B: Eval forward with mask-derived G → record loss at pos k
                logits = model.forward_for_eval_mask_pdlm(eval_inputs, targets, attn_mask=attn_mask)
                log_probs = F.log_softmax(logits.float(), dim=-1)
                probs = log_probs.exp()
                target_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                entropy = -(probs * log_probs).sum(dim=-1)
                del log_probs

                nll = -target_log_probs[pos_masks[k]]
                nll_by_pos[k]["nll"] += nll.sum().item()
                nll_by_pos[k]["entropy"] += entropy[pos_masks[k]].sum().item()
                nll_by_pos[k]["tokens"] += pos_masks[k].sum().item()
                preds = logits[pos_masks[k]].argmax(dim=-1)
                nll_by_pos[k]["correct"] += (preds == targets[pos_masks[k]]).sum().item()

                # Top-5 confidence metrics
                top5_probs, _ = probs.topk(5, dim=-1)  # (B, T, 5)
                pos_top5 = top5_probs[pos_masks[k]]    # (N, 5)
                nll_by_pos[k]["argmax_prob"] += pos_top5[:, 0].sum().item()
                nll_by_pos[k]["second_prob"] += pos_top5[:, 1].sum().item()
                nll_by_pos[k]["third_prob"]  += pos_top5[:, 2].sum().item()
                nll_by_pos[k]["top3_sum"]    += pos_top5[:, :3].sum().item()
                nll_by_pos[k]["top5_sum"]    += pos_top5[:, :5].sum().item()
                del probs

                # Teacher-force pos k to ground truth
                eval_inputs[pos_masks[k]] = targets[pos_masks[k]]

                # Collapse remaining positions from eval forward for next iter's starting G
                if k < block_size - 1:
                    g_next = model.collapse_pure_to_group(logits)
                    for rp in range(k + 1, block_size):
                        eval_inputs[pos_masks[rp]] = g_next[pos_masks[rp]] + group_offset
                del logits

    return build_result_dict(nll_by_pos, block_size, include_accuracy=True)


def eval_mask_pdlm_fresh_mask_g(
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
    Evaluate mask_pdlm with fresh mask-derived G tokens at each step.

    At each denoising step k, remaining positions are reset to MASK tokens before
    collapsing to G. This removes the trajectory inertia caused by stale G tokens
    carried from previous steps (unlike refresh eval which uses previous G as context).

    If fresh_acc at pos1+ recovers toward p=0 levels, it confirms the accuracy drop
    in normal end2end eval is caused by G-token inertia, not model capability.

    Also reports top-5 confidence metrics per position to show that higher soft_p
    leads to more confident predictions (higher argmax prob, sharper top-k distribution).

    Returns:
        {
            "stage": "mask_pdlm",
            "end2end_fresh_mask_g": result_dict,
        }
    """
    with model_eval_context(model):
        with torch.no_grad():
            cached_batches = [next(val_loader) for _ in range(num_batches)]
            fresh_result = _eval_end2end_fresh_mask_g(
                model, cached_batches, block_size, attn_mask, device, autocast_ctx,
            )

    return {
        "stage": "mask_pdlm",
        "end2end_fresh_mask_g": fresh_result,
    }


def _eval_threshold_decode(
    model, cached_batches, block_size, attn_mask, device, autocast_ctx, threshold,
):
    """
    Threshold-based parallel decoding evaluation for 4-state mask_pdlm.

    At each step:
    - Forward pass → get argmax_prob at all remaining positions
    - Decode all positions where argmax_prob > threshold (teacher-force to GT)
    - Fallback: if no position in a block passes threshold, decode the leftmost
      remaining position in that block (not most confident — matches mask_pdlm L→R bias)
    - Collapse remaining positions to group tokens
    - Repeat until all positions decoded

    Key differences from BD3-LM version:
    - eval_inputs init: targets.clone() + mask target blocks (not all-MASK)
    - After decode: collapse remaining to groups (not leave as MASK)
    - Fallback: leftmost (argmin on position indices) not most confident (argmax on probs)
    - Target blocks start at 2 (not 1)

    Tracks avg_steps: average number of forward passes to fully decode a block.
    """
    pure_vocab_size = model.config.pure_vocab_size
    num_groups = model.config.num_groups
    mask_token_id = pure_vocab_size + num_groups
    group_offset = pure_vocab_size

    total_steps = 0.0
    total_blocks = 0

    for inputs, targets, loss_extras, _ in cached_batches:
        B, T = targets.shape
        num_blocks = T // block_size
        num_target_blocks = num_blocks - 2  # skip blocks 0 and 1

        if num_target_blocks <= 0:
            continue

        with autocast_ctx:
            # remaining_blk[b, bi, p] = True if position p in block bi is not yet decoded
            remaining_blk = torch.zeros(B, num_blocks, block_size, dtype=torch.bool, device=device)
            remaining_blk[:, 2:, :] = True  # target blocks: 2..N-1

            # Init: teacher-force with ground truth, mask target blocks
            eval_inputs = targets.clone()
            target_mask = remaining_blk.view(B, T)
            eval_inputs[target_mask] = mask_token_id

            # Track steps per block: (B, num_target_blocks)
            block_steps = torch.zeros(B, num_target_blocks, device=device)
            block_done = torch.zeros(B, num_target_blocks, dtype=torch.bool, device=device)

            for step in range(1, block_size + 1):
                if not remaining_blk.any():
                    break

                logits = model.forward_for_eval_mask_pdlm(eval_inputs, targets, attn_mask=attn_mask)
                probs = torch.softmax(logits.float(), dim=-1)
                argmax_probs = probs.max(dim=-1).values  # (B, T)

                # Reshape to block view: (B, num_blocks, block_size)
                argmax_probs_blk = argmax_probs.view(B, num_blocks, block_size)

                # Positions passing threshold (only within remaining)
                decode_blk = remaining_blk & (argmax_probs_blk > threshold)

                # Fallback: for blocks with remaining positions but none passing threshold,
                # decode the leftmost remaining position
                has_remaining = remaining_blk[:, 2:, :].any(dim=-1)  # (B, num_target_blocks)
                has_decoded = decode_blk[:, 2:, :].any(dim=-1)       # (B, num_target_blocks)
                needs_fallback = has_remaining & ~has_decoded          # (B, num_target_blocks)

                if needs_fallback.any():
                    # For fallback blocks: find leftmost remaining position
                    # Set non-remaining positions to inf, take argmin to get leftmost
                    fb_pos = torch.arange(block_size, device=device).float()
                    fb_pos = fb_pos.unsqueeze(0).unsqueeze(0).expand(B, num_target_blocks, block_size)
                    fb_pos = fb_pos.clone()
                    fb_pos[~remaining_blk[:, 2:, :]] = float('inf')
                    fb_leftmost = fb_pos.argmin(dim=-1)               # (B, num_target_blocks)
                    # One-hot for the leftmost position per block
                    fb_onehot = F.one_hot(fb_leftmost, block_size).bool()  # (B, num_target_blocks, block_size)
                    # Only apply fallback where needed
                    fb_mask = needs_fallback.unsqueeze(-1) & fb_onehot
                    decode_blk[:, 2:, :] |= fb_mask

                # Flatten back to (B, T) for teacher-forcing
                decode_flat = decode_blk.view(B, T)
                eval_inputs[decode_flat] = targets[decode_flat]
                remaining_blk[decode_blk] = False

                # Collapse remaining positions to groups
                if remaining_blk.any():
                    all_groups = model.collapse_pure_to_group(logits)
                    remaining_flat = remaining_blk.view(B, T)
                    eval_inputs[remaining_flat] = all_groups[remaining_flat] + group_offset

                # Update block_steps for blocks that just finished
                newly_done = has_remaining & ~remaining_blk[:, 2:, :].any(dim=-1) & ~block_done
                block_steps[newly_done] = step
                block_done |= newly_done

        total_steps += block_steps.sum().item()
        total_blocks += B * num_target_blocks

    avg_steps = total_steps / total_blocks if total_blocks > 0 else 0.0
    return {"threshold": threshold, "avg_steps": avg_steps, "total_blocks": total_blocks}


def eval_mask_pdlm_threshold(
    model,
    val_loader,
    block_size,
    num_batches,
    attn_mask,
    device,
    autocast_ctx,
    prefix_pure_tokens=0,
    thresholds=None,
):
    """
    Run threshold-based parallel decoding evaluation for mask_pdlm across multiple thresholds.

    Caches batches once, then sweeps over all threshold values.

    Returns:
        {"stage": "mask_pdlm", "threshold_decode": {tau: {"avg_steps": X, "total_blocks": N}}}
    """
    if thresholds is None:
        thresholds = [i / 10.0 for i in range(11)]  # 0.0, 0.1, ..., 1.0

    with model_eval_context(model):
        with torch.no_grad():
            cached_batches = [next(val_loader) for _ in range(num_batches)]

            results = {}
            for tau in thresholds:
                result = _eval_threshold_decode(
                    model, cached_batches, block_size,
                    attn_mask, device, autocast_ctx, tau,
                )
                results[tau] = result

    return {"stage": "mask_pdlm", "threshold_decode": results}


def eval_mask_pdlm_parallel(
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
    Evaluate mask_pdlm with parallel decode variants.

    Runs two variants (parallel_mid and parallel_end) without re-running unified loss.
    - parallel_mid: decode the middle pair of positions in one step
    - parallel_end: decode the last pair of positions in one step

    For block_size=4:
      parallel_mid schedule: [(0,), (1,2), (3,)]   — 4 total forward passes
      parallel_end schedule: [(0,), (1,), (2,3)]   — 4 total forward passes

    Returns:
        {
            "stage": "mask_pdlm",
            "parallel_mid": result_dict or None,
            "parallel_end": result_dict or None,
        }
    """
    schedule_mid, schedule_end = _make_parallel_schedules(block_size)

    with model_eval_context(model):
        with torch.no_grad():
            cached_batches = [next(val_loader) for _ in range(num_batches)]

            result_mid = (
                _eval_end2end_with_schedule(
                    model, cached_batches, block_size, attn_mask, device, autocast_ctx, schedule_mid,
                )
                if schedule_mid is not None else None
            )
            result_end = (
                _eval_end2end_with_schedule(
                    model, cached_batches, block_size, attn_mask, device, autocast_ctx, schedule_end,
                )
                if schedule_end is not None else None
            )

    return {
        "stage": "mask_pdlm",
        "parallel_mid": result_mid,
        "parallel_end": result_end,
    }
