"""
BD3LM Evaluation Module.

Computes loss and perplexity for BD3LM models on validation data.

Evaluation setup:
- Input is [xt | x0] of length 2L
- xt (first L): ALL tokens are MASKED (or partially masked with suffix clear)
- x0 (second L): ALL tokens are clean
- Skip block 0 for loss computation (no previous context to condition on)
- Compute loss from blocks 1, 2, ... (they have clean context via cross-attention)

Two evaluation modes:
- Normal mode (target_shift < 0): compute overall + per-position metrics
- Target_shift mode (target_shift >= 1): compute metrics only at specific position

Suffix metrics:
- In addition to all-masked eval, we also report metrics with N suffix tokens clear
- For position p, suffix tokens are positions p+1, p+2, ..., block_size-1
- This shows how much the model benefits from seeing clean context after prediction target

Example (L=16, block_size=4, 4 blocks, target_shift=1):
    All masked:   [MASK MASK MASK MASK | MASK MASK MASK MASK | ...]
    1 suffix:     [MASK clean MASK MASK | MASK clean MASK MASK | ...]  (pos 1 clear)
    2 suffix:     [MASK clean clean MASK | MASK clean clean MASK | ...]  (pos 1,2 clear)
    3 suffix:     [MASK clean clean clean | MASK clean clean clean | ...]  (pos 1,2,3 clear)
"""

import torch
import torch.nn.functional as F

from nanochat.bd3lm_utils.prime_encoding import encode as prime_encode


def eval_bd3lm(
    model,
    val_loader,
    block_size,
    target_shift,
    num_batches,
    attn_mask,
    device,
    autocast_ctx,
    mask_token_id,
    left_to_right=False,
):
    """
    Evaluate BD3LM model on validation set.

    Args:
        model: BD3LM model
        val_loader: validation data loader (yields inputs, targets, loss_extras, state_dict)
                    We only use targets from the loader; inputs are replaced with all-MASK
        block_size: block size for BD3LM
        target_shift: if >= 1, evaluate only position (target_shift-1); if < 0, evaluate all positions
        num_batches: number of batches to evaluate
        attn_mask: attention mask for the model (from gen_mask)
        device: device to run on
        autocast_ctx: autocast context for mixed precision
        mask_token_id: token id for MASK token
        left_to_right: if True and target_shift < 0, run left-to-right teacher-forced eval
                       (one forward per position k, reveals prefix 0..k-1 within each block)

    Returns:
        dict with evaluation results:
        - target_shift >= 1 (e.g., ts=1, block_size=4):
            {
                "loss": float, "ppl": float,              # all masked (core metric)
                "entropy_ppl": float,                     # distribution entropy at target pos
                "argmax_prob": float,                     # avg prob of top-1 prediction
                "loss_1suffix": float, "ppl_1suffix": float,
                "entropy_ppl_1suffix": float, "argmax_prob_1suffix": float,
                ...
            }
        - target_shift < 0, left_to_right=False (normal mode):
            {
                "overall_loss": float, "overall_ppl": float,
                "positions": {
                    0: {"loss": X, "ppl": Y, "entropy_ppl": Z, "argmax_prob": W,
                        "loss_1suffix": A, "ppl_1suffix": B,
                        "entropy_ppl_1suffix": C, "argmax_prob_1suffix": D, ...},
                    1: {"loss": X, "ppl": Y, "entropy_ppl": Z, "argmax_prob": W, ...},
                    ...
                },
                "suffix_overall": {
                    1: {"positions": [0, 1, 2], "overall_loss": float, "overall_ppl": float},
                    2: {...}, ...
                }
            }
        - target_shift < 0, left_to_right=True:
            {
                "stage": "bd3lm",
                "left_to_right": {
                    "overall_loss": float, "overall_ppl": float, "overall_accuracy": float,
                    "positions": {
                        0: {"loss": float, "ppl": float, "accuracy": float},
                        ...
                    }
                }
            }
    """
    was_training = model.training
    model.eval()

    with torch.no_grad():
        if left_to_right and target_shift < 0:
            ltr_result = _eval_left_to_right_mode(
                model, val_loader, block_size,
                num_batches, attn_mask, device, autocast_ctx, mask_token_id
            )
            result = {"stage": "bd3lm", "left_to_right": ltr_result}
        elif target_shift >= 1:
            result = _eval_target_shift_mode(
                model, val_loader, block_size, target_shift,
                num_batches, attn_mask, device, autocast_ctx, mask_token_id
            )
        else:
            result = _eval_normal_mode(
                model, val_loader, block_size,
                num_batches, attn_mask, device, autocast_ctx, mask_token_id
            )

    if was_training:
        model.train()
    return result


def _prepare_ltr_batch(targets, mask_token_id, block_size, num_prefix):
    """
    Prepare left-to-right eval batch: reveal positions 0..num_prefix-1 in each block.

    For each block, the first num_prefix positions are clean (revealed),
    all remaining positions are MASK.

    Args:
        targets: (B, L) clean target tokens (already on device)
        mask_token_id: token id for MASK
        block_size: size of each block
        num_prefix: number of prefix positions to reveal (0 = all masked)

    Returns:
        inputs: (B, L) tokens with prefix positions revealed, rest MASK
    """
    B, L = targets.shape
    num_blocks = L // block_size

    inputs = torch.full((B, L), mask_token_id, dtype=torch.long, device=targets.device)

    if num_prefix > 0:
        for block_idx in range(num_blocks):
            block_start = block_idx * block_size
            for j in range(num_prefix):
                inputs[:, block_start + j] = targets[:, block_start + j]

    return inputs


def _eval_left_to_right_mode(
    model, val_loader, block_size,
    num_batches, attn_mask, device, autocast_ctx, mask_token_id
):
    """
    Evaluate BD3LM in left-to-right teacher-forced mode.

    Mirrors PDLM p=0 end2end eval: at each position k within a block, predicts
    the token given ground-truth prefix (positions 0..k-1 in the block) plus
    clean tokens from all previous blocks (available via x0 cross-attention).

    One forward pass per k covers all blocks simultaneously (O(block_size) passes).
    Block 0 is skipped (no previous context).

    Returns:
        {
            "overall_loss": float, "overall_ppl": float, "overall_accuracy": float,
            "positions": {
                0: {"loss": float, "ppl": float, "accuracy": float},
                1: {...}, ...
            }
        }
    """
    nll_data = {k: {"nll": 0.0, "entropy": 0.0, "argmax_prob": 0.0, "tokens": 0} for k in range(block_size)}
    acc_data = {k: {"correct": 0, "total": 0} for k in range(block_size)}

    # Collect all batches (need to iterate block_size times)
    all_targets = []
    for _ in range(num_batches):
        _, targets_batch, _, _ = next(val_loader)
        all_targets.append(targets_batch)

    for k in range(block_size):
        for targets_batch in all_targets:
            B, L = targets_batch.shape
            num_blocks = L // block_size

            inputs = _prepare_ltr_batch(targets_batch, mask_token_id, block_size, num_prefix=k)

            with autocast_ctx:
                logits = model.forward_for_eval(inputs, targets_batch, attn_mask=attn_mask)
                log_probs = F.log_softmax(logits.float(), dim=-1)
                probs = log_probs.exp()
                entropy = -(probs * log_probs).sum(dim=-1)  # (B, L)
                argmax_prob = probs.max(dim=-1)[0]           # (B, L)

                # Evaluate position k in each block, skipping block 0
                for block_idx in range(1, num_blocks):
                    pos_in_seq = block_idx * block_size + k

                    nll = -log_probs[:, pos_in_seq, :].gather(
                        -1, targets_batch[:, pos_in_seq].unsqueeze(-1)
                    ).squeeze(-1)
                    nll_data[k]["nll"] += nll.sum().item()
                    nll_data[k]["entropy"] += entropy[:, pos_in_seq].sum().item()
                    nll_data[k]["argmax_prob"] += argmax_prob[:, pos_in_seq].sum().item()
                    nll_data[k]["tokens"] += B

                    preds = logits[:, pos_in_seq, :].argmax(dim=-1)
                    acc_data[k]["correct"] += (preds == targets_batch[:, pos_in_seq]).sum().item()
                    acc_data[k]["total"] += B

    total_nll = sum(nll_data[k]["nll"] for k in range(block_size))
    total_tokens = sum(nll_data[k]["tokens"] for k in range(block_size))
    total_correct = sum(acc_data[k]["correct"] for k in range(block_size))

    total_entropy = sum(nll_data[k]["entropy"] for k in range(block_size))
    total_argmax_prob = sum(nll_data[k]["argmax_prob"] for k in range(block_size))
    overall_loss = total_nll / total_tokens if total_tokens > 0 else 0.0
    overall_entropy = total_entropy / total_tokens if total_tokens > 0 else 0.0
    result = {
        "overall_loss": overall_loss,
        "overall_ppl": torch.exp(torch.tensor(overall_loss)).item(),
        "overall_entropy_ppl": torch.exp(torch.tensor(overall_entropy)).item(),
        "overall_accuracy": total_correct / total_tokens if total_tokens > 0 else 0.0,
        "overall_argmax_prob": total_argmax_prob / total_tokens if total_tokens > 0 else 0.0,
        "positions": {},
    }

    for k in range(block_size):
        tokens_k = nll_data[k]["tokens"]
        loss_k = nll_data[k]["nll"] / tokens_k if tokens_k > 0 else 0.0
        entropy_k = nll_data[k]["entropy"] / tokens_k if tokens_k > 0 else 0.0
        acc_k = acc_data[k]["correct"] / acc_data[k]["total"] if acc_data[k]["total"] > 0 else 0.0
        result["positions"][k] = {
            "loss": loss_k,
            "ppl": torch.exp(torch.tensor(loss_k)).item(),
            "entropy_ppl": torch.exp(torch.tensor(entropy_k)).item(),
            "argmax_prob": nll_data[k]["argmax_prob"] / tokens_k if tokens_k > 0 else 0.0,
            "accuracy": acc_k,
        }

    return result


def _eval_threshold_decode(
    model, cached_batches, block_size,
    attn_mask, device, autocast_ctx, mask_token_id, threshold,
):
    """
    Threshold-based parallel decoding evaluation for BD3LM.

    At each step:
    - Forward pass → get argmax_prob at all remaining positions
    - Decode all positions where argmax_prob > threshold (use model's argmax prediction)
    - Fallback: if no position in a block passes threshold, decode the most
      confident remaining position in that block
    - Remaining positions stay as MASK (no group collapse in BD3LM)
    - Repeat until all positions decoded

    Tracks avg_steps: average number of forward passes to fully decode a block.

    Args:
        model: BD3LM model
        cached_batches: list of (targets,) tuples (pre-collected from val_loader)
        block_size: block size
        attn_mask: attention mask
        device: device
        autocast_ctx: autocast context
        mask_token_id: MASK token id
        threshold: confidence threshold τ (0.0 to 1.0)

    Returns:
        {"threshold": float, "avg_steps": float, "total_blocks": int}
    """
    total_steps = 0.0
    total_blocks = 0

    for (targets_batch,) in cached_batches:
        B, L = targets_batch.shape
        num_blocks = L // block_size
        num_target_blocks = num_blocks - 1  # skip block 0

        with autocast_ctx:
            # Reshape to (B, num_blocks, block_size) for vectorized block operations
            # remaining_blk[b, bi, p] = True if position p in target block bi is not yet decoded
            remaining_blk = torch.zeros(B, num_blocks, block_size, dtype=torch.bool, device=device)
            remaining_blk[:, 1:, :] = True  # skip block 0

            # All positions start as MASK
            eval_inputs = torch.full((B, L), mask_token_id, dtype=torch.long, device=device)

            # Track steps per block: (B, num_target_blocks)
            block_steps = torch.zeros(B, num_target_blocks, device=device)
            block_done = torch.zeros(B, num_target_blocks, dtype=torch.bool, device=device)

            for step in range(1, block_size + 1):
                if not remaining_blk.any():
                    break

                logits = model.forward_for_eval(eval_inputs, targets_batch, attn_mask=attn_mask)
                probs = torch.softmax(logits.float(), dim=-1)
                argmax_probs = probs.max(dim=-1).values  # (B, L)

                # Reshape to block view: (B, num_blocks, block_size)
                argmax_probs_blk = argmax_probs.view(B, num_blocks, block_size)

                # Positions passing threshold (only within remaining)
                decode_blk = remaining_blk & (argmax_probs_blk > threshold)

                # Fallback: for blocks with remaining positions but none passing threshold,
                # decode the most confident remaining position
                has_remaining = remaining_blk[:, 1:, :].any(dim=-1)  # (B, num_target_blocks)
                has_decoded = decode_blk[:, 1:, :].any(dim=-1)       # (B, num_target_blocks)
                needs_fallback = has_remaining & ~has_decoded          # (B, num_target_blocks)

                if needs_fallback.any():
                    # For fallback blocks: mask out non-remaining positions, find argmax
                    fb_probs = argmax_probs_blk[:, 1:, :].clone()     # (B, num_target_blocks, block_size)
                    fb_probs[~remaining_blk[:, 1:, :]] = -1.0
                    fb_best = fb_probs.argmax(dim=-1)                  # (B, num_target_blocks)
                    # One-hot for the best position per block
                    fb_onehot = F.one_hot(fb_best, block_size).bool()  # (B, num_target_blocks, block_size)
                    # Only apply fallback where needed
                    fb_mask = needs_fallback.unsqueeze(-1) & fb_onehot
                    decode_blk[:, 1:, :] |= fb_mask

                # Flatten back to (B, L) — use model's argmax prediction (non-oracle)
                decode_flat = decode_blk.view(B, L)
                eval_inputs[decode_flat] = logits.argmax(dim=-1)[decode_flat]
                remaining_blk[decode_blk] = False

                # Update block_steps for blocks that just finished
                newly_done = has_remaining & ~remaining_blk[:, 1:, :].any(dim=-1) & ~block_done
                block_steps[newly_done] = step
                block_done |= newly_done

        total_steps += block_steps.sum().item()
        total_blocks += B * num_target_blocks

    avg_steps = total_steps / total_blocks if total_blocks > 0 else 0.0
    return {"threshold": threshold, "avg_steps": avg_steps, "total_blocks": total_blocks}


def _eval_threshold_decode_two_tier(
    model, cached_batches, block_size,
    attn_mask, device, autocast_ctx, mask_token_id,
    tau1, tau2, base, mask_sub_token, target_length,
):
    """
    Two-tier threshold decode for BD3-LM-Prime.

    Three states per position:
    - 0: fully masked (both sub-tokens = mask_sub_token)
    - 1: half-decoded (sub-token 0 revealed, sub-token 1 = mask_sub_token)
    - 2: fully decoded (both sub-tokens clean)

    High confidence (> tau1): full decode.
    Medium confidence (tau2 < prob <= tau1, state==0): half decode (reveal MSB only).
    Half-decoded positions promoted to full on subsequent passes with consistency filtering.

    Args:
        model: BD3-LM-Prime model (must have forward_for_eval_sub)
        cached_batches: list of (targets,) tuples
        block_size: block size
        attn_mask: attention mask
        device: device
        autocast_ctx: autocast context
        mask_token_id: MASK token id (original token space)
        tau1: high confidence threshold
        tau2: low confidence threshold
        base: sub-token base
        mask_sub_token: sub-token mask id
        target_length: number of sub-tokens per token (l)

    Returns:
        {"threshold_high": tau1, "threshold_low": tau2, "avg_steps": float, "total_blocks": int}
    """
    l = target_length
    pure_vocab_size = model.config.pure_vocab_size

    # Precompute: token -> sub-token 0 mapping for consistency filtering
    token_to_s0 = torch.arange(pure_vocab_size, device=device) // base  # (pure_vocab_size,)

    total_steps = 0.0
    total_blocks = 0

    for (targets_batch,) in cached_batches:
        B, L = targets_batch.shape
        num_blocks = L // block_size
        num_target_blocks = num_blocks - 1  # skip block 0

        with autocast_ctx:
            # Sub-token level inputs: (B, L * l)
            eval_sub = torch.full((B, L * l), mask_sub_token, dtype=torch.long, device=device)

            # Set block 0 to clean sub-tokens
            targets_sub = prime_encode(targets_batch, base, l,
                                       mask_token_id=mask_token_id,
                                       mask_sub_token=mask_sub_token)  # (B, L*l)
            eval_sub[:, :block_size * l] = targets_sub[:, :block_size * l]

            # State tracking: (B, num_blocks, block_size) with values {0, 1, 2}
            state_blk = torch.zeros(B, num_blocks, block_size, dtype=torch.long, device=device)
            state_blk[:, 0, :] = 2  # block 0 is fully decoded

            # Precompute position -> sub-token index mappings
            pos_L = torch.arange(L, device=device)
            sub0_idx = pos_L * l        # (L,) indices of sub-token 0
            sub1_idx = pos_L * l + 1    # (L,) indices of sub-token 1

            # Track steps per block
            block_steps = torch.zeros(B, num_target_blocks, device=device)
            block_done = torch.zeros(B, num_target_blocks, dtype=torch.bool, device=device)

            for step in range(1, block_size + 1):
                # Check if all target blocks are done
                remaining = (state_blk[:, 1:, :] < 2)  # (B, num_target_blocks, block_size)
                if not remaining.any():
                    break
                had_remaining_blk = remaining.any(dim=-1)  # (B, num_target_blocks) — snapshot before updates

                # Forward pass with sub-token inputs
                logits = model.forward_for_eval_sub(eval_sub, targets_batch, attn_mask=attn_mask)
                probs = torch.softmax(logits.float(), dim=-1)  # (B, L, pure_vocab_size)
                argmax_probs, argmax_tokens = probs.max(dim=-1)  # (B, L)

                # Reshape to block view: (B, num_blocks, block_size)
                argmax_probs_blk = argmax_probs.view(B, num_blocks, block_size)

                # === Decision masks (all over target blocks only) ===
                state_tgt = state_blk[:, 1:, :]  # (B, num_target_blocks, block_size)
                probs_tgt = argmax_probs_blk[:, 1:, :]

                full_from_0 = (state_tgt == 0) & (probs_tgt > tau1)
                half_to_full_mask = (state_tgt == 1) & (probs_tgt > tau1)
                half_decode_mask = (state_tgt == 0) & (probs_tgt > tau2) & (probs_tgt <= tau1)

                # Helper: map (batch, target_block, pos) -> flat L-space position
                # target block i -> actual block i+1 -> flat_pos = (i+1)*block_size + p
                def _to_flat(batch_idx, tbi_idx, p_idx):
                    return (tbi_idx + 1) * block_size + p_idx

                # === Full decode from state 0 (vectorized) ===
                if full_from_0.any():
                    bi, tbi, pi = full_from_0.nonzero(as_tuple=True)
                    fp = _to_flat(bi, tbi, pi)
                    tokens = argmax_tokens[bi, fp]
                    eval_sub[bi, sub0_idx[fp]] = tokens // base
                    eval_sub[bi, sub1_idx[fp]] = tokens % base
                    state_blk[bi, tbi + 1, pi] = 2

                # === Half→full with consistency filter (vectorized) ===
                if half_to_full_mask.any():
                    bi, tbi, pi = half_to_full_mask.nonzero(as_tuple=True)
                    fp = _to_flat(bi, tbi, pi)
                    committed_s0 = eval_sub[bi, sub0_idx[fp]]  # (N,)
                    # Batch consistency filter: mask invalid tokens per position
                    pos_logits = logits[bi, fp, :].clone()  # (N, pure_vocab_size)
                    invalid = token_to_s0.unsqueeze(0) != committed_s0.unsqueeze(1)  # (N, V)
                    pos_logits[invalid] = float('-inf')
                    consistent_tokens = pos_logits.argmax(dim=-1)  # (N,)
                    eval_sub[bi, sub0_idx[fp]] = consistent_tokens // base
                    eval_sub[bi, sub1_idx[fp]] = consistent_tokens % base
                    state_blk[bi, tbi + 1, pi] = 2

                # === Half decode: write sub-token 0 only (vectorized) ===
                if half_decode_mask.any():
                    bi, tbi, pi = half_decode_mask.nonzero(as_tuple=True)
                    fp = _to_flat(bi, tbi, pi)
                    tokens = argmax_tokens[bi, fp]
                    eval_sub[bi, sub0_idx[fp]] = tokens // base
                    # sub1 stays as mask_sub_token
                    state_blk[bi, tbi + 1, pi] = 1

                # === Fallback: full-decode most confident remaining if no full decode happened ===
                any_full_decode = full_from_0 | half_to_full_mask
                has_remaining_after = (state_blk[:, 1:, :] < 2)
                has_remaining_blk = has_remaining_after.any(dim=-1)
                needs_fallback = has_remaining_blk & ~any_full_decode.any(dim=-1)

                if needs_fallback.any():
                    fb_probs = argmax_probs_blk[:, 1:, :].clone()
                    fb_probs[~has_remaining_after] = -1.0
                    fb_best = fb_probs.argmax(dim=-1)  # (B, num_target_blocks)

                    fb_bi, fb_tbi = needs_fallback.nonzero(as_tuple=True)
                    fb_pi = fb_best[fb_bi, fb_tbi]
                    fb_fp = _to_flat(fb_bi, fb_tbi, fb_pi)
                    fb_state = state_blk[fb_bi, fb_tbi + 1, fb_pi]

                    # Split into state==0 (direct full) and state==1 (consistency filter)
                    is_s0 = (fb_state == 0)
                    is_s1 = (fb_state == 1)

                    if is_s0.any():
                        s0_bi, s0_fp = fb_bi[is_s0], fb_fp[is_s0]
                        s0_tbi, s0_pi = fb_tbi[is_s0], fb_pi[is_s0]
                        tokens = argmax_tokens[s0_bi, s0_fp]
                        eval_sub[s0_bi, sub0_idx[s0_fp]] = tokens // base
                        eval_sub[s0_bi, sub1_idx[s0_fp]] = tokens % base
                        state_blk[s0_bi, s0_tbi + 1, s0_pi] = 2

                    if is_s1.any():
                        s1_bi, s1_fp = fb_bi[is_s1], fb_fp[is_s1]
                        s1_tbi, s1_pi = fb_tbi[is_s1], fb_pi[is_s1]
                        committed_s0 = eval_sub[s1_bi, sub0_idx[s1_fp]]
                        pos_logits = logits[s1_bi, s1_fp, :].clone()
                        invalid = token_to_s0.unsqueeze(0) != committed_s0.unsqueeze(1)
                        pos_logits[invalid] = float('-inf')
                        consistent_tokens = pos_logits.argmax(dim=-1)
                        eval_sub[s1_bi, sub0_idx[s1_fp]] = consistent_tokens // base
                        eval_sub[s1_bi, sub1_idx[s1_fp]] = consistent_tokens % base
                        state_blk[s1_bi, s1_tbi + 1, s1_pi] = 2

                # Update block_steps for blocks that just finished
                newly_done = had_remaining_blk & ~(state_blk[:, 1:, :] < 2).any(dim=-1) & ~block_done
                block_steps[newly_done] = step
                block_done |= newly_done

        total_steps += block_steps.sum().item()
        total_blocks += B * num_target_blocks

    avg_steps = total_steps / total_blocks if total_blocks > 0 else 0.0
    return {"threshold_high": tau1, "threshold_low": tau2, "avg_steps": avg_steps, "total_blocks": total_blocks}


def eval_bd3lm_threshold(
    model, val_loader, block_size, num_batches,
    attn_mask, device, autocast_ctx, mask_token_id,
    thresholds=None, two_tier=False, tau2_delta=0.2,
):
    """
    Run threshold-based parallel decoding evaluation across multiple thresholds.

    Caches batches once, then sweeps over all threshold values.

    Args:
        model: BD3LM model
        val_loader: validation data loader
        block_size: block size
        num_batches: number of batches to evaluate
        attn_mask: attention mask
        device: device
        autocast_ctx: autocast context
        mask_token_id: MASK token id
        thresholds: list of threshold values (default: 0.0 to 1.0 in steps of 0.1)
        two_tier: if True, use two-tier threshold decode for BD3-LM-Prime
        tau2_delta: delta between tau1 and tau2 (tau2 = max(0, tau1 - tau2_delta))

    Returns:
        {"stage": "bd3lm", "threshold_decode": {tau: {"avg_steps": X, "total_blocks": N}}}
    """
    if thresholds is None:
        thresholds = [i / 10.0 for i in range(11)]  # 0.0, 0.1, ..., 1.0

    was_training = model.training
    model.eval()

    # Cache batches (only need targets)
    cached_batches = []
    for _ in range(num_batches):
        _, targets_batch, _, _ = next(val_loader)
        cached_batches.append((targets_batch,))

    results = {}
    with torch.no_grad():
        if two_tier:
            # Two-tier threshold decode for BD3-LM-Prime
            # Skip tau1 <= tau2_delta: tau2 would be 0, making two-tier degenerate
            config = model.config
            for tau1 in thresholds:
                if tau1 <= tau2_delta:
                    continue
                tau2 = max(0.0, tau1 - tau2_delta)
                result = _eval_threshold_decode_two_tier(
                    model, cached_batches, block_size,
                    attn_mask, device, autocast_ctx, mask_token_id,
                    tau1, tau2, config.base, config.mask_sub_token, config.target_length,
                )
                results[tau1] = result
        else:
            for tau in thresholds:
                result = _eval_threshold_decode(
                    model, cached_batches, block_size,
                    attn_mask, device, autocast_ctx, mask_token_id, tau,
                )
                results[tau] = result

    if was_training:
        model.train()
    return {"stage": "bd3lm", "threshold_decode": results}


def _build_ltr_sub_input(targets_sub, block_size, l, mask_sub_token, k, half_state, half_s0):
    """Build sub-token input for L2R measurement forward.

    Args:
        targets_sub: (B, L*l) clean sub-token IDs
        block_size: block size
        l: target_length (sub-tokens per token)
        mask_sub_token: mask sub-token ID
        k: current position being decoded (0-indexed within block)
        half_state: (B, num_blocks, block_size) bool — True if position is half-decoded
        half_s0: (B, num_blocks, block_size) long — committed MSB sub-token values

    Returns:
        eval_sub: (B, L*l) sub-token input
            - Block 0: clean from targets_sub
            - Positions [0..k-1] in blocks 1+: clean (teacher forcing)
            - Position k in blocks 1+: MASK (both sub-tokens)
            - Positions [k+1..end] in blocks 1+: half-decoded or fully masked
    """
    B = targets_sub.shape[0]
    L_sub = targets_sub.shape[1]
    L = L_sub // l
    num_blocks = L // block_size
    device = targets_sub.device

    # Start all masked
    eval_sub = torch.full((B, L_sub), mask_sub_token, dtype=torch.long, device=device)

    # Block 0: fully clean
    eval_sub[:, :block_size * l] = targets_sub[:, :block_size * l]

    # Positions [0..k-1] in blocks 1+: ground truth (teacher forcing)
    if k > 0:
        # Build mask for prefix positions in all target blocks
        pos_in_block = torch.arange(block_size, device=device)
        block_starts = torch.arange(1, num_blocks, device=device) * block_size  # (num_target_blocks,)
        # Flat token positions for prefix: block_start + p for p in [0, k)
        prefix_positions = (block_starts.unsqueeze(1) + pos_in_block[:k].unsqueeze(0)).reshape(-1)  # (num_target_blocks * k,)
        # Sub-token indices
        for si in range(l):
            sub_idx = prefix_positions * l + si
            eval_sub[:, sub_idx] = targets_sub[:, sub_idx]

    # Positions [k+1..end] in blocks 1+: apply half-decodes where half_state is True
    if k < block_size - 1:
        # half_state: (B, num_blocks, block_size)
        # Only target blocks (1:), future positions (k+1:)
        for p in range(k + 1, block_size):
            # half_state[:, 1:, p] gives (B, num_target_blocks) mask
            hs = half_state[:, 1:, p]  # (B, num_target_blocks)
            if hs.any():
                block_starts = torch.arange(1, num_blocks, device=device)  # (num_target_blocks,)
                flat_pos = block_starts + p  # (num_target_blocks,) token positions
                sub0 = flat_pos * l  # sub-token 0 indices

                # Expand for batch: (B, num_target_blocks)
                b_idx, tb_idx = hs.nonzero(as_tuple=True)
                eval_sub[b_idx, sub0[tb_idx]] = half_s0[b_idx, tb_idx + 1, p]

    return eval_sub


def _build_ltr_sub_input_clean(targets_sub, block_size, l, mask_sub_token, k):
    """Build clean sub-token input for L2R lookahead forward (no half-decodes).

    Args:
        targets_sub: (B, L*l) clean sub-token IDs
        block_size: block size
        l: target_length
        mask_sub_token: mask sub-token ID
        k: current position (positions [0..k-1] are ground truth prefix)

    Returns:
        eval_sub: (B, L*l) sub-token input
            - Block 0: clean
            - Positions [0..k-1] in blocks 1+: clean
            - Positions [k..end] in blocks 1+: all MASK
    """
    B = targets_sub.shape[0]
    L_sub = targets_sub.shape[1]
    L = L_sub // l
    num_blocks = L // block_size
    device = targets_sub.device

    # Start all masked
    eval_sub = torch.full((B, L_sub), mask_sub_token, dtype=torch.long, device=device)

    # Block 0: fully clean
    eval_sub[:, :block_size * l] = targets_sub[:, :block_size * l]

    # Positions [0..k-1] in blocks 1+: ground truth
    if k > 0:
        pos_in_block = torch.arange(block_size, device=device)
        block_starts = torch.arange(1, num_blocks, device=device) * block_size
        prefix_positions = (block_starts.unsqueeze(1) + pos_in_block[:k].unsqueeze(0)).reshape(-1)
        for si in range(l):
            sub_idx = prefix_positions * l + si
            eval_sub[:, sub_idx] = targets_sub[:, sub_idx]

    return eval_sub


def _eval_ltr_sub_token_lookahead(
    model, cached_batches, block_size,
    attn_mask, device, autocast_ctx, mask_token_id, threshold,
):
    """Teacher-forced L2R eval with sub-token lookahead half-decoding.

    At each step k, two forwards:
    1. Measurement: [gt_prefix, half?/mask future] → record loss/accuracy at position k
    2. Lookahead: [gt_prefix, clean MASK future] → decide half-decodes for next step

    Args:
        model: BD3-LM-Prime model
        cached_batches: list of (targets,) tuples
        block_size: block size
        attn_mask: attention mask
        device: device
        autocast_ctx: autocast context
        mask_token_id: MASK token id
        threshold: τ for half-decode decisions (float('inf') = baseline, no half-decodes)

    Returns:
        {"overall_loss": float, "overall_ppl": float, "overall_accuracy": float,
         "positions": {k: {"loss", "ppl", "accuracy", "argmax_prob"}}}
    """
    config = model.config
    base = config.base
    l = config.target_length
    mask_sub_token = config.mask_sub_token

    nll_data = {k: {"nll": 0.0, "entropy": 0.0, "argmax_prob": 0.0, "tokens": 0} for k in range(block_size)}
    acc_data = {k: {"correct": 0, "total": 0} for k in range(block_size)}

    is_baseline = threshold == float('inf')

    for (targets_batch,) in cached_batches:
        B, L = targets_batch.shape
        num_blocks = L // block_size

        with autocast_ctx:
            targets_sub = prime_encode(targets_batch, base, l,
                                       mask_token_id=mask_token_id,
                                       mask_sub_token=mask_sub_token)

            # Half-decode state (reset per batch, accumulates across k)
            half_state = torch.zeros(B, num_blocks, block_size, dtype=torch.bool, device=device)
            half_s0 = torch.zeros(B, num_blocks, block_size, dtype=torch.long, device=device)

            for k in range(block_size):
                # --- Forward 1 (measurement) ---
                if is_baseline:
                    eval_sub_meas = _build_ltr_sub_input_clean(targets_sub, block_size, l, mask_sub_token, k)
                else:
                    eval_sub_meas = _build_ltr_sub_input(targets_sub, block_size, l, mask_sub_token, k, half_state, half_s0)

                logits = model.forward_for_eval_sub(eval_sub_meas, targets_batch, attn_mask=attn_mask)
                log_probs = F.log_softmax(logits.float(), dim=-1)
                probs = log_probs.exp()
                entropy = -(probs * log_probs).sum(dim=-1)
                argmax_prob_all = probs.max(dim=-1)[0]

                # Record metrics at position k in blocks 1+
                for block_idx in range(1, num_blocks):
                    pos_in_seq = block_idx * block_size + k
                    nll = -log_probs[:, pos_in_seq, :].gather(
                        -1, targets_batch[:, pos_in_seq].unsqueeze(-1)
                    ).squeeze(-1)
                    nll_data[k]["nll"] += nll.sum().item()
                    nll_data[k]["entropy"] += entropy[:, pos_in_seq].sum().item()
                    nll_data[k]["argmax_prob"] += argmax_prob_all[:, pos_in_seq].sum().item()
                    nll_data[k]["tokens"] += B

                    preds = logits[:, pos_in_seq, :].argmax(dim=-1)
                    acc_data[k]["correct"] += (preds == targets_batch[:, pos_in_seq]).sum().item()
                    acc_data[k]["total"] += B

                # --- Forward 2 (lookahead) — skip if baseline or last position ---
                if not is_baseline and k < block_size - 1:
                    eval_sub_look = _build_ltr_sub_input_clean(targets_sub, block_size, l, mask_sub_token, k)
                    logits_look = model.forward_for_eval_sub(eval_sub_look, targets_batch, attn_mask=attn_mask)
                    probs_look = torch.softmax(logits_look.float(), dim=-1)
                    argmax_probs_look, argmax_tokens_look = probs_look.max(dim=-1)

                    # Decide half-decodes for future positions [k+1, block_size-1]
                    for p in range(k + 1, block_size):
                        for block_idx in range(1, num_blocks):
                            pos_in_seq = block_idx * block_size + p
                            # Vectorize over batch
                            should_half = (argmax_probs_look[:, pos_in_seq] > threshold) & ~half_state[:, block_idx, p]
                            if should_half.any():
                                half_state[should_half, block_idx, p] = True
                                half_s0[should_half, block_idx, p] = argmax_tokens_look[should_half, pos_in_seq] // base

    # Aggregate results
    total_nll = sum(nll_data[k]["nll"] for k in range(block_size))
    total_tokens = sum(nll_data[k]["tokens"] for k in range(block_size))
    total_correct = sum(acc_data[k]["correct"] for k in range(block_size))
    total_entropy = sum(nll_data[k]["entropy"] for k in range(block_size))
    total_argmax_prob = sum(nll_data[k]["argmax_prob"] for k in range(block_size))

    overall_loss = total_nll / total_tokens if total_tokens > 0 else 0.0
    result = {
        "overall_loss": overall_loss,
        "overall_ppl": torch.exp(torch.tensor(overall_loss)).item(),
        "overall_entropy_ppl": torch.exp(torch.tensor(total_entropy / total_tokens)).item() if total_tokens > 0 else 0.0,
        "overall_accuracy": total_correct / total_tokens if total_tokens > 0 else 0.0,
        "overall_argmax_prob": total_argmax_prob / total_tokens if total_tokens > 0 else 0.0,
        "positions": {},
    }

    for k in range(block_size):
        tokens_k = nll_data[k]["tokens"]
        loss_k = nll_data[k]["nll"] / tokens_k if tokens_k > 0 else 0.0
        entropy_k = nll_data[k]["entropy"] / tokens_k if tokens_k > 0 else 0.0
        acc_k = acc_data[k]["correct"] / acc_data[k]["total"] if acc_data[k]["total"] > 0 else 0.0
        result["positions"][k] = {
            "loss": loss_k,
            "ppl": torch.exp(torch.tensor(loss_k)).item(),
            "entropy_ppl": torch.exp(torch.tensor(entropy_k)).item(),
            "argmax_prob": nll_data[k]["argmax_prob"] / tokens_k if tokens_k > 0 else 0.0,
            "accuracy": acc_k,
        }

    return result


def eval_bd3lm_ltr_lookahead(
    model, val_loader, block_size, num_batches,
    attn_mask, device, autocast_ctx, mask_token_id,
    thresholds=None,
):
    """Run sub-token lookahead L2R eval across multiple thresholds.

    Caches batches once, then sweeps thresholds. Includes baseline (τ=∞).

    Returns:
        {"stage": "bd3lm", "ltr_sub_lookahead": {tau: {results}}}
    """
    if thresholds is None:
        thresholds = [i / 10.0 for i in range(11)]

    was_training = model.training
    model.eval()

    cached_batches = []
    for _ in range(num_batches):
        _, targets_batch, _, _ = next(val_loader)
        cached_batches.append((targets_batch,))

    results = {}
    with torch.no_grad():
        # Baseline first (τ=∞, no half-decodes)
        result = _eval_ltr_sub_token_lookahead(
            model, cached_batches, block_size,
            attn_mask, device, autocast_ctx, mask_token_id, float('inf'),
        )
        results["inf"] = result

        for tau in thresholds:
            result = _eval_ltr_sub_token_lookahead(
                model, cached_batches, block_size,
                attn_mask, device, autocast_ctx, mask_token_id, tau,
            )
            results[tau] = result

    if was_training:
        model.train()
    return {"stage": "bd3lm", "ltr_sub_lookahead": results}


def _eval_ltr_sub_token_lookahead_fresh(
    model, cached_batches, block_size,
    attn_mask, device, autocast_ctx, mask_token_id, threshold,
):
    """Teacher-forced L2R eval with FRESH sub-token lookahead half-decoding.

    Like _eval_ltr_sub_token_lookahead, but re-derives half-decode decisions
    at each step k from scratch using the current ground-truth prefix,
    rather than inheriting accumulated half_state/half_s0 from earlier steps.
    This removes trajectory inertia from inherited sub-token reveals.

    At each step k, two forwards:
    1. Lookahead: [gt_prefix 0..k-1, MASK k..end] → decide fresh half-decodes
    2. Measurement: [gt_prefix, fresh half?/mask future] → record loss/accuracy at position k

    Args:
        model: BD3-LM-Prime model
        cached_batches: list of (targets,) tuples
        block_size: block size
        attn_mask: attention mask
        device: device
        autocast_ctx: autocast context
        mask_token_id: MASK token id
        threshold: τ for half-decode decisions (float('inf') = baseline, no half-decodes)

    Returns:
        {"overall_loss": float, "overall_ppl": float, "overall_accuracy": float,
         "positions": {k: {"loss", "ppl", "accuracy", "argmax_prob"}}}
    """
    config = model.config
    base = config.base
    l = config.target_length
    mask_sub_token = config.mask_sub_token

    nll_data = {k: {"nll": 0.0, "entropy": 0.0, "argmax_prob": 0.0, "tokens": 0} for k in range(block_size)}
    acc_data = {k: {"correct": 0, "total": 0} for k in range(block_size)}

    is_baseline = threshold == float('inf')

    for (targets_batch,) in cached_batches:
        B, L = targets_batch.shape
        num_blocks = L // block_size

        with autocast_ctx:
            targets_sub = prime_encode(targets_batch, base, l,
                                       mask_token_id=mask_token_id,
                                       mask_sub_token=mask_sub_token)

            for k in range(block_size):
                # --- Forward 1 (fresh lookahead) — derive half-decodes from current prefix ---
                # Fresh half_state/half_s0: zeroed each k, not accumulated
                fresh_half_state = torch.zeros(B, num_blocks, block_size, dtype=torch.bool, device=device)
                fresh_half_s0 = torch.zeros(B, num_blocks, block_size, dtype=torch.long, device=device)

                if not is_baseline and k < block_size - 1:
                    eval_sub_look = _build_ltr_sub_input_clean(targets_sub, block_size, l, mask_sub_token, k)
                    logits_look = model.forward_for_eval_sub(eval_sub_look, targets_batch, attn_mask=attn_mask)
                    probs_look = torch.softmax(logits_look.float(), dim=-1)
                    argmax_probs_look, argmax_tokens_look = probs_look.max(dim=-1)

                    # Decide half-decodes for future positions [k+1, block_size-1]
                    for p in range(k + 1, block_size):
                        for block_idx in range(1, num_blocks):
                            pos_in_seq = block_idx * block_size + p
                            should_half = argmax_probs_look[:, pos_in_seq] > threshold
                            if should_half.any():
                                fresh_half_state[should_half, block_idx, p] = True
                                fresh_half_s0[should_half, block_idx, p] = argmax_tokens_look[should_half, pos_in_seq] // base

                # --- Forward 2 (measurement with fresh half-decodes) ---
                if is_baseline:
                    eval_sub_meas = _build_ltr_sub_input_clean(targets_sub, block_size, l, mask_sub_token, k)
                else:
                    eval_sub_meas = _build_ltr_sub_input(targets_sub, block_size, l, mask_sub_token, k, fresh_half_state, fresh_half_s0)

                logits = model.forward_for_eval_sub(eval_sub_meas, targets_batch, attn_mask=attn_mask)
                log_probs = F.log_softmax(logits.float(), dim=-1)
                probs = log_probs.exp()
                entropy = -(probs * log_probs).sum(dim=-1)
                argmax_prob_all = probs.max(dim=-1)[0]

                # Record metrics at position k in blocks 1+
                for block_idx in range(1, num_blocks):
                    pos_in_seq = block_idx * block_size + k
                    nll = -log_probs[:, pos_in_seq, :].gather(
                        -1, targets_batch[:, pos_in_seq].unsqueeze(-1)
                    ).squeeze(-1)
                    nll_data[k]["nll"] += nll.sum().item()
                    nll_data[k]["entropy"] += entropy[:, pos_in_seq].sum().item()
                    nll_data[k]["argmax_prob"] += argmax_prob_all[:, pos_in_seq].sum().item()
                    nll_data[k]["tokens"] += B

                    preds = logits[:, pos_in_seq, :].argmax(dim=-1)
                    acc_data[k]["correct"] += (preds == targets_batch[:, pos_in_seq]).sum().item()
                    acc_data[k]["total"] += B

    # Aggregate results
    total_nll = sum(nll_data[k]["nll"] for k in range(block_size))
    total_tokens = sum(nll_data[k]["tokens"] for k in range(block_size))
    total_correct = sum(acc_data[k]["correct"] for k in range(block_size))
    total_entropy = sum(nll_data[k]["entropy"] for k in range(block_size))
    total_argmax_prob = sum(nll_data[k]["argmax_prob"] for k in range(block_size))

    overall_loss = total_nll / total_tokens if total_tokens > 0 else 0.0
    result = {
        "overall_loss": overall_loss,
        "overall_ppl": torch.exp(torch.tensor(overall_loss)).item(),
        "overall_entropy_ppl": torch.exp(torch.tensor(total_entropy / total_tokens)).item() if total_tokens > 0 else 0.0,
        "overall_accuracy": total_correct / total_tokens if total_tokens > 0 else 0.0,
        "overall_argmax_prob": total_argmax_prob / total_tokens if total_tokens > 0 else 0.0,
        "positions": {},
    }

    for k in range(block_size):
        tokens_k = nll_data[k]["tokens"]
        loss_k = nll_data[k]["nll"] / tokens_k if tokens_k > 0 else 0.0
        entropy_k = nll_data[k]["entropy"] / tokens_k if tokens_k > 0 else 0.0
        acc_k = acc_data[k]["correct"] / acc_data[k]["total"] if acc_data[k]["total"] > 0 else 0.0
        result["positions"][k] = {
            "loss": loss_k,
            "ppl": torch.exp(torch.tensor(loss_k)).item(),
            "entropy_ppl": torch.exp(torch.tensor(entropy_k)).item(),
            "argmax_prob": nll_data[k]["argmax_prob"] / tokens_k if tokens_k > 0 else 0.0,
            "accuracy": acc_k,
        }

    return result


def eval_bd3lm_ltr_lookahead_fresh(
    model, val_loader, block_size, num_batches,
    attn_mask, device, autocast_ctx, mask_token_id,
    thresholds=None,
):
    """Run fresh sub-token lookahead L2R eval across multiple thresholds.

    Like eval_bd3lm_ltr_lookahead, but re-derives half-decode decisions
    at each step from the current prefix (no inherited state / inertia).

    Caches batches once, then sweeps thresholds. Includes baseline (τ=∞).

    Returns:
        {"stage": "bd3lm", "ltr_sub_lookahead": {tau: {results}}}
    """
    if thresholds is None:
        thresholds = [i / 10.0 for i in range(11)]

    was_training = model.training
    model.eval()

    cached_batches = []
    for _ in range(num_batches):
        _, targets_batch, _, _ = next(val_loader)
        cached_batches.append((targets_batch,))

    results = {}
    with torch.no_grad():
        # Baseline first (τ=∞, no half-decodes)
        result = _eval_ltr_sub_token_lookahead_fresh(
            model, cached_batches, block_size,
            attn_mask, device, autocast_ctx, mask_token_id, float('inf'),
        )
        results["inf"] = result

        for tau in thresholds:
            result = _eval_ltr_sub_token_lookahead_fresh(
                model, cached_batches, block_size,
                attn_mask, device, autocast_ctx, mask_token_id, tau,
            )
            results[tau] = result

    if was_training:
        model.train()
    return {"stage": "bd3lm", "ltr_sub_lookahead": results}


def _prepare_eval_batch(targets, mask_token_id):
    """
    Prepare evaluation batch where ALL positions in xt are masked.

    Args:
        targets: (B, L) clean target tokens (already on device)
        mask_token_id: token id for MASK

    Returns:
        inputs: (B, L) all MASK tokens
        targets: (B, L) clean targets (unchanged)
    """
    B, L = targets.shape
    # All positions in xt are masked
    inputs = torch.full((B, L), mask_token_id, dtype=torch.long, device=targets.device)
    return inputs, targets


def _prepare_eval_batch_with_suffix(targets, mask_token_id, block_size, pred_position, num_suffix_clear):
    """
    Prepare evaluation batch with N suffix positions revealed (not masked).

    For each block, positions 0 to pred_position are masked, and positions
    (pred_position+1) to (pred_position+num_suffix_clear) are revealed (clean).
    Remaining positions after the suffix are still masked.

    Args:
        targets: (B, L) clean target tokens (already on device)
        mask_token_id: token id for MASK
        block_size: size of each block
        pred_position: the position being predicted (0-indexed within block)
        num_suffix_clear: number of suffix positions to reveal (0 = all masked)

    Returns:
        inputs: (B, L) tokens with suffix positions revealed
        targets: (B, L) clean targets (unchanged)

    Example (block_size=4, pred_position=0, num_suffix_clear=2):
        Block pattern: [MASK, clean, clean, MASK]
                        ^pred  ^suf1  ^suf2  ^still masked
    """
    B, L = targets.shape
    num_blocks = L // block_size

    # Start with all masked
    inputs = torch.full((B, L), mask_token_id, dtype=torch.long, device=targets.device)

    # Reveal suffix positions in each block
    if num_suffix_clear > 0:
        for block_idx in range(num_blocks):
            block_start = block_idx * block_size
            # Suffix positions: pred_position+1, pred_position+2, ..., pred_position+num_suffix_clear
            for suffix_offset in range(1, num_suffix_clear + 1):
                suffix_pos = pred_position + suffix_offset
                if suffix_pos < block_size:  # don't go beyond block boundary
                    abs_pos = block_start + suffix_pos
                    inputs[:, abs_pos] = targets[:, abs_pos]

    return inputs, targets


def _eval_target_shift_mode(
    model, val_loader, block_size, target_shift,
    num_batches, attn_mask, device, autocast_ctx, mask_token_id
):
    """
    Evaluate in target_shift mode: compute loss only at position (target_shift-1).
    Skip block 0, compute from blocks 1 onwards.

    Also computes suffix metrics: with N suffix positions revealed.
    For target_shift=1 (pred pos 0), max suffix = block_size - 1.
    For target_shift=4 (pred pos 3, last), max suffix = 0.
    """
    position = target_shift - 1  # convert to 0-indexed
    max_suffix = block_size - 1 - position  # how many suffix positions available

    # Accumulators for each suffix count (0 = all masked, 1 = 1 suffix clear, etc.)
    nll_by_suffix = {s: 0.0 for s in range(max_suffix + 1)}
    entropy_by_suffix = {s: 0.0 for s in range(max_suffix + 1)}
    argmax_prob_by_suffix = {s: 0.0 for s in range(max_suffix + 1)}
    tokens_by_suffix = {s: 0 for s in range(max_suffix + 1)}

    # Collect all batches first (we need to iterate multiple times for different suffix counts)
    all_targets = []
    for batch_idx in range(num_batches):
        _, targets_batch, _, _ = next(val_loader)
        all_targets.append(targets_batch)

    # Evaluate for each suffix count
    for num_suffix in range(max_suffix + 1):
        for targets_batch in all_targets:
            B, L = targets_batch.shape
            num_blocks = L // block_size

            # Prepare inputs with appropriate suffix clearing
            if num_suffix == 0:
                inputs, targets = _prepare_eval_batch(targets_batch, mask_token_id)
            else:
                inputs, targets = _prepare_eval_batch_with_suffix(
                    targets_batch, mask_token_id, block_size, position, num_suffix
                )

            with autocast_ctx:
                # Forward pass
                logits = model.forward_for_eval(inputs, targets, attn_mask=attn_mask)

                # Compute log probabilities
                log_probs = F.log_softmax(logits.float(), dim=-1)

                # Gather log probs for target tokens
                target_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

                # Entropy and argmax prob over the full distribution
                probs = log_probs.exp()
                entropy = -(probs * log_probs).sum(dim=-1)  # (B, L)
                argmax_prob = probs.max(dim=-1)[0]           # (B, L)

                # Compute metrics for the specific position in each block
                # Skip block 0, compute from blocks 1 onwards
                for block_idx in range(1, num_blocks):
                    pos_in_seq = block_idx * block_size + position
                    nll = -target_log_probs[:, pos_in_seq]
                    nll_by_suffix[num_suffix] += nll.sum().item()
                    entropy_by_suffix[num_suffix] += entropy[:, pos_in_seq].sum().item()
                    argmax_prob_by_suffix[num_suffix] += argmax_prob[:, pos_in_seq].sum().item()
                    tokens_by_suffix[num_suffix] += B

    # Build result dict
    result = {}

    # Core metrics (all masked)
    n = tokens_by_suffix[0]
    avg_loss = nll_by_suffix[0] / n if n > 0 else 0.0
    avg_entropy = entropy_by_suffix[0] / n if n > 0 else 0.0
    result["loss"] = avg_loss
    result["ppl"] = torch.exp(torch.tensor(avg_loss)).item()
    result["entropy_ppl"] = torch.exp(torch.tensor(avg_entropy)).item()
    result["argmax_prob"] = argmax_prob_by_suffix[0] / n if n > 0 else 0.0

    # Suffix metrics
    for num_suffix in range(1, max_suffix + 1):
        n_s = tokens_by_suffix[num_suffix]
        avg_loss_s = nll_by_suffix[num_suffix] / n_s if n_s > 0 else 0.0
        avg_entropy_s = entropy_by_suffix[num_suffix] / n_s if n_s > 0 else 0.0
        result[f"loss_{num_suffix}suffix"] = avg_loss_s
        result[f"ppl_{num_suffix}suffix"] = torch.exp(torch.tensor(avg_loss_s)).item()
        result[f"entropy_ppl_{num_suffix}suffix"] = torch.exp(torch.tensor(avg_entropy_s)).item()
        result[f"argmax_prob_{num_suffix}suffix"] = argmax_prob_by_suffix[num_suffix] / n_s if n_s > 0 else 0.0

    return result


def _eval_normal_mode(
    model, val_loader, block_size,
    num_batches, attn_mask, device, autocast_ctx, mask_token_id
):
    """
    Evaluate in normal mode: compute overall + per-position metrics.
    Skip block 0, compute from blocks 1 onwards.

    Also computes suffix metrics for each position that has suffixes:
    - Position 0: can have up to block_size-1 suffixes
    - Position 1: can have up to block_size-2 suffixes
    - ...
    - Position block_size-1: no suffixes
    """
    max_suffix = block_size - 1  # max possible suffix count (for position 0)

    # Collect all batches first (we need to iterate multiple times)
    all_targets = []
    for batch_idx in range(num_batches):
        _, targets_batch, _, _ = next(val_loader)
        all_targets.append(targets_batch)

    # Track metrics: nll_data[num_suffix][pos]
    # num_suffix=0 means all masked (original eval)
    nll_data = {
        s: {p: {"nll": 0.0, "entropy": 0.0, "argmax_prob": 0.0, "tokens": 0}
            for p in range(block_size)}
        for s in range(max_suffix + 1)
    }

    # Evaluate for each suffix count
    for num_suffix in range(max_suffix + 1):
        # For this suffix count, only evaluate positions that have enough suffixes
        # Position p has (block_size - 1 - p) suffixes available
        valid_positions = [p for p in range(block_size) if (block_size - 1 - p) >= num_suffix]

        if not valid_positions:
            continue

        for targets_batch in all_targets:
            B, L = targets_batch.shape
            num_blocks = L // block_size

            # For each valid position, prepare batch and compute NLL
            for pred_pos in valid_positions:
                if num_suffix == 0:
                    inputs, targets = _prepare_eval_batch(targets_batch, mask_token_id)
                else:
                    inputs, targets = _prepare_eval_batch_with_suffix(
                        targets_batch, mask_token_id, block_size, pred_pos, num_suffix
                    )

                with autocast_ctx:
                    # Forward pass
                    logits = model.forward_for_eval(inputs, targets, attn_mask=attn_mask)

                    # Compute log probabilities
                    log_probs = F.log_softmax(logits.float(), dim=-1)

                    # Gather log probs for target tokens
                    target_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

                    # Entropy and argmax prob over the full distribution
                    probs = log_probs.exp()
                    entropy = -(probs * log_probs).sum(dim=-1)  # (B, L)
                    argmax_prob = probs.max(dim=-1)[0]           # (B, L)

                    # Compute metrics at this position in each block (skip block 0)
                    for block_idx in range(1, num_blocks):
                        pos_in_seq = block_idx * block_size + pred_pos
                        nll = -target_log_probs[:, pos_in_seq]
                        nll_data[num_suffix][pred_pos]["nll"] += nll.sum().item()
                        nll_data[num_suffix][pred_pos]["entropy"] += entropy[:, pos_in_seq].sum().item()
                        nll_data[num_suffix][pred_pos]["argmax_prob"] += argmax_prob[:, pos_in_seq].sum().item()
                        nll_data[num_suffix][pred_pos]["tokens"] += B

    # Build position-centric result dict
    result = {}

    # Original metrics (all masked, num_suffix=0)
    total_nll = sum(nll_data[0][p]["nll"] for p in range(block_size))
    total_tokens = sum(nll_data[0][p]["tokens"] for p in range(block_size))
    result["overall_loss"] = total_nll / total_tokens if total_tokens > 0 else 0.0
    result["overall_ppl"] = torch.exp(torch.tensor(result["overall_loss"])).item()

    # Position-centric data: for each position, include base metrics + all suffix metrics
    result["positions"] = {}
    for pos in range(block_size):
        pos_data = {}

        # Base metrics (all masked)
        n = nll_data[0][pos]["tokens"]
        base_loss = nll_data[0][pos]["nll"] / n if n > 0 else 0.0
        base_entropy = nll_data[0][pos]["entropy"] / n if n > 0 else 0.0
        pos_data["loss"] = base_loss
        pos_data["ppl"] = torch.exp(torch.tensor(base_loss)).item()
        pos_data["entropy_ppl"] = torch.exp(torch.tensor(base_entropy)).item()
        pos_data["argmax_prob"] = nll_data[0][pos]["argmax_prob"] / n if n > 0 else 0.0

        # Suffix metrics for this position
        # Position pos has (block_size - 1 - pos) suffixes available
        max_suffix_for_pos = block_size - 1 - pos
        for s in range(1, max_suffix_for_pos + 1):
            n_s = nll_data[s][pos]["tokens"]
            suffix_loss = nll_data[s][pos]["nll"] / n_s if n_s > 0 else 0.0
            suffix_entropy = nll_data[s][pos]["entropy"] / n_s if n_s > 0 else 0.0
            pos_data[f"loss_{s}suffix"] = suffix_loss
            pos_data[f"ppl_{s}suffix"] = torch.exp(torch.tensor(suffix_loss)).item()
            pos_data[f"entropy_ppl_{s}suffix"] = torch.exp(torch.tensor(suffix_entropy)).item()
            pos_data[f"argmax_prob_{s}suffix"] = nll_data[s][pos]["argmax_prob"] / n_s if n_s > 0 else 0.0

        result["positions"][pos] = pos_data

    # Overall suffix metrics (kept for backward compatibility and wandb logging)
    result["suffix_overall"] = {}
    for num_suffix in range(1, max_suffix + 1):
        # Positions that have at least num_suffix suffixes
        valid_positions = [p for p in range(block_size) if (block_size - 1 - p) >= num_suffix]

        if not valid_positions:
            continue

        total_nll_suffix = sum(nll_data[num_suffix][p]["nll"] for p in valid_positions)
        total_tokens_suffix = sum(nll_data[num_suffix][p]["tokens"] for p in valid_positions)
        overall_loss_suffix = total_nll_suffix / total_tokens_suffix if total_tokens_suffix > 0 else 0.0
        result["suffix_overall"][num_suffix] = {
            "positions": valid_positions,
            "overall_loss": overall_loss_suffix,
            "overall_ppl": torch.exp(torch.tensor(overall_loss_suffix)).item(),
        }

    return result
