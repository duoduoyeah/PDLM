"""
PDLM Evaluation Module.

Supports both Stage 1 MASK and Stage 2 evaluation.

Stage 1 MASK (Pure/MASK → Group):
- Input: [prefix, MASK, MASK, ...] at xt positions
- Output: predict group tokens for each MASK position
- Loss: any-correct over overlap_k valid groups
- Metrics: loss, PPL, accuracy per position

Stage 2 (Group → Pure):
- Input is [xt | x0] of length 2L (concatenated inside model.forward)
- xt (first L): group tokens at block positions, pure tokens at prefix
- x0 (second L): pure tokens everywhere (clean reference)
- Skip block 0 for loss computation (no previous context to condition on)
- Compute loss from blocks 1, 2, ... (they have clean context via cross-attention)

Metrics:
1. Loss/PPL: Cross-entropy loss on token prediction at block positions
2. Per-position breakdown: loss_pos_0, loss_pos_1, ... within each block
3. Accuracy: for Stage 1, any-correct accuracy (any valid group counts)
4. Compatibility: Check if parallel predictions are mutually consistent (Stage 2 only)

Perplexity note:
- PPL = exp(avg_cross_entropy_loss)
- Stage 1: output vocab is num_groups
- Stage 2: output vocab is pure_vocab_size
- Loss is computed only on block positions
"""

import torch
import torch.nn.functional as F

from nanochat.group_tokenizer.token_map import get_token_map


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

    Stage 1 MASK: [prefix, MASK, MASK, ...] → group tokens
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
    was_training = model.training
    model.eval()

    with torch.no_grad():
        result = _eval_stage1_mask_per_position(
            model, val_loader, block_size, num_batches,
            attn_mask, device, autocast_ctx, prefix_pure_tokens
        )

    if was_training:
        model.train()
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

    # Build result dict
    result = {}

    # Overall metrics (sum across all positions)
    total_nll = sum(metrics_by_pos[p]["nll"] for p in range(block_size))
    total_correct = sum(metrics_by_pos[p]["correct"] for p in range(block_size))
    total_tokens = sum(metrics_by_pos[p]["tokens"] for p in range(block_size))
    result["overall_loss"] = total_nll / total_tokens if total_tokens > 0 else 0.0
    result["overall_ppl"] = torch.exp(torch.tensor(result["overall_loss"])).item()
    result["overall_accuracy"] = total_correct / total_tokens if total_tokens > 0 else 0.0
    result["num_tokens_evaluated"] = total_tokens

    # Per-position metrics
    result["positions"] = {}
    for pos in range(block_size):
        pos_nll = metrics_by_pos[pos]["nll"]
        pos_correct = metrics_by_pos[pos]["correct"]
        pos_tokens = metrics_by_pos[pos]["tokens"]
        pos_loss = pos_nll / pos_tokens if pos_tokens > 0 else 0.0
        result["positions"][pos] = {
            "loss": pos_loss,
            "ppl": torch.exp(torch.tensor(pos_loss)).item(),
            "accuracy": pos_correct / pos_tokens if pos_tokens > 0 else 0.0,
            "tokens": pos_tokens,
        }

    return result


def eval_pdlm(
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
    Evaluate PDLM Stage 2 model on validation set.

    Args:
        model: PDLM model (stage2)
        val_loader: validation data loader (yields inputs, targets, loss_extras, state_dict)
                    inputs: group tokens at block positions
                    targets: pure tokens everywhere
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
            "positions": {
                0: {"loss": X, "ppl": Y},
                1: {"loss": X, "ppl": Y},
                ...
            },
            "num_tokens_evaluated": int,
        }
    """
    was_training = model.training
    model.eval()

    with torch.no_grad():
        result = _eval_loss_per_position(
            model, val_loader, block_size, num_batches,
            attn_mask, device, autocast_ctx, prefix_pure_tokens
        )

    if was_training:
        model.train()
    return result


def _eval_loss_per_position(
    model, val_loader, block_size, num_batches,
    attn_mask, device, autocast_ctx, prefix_pure_tokens,
    cache_batches=False,
):
    """
    Evaluate loss broken down by position within block.
    Skip block 0 (no context), compute from blocks 1 onwards.

    Args:
        cache_batches: If True, return (result, cached_batches) where cached_batches
                       is a list of (inputs, targets, loss_extras, state) tuples.
    """
    # Accumulators: nll_by_pos[pos] = (total_nll, total_tokens)
    nll_by_pos = {p: {"nll": 0.0, "tokens": 0} for p in range(block_size)}
    cached_batches = [] if cache_batches else None

    for batch_idx in range(num_batches):
        inputs, targets, loss_extras, state = next(val_loader)

        if cache_batches:
            cached_batches.append((inputs, targets, loss_extras, state))
        # inputs: (B, T) - group tokens at block positions, pure at prefix
        # targets: (B, T) - pure tokens everywhere
        # loss_extras: {"loss_mask": (B, T)} - True at block positions

        B, T = inputs.shape
        loss_mask = loss_extras.get("loss_mask", None)

        with autocast_ctx:
            # Forward pass: model concatenates [inputs | targets] internally
            # Returns logits for first T positions (xt half)
            logits = model.forward_for_eval(inputs, targets, attn_mask=attn_mask)
            # logits: (B, T, pure_vocab_size)

            # Compute log probabilities
            log_probs = F.log_softmax(logits.float(), dim=-1)

            # Gather log probs for target tokens
            target_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            # target_log_probs: (B, T)

            # Compute NLL for each position within blocks
            # Skip block 0, compute from blocks 1 onwards
            # Also respect prefix_pure_tokens and loss_mask

            # Determine block boundaries
            # With prefix_sliding_tokens from dataloader, blocks start after prefix
            # For eval, we use prefix_sliding_tokens=0, so blocks align with T

            # Number of complete blocks
            num_blocks = T // block_size

            for pos in range(block_size):
                # For each block (starting from block 1 to skip block 0)
                for block_idx in range(1, num_blocks):
                    pos_in_seq = block_idx * block_size + pos

                    # Check if this position should have loss (within loss_mask)
                    if loss_mask is not None:
                        # Only count positions where loss_mask is True
                        mask_at_pos = loss_mask[:, pos_in_seq]  # (B,)
                        nll = -target_log_probs[:, pos_in_seq]  # (B,)
                        nll_masked = nll * mask_at_pos.float()
                        nll_by_pos[pos]["nll"] += nll_masked.sum().item()
                        nll_by_pos[pos]["tokens"] += mask_at_pos.sum().item()
                    else:
                        nll = -target_log_probs[:, pos_in_seq]
                        nll_by_pos[pos]["nll"] += nll.sum().item()
                        nll_by_pos[pos]["tokens"] += B

    # Build result dict
    result = {}

    # Overall metrics (sum across all positions)
    total_nll = sum(nll_by_pos[p]["nll"] for p in range(block_size))
    total_tokens = sum(nll_by_pos[p]["tokens"] for p in range(block_size))
    result["overall_loss"] = total_nll / total_tokens if total_tokens > 0 else 0.0
    result["overall_ppl"] = torch.exp(torch.tensor(result["overall_loss"])).item()
    result["num_tokens_evaluated"] = total_tokens

    # Per-position metrics
    result["positions"] = {}
    for pos in range(block_size):
        pos_nll = nll_by_pos[pos]["nll"]
        pos_tokens = nll_by_pos[pos]["tokens"]
        pos_loss = pos_nll / pos_tokens if pos_tokens > 0 else 0.0
        result["positions"][pos] = {
            "loss": pos_loss,
            "ppl": torch.exp(torch.tensor(pos_loss)).item(),
            "tokens": pos_tokens,
        }

    if cache_batches:
        return result, cached_batches
    return result


def eval_pdlm_compatibility(
    model,
    val_loader,
    block_size,
    num_batches,
    attn_mask,
    device,
    autocast_ctx,
    token_map=None,
    cached_batches=None,
):
    """
    Evaluate compatibility of parallel predictions.

    For each block:
    1. Generate x1, x2, x3, x4 in parallel from G1, G2, G3, G4
    2. For each position i, re-predict x_i given OTHER pure tokens revealed
       e.g., predict x4' given [prefix, x1, x2, x3, G4]
    3. Check if x_i == x_i'

    Args:
        model: PDLM model (stage2)
        val_loader: validation data loader (can be None if cached_batches provided)
        block_size: block size for PDLM
        num_batches: number of batches to evaluate (ignored if cached_batches provided)
        attn_mask: attention mask for the model
        device: device to run on
        autocast_ctx: autocast context for mixed precision
        token_map: TokenMap for pure<->group conversion (loaded if None)
        cached_batches: pre-cached list of (inputs, targets, loss_extras, state) tuples.
                        If provided, uses these instead of consuming from val_loader.

    Returns:
        dict with compatibility results:
        {
            "overall_compatibility": float,  # % positions where prediction unchanged
            "positions": {
                0: {"compatibility": float, "total": int, "matched": int},
                ...
            },
        }
    """
    was_training = model.training
    model.eval()

    if token_map is None:
        token_map = get_token_map(device=device)

    # Accumulators
    compat_by_pos = {p: {"matched": 0, "total": 0} for p in range(block_size)}

    # Use cached batches if provided, otherwise consume from loader
    def get_batch(idx):
        if cached_batches is not None:
            return cached_batches[idx]
        return next(val_loader)

    # Determine actual number of batches
    actual_batches = len(cached_batches) if cached_batches is not None else num_batches

    with torch.no_grad():
        for batch_idx in range(actual_batches):
            inputs, targets, loss_extras, _ = get_batch(batch_idx)
            # inputs: (B, T) - group tokens at block positions
            # targets: (B, T) - pure tokens

            B, T = inputs.shape
            num_blocks = T // block_size
            pure_vocab_size = token_map.pure_vocab_size

            with autocast_ctx:
                # Step 1: Get initial predictions (all group tokens -> pure predictions)
                logits = model.forward_for_eval(inputs, targets, attn_mask=attn_mask)
                initial_preds = logits.argmax(dim=-1)  # (B, T)

                # Step 2: For each position, reveal other pure tokens and re-predict
                for block_idx in range(1, num_blocks):  # Skip block 0
                    block_start = block_idx * block_size
                    block_end = block_start + block_size

                    for reveal_except_pos in range(block_size):
                        # Create modified input: reveal all positions EXCEPT reveal_except_pos
                        modified_inputs = inputs.clone()

                        for pos in range(block_size):
                            abs_pos = block_start + pos
                            if pos != reveal_except_pos:
                                # Reveal: use initial prediction (pure token)
                                modified_inputs[:, abs_pos] = initial_preds[:, abs_pos]
                            # else: keep as group token

                        # Re-predict
                        modified_logits = model.forward_for_eval(
                            modified_inputs, targets, attn_mask=attn_mask
                        )
                        modified_preds = modified_logits.argmax(dim=-1)

                        # Compare at the position that was NOT revealed
                        target_pos = block_start + reveal_except_pos
                        original_pred = initial_preds[:, target_pos]
                        new_pred = modified_preds[:, target_pos]

                        matched = (original_pred == new_pred).sum().item()
                        compat_by_pos[reveal_except_pos]["matched"] += matched
                        compat_by_pos[reveal_except_pos]["total"] += B

    if was_training:
        model.train()

    # Build result dict
    result = {}

    total_matched = sum(compat_by_pos[p]["matched"] for p in range(block_size))
    total_count = sum(compat_by_pos[p]["total"] for p in range(block_size))
    result["overall_compatibility"] = total_matched / total_count if total_count > 0 else 0.0

    result["positions"] = {}
    for pos in range(block_size):
        matched = compat_by_pos[pos]["matched"]
        total = compat_by_pos[pos]["total"]
        result["positions"][pos] = {
            "compatibility": matched / total if total > 0 else 0.0,
            "matched": matched,
            "total": total,
        }

    return result


def eval_pdlm_full(
    model,
    val_loader,
    block_size,
    num_batches,
    attn_mask,
    device,
    autocast_ctx,
    prefix_pure_tokens=0,
    run_compatibility=True,
    compatibility_batches=None,
):
    """
    Run full PDLM evaluation: loss + perplexity + optional compatibility.

    Args:
        model: PDLM model
        val_loader: validation data loader
        block_size: block size
        num_batches: number of batches for loss eval
        attn_mask: attention mask
        device: device
        autocast_ctx: autocast context
        prefix_pure_tokens: pure prefix tokens
        run_compatibility: whether to run compatibility eval
        compatibility_batches: batches for compatibility (default: num_batches // 4)

    Returns:
        dict with all metrics
    """
    # Run loss evaluation with batch caching for consistency
    was_training = model.training
    model.eval()

    with torch.no_grad():
        loss_result, cached_batches = _eval_loss_per_position(
            model, val_loader, block_size, num_batches,
            attn_mask, device, autocast_ctx, prefix_pure_tokens,
            cache_batches=True,
        )

    result = loss_result

    # Run compatibility evaluation on the same cached batches
    if run_compatibility:
        if compatibility_batches is None:
            compatibility_batches = max(1, num_batches // 4)

        # Use first compatibility_batches from cache (same data as loss eval)
        compat_batches = cached_batches[:compatibility_batches]

        compat_result = eval_pdlm_compatibility(
            model, None, block_size, len(compat_batches),
            attn_mask, device, autocast_ctx,
            cached_batches=compat_batches,
        )
        result["compatibility"] = compat_result

    if was_training:
        model.train()

    return result


def dump_batch_to_file(
    model,
    val_loader,
    block_size,
    attn_mask,
    device,
    autocast_ctx,
    output_path,
    tokenizer_dir=None,
):
    """
    Dump 1 batch showing input group tokens and model predictions.

    Args:
        model: PDLM model
        val_loader: validation data loader
        block_size: block size
        attn_mask: attention mask
        device: device
        autocast_ctx: autocast context
        output_path: path to write txt file
        tokenizer_dir: path to tokenizer dir (default: uses get_base_dir())

    Output file format (per sequence):
        === Sequence 0 ===
        Input (noised):  [<|G_12|>, <|G_45|>, ...]
        Output (preds):  ['hello', ' world', ...]
        Target (truth):  ['hello', ' world', ...]
    """
    from nanochat.group_tokenizer.dump import load_tokenizer

    was_training = model.training
    model.eval()

    # Load tokenizer for decoding
    if tokenizer_dir is None:
        from nanochat.common import get_base_dir
        import os
        tokenizer_dir = os.path.join(get_base_dir(), "tokenizer")

    tokenizer = load_tokenizer(tokenizer_dir)
    token_map = get_token_map(tokenizer_dir, device=device)

    with torch.no_grad():
        inputs, targets, loss_extras, _ = next(val_loader)
        # inputs: (B, T) - group tokens at block positions
        # targets: (B, T) - pure tokens

        B, T = inputs.shape

        with autocast_ctx:
            logits = model.forward_for_eval(inputs, targets, attn_mask=attn_mask)
            preds = logits.argmax(dim=-1)  # (B, T)

    if was_training:
        model.train()

    # Helper to format group token
    def fmt_group(tid):
        if token_map.is_group(torch.tensor(tid)):
            gid = tid - token_map.group_start_id
            return f"<|G_{gid}|>"
        elif tokenizer:
            return repr(tokenizer.decode([tid]))
        return f"[{tid}]"

    # Helper to format pure token
    def fmt_pure(tid):
        if tokenizer:
            return repr(tokenizer.decode([tid]))
        return f"[{tid}]"

    # Write output
    with open(output_path, "w") as f:
        for b in range(B):
            f.write(f"=== Sequence {b} ===\n")

            # Input tokens (group or pure)
            input_strs = [fmt_group(inputs[b, t].item()) for t in range(T)]
            f.write(f"Input (noised):  [{', '.join(input_strs)}]\n")

            # Predicted tokens (argmax)
            pred_strs = [fmt_pure(preds[b, t].item()) for t in range(T)]
            f.write(f"Output (preds):  [{', '.join(pred_strs)}]\n")

            # Target tokens (ground truth)
            tgt_strs = [fmt_pure(targets[b, t].item()) for t in range(T)]
            f.write(f"Target (truth):  [{', '.join(tgt_strs)}]\n")

            f.write("\n")

    print(f"Dumped {B} sequences to {output_path}")