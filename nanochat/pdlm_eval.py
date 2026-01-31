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


def eval_pdlm_stage1_block(
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
    Evaluate PDLM Stage 1 Block model with pure-target mode on validation set.

    Stage 1 Block (pure-target mode): pure tokens in, L×L block-causal mask,
    lm_head outputs pure vocab logits. No 2L structure — model is called directly.

    Loss: CE against pure tokens
    Accuracy: Group-level - predicted group contains target token?

    Args:
        model: PDLM model (stage1_block)
        val_loader: validation data loader (yields inputs, targets, loss_extras, state_dict)
                    inputs: pure tokens
                    targets: pure tokens
                    loss_extras: {"loss_mask", "pure_targets", "pure_to_group"}
        block_size: block size for PDLM
        num_batches: number of batches to evaluate
        attn_mask: L×L block-causal attention mask
        device: device to run on
        autocast_ctx: autocast context for mixed precision
        prefix_pure_tokens: number of pure prefix tokens (no loss on these)

    Returns:
        dict with evaluation results:
        {
            "overall_loss": float,  # CE against pure tokens
            "overall_ppl": float,
            "overall_accuracy": float,  # group-level accuracy
            "positions": {...},
            "num_tokens_evaluated": int,
        }
    """
    was_training = model.training
    model.eval()

    with torch.no_grad():
        result = _eval_stage1_block_pure_target(
            model, val_loader, block_size, num_batches,
            attn_mask, device, autocast_ctx, prefix_pure_tokens
        )

    if was_training:
        model.train()
    return result


def _eval_stage1_block_pure_target(
    model, val_loader, block_size, num_batches,
    attn_mask, device, autocast_ctx, prefix_pure_tokens,
):
    """
    Pure-target mode evaluation for stage1_block.

    Loss: Standard CE against pure tokens.
    Accuracy: Collapse logits to groups, check if target token is in predicted group.
    """
    metrics_by_pos = {p: {"nll": 0.0, "correct": 0, "tokens": 0} for p in range(block_size)}

    # Get group head for accuracy computation: (num_groups, n_embd)
    # group_logits = pure_logits @ group_to_pure_mask.T (sum pure logits per group)
    group_to_pure_mask = model.group_to_pure_mask  # (num_groups, pure_vocab)

    for batch_idx in range(num_batches):
        inputs, targets, loss_extras, state = next(val_loader)
        # inputs: (B, T) - pure tokens
        # loss_extras: {"loss_mask": (B, T), "pure_targets": (B, T), "pure_to_group": (pure_vocab, overlap_k)}

        B, T = inputs.shape
        loss_mask = loss_extras.get("loss_mask", None)
        pure_targets = loss_extras.get("pure_targets", None)
        pure_to_group = loss_extras.get("pure_to_group", None)

        if pure_targets is None:
            raise ValueError("Stage 1 Block eval requires pure_targets in loss_extras")
        if pure_to_group is None:
            raise ValueError("Stage 1 Block eval requires pure_to_group in loss_extras")

        overlap_k = pure_to_group.size(-1)

        with autocast_ctx:
            # Forward pass: direct call (no 2L, no forward_for_eval)
            logits = model(inputs, attn_mask=attn_mask)
            # logits: (B, T, pure_vocab_size)

            # Loss: standard CE against pure targets
            log_probs = F.log_softmax(logits.float(), dim=-1)
            target_log_probs = log_probs.gather(-1, pure_targets.unsqueeze(-1)).squeeze(-1)

            # For accuracy: collapse logits to group logits
            # group_logits[g] = sum of pure_logits for all tokens in group g
            # This is equivalent to: group_logits = logits @ group_to_pure_mask.T
            group_logits = logits @ group_to_pure_mask.T  # (B, T, num_groups)
            pred_groups = group_logits.argmax(dim=-1)  # (B, T)

            # Check if target token is in predicted group
            target_groups = pure_to_group[pure_targets]  # (B, T, overlap_k)

            num_blocks = T // block_size

            for pos in range(block_size):
                # Skip block 0 and last block
                for block_idx in range(1, num_blocks - 1):
                    pos_in_seq = block_idx * block_size + pos

                    if loss_mask is not None:
                        mask_at_pos = loss_mask[:, pos_in_seq]  # (B,)
                    else:
                        mask_at_pos = torch.ones(B, dtype=torch.bool, device=device)

                    # NLL
                    nll = -target_log_probs[:, pos_in_seq]
                    nll_masked = nll * mask_at_pos.float()
                    metrics_by_pos[pos]["nll"] += nll_masked.sum().item()
                    metrics_by_pos[pos]["tokens"] += mask_at_pos.sum().item()

                    # Group accuracy: check if predicted group contains target token
                    pred_g = pred_groups[:, pos_in_seq]  # (B,)
                    valid_groups = target_groups[:, pos_in_seq, :]  # (B, overlap_k)
                    valid_mask = valid_groups >= 0
                    any_match = ((valid_groups == pred_g.unsqueeze(-1)) & valid_mask).any(dim=-1)
                    correct_masked = any_match & mask_at_pos
                    metrics_by_pos[pos]["correct"] += correct_masked.sum().item()

    # Build result dict
    result = {}
    total_nll = sum(metrics_by_pos[p]["nll"] for p in range(block_size))
    total_correct = sum(metrics_by_pos[p]["correct"] for p in range(block_size))
    total_tokens = sum(metrics_by_pos[p]["tokens"] for p in range(block_size))
    result["overall_loss"] = total_nll / total_tokens if total_tokens > 0 else 0.0
    result["overall_ppl"] = torch.exp(torch.tensor(result["overall_loss"])).item()
    result["overall_accuracy"] = total_correct / total_tokens if total_tokens > 0 else 0.0
    result["num_tokens_evaluated"] = total_tokens

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

    Optimization: Tests all blocks simultaneously for each position.
    Instead of (num_blocks × block_size) forward passes per batch,
    uses only (1 + block_size) passes: 1 for initial predictions + 4 for position tests.

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

            with autocast_ctx:
                # Step 1: Get initial predictions (1 forward pass)
                logits = model.forward_for_eval(inputs, targets, attn_mask=attn_mask)
                initial_preds = logits.argmax(dim=-1)  # (B, T)

                # Step 2: Test each position across ALL blocks (block_size forward passes)
                for test_pos in range(block_size):
                    # Create input: reveal predictions at OTHER positions, keep group at test_pos
                    # This tests all blocks simultaneously
                    modified = inputs.clone()

                    for block_idx in range(1, num_blocks):  # Skip block 0
                        for pos in range(block_size):
                            abs_pos = block_idx * block_size + pos
                            if pos != test_pos:
                                # Reveal: use initial prediction
                                modified[:, abs_pos] = initial_preds[:, abs_pos]
                            # else: keep group token at test_pos

                    # Single forward pass for all blocks
                    modified_logits = model.forward_for_eval(
                        modified, targets, attn_mask=attn_mask
                    )
                    modified_preds = modified_logits.argmax(dim=-1)

                    # Compare at all test positions across all blocks
                    for block_idx in range(1, num_blocks):
                        abs_pos = block_idx * block_size + test_pos
                        original_pred = initial_preds[:, abs_pos]
                        new_pred = modified_preds[:, abs_pos]

                        matched = (original_pred == new_pred).sum().item()
                        compat_by_pos[test_pos]["matched"] += matched
                        compat_by_pos[test_pos]["total"] += B

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


def eval_pdlm_oracle_accuracy(
    model,
    val_loader,
    block_size,
    num_batches,
    attn_mask,
    device,
    autocast_ctx,
    cached_batches=None,
):
    """
    Test accuracy when given ground truth context (oracle test).

    For each position i in a block:
    - Input: [p1, p2, p3, G_i] (ground truth pure tokens at other positions, group token at i)
    - Predict: x_i'
    - Compare: p_i == x_i'?

    This tests the model's capability to predict correctly given perfect context,
    which helps interpret compatibility results:
    - High oracle accuracy + low compatibility → model explores valid alternatives
    - Low oracle accuracy → model fundamentally struggles even with perfect context

    Optimization: Tests all blocks simultaneously for each position.
    Instead of (num_blocks × block_size) forward passes per batch,
    uses only block_size passes (4 for block_size=4).

    Args:
        model: PDLM model (stage2)
        val_loader: validation data loader (can be None if cached_batches provided)
        block_size: block size for PDLM
        num_batches: number of batches to evaluate (ignored if cached_batches provided)
        attn_mask: attention mask for the model
        device: device to run on
        autocast_ctx: autocast context for mixed precision
        cached_batches: pre-cached list of (inputs, targets, loss_extras, state) tuples.

    Returns:
        dict with oracle accuracy results:
        {
            "overall_accuracy": float,  # % positions where prediction matches ground truth
            "positions": {
                0: {"accuracy": float, "total": int, "matched": int},
                ...
            },
        }
    """
    was_training = model.training
    model.eval()

    # Accumulators
    acc_by_pos = {p: {"matched": 0, "total": 0} for p in range(block_size)}

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
            # targets: (B, T) - pure tokens (ground truth)

            B, T = inputs.shape
            num_blocks = T // block_size

            with autocast_ctx:
                # Test each position across ALL blocks (block_size forward passes total)
                for test_pos in range(block_size):
                    # Create input: ground truth everywhere, EXCEPT test_pos in each block
                    # has group token
                    modified = targets.clone()

                    for block_idx in range(1, num_blocks):  # Skip block 0
                        abs_pos = block_idx * block_size + test_pos
                        modified[:, abs_pos] = inputs[:, abs_pos]  # put group token here

                    # Single forward pass for all blocks
                    logits = model.forward_for_eval(modified, targets, attn_mask=attn_mask)
                    preds = logits.argmax(dim=-1)

                    # Compare at all test positions across all blocks
                    for block_idx in range(1, num_blocks):
                        abs_pos = block_idx * block_size + test_pos
                        matched = (preds[:, abs_pos] == targets[:, abs_pos]).sum().item()
                        acc_by_pos[test_pos]["matched"] += matched
                        acc_by_pos[test_pos]["total"] += B

    if was_training:
        model.train()

    # Build result dict
    result = {}

    total_matched = sum(acc_by_pos[p]["matched"] for p in range(block_size))
    total_count = sum(acc_by_pos[p]["total"] for p in range(block_size))
    result["overall_accuracy"] = total_matched / total_count if total_count > 0 else 0.0

    result["positions"] = {}
    for pos in range(block_size):
        matched = acc_by_pos[pos]["matched"]
        total = acc_by_pos[pos]["total"]
        result["positions"][pos] = {
            "accuracy": matched / total if total > 0 else 0.0,
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
    run_oracle_accuracy=True,
    oracle_accuracy_batches=None,
):
    """
    Run full PDLM evaluation: loss + perplexity + optional compatibility + optional oracle accuracy.

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
        run_oracle_accuracy: whether to run oracle accuracy eval
        oracle_accuracy_batches: batches for oracle accuracy (default: num_batches // 4)

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

    # Run oracle accuracy evaluation on the same cached batches
    if run_oracle_accuracy:
        if oracle_accuracy_batches is None:
            oracle_accuracy_batches = max(1, num_batches // 4)

        # Use first oracle_accuracy_batches from cache (same data as loss eval)
        oracle_batches = cached_batches[:oracle_accuracy_batches]

        oracle_result = eval_pdlm_oracle_accuracy(
            model, None, block_size, len(oracle_batches),
            attn_mask, device, autocast_ctx,
            cached_batches=oracle_batches,
        )
        result["oracle_accuracy"] = oracle_result

    if was_training:
        model.train()

    return result


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
    Evaluate PDLM both_block model: combined Stage 1 (block→block) + Stage 2 (denoise).

    Returns a dict with three sections: Overall, Stage 1, Stage 2, plus optional
    compatibility and oracle accuracy.

    Args:
        model: PDLM model (both_block stage)
        val_loader: validation data loader
        block_size: block size for PDLM
        num_batches: number of batches to evaluate
        attn_mask: 2L×2L attention mask
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
    was_training = model.training
    model.eval()

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

    if was_training:
        model.train()

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
    Evaluate Stage 1 (block→block group prediction) from both_block model.
    Uses forward_for_eval_both_block to get stage1_logits from x0 half.
    Skip block 0 and last block for loss computation.
    """
    from nanochat.pdlm import any_correct_ce_loss

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
            # stage1_logits: (B, T, num_groups) — group logits from x0 half

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

    # Build result
    total_nll = sum(metrics_by_pos[p]["nll"] for p in range(block_size))
    total_correct = sum(metrics_by_pos[p]["correct"] for p in range(block_size))
    total_tokens = sum(metrics_by_pos[p]["tokens"] for p in range(block_size))

    result = {
        "overall_loss": total_nll / total_tokens if total_tokens > 0 else 0.0,
        "overall_ppl": torch.exp(torch.tensor(total_nll / total_tokens if total_tokens > 0 else 0.0)).item(),
        "overall_accuracy": total_correct / total_tokens if total_tokens > 0 else 0.0,
        "num_tokens_evaluated": total_tokens,
        "positions": {},
    }
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


def _eval_both_block_stage2(
    model, cached_batches, block_size, attn_mask, device, autocast_ctx,
):
    """
    Evaluate Stage 2 (group→pure denoising) from both_block model.
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

    # Build result
    total_nll = sum(nll_by_pos[p]["nll"] for p in range(block_size))
    total_tokens = sum(nll_by_pos[p]["tokens"] for p in range(block_size))

    result = {
        "overall_loss": total_nll / total_tokens if total_tokens > 0 else 0.0,
        "overall_ppl": torch.exp(torch.tensor(total_nll / total_tokens if total_tokens > 0 else 0.0)).item(),
        "num_tokens_evaluated": total_tokens,
        "positions": {},
    }
    for pos in range(block_size):
        pos_nll = nll_by_pos[pos]["nll"]
        pos_tokens = nll_by_pos[pos]["tokens"]
        pos_loss = pos_nll / pos_tokens if pos_tokens > 0 else 0.0
        result["positions"][pos] = {
            "loss": pos_loss,
            "ppl": torch.exp(torch.tensor(pos_loss)).item(),
            "tokens": pos_tokens,
        }
    return result


def _eval_both_block_end2end(
    model, cached_batches, block_size, attn_mask, device, autocast_ctx,
):
    """
    End-to-end two-step inference evaluation for both_block.

    1. Forward pass 1: get stage1_logits → argmax predicted groups
    2. Build new_inputs with predicted group tokens at block positions
    3. Forward pass 2: forward_for_eval with new_inputs → stage2 logits
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


def dump_batch_to_file(
    model,
    val_loader,
    block_size,
    attn_mask,
    device,
    autocast_ctx,
    output_path,
    tokenizer_dir=None,
    num_sequences=2,
    num_blocks_to_show=3,
):
    """
    Dump 1 batch showing input group tokens, model predictions, compatibility, and oracle tests.

    Args:
        model: PDLM model
        val_loader: validation data loader
        block_size: block size
        attn_mask: attention mask
        device: device
        autocast_ctx: autocast context
        output_path: path to write txt file
        tokenizer_dir: path to tokenizer dir (default: uses get_base_dir())
        num_sequences: number of sequences to dump (default: 2)
        num_blocks_to_show: number of blocks to show detailed analysis for (default: 3)

    Output file format (per sequence):
        === Sequence 0 ===
        [Basic predictions]
        [Compatibility tests for blocks 1,2,3]
        [Oracle accuracy tests for blocks 1,2,3]
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

    # Helper to format group token
    def fmt_group(tid):
        if token_map.is_group(torch.tensor(tid)):
            gid = tid - token_map.group_start_id
            return f"G{gid}"
        elif tokenizer:
            return repr(tokenizer.decode([tid]))
        return f"[{tid}]"

    # Helper to format pure token
    def fmt_pure(tid):
        if tokenizer:
            return repr(tokenizer.decode([tid]))
        return f"[{tid}]"

    # Helper to format a block slice
    def fmt_block(tensor, b, block_start, is_input=False):
        tokens = []
        for pos in range(block_size):
            tid = tensor[b, block_start + pos].item()
            if is_input:
                tokens.append(fmt_group(tid))
            else:
                tokens.append(fmt_pure(tid))
        return "[" + ", ".join(tokens) + "]"

    with torch.no_grad():
        inputs, targets, loss_extras, _ = next(val_loader)
        # inputs: (B, T) - group tokens at block positions
        # targets: (B, T) - pure tokens

        B, T = inputs.shape
        num_blocks = T // block_size
        num_sequences = min(num_sequences, B)
        num_blocks_to_show = min(num_blocks_to_show, num_blocks - 1)  # skip block 0

        with autocast_ctx:
            # Step 1: Get initial predictions (parallel decode from group tokens)
            logits = model.forward_for_eval(inputs, targets, attn_mask=attn_mask)
            initial_preds = logits.argmax(dim=-1)  # (B, T)

            # Step 2: Compatibility tests - for each position, reveal other predictions
            # compat_preds[test_pos] = predictions when other positions revealed
            compat_preds = {}
            for test_pos in range(block_size):
                modified = inputs.clone()
                for block_idx in range(1, num_blocks):
                    for pos in range(block_size):
                        abs_pos = block_idx * block_size + pos
                        if pos != test_pos:
                            modified[:, abs_pos] = initial_preds[:, abs_pos]
                modified_logits = model.forward_for_eval(modified, targets, attn_mask=attn_mask)
                compat_preds[test_pos] = modified_logits.argmax(dim=-1)

            # Step 3: Oracle tests - for each position, give ground truth at other positions
            # oracle_preds[test_pos] = predictions when ground truth given at other positions
            oracle_preds = {}
            for test_pos in range(block_size):
                modified = targets.clone()
                for block_idx in range(1, num_blocks):
                    abs_pos = block_idx * block_size + test_pos
                    modified[:, abs_pos] = inputs[:, abs_pos]  # keep group token
                modified_logits = model.forward_for_eval(modified, targets, attn_mask=attn_mask)
                oracle_preds[test_pos] = modified_logits.argmax(dim=-1)

    if was_training:
        model.train()

    # Write output
    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("PDLM BATCH DUMP - Compatibility & Oracle Analysis\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Block size: {block_size}, Sequence length: {T}, Num blocks: {num_blocks}\n")
        f.write(f"Showing {num_sequences} sequences, {num_blocks_to_show} blocks each\n\n")

        for b in range(num_sequences):
            f.write("=" * 80 + "\n")
            f.write(f"SEQUENCE {b}\n")
            f.write("=" * 80 + "\n\n")

            # Show first few blocks overview
            f.write("--- BLOCK OVERVIEW (first 5 blocks) ---\n")
            for block_idx in range(min(5, num_blocks)):
                block_start = block_idx * block_size
                f.write(f"Block {block_idx}: ")
                f.write(f"Input={fmt_block(inputs, b, block_start, is_input=True)} ")
                f.write(f"Pred={fmt_block(initial_preds, b, block_start)} ")
                f.write(f"Truth={fmt_block(targets, b, block_start)}\n")
            f.write("\n")

            # Detailed analysis for selected blocks
            for block_idx in range(1, 1 + num_blocks_to_show):
                block_start = block_idx * block_size
                f.write("-" * 60 + "\n")
                f.write(f"BLOCK {block_idx} DETAILED ANALYSIS (positions {block_start}-{block_start + block_size - 1})\n")
                f.write("-" * 60 + "\n\n")

                # Show the block
                f.write(f"Input (group tokens):  {fmt_block(inputs, b, block_start, is_input=True)}\n")
                f.write(f"Initial prediction:    {fmt_block(initial_preds, b, block_start)}\n")
                f.write(f"Ground truth:          {fmt_block(targets, b, block_start)}\n\n")

                # Compatibility analysis
                f.write("COMPATIBILITY TEST: Re-predict each position after revealing others\n")
                f.write("  (Does the model stick with its prediction when seeing its other outputs?)\n\n")
                for test_pos in range(block_size):
                    abs_pos = block_start + test_pos
                    init_pred = initial_preds[b, abs_pos].item()
                    new_pred = compat_preds[test_pos][b, abs_pos].item()
                    matched = "✓ SAME" if init_pred == new_pred else "✗ CHANGED"

                    # Show what the input looked like for this test
                    input_desc = []
                    for pos in range(block_size):
                        if pos == test_pos:
                            input_desc.append(fmt_group(inputs[b, block_start + pos].item()))
                        else:
                            input_desc.append(fmt_pure(initial_preds[b, block_start + pos].item()))
                    input_str = "[" + ", ".join(input_desc) + "]"

                    f.write(f"  Pos {test_pos}: Input={input_str}\n")
                    f.write(f"         Initial={fmt_pure(init_pred)}, Re-pred={fmt_pure(new_pred)} → {matched}\n")
                f.write("\n")

                # Oracle accuracy analysis
                f.write("ORACLE TEST: Predict each position given ground truth at others\n")
                f.write("  (Can the model predict correctly with perfect context?)\n\n")
                for test_pos in range(block_size):
                    abs_pos = block_start + test_pos
                    pred = oracle_preds[test_pos][b, abs_pos].item()
                    truth = targets[b, abs_pos].item()
                    matched = "✓ CORRECT" if pred == truth else "✗ WRONG"

                    # Show what the input looked like for this test
                    input_desc = []
                    for pos in range(block_size):
                        if pos == test_pos:
                            input_desc.append(fmt_group(inputs[b, block_start + pos].item()))
                        else:
                            input_desc.append(fmt_pure(targets[b, block_start + pos].item()))
                    input_str = "[" + ", ".join(input_desc) + "]"

                    f.write(f"  Pos {test_pos}: Input={input_str}\n")
                    f.write(f"         Pred={fmt_pure(pred)}, Truth={fmt_pure(truth)} → {matched}\n")
                f.write("\n")

            f.write("\n")

    print(f"Dumped {num_sequences} sequences to {output_path}")