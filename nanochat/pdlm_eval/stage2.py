"""
Stage 2 evaluation for PDLM.

Stage 2 (Group -> Pure):
- Input is [xt | x0] of length 2L (concatenated inside model.forward)
- xt (first L): group tokens at block positions, pure tokens at prefix
- x0 (second L): pure tokens everywhere (clean reference)
- Skip block 0 for loss computation (no previous context to condition on)
- Compute loss from blocks 1, 2, ... (they have clean context via cross-attention)

Metrics:
1. Loss/PPL: Cross-entropy loss on token prediction at block positions
2. Per-position breakdown: loss_pos_0, loss_pos_1, ... within each block
3. Compatibility: Check if parallel predictions are mutually consistent
4. Oracle accuracy: Accuracy when given ground truth context
"""

import torch
import torch.nn.functional as F

from nanochat.group_tokenizer.token_map import get_token_map
from nanochat.pdlm_eval.common import model_eval_context


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
    with model_eval_context(model):
        with torch.no_grad():
            result = _eval_loss_per_position(
                model, val_loader, block_size, num_batches,
                attn_mask, device, autocast_ctx, prefix_pure_tokens
            )
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
    Instead of (num_blocks x block_size) forward passes per batch,
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
    with model_eval_context(model):
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
    - High oracle accuracy + low compatibility -> model explores valid alternatives
    - Low oracle accuracy -> model fundamentally struggles even with perfect context

    Optimization: Tests all blocks simultaneously for each position.
    Instead of (num_blocks x block_size) forward passes per batch,
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
    with model_eval_context(model):
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
    with model_eval_context(model):
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

    return result
