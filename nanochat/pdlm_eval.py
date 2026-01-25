"""
PDLM Stage 2 Evaluation Module.

Computes loss, perplexity, and compatibility metrics for PDLM models on validation data.

Evaluation setup for Stage 2 (Group → Pure):
- Input is [xt | x0] of length 2L (concatenated inside model.forward)
- xt (first L): group tokens at block positions, pure tokens at prefix
- x0 (second L): pure tokens everywhere (clean reference)
- Skip block 0 for loss computation (no previous context to condition on)
- Compute loss from blocks 1, 2, ... (they have clean context via cross-attention)

Metrics:
1. Loss/PPL: Cross-entropy loss on pure token prediction at block positions
2. Per-position breakdown: loss_pos_0, loss_pos_1, ... within each block
3. Compatibility: Check if parallel predictions are mutually consistent

Perplexity note:
- PPL = exp(avg_cross_entropy_loss)
- The output vocab is pure_vocab_size (not pure + groups)
- Loss is computed only on block positions (where model predicts pure from group)
"""

import torch
import torch.nn.functional as F

from nanochat.group_tokenizer.token_map import get_token_map


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
    attn_mask, device, autocast_ctx, prefix_pure_tokens
):
    """
    Evaluate loss broken down by position within block.
    Skip block 0 (no context), compute from blocks 1 onwards.
    """
    # Accumulators: nll_by_pos[pos] = (total_nll, total_tokens)
    nll_by_pos = {p: {"nll": 0.0, "tokens": 0} for p in range(block_size)}

    for batch_idx in range(num_batches):
        inputs, targets, loss_extras, _ = next(val_loader)
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
                # Skip positions in prefix_pure_tokens
                if pos < prefix_pure_tokens:
                    continue

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
        val_loader: validation data loader
        block_size: block size for PDLM
        num_batches: number of batches to evaluate
        attn_mask: attention mask for the model
        device: device to run on
        autocast_ctx: autocast context for mixed precision
        token_map: TokenMap for pure<->group conversion (loaded if None)

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

    with torch.no_grad():
        for batch_idx in range(num_batches):
            inputs, targets, loss_extras, _ = next(val_loader)
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
    # Run loss evaluation
    result = eval_pdlm(
        model, val_loader, block_size, num_batches,
        attn_mask, device, autocast_ctx, prefix_pure_tokens
    )

    # Run compatibility evaluation
    if run_compatibility:
        if compatibility_batches is None:
            compatibility_batches = max(1, num_batches // 4)

        # Need a fresh loader for compatibility eval
        # The caller should provide a way to reset or create new loader
        # For now, we continue with the same loader (will use next batches)
        compat_result = eval_pdlm_compatibility(
            model, val_loader, block_size, compatibility_batches,
            attn_mask, device, autocast_ctx
        )
        result["compatibility"] = compat_result

    return result
