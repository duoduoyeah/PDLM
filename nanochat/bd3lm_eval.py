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

    Returns:
        dict with evaluation results:
        - target_shift >= 1 (e.g., ts=1, block_size=4):
            {
                "loss": float, "ppl": float,  # all masked (core metric)
                "loss_1suffix": float, "ppl_1suffix": float,  # pos 1 clear
                "loss_2suffix": float, "ppl_2suffix": float,  # pos 1,2 clear
                "loss_3suffix": float, "ppl_3suffix": float,  # pos 1,2,3 clear
            }
        - target_shift < 0 (normal mode):
            {
                "overall_loss": float, "overall_ppl": float,
                "positions": {
                    0: {"loss": X, "ppl": Y, "loss_1suffix": A, "ppl_1suffix": B, ...},
                    1: {"loss": X, "ppl": Y, "loss_1suffix": A, ...},
                    ...
                },
                "suffix_overall": {
                    1: {"positions": [0, 1, 2], "overall_loss": float, "overall_ppl": float},
                    2: {...}, ...
                }
            }
    """
    was_training = model.training
    model.eval()

    with torch.no_grad():
        if target_shift >= 1:
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

                # Compute NLL for the specific position in each block
                # Skip block 0, compute from blocks 1 onwards
                for block_idx in range(1, num_blocks):
                    pos_in_seq = block_idx * block_size + position
                    nll = -target_log_probs[:, pos_in_seq]
                    nll_by_suffix[num_suffix] += nll.sum().item()
                    tokens_by_suffix[num_suffix] += B

    # Build result dict
    result = {}

    # Core metrics (all masked)
    avg_loss = nll_by_suffix[0] / tokens_by_suffix[0] if tokens_by_suffix[0] > 0 else 0.0
    result["loss"] = avg_loss
    result["ppl"] = torch.exp(torch.tensor(avg_loss)).item()

    # Suffix metrics
    for num_suffix in range(1, max_suffix + 1):
        avg_loss_suffix = nll_by_suffix[num_suffix] / tokens_by_suffix[num_suffix] if tokens_by_suffix[num_suffix] > 0 else 0.0
        result[f"loss_{num_suffix}suffix"] = avg_loss_suffix
        result[f"ppl_{num_suffix}suffix"] = torch.exp(torch.tensor(avg_loss_suffix)).item()

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

    # Track metrics: nll_data[num_suffix][pos] = (total_nll, total_tokens)
    # num_suffix=0 means all masked (original eval)
    nll_data = {
        s: {p: {"nll": 0.0, "tokens": 0} for p in range(block_size)}
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

                    # Compute NLL at this position in each block (skip block 0)
                    for block_idx in range(1, num_blocks):
                        pos_in_seq = block_idx * block_size + pred_pos
                        nll = -target_log_probs[:, pos_in_seq]
                        nll_data[num_suffix][pred_pos]["nll"] += nll.sum().item()
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
        base_loss = nll_data[0][pos]["nll"] / nll_data[0][pos]["tokens"] if nll_data[0][pos]["tokens"] > 0 else 0.0
        pos_data["loss"] = base_loss
        pos_data["ppl"] = torch.exp(torch.tensor(base_loss)).item()

        # Suffix metrics for this position
        # Position pos has (block_size - 1 - pos) suffixes available
        max_suffix_for_pos = block_size - 1 - pos
        for s in range(1, max_suffix_for_pos + 1):
            suffix_loss = nll_data[s][pos]["nll"] / nll_data[s][pos]["tokens"] if nll_data[s][pos]["tokens"] > 0 else 0.0
            pos_data[f"loss_{s}suffix"] = suffix_loss
            pos_data[f"ppl_{s}suffix"] = torch.exp(torch.tensor(suffix_loss)).item()

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
