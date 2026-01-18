"""
BD3LM Evaluation Module.

Computes loss and perplexity for BD3LM models on validation data.

Evaluation setup:
- Input is [xt | x0] of length 2L
- xt (first L): ALL tokens are MASKED
- x0 (second L): ALL tokens are clean
- Skip block 0 for loss computation (no previous context to condition on)
- Compute loss from blocks 1, 2, ... (they have clean context via cross-attention)

Two evaluation modes:
- Normal mode (target_shift < 0): compute overall + per-position metrics
- Target_shift mode (target_shift >= 1): compute metrics only at specific position

Example (L=16, block_size=4, 4 blocks):
    xt:  [MASK MASK MASK MASK | MASK MASK MASK MASK | MASK MASK MASK MASK | MASK MASK MASK MASK]
          ^-- block 0 (skip) --^ ^---- block 1 ----^ ^---- block 2 ----^ ^---- block 3 ----^
    x0:  [clean clean clean clean | clean clean clean clean | ...]
    Loss computed from blocks 1, 2, 3 only.
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
        - target_shift >= 1:
            {"loss": float, "ppl": float}
        - target_shift < 0 (normal mode):
            {
                "overall_loss": float,
                "overall_ppl": float,
                "per_pos_loss": [loss_0, loss_1, ..., loss_{block_size-1}],
                "per_pos_ppl": [ppl_0, ppl_1, ..., ppl_{block_size-1}],
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


def _eval_target_shift_mode(
    model, val_loader, block_size, target_shift,
    num_batches, attn_mask, device, autocast_ctx, mask_token_id
):
    """
    Evaluate in target_shift mode: compute loss only at position (target_shift-1).
    Skip block 0, compute from blocks 1 onwards.
    """
    position = target_shift - 1  # convert to 0-indexed

    total_nll = 0.0
    total_tokens = 0

    for batch_idx in range(num_batches):
        # Get batch from val_loader
        # val_loader yields (inputs, targets, loss_extras, state_dict)
        # We ignore inputs and create our own (all masked)
        _, targets_batch, _, _ = next(val_loader)

        B, L = targets_batch.shape
        num_blocks = L // block_size

        # Prepare inputs: all masked
        inputs, targets = _prepare_eval_batch(targets_batch, mask_token_id)

        with autocast_ctx:
            # Forward pass: returns logits of shape (B, L, vocab_size)
            logits = model.forward_for_eval(inputs, targets, attn_mask=attn_mask)

            # Compute log probabilities
            log_probs = F.log_softmax(logits.float(), dim=-1)

            # Gather log probs for target tokens: (B, L)
            target_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

            # Compute NLL for the specific position in each block
            # Skip block 0, compute from blocks 1 onwards
            for block_idx in range(1, num_blocks):
                pos_in_seq = block_idx * block_size + position
                nll = -target_log_probs[:, pos_in_seq]  # (B,)
                total_nll += nll.sum().item()
                total_tokens += B

    avg_loss = total_nll / total_tokens if total_tokens > 0 else 0.0
    ppl = torch.exp(torch.tensor(avg_loss)).item()

    return {
        "loss": avg_loss,
        "ppl": ppl,
    }


def _eval_normal_mode(
    model, val_loader, block_size,
    num_batches, attn_mask, device, autocast_ctx, mask_token_id
):
    """
    Evaluate in normal mode: compute overall + per-position metrics.
    Skip block 0, compute from blocks 1 onwards.
    """
    # Track overall metrics
    total_nll = 0.0
    total_tokens = 0

    # Track per-position metrics
    per_pos_nll = [0.0] * block_size
    per_pos_tokens = [0] * block_size

    for batch_idx in range(num_batches):
        # Get batch from val_loader
        _, targets_batch, _, _ = next(val_loader)

        B, L = targets_batch.shape
        num_blocks = L // block_size

        # Prepare inputs: all masked
        inputs, targets = _prepare_eval_batch(targets_batch, mask_token_id)

        with autocast_ctx:
            # Forward pass
            logits = model.forward_for_eval(inputs, targets, attn_mask=attn_mask)

            # Compute log probabilities
            log_probs = F.log_softmax(logits.float(), dim=-1)

            # Gather log probs for target tokens: (B, L)
            target_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

            # Compute NLL = -log_prob
            nll = -target_log_probs  # (B, L)

            # Skip block 0, compute from blocks 1 onwards
            for block_idx in range(1, num_blocks):
                block_start = block_idx * block_size
                block_end = block_start + block_size

                # Overall metrics: sum all positions in this block
                block_nll = nll[:, block_start:block_end]  # (B, block_size)
                total_nll += block_nll.sum().item()
                total_tokens += B * block_size

                # Per-position metrics
                for pos in range(block_size):
                    pos_in_seq = block_start + pos
                    pos_nll = nll[:, pos_in_seq]  # (B,)
                    per_pos_nll[pos] += pos_nll.sum().item()
                    per_pos_tokens[pos] += B

    # Compute averages
    overall_loss = total_nll / total_tokens if total_tokens > 0 else 0.0
    overall_ppl = torch.exp(torch.tensor(overall_loss)).item()

    per_pos_avg_loss = [
        per_pos_nll[i] / per_pos_tokens[i] if per_pos_tokens[i] > 0 else 0.0
        for i in range(block_size)
    ]
    per_pos_ppl = [
        torch.exp(torch.tensor(loss)).item() for loss in per_pos_avg_loss
    ]

    return {
        "overall_loss": overall_loss,
        "overall_ppl": overall_ppl,
        "per_pos_loss": per_pos_avg_loss,
        "per_pos_ppl": per_pos_ppl,
    }
