"""
MTP Stage 1 Evaluation Module.

Computes loss, perplexity, and transition accuracy metrics for MTP models.

Evaluation metrics:
1. Overall: loss, PPL, transition_accuracy
2. Per-position (k=0..K-1): loss, PPL, transition_accuracy
3. Per-group: transition_accuracy for each of the ~64 groups

Note:
- Uses teacher forcing (matches training behavior)
- Transition accuracy = argmax(logits) == target_group (not pure token accuracy)
"""

import torch
import torch.nn.functional as F


def eval_mtp(
    model,
    val_loader,
    num_batches,
    device,
    autocast_ctx,
):
    """
    Evaluate MTP Stage 1 model on validation set.

    Args:
        model: GPTMTP model
        val_loader: validation data loader yielding (inputs, targets, loss_extras, state_dict)
                    inputs: (B, T) pure token ids
                    targets: (B, T, K) K future group tokens per position
        num_batches: number of batches to evaluate
        device: device to run on
        autocast_ctx: autocast context for mixed precision

    Returns:
        dict with evaluation results:
        {
            "overall_loss": float,
            "overall_ppl": float,
            "overall_accuracy": float,
            "positions": {
                0: {"loss": X, "ppl": Y, "accuracy": Z, "tokens": N},
                ...
            },
            "per_group_accuracy": {
                0: {"accuracy": X, "correct": C, "total": T},
                ...
            },
            "num_tokens_evaluated": int,
        }
    """
    was_training = model.training
    model.eval()

    K = model.config.n_future_tokens
    num_groups = model.config.num_groups

    # Accumulators per position k
    stats_by_pos = {k: {"nll": 0.0, "correct": 0, "tokens": 0} for k in range(K)}

    # Accumulators per group (for transition accuracy)
    stats_by_group = {g: {"correct": 0, "total": 0} for g in range(num_groups)}

    with torch.no_grad():
        for batch_idx in range(num_batches):
            inputs, targets, loss_extras, state = next(val_loader)
            # inputs: (B, T) pure tokens
            # targets: (B, T, K) group tokens

            B, T = inputs.shape

            with autocast_ctx:
                # Forward pass
                main_logits, mtp_logits = model.forward_for_eval(inputs, targets)
                # main_logits: (B, T, num_groups) for k=0
                # mtp_logits: (B, T, K-1, num_groups) for k=1..K-1

                # Compute metrics for k=0 (main model prediction)
                _update_position_stats(
                    stats_by_pos[0],
                    stats_by_group,
                    main_logits,
                    targets[:, :, 0],
                )

                # Compute metrics for k=1..K-1 (MTP head predictions)
                for k in range(1, K):
                    step_logits = mtp_logits[:, :, k - 1, :]  # (B, T, num_groups)
                    step_targets = targets[:, :, k]  # (B, T)
                    _update_position_stats(
                        stats_by_pos[k],
                        stats_by_group,
                        step_logits,
                        step_targets,
                    )

    if was_training:
        model.train()

    return _build_result(stats_by_pos, stats_by_group, K, num_groups)


def _update_position_stats(pos_stats, group_stats, logits, targets):
    """Update per-position and per-group statistics."""
    B, T, V = logits.shape

    # Compute log probabilities and NLL
    log_probs = F.log_softmax(logits.float(), dim=-1)
    target_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    nll = -target_log_probs  # (B, T)

    # Compute predictions and accuracy
    preds = logits.argmax(dim=-1)  # (B, T)
    correct = (preds == targets)  # (B, T)

    # Mask out invalid targets (if any are -1)
    valid_mask = (targets != -1)
    nll_masked = nll * valid_mask.float()
    correct_masked = correct * valid_mask

    # Update position stats
    pos_stats["nll"] += nll_masked.sum().item()
    pos_stats["correct"] += correct_masked.sum().item()
    pos_stats["tokens"] += valid_mask.sum().item()

    # Update per-group stats
    # For each group g, count how many times target was g and prediction was correct
    targets_flat = targets.reshape(-1)
    correct_flat = correct_masked.reshape(-1)
    valid_flat = valid_mask.reshape(-1)

    for g in range(logits.shape[-1]):
        group_mask = (targets_flat == g) & valid_flat
        group_stats[g]["total"] += group_mask.sum().item()
        group_stats[g]["correct"] += (group_mask & correct_flat).sum().item()


def _build_result(stats_by_pos, stats_by_group, K, num_groups):
    """Build the result dictionary from accumulated statistics."""
    result = {}

    # Overall metrics
    total_nll = sum(stats_by_pos[k]["nll"] for k in range(K))
    total_correct = sum(stats_by_pos[k]["correct"] for k in range(K))
    total_tokens = sum(stats_by_pos[k]["tokens"] for k in range(K))

    result["overall_loss"] = total_nll / total_tokens if total_tokens > 0 else 0.0
    result["overall_ppl"] = torch.exp(torch.tensor(result["overall_loss"])).item()
    result["overall_accuracy"] = total_correct / total_tokens if total_tokens > 0 else 0.0
    result["num_tokens_evaluated"] = total_tokens

    # Per-position metrics
    result["positions"] = {}
    for k in range(K):
        pos_nll = stats_by_pos[k]["nll"]
        pos_correct = stats_by_pos[k]["correct"]
        pos_tokens = stats_by_pos[k]["tokens"]

        pos_loss = pos_nll / pos_tokens if pos_tokens > 0 else 0.0
        pos_accuracy = pos_correct / pos_tokens if pos_tokens > 0 else 0.0

        result["positions"][k] = {
            "loss": pos_loss,
            "ppl": torch.exp(torch.tensor(pos_loss)).item(),
            "accuracy": pos_accuracy,
            "tokens": pos_tokens,
        }

    # Per-group accuracy
    result["per_group_accuracy"] = {}
    for g in range(num_groups):
        correct = stats_by_group[g]["correct"]
        total = stats_by_group[g]["total"]
        result["per_group_accuracy"][g] = {
            "accuracy": correct / total if total > 0 else 0.0,
            "correct": correct,
            "total": total,
        }

    return result
