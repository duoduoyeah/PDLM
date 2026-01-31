"""
MTP Stage 1 Evaluation Module.

Computes loss, perplexity, and transition accuracy metrics for MTP models.

Evaluation metrics:
1. Overall: loss, PPL, transition_accuracy
2. Per-position (k=0..K-1): loss, PPL, transition_accuracy
3. Per-group: transition_accuracy for each of the ~64 groups

Supports two target modes:
- Pure mode: targets are (B, T, K) pure token IDs, loss is standard CE,
             accuracy computed by collapsing predictions to groups
- Group mode: targets are (B, T, K, overlap_k) group token IDs,
             loss is any-correct CE, accuracy is group prediction accuracy

Note:
- Uses teacher forcing (matches training behavior)
- Transition accuracy = group-level accuracy in both modes
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
                    targets: depends on stage1_target_mode:
                             - pure mode: (B, T, K) pure token ids
                             - group mode: (B, T, K, overlap_k) group token ids
        num_batches: number of batches to evaluate
        device: device to run on
        autocast_ctx: autocast context for mixed precision

    Returns:
        dict with evaluation results:
        {
            "overall_loss": float,
            "overall_ppl": float,
            "overall_accuracy": float,  # group-level accuracy
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
    target_mode = getattr(model.config, "stage1_target_mode", "group")

    # Accumulators per position k
    stats_by_pos = {k: {"nll": 0.0, "correct": 0, "tokens": 0} for k in range(K)}

    # Accumulators per group (for transition accuracy)
    stats_by_group = {g: {"correct": 0, "total": 0} for g in range(num_groups)}

    with torch.no_grad():
        for batch_idx in range(num_batches):
            inputs, targets, loss_extras, state = next(val_loader)
            # inputs: (B, T) pure tokens

            B, T = inputs.shape

            with autocast_ctx:
                if target_mode == "pure":
                    # Pure mode: targets are (B, T, K) pure token IDs
                    pure_to_group = loss_extras.get("pure_to_group") if loss_extras else None
                    assert pure_to_group is not None, "pure_to_group required for pure mode eval"

                    # Forward pass
                    main_logits, mtp_logits = model.forward_for_eval(
                        inputs, targets, pure_to_group=pure_to_group
                    )
                    # main_logits: (B, T, pure_vocab_size) for k=0
                    # mtp_logits: (B, T, K-1, pure_vocab_size) for k=1..K-1

                    # Get collapsed group head for accuracy computation
                    group_head = model.get_group_head_weight()  # (num_groups, n_embd)

                    # Compute metrics for k=0 (main model prediction)
                    _update_position_stats_pure(
                        stats_by_pos[0],
                        stats_by_group,
                        main_logits,
                        targets[:, :, 0],  # (B, T) pure token targets
                        pure_to_group,
                        group_head,
                        model,
                    )

                    # Compute metrics for k=1..K-1 (MTP head predictions)
                    for k in range(1, K):
                        step_logits = mtp_logits[:, :, k - 1, :]  # (B, T, pure_vocab_size)
                        step_targets = targets[:, :, k]  # (B, T) pure token targets
                        _update_position_stats_pure(
                            stats_by_pos[k],
                            stats_by_group,
                            step_logits,
                            step_targets,
                            pure_to_group,
                            group_head,
                            model,
                        )
                else:
                    # Group mode: targets are (B, T, K, overlap_k) group token IDs
                    # Forward pass
                    main_logits, mtp_logits = model.forward_for_eval(inputs, targets)
                    # main_logits: (B, T, num_groups) for k=0
                    # mtp_logits: (B, T, K-1, num_groups) for k=1..K-1

                    # Compute metrics for k=0 (main model prediction)
                    _update_position_stats_group(
                        stats_by_pos[0],
                        stats_by_group,
                        main_logits,
                        targets[:, :, 0, :],  # (B, T, overlap_k)
                    )

                    # Compute metrics for k=1..K-1 (MTP head predictions)
                    for k in range(1, K):
                        step_logits = mtp_logits[:, :, k - 1, :]  # (B, T, num_groups)
                        step_targets = targets[:, :, k, :]  # (B, T, overlap_k)
                        _update_position_stats_group(
                            stats_by_pos[k],
                            stats_by_group,
                            step_logits,
                            step_targets,
                        )

    if was_training:
        model.train()

    return _build_result(stats_by_pos, stats_by_group, K, num_groups)


def _update_position_stats_pure(pos_stats, group_stats, logits, targets,
                                pure_to_group, group_head, model):
    """
    Update per-position and per-group statistics for pure target mode.

    Loss is standard CE against pure tokens.
    Accuracy is computed by collapsing predictions to groups.

    Args:
        pos_stats: dict to accumulate position-level stats
        group_stats: dict to accumulate per-group stats
        logits: (B, T, pure_vocab_size) prediction logits over pure tokens
        targets: (B, T) pure token targets
        pure_to_group: (pure_vocab_size, overlap_k) mapping
        group_head: (num_groups, n_embd) collapsed group head for accuracy
        model: GPTMTP model for getting hidden states
    """
    B, T, V = logits.shape

    # Compute log probabilities for loss
    log_probs = F.log_softmax(logits.float(), dim=-1)  # (B, T, V)

    # Standard CE loss: gather log prob of target
    targets_expanded = targets.unsqueeze(-1)  # (B, T, 1)
    nll = -log_probs.gather(-1, targets_expanded).squeeze(-1)  # (B, T)

    # For accuracy: collapse logits to group predictions
    # Method: get predicted token, then check if its group matches target's group
    pred_pure = logits.argmax(dim=-1)  # (B, T) predicted pure tokens
    pred_groups = pure_to_group[pred_pure]  # (B, T, overlap_k) possible groups of predicted token

    # Get groups for target tokens
    target_groups = pure_to_group[targets]  # (B, T, overlap_k) valid groups for target

    # Accuracy: check if any predicted group matches any target group
    # pred_groups[:, :, :, None]: (B, T, overlap_k, 1)
    # target_groups[:, :, None, :]: (B, T, 1, overlap_k)
    pred_groups_exp = pred_groups.unsqueeze(-1)  # (B, T, overlap_k, 1)
    target_groups_exp = target_groups.unsqueeze(-2)  # (B, T, 1, overlap_k)

    # Mask out invalid groups (-1)
    valid_pred = pred_groups_exp >= 0
    valid_target = target_groups_exp >= 0

    # Check if any valid pred group matches any valid target group
    matches = (pred_groups_exp == target_groups_exp) & valid_pred & valid_target
    correct = matches.any(dim=-1).any(dim=-1)  # (B, T)

    # Update position stats
    pos_stats["nll"] += nll.sum().item()
    pos_stats["correct"] += correct.sum().item()
    pos_stats["tokens"] += B * T

    # Update per-group stats using first valid target group
    first_target_group = target_groups[:, :, 0]  # (B, T)
    targets_flat = first_target_group.reshape(-1)
    correct_flat = correct.reshape(-1)
    valid_flat = first_target_group.reshape(-1) >= 0

    num_groups = len(group_stats)
    for g in range(num_groups):
        group_mask = (targets_flat == g) & valid_flat
        group_stats[g]["total"] += group_mask.sum().item()
        group_stats[g]["correct"] += (group_mask & correct_flat).sum().item()


def _update_position_stats_group(pos_stats, group_stats, logits, targets):
    """
    Update per-position and per-group statistics for group target mode.

    Loss is any-correct CE against group tokens.
    Accuracy is group prediction accuracy.

    Args:
        pos_stats: dict to accumulate position-level stats
        group_stats: dict to accumulate per-group stats
        logits: (B, T, num_groups) prediction logits
        targets: (B, T, overlap_k) all valid target groups for any-correct loss
    """
    B, T, V = logits.shape

    # Compute log probabilities
    log_probs = F.log_softmax(logits.float(), dim=-1)  # (B, T, V)

    # Any-correct loss: -log(Σ P(valid_g)) = -logsumexp(log_probs for valid groups)
    # targets: (B, T, overlap_k) where invalid entries are -1
    valid_mask = targets >= 0  # (B, T, overlap_k)
    safe_targets = targets.clamp(min=0)  # Replace -1 with 0 for gather (will be masked)

    # Gather log probs for all target positions
    valid_log_probs = log_probs.gather(-1, safe_targets)  # (B, T, overlap_k)

    # Mask out invalid targets with -inf so they don't contribute to logsumexp
    valid_log_probs = valid_log_probs.masked_fill(~valid_mask, float("-inf"))

    # NLL using any-correct formula: -logsumexp over valid groups
    nll = -torch.logsumexp(valid_log_probs, dim=-1)  # (B, T)

    # Accuracy: correct if prediction matches ANY valid target
    preds = logits.argmax(dim=-1)  # (B, T)
    correct = (preds.unsqueeze(-1) == targets).any(dim=-1)  # (B, T)

    # A position is valid if it has at least one valid target (first target is valid)
    position_valid = valid_mask[:, :, 0]  # (B, T)
    nll_masked = nll * position_valid.float()
    correct_masked = correct & position_valid

    # Update position stats
    pos_stats["nll"] += nll_masked.sum().item()
    pos_stats["correct"] += correct_masked.sum().item()
    pos_stats["tokens"] += position_valid.sum().item()

    # Update per-group stats using first valid target for group tracking
    first_target = targets[:, :, 0]  # (B, T)
    targets_flat = first_target.reshape(-1)
    correct_flat = correct_masked.reshape(-1)
    valid_flat = position_valid.reshape(-1)

    for g in range(V):
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
