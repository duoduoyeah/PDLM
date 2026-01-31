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

                    # Get group_to_pure_mask for collapsing logits to group predictions
                    group_to_pure_mask = model.group_to_pure_mask  # (num_groups, pure_vocab)

                    # Compute metrics for k=0 (main model prediction)
                    _update_position_stats_pure(
                        stats_by_pos[0],
                        stats_by_group,
                        main_logits,
                        targets[:, :, 0],  # (B, T) pure token targets
                        pure_to_group,
                        group_to_pure_mask,
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
                            group_to_pure_mask,
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
                                pure_to_group, group_to_pure_mask):
    """
    Update per-position and per-group statistics for pure target mode.

    Loss is standard CE against pure tokens.
    Accuracy is computed by collapsing logits to group space (via group_to_pure_mask),
    taking argmax to get the predicted group, then checking if that group is among
    the target token's valid groups.

    Args:
        pos_stats: dict to accumulate position-level stats
        group_stats: dict to accumulate per-group stats
        logits: (B, T, pure_vocab_size) prediction logits over pure tokens
        targets: (B, T) pure token targets
        pure_to_group: (pure_vocab_size, overlap_k) mapping from pure token to its valid groups
        group_to_pure_mask: (num_groups, pure_vocab_size) mask for collapsing logits to groups
    """
    B, T, V = logits.shape

    # Compute log probabilities for loss
    log_probs = F.log_softmax(logits.float(), dim=-1)  # (B, T, V)

    # Standard CE loss: gather log prob of target
    targets_expanded = targets.unsqueeze(-1)  # (B, T, 1)
    nll = -log_probs.gather(-1, targets_expanded).squeeze(-1)  # (B, T)

    # For accuracy: collapse logits to group space, then argmax
    # group_to_pure_mask: (num_groups, pure_vocab) -> transpose to (pure_vocab, num_groups)
    group_logits = logits @ group_to_pure_mask.T  # (B, T, num_groups)
    pred_group = group_logits.argmax(dim=-1)  # (B, T) predicted group

    # Get valid groups for each target token
    target_groups = pure_to_group[targets]  # (B, T, overlap_k)
    valid_mask = target_groups >= 0  # (B, T, overlap_k)

    # Correct if predicted group matches any valid target group
    correct = (pred_group.unsqueeze(-1) == target_groups).any(dim=-1) & valid_mask.any(dim=-1)  # (B, T)

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


def dump_mtp_batch(
    model,
    val_loader,
    device,
    autocast_ctx,
    output_path,
    tokenizer_dir=None,
    num_sequences=5,
    num_positions_to_show=8,
    start_pos=5,
):
    """
    Dump MTP predictions showing K future predictions per position.

    For each position shows predictions for k=0..K-1 with:
    - Predicted pure token (argmax of pure logits)
    - Predicted group (argmax after collapse to groups)
    - True pure token
    - Match status (PURE, GROUP, or WRONG)

    Args:
        model: GPTMTP model (pure mode)
        val_loader: validation data loader
        device: device
        autocast_ctx: autocast context
        output_path: path to write txt file
        tokenizer_dir: path to tokenizer dir (default: uses get_base_dir())
        num_sequences: number of sequences to dump (default: 5)
        num_positions_to_show: number of positions to show per sequence (default: 8)
        start_pos: starting position index for display (default: 5)
    """
    from nanochat.group_tokenizer.dump import load_tokenizer

    was_training = model.training
    model.eval()

    K = model.config.n_future_tokens
    target_mode = getattr(model.config, "stage1_target_mode", "group")
    assert target_mode == "pure", f"dump_mtp_batch only supports pure mode, got {target_mode}"

    # Load tokenizer for decoding
    if tokenizer_dir is None:
        import os
        from nanochat.common import get_base_dir
        tokenizer_dir = os.path.join(get_base_dir(), "tokenizer")

    tokenizer = load_tokenizer(tokenizer_dir)

    # Get group mask from model
    group_to_pure_mask = model.group_to_pure_mask  # (num_groups, pure_vocab)

    def fmt_pure(tid):
        if tokenizer:
            try:
                return repr(tokenizer.decode([tid]))
            except Exception:
                return f"[{tid}]"
        return f"[{tid}]"

    with torch.no_grad():
        inputs, targets, loss_extras, _ = next(val_loader)
        # inputs: (B, T) pure tokens
        # targets: (B, T, K) pure token IDs (pure mode)

        B, T = inputs.shape
        pure_to_group = loss_extras.get("pure_to_group", None)
        assert pure_to_group is not None, "pure_to_group required for MTP dump"

        overlap_k = pure_to_group.size(-1)
        num_sequences = min(num_sequences, B)

        # Clamp positions to valid range (need start_pos + num_positions_to_show + K - 1 < T)
        max_start = T - K - num_positions_to_show
        if start_pos > max_start:
            start_pos = max(0, max_start)
        end_pos = min(start_pos + num_positions_to_show, T - K)

        with autocast_ctx:
            # Forward pass
            main_logits, mtp_logits = model.forward_for_eval(
                inputs, targets, pure_to_group=pure_to_group
            )
            # main_logits: (B, T, pure_vocab_size) for k=0
            # mtp_logits: (B, T, K-1, pure_vocab_size) for k=1..K-1

            # Stack into (B, T, K, V)
            all_logits = torch.cat([
                main_logits.unsqueeze(2),  # (B, T, 1, V)
                mtp_logits,                # (B, T, K-1, V)
            ], dim=2)  # (B, T, K, V)

            # Pure predictions: argmax over pure vocab
            pred_pure = all_logits.argmax(dim=-1)  # (B, T, K)

            # Group predictions: collapse logits via group_to_pure_mask
            # group_to_pure_mask: (num_groups, pure_vocab) -> transpose to (pure_vocab, num_groups)
            group_logits = all_logits @ group_to_pure_mask.T  # (B, T, K, num_groups)
            pred_groups = group_logits.argmax(dim=-1)  # (B, T, K)

    # Write output
    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("MTP STAGE 1 DUMP - Multi-Token Predictions\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Sequence length: {T}, K: {K}, Target mode: {target_mode}\n")
        f.write(f"Overlap_k: {overlap_k}\n")
        f.write(f"Showing {num_sequences} sequences, {end_pos - start_pos} positions each\n\n")

        # Column widths
        k_w = 3
        pred_pure_w = 20
        pred_group_w = 10
        true_pure_w = 20
        status_w = 10

        for b in range(num_sequences):
            f.write("=" * 80 + "\n")
            f.write(f"=== SEQUENCE {b} ===\n")
            f.write("=" * 80 + "\n\n")

            # Show input tokens around the positions we'll display
            show_start = max(0, start_pos - 2)
            show_end = min(T, end_pos + K + 2)
            input_tokens = [fmt_pure(inputs[b, i].item()) for i in range(show_start, show_end)]
            f.write(f"Input: {' '.join(input_tokens)}\n\n")

            for pos in range(start_pos, end_pos):
                input_str = fmt_pure(inputs[b, pos].item())
                target_end = pos + K
                f.write(f"--- POSITION {pos} (input: {input_str}) → predicts tokens {pos + 1}-{target_end} ---\n")

                # Header
                header = (
                    f"{'k':>{k_w}} | "
                    f"{'Pred Pure':<{pred_pure_w}} | "
                    f"{'Pred Grp':<{pred_group_w}} | "
                    f"{'True Pure':<{true_pure_w}} | "
                    f"{'Status':<{status_w}}"
                )
                f.write("  " + header + "\n")

                # Stats for this position
                pure_matches = 0
                group_matches = 0
                wrong = 0

                for k in range(K):
                    # Predicted pure token
                    pred_pure_tid = pred_pure[b, pos, k].item()
                    pred_pure_str = fmt_pure(pred_pure_tid)

                    # Predicted group
                    pred_group_id = pred_groups[b, pos, k].item()
                    pred_group_str = f"G{pred_group_id}"

                    # True pure token: targets[b, pos, k]
                    true_pure_tid = targets[b, pos, k].item()
                    true_pure_str = fmt_pure(true_pure_tid)

                    # True groups for this target token
                    true_group_ids = pure_to_group[true_pure_tid]  # (overlap_k,)
                    valid_mask = true_group_ids >= 0

                    # Determine match status
                    if pred_pure_tid == true_pure_tid:
                        status = "PURE"
                        pure_matches += 1
                    elif valid_mask.any() and (true_group_ids[valid_mask] == pred_group_id).any():
                        status = "GROUP"
                        group_matches += 1
                    else:
                        status = "WRONG"
                        wrong += 1

                    row = (
                        f"{k:>{k_w}} | "
                        f"{pred_pure_str:<{pred_pure_w}} | "
                        f"{pred_group_str:<{pred_group_w}} | "
                        f"{true_pure_str:<{true_pure_w}} | "
                        f"{status:<{status_w}}"
                    )
                    f.write("  " + row + "\n")

                f.write(f"\nPosition {pos} summary: ")
                f.write(f"Pure={pure_matches}/{K} ")
                f.write(f"Group={group_matches}/{K} ")
                f.write(f"Wrong={wrong}/{K}\n\n")

            f.write("\n")

    if was_training:
        model.train()

    print(f"Dumped {num_sequences} sequences to {output_path}")
