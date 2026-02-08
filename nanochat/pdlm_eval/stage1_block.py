"""
Stage 1 Block evaluation for PDLM.

Supports two target modes:
- pure (default): lm_head outputs pure vocab logits, CE against pure tokens,
  accuracy via group collapse.
- group: lm_head outputs group logits, any_correct_ce_loss against group targets,
  direct argmax accuracy.

No 2L structure - model is called directly with LxL block-causal mask.
"""

import torch
import torch.nn.functional as F

from nanochat.pdlm_eval.common import model_eval_context, build_result_dict


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

    Stage 1 Block (pure-target mode): pure tokens in, LxL block-causal mask,
    lm_head outputs pure vocab logits. No 2L structure - model is called directly.

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
        attn_mask: LxL block-causal attention mask
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
    target_mode = getattr(model.config, "stage1_target_mode", "pure")
    with model_eval_context(model):
        with torch.no_grad():
            if target_mode == "group":
                result = _eval_stage1_block_group_target(
                    model, val_loader, block_size, num_batches,
                    attn_mask, device, autocast_ctx, prefix_pure_tokens
                )
            else:
                result = _eval_stage1_block_pure_target(
                    model, val_loader, block_size, num_batches,
                    attn_mask, device, autocast_ctx, prefix_pure_tokens
                )
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

    return build_result_dict(metrics_by_pos, block_size, include_accuracy=True)


def _eval_stage1_block_group_target(
    model, val_loader, block_size, num_batches,
    attn_mask, device, autocast_ctx, prefix_pure_tokens,
):
    """
    Group-target mode evaluation for stage1_block.

    Loss: any_correct_ce_loss against group targets (block_targets).
    Accuracy: Direct argmax vs block_targets (any-correct).

    Matches _eval_both_block_stage1 in both_block.py.
    """
    metrics_by_pos = {p: {"nll": 0.0, "correct": 0, "tokens": 0} for p in range(block_size)}

    for batch_idx in range(num_batches):
        inputs, targets, loss_extras, state = next(val_loader)
        B, T = inputs.shape
        block_targets = loss_extras.get("block_targets", None)
        block_loss_mask = loss_extras.get("block_loss_mask", None)

        if block_targets is None:
            raise ValueError("Group-target stage1_block eval requires block_targets in loss_extras")

        overlap_k = block_targets.size(-1)

        with autocast_ctx:
            # Forward pass: direct call (no 2L structure)
            logits = model(inputs, attn_mask=attn_mask)
            # logits: (B, T, num_groups)

            log_probs = F.log_softmax(logits.float(), dim=-1)
            preds = logits.argmax(dim=-1)  # (B, T)

            num_blocks = T // block_size

            for pos in range(block_size):
                # Skip block 0 (no prior block context)
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

    return build_result_dict(metrics_by_pos, block_size, include_accuracy=True)


def dump_stage1_block_batch(
    model,
    val_loader,
    block_size,
    attn_mask,
    device,
    autocast_ctx,
    output_path,
    tokenizer_dir=None,
    num_sequences=5,
    num_blocks_to_show=8,
):
    """
    Dump stage1_block predictions showing input, predicted, and true tokens.

    For each position shows:
    - Input pure token (at position i)
    - Predicted pure token (argmax of pure logits)
    - Predicted group (argmax after collapse to groups)
    - True pure token (pure_targets[i] = token at i + block_size)
    - Match status (pure match, group match, or wrong)

    Args:
        model: PDLM model (stage1_block)
        val_loader: validation data loader
        block_size: block size
        attn_mask: LxL block-causal attention mask
        device: device
        autocast_ctx: autocast context
        output_path: path to write txt file
        tokenizer_dir: path to tokenizer dir (default: uses get_base_dir())
        num_sequences: number of sequences to dump (default: 10)
        num_blocks_to_show: number of blocks to show detailed analysis for (default: 3)

    Output file format:
        For each sequence and block:
        Pos | Input Pure | Pred Pure | Pred Group | True Pure | Status
        ----|------------|-----------|------------|-----------|--------
          0 | "the"      | "a"       | G12        | "a"       | PURE
          1 | "cat"      | "dog"     | G45        | "fox"     | GROUP
          2 | "sat"      | "ran"     | G67        | "flew"    | WRONG
    """
    from nanochat.group_tokenizer.dump import load_tokenizer

    with model_eval_context(model):
        # Load tokenizer for decoding
        if tokenizer_dir is None:
            from nanochat.common import get_base_dir
            import os
            tokenizer_dir = os.path.join(get_base_dir(), "tokenizer")

        tokenizer = load_tokenizer(tokenizer_dir)

        # Get group mappings from model
        group_to_pure_mask = model.group_to_pure_mask  # (num_groups, pure_vocab)

        # Helper to format pure token
        def fmt_pure(tid):
            if tokenizer:
                try:
                    return repr(tokenizer.decode([tid]))
                except Exception:
                    return f"[{tid}]"
            return f"[{tid}]"

        with torch.no_grad():
            inputs, targets, loss_extras, _ = next(val_loader)
            # inputs: (B, T) - pure tokens
            # loss_extras: {"pure_targets": (B, T), "pure_to_group": (pure_vocab, overlap_k)}

            B, T = inputs.shape
            pure_targets = loss_extras.get("pure_targets", None)
            pure_to_group = loss_extras.get("pure_to_group", None)

            if pure_targets is None:
                raise ValueError("Stage 1 Block dump requires pure_targets in loss_extras")
            if pure_to_group is None:
                raise ValueError("Stage 1 Block dump requires pure_to_group in loss_extras")

            overlap_k = pure_to_group.size(-1)
            num_blocks = T // block_size
            num_sequences = min(num_sequences, B)
            num_blocks_to_show = min(num_blocks_to_show, num_blocks - 2)  # skip block 0 and last

            with autocast_ctx:
                # Forward pass: direct call (no 2L structure)
                logits = model(inputs, attn_mask=attn_mask)
                # logits: (B, T, pure_vocab_size)

                # Get pure predictions
                pred_pure = logits.argmax(dim=-1)  # (B, T)

                # Get group predictions by collapsing logits
                group_logits = logits @ group_to_pure_mask.T  # (B, T, num_groups)
                pred_groups = group_logits.argmax(dim=-1)  # (B, T)

                # Get true groups for each target token
                true_groups = pure_to_group[pure_targets]  # (B, T, overlap_k)

    # Write output
    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("PDLM STAGE 1 BLOCK DUMP - Pure-Target Mode Predictions\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Block size: {block_size}, Sequence length: {T}, Num blocks: {num_blocks}\n")
        f.write(f"Overlap_k: {overlap_k}\n")
        f.write(f"Showing {num_sequences} sequences, {num_blocks_to_show} blocks each\n\n")

        # Column widths
        pos_w = 4
        input_w = 20
        pred_pure_w = 20
        pred_group_w = 10
        true_pure_w = 20
        status_w = 15

        for b in range(num_sequences):
            f.write("=" * 80 + "\n")
            f.write(f"=== SEQUENCE {b} ===\n")
            f.write("=" * 80 + "\n\n")

            # Show full input sequence (blocks 0 through num_blocks_to_show)
            total_tokens_shown = (num_blocks_to_show + 1) * block_size
            total_tokens_shown = min(total_tokens_shown, T)
            input_tokens = [fmt_pure(inputs[b, i].item()) for i in range(total_tokens_shown)]
            f.write(f"Input: {' '.join(input_tokens)}\n\n")

            # Show block 0 (context block, not used for eval)
            f.write("-" * 80 + "\n")
            f.write(f"--- BLOCK 0 (positions 0-{block_size - 1}, context only - not evaluated) ---\n")
            f.write("-" * 80 + "\n")
            block0_tokens = [fmt_pure(inputs[b, i].item()) for i in range(block_size)]
            f.write(f"  {' '.join(block0_tokens)}\n\n")

            # For each block (skip block 0 and last block)
            for block_idx in range(1, 1 + num_blocks_to_show):
                block_start = block_idx * block_size
                # Predictions at block_start predict tokens at block_start + block_size
                target_start = block_start + block_size

                f.write("-" * 80 + "\n")
                f.write(f"--- BLOCK {block_idx} (positions {block_start}-{block_start + block_size - 1}, ")
                f.write(f"predicts tokens at {target_start}-{target_start + block_size - 1}) ---\n")
                f.write("-" * 80 + "\n")

                # Header
                header = (
                    f"{'Pos':>{pos_w}} | "
                    f"{'Input Pure Token':<{input_w}} | "
                    f"{'Pred Pure':<{pred_pure_w}} | "
                    f"{'Pred Grp':<{pred_group_w}} | "
                    f"{'True Pure':<{true_pure_w}} | "
                    f"{'Status':<{status_w}}"
                )
                f.write(header + "\n")
                f.write("-" * len(header) + "\n")

                # Stats for this block
                pure_matches = 0
                group_matches = 0
                wrong = 0

                for pos in range(block_size):
                    pos_in_seq = block_start + pos
                    target_pos = target_start + pos

                    # Input pure token at this position
                    input_tid = inputs[b, pos_in_seq].item()
                    input_str = fmt_pure(input_tid)

                    # Predicted pure token
                    pred_pure_tid = pred_pure[b, pos_in_seq].item()
                    pred_pure_str = fmt_pure(pred_pure_tid)

                    # Predicted group
                    pred_group_id = pred_groups[b, pos_in_seq].item()
                    pred_group_str = f"G{pred_group_id}"

                    # True pure token
                    true_pure_tid = pure_targets[b, pos_in_seq].item()
                    true_pure_str = fmt_pure(true_pure_tid)

                    # True groups (all valid groups for the true token)
                    true_group_ids = true_groups[b, pos_in_seq]  # (overlap_k,)
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
                        f"{pos:>{pos_w}} | "
                        f"{input_str:<{input_w}} | "
                        f"{pred_pure_str:<{pred_pure_w}} | "
                        f"{pred_group_str:<{pred_group_w}} | "
                        f"{true_pure_str:<{true_pure_w}} | "
                        f"{status:<{status_w}}"
                    )
                    f.write(row + "\n")

                # Block summary
                f.write("\n")
                f.write(f"Block {block_idx} summary: ")
                f.write(f"Pure={pure_matches}/{block_size} ")
                f.write(f"Group={group_matches}/{block_size} ")
                f.write(f"Wrong={wrong}/{block_size}\n")
                f.write("\n")

            f.write("\n")

    print(f"Dumped {num_sequences} sequences to {output_path}")
