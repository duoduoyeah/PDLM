"""
Batch dump utilities for PDLM evaluation.

Dumps detailed predictions showing input tokens, model outputs,
compatibility tests, and oracle tests to text files.
"""

import torch

from nanochat.group_tokenizer.token_map import get_token_map
from nanochat.pdlm_eval.common import model_eval_context


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

    with model_eval_context(model):
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
                    matched = "SAME" if init_pred == new_pred else "CHANGED"

                    # Show what the input looked like for this test
                    input_desc = []
                    for pos in range(block_size):
                        if pos == test_pos:
                            input_desc.append(fmt_group(inputs[b, block_start + pos].item()))
                        else:
                            input_desc.append(fmt_pure(initial_preds[b, block_start + pos].item()))
                    input_str = "[" + ", ".join(input_desc) + "]"

                    f.write(f"  Pos {test_pos}: Input={input_str}\n")
                    f.write(f"         Initial={fmt_pure(init_pred)}, Re-pred={fmt_pure(new_pred)} -> {matched}\n")
                f.write("\n")

                # Oracle accuracy analysis
                f.write("ORACLE TEST: Predict each position given ground truth at others\n")
                f.write("  (Can the model predict correctly with perfect context?)\n\n")
                for test_pos in range(block_size):
                    abs_pos = block_start + test_pos
                    pred = oracle_preds[test_pos][b, abs_pos].item()
                    truth = targets[b, abs_pos].item()
                    matched = "CORRECT" if pred == truth else "WRONG"

                    # Show what the input looked like for this test
                    input_desc = []
                    for pos in range(block_size):
                        if pos == test_pos:
                            input_desc.append(fmt_group(inputs[b, block_start + pos].item()))
                        else:
                            input_desc.append(fmt_pure(targets[b, block_start + pos].item()))
                    input_str = "[" + ", ".join(input_desc) + "]"

                    f.write(f"  Pos {test_pos}: Input={input_str}\n")
                    f.write(f"         Pred={fmt_pure(pred)}, Truth={fmt_pure(truth)} -> {matched}\n")
                f.write("\n")

            f.write("\n")

    print(f"Dumped {num_sequences} sequences to {output_path}")
