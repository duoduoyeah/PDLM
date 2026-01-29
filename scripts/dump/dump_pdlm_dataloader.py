"""
Dump PDLM dataloader output to verify the data structure.

Usage (on Colab, after run_expc.sh setup):
    export NANOCHAT_BASE_DIR=/content/pdlm_temp_train/expc_d4_b4_n64_k1_g64_r10_test
    python -m scripts.dump.dump_pdlm_dataloader
"""

import os
import argparse
import torch
from nanochat.pdlm import PDLMConfig
from nanochat.dataloader import get_data_loader
from nanochat.group_tokenizer.token_map import get_token_map
from nanochat.tokenizer import get_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default=None,
                        help="Base dir containing tokenizer/ (default: use NANOCHAT_BASE_DIR env)")
    parser.add_argument("--block_size", type=int, default=4)
    parser.add_argument("--prefix_pure_tokens", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=512)
    args = parser.parse_args()

    # Set NANOCHAT_BASE_DIR if provided
    if args.base_dir:
        os.environ["NANOCHAT_BASE_DIR"] = args.base_dir
        print(f"Set NANOCHAT_BASE_DIR={args.base_dir}")

    # Config matching run_expc.sh defaults
    block_size = args.block_size
    prefix_pure_tokens = args.prefix_pure_tokens
    max_seq_len = args.max_seq_len
    batch_size = 2

    # Get token map info
    token_map = get_token_map(device="cpu")
    pure_vocab_size = token_map.pure_vocab_size
    num_groups = token_map.num_groups
    group_start = pure_vocab_size
    group_end = pure_vocab_size + num_groups

    print(f"Token map: pure_vocab={pure_vocab_size}, num_groups={num_groups}")
    print(f"Group token range: [{group_start}, {group_end})")
    print()

    # Create model config
    model_config = PDLMConfig(
        sequence_len=max_seq_len,
        pure_vocab_size=pure_vocab_size,
        num_groups=num_groups,
        stage="stage2",
        bucket_size=block_size,
        prefix_pure_tokens=prefix_pure_tokens,
    )

    # Create dataloader (val split)
    val_loader = get_data_loader(
        batch_size,
        max_seq_len,
        split="val",
        device="cpu",
        model_config=model_config,
        resume_state_dict=None,
    )

    # Get one batch
    inputs, targets, loss_extras, state_dict = next(val_loader)
    loss_mask = loss_extras.get("loss_mask", None)

    print(f"inputs shape: {inputs.shape}")
    print(f"targets shape: {targets.shape}")
    print(f"loss_mask shape: {loss_mask.shape if loss_mask is not None else 'None'}")
    print()

    # Analyze first sequence
    inp = inputs[0]  # (T,)
    tgt = targets[0]  # (T,)
    mask = loss_mask[0] if loss_mask is not None else None  # (T,)

    # Check token types
    def token_type(tok_id):
        if tok_id < pure_vocab_size:
            return "pure"
        elif tok_id < group_end:
            return f"group_{tok_id - group_start}"
        else:
            return "other"

    print("=== First 20 positions ===")
    print(f"{'pos':>4} | {'input':>8} | {'target':>8} | {'inp_type':>10} | {'loss_mask':>10}")
    print("-" * 60)
    for i in range(min(20, len(inp))):
        inp_type = token_type(inp[i].item())
        mask_val = mask[i].item() if mask is not None else "N/A"
        print(f"{i:>4} | {inp[i].item():>8} | {tgt[i].item():>8} | {inp_type:>10} | {mask_val!s:>10}")

    print()
    print("=== Per-block analysis (first 3 blocks) ===")
    for block_idx in range(min(3, max_seq_len // block_size)):
        block_start = block_idx * block_size
        block_end = block_start + block_size
        print(f"\nBlock {block_idx} (positions {block_start}-{block_end-1}):")

        for pos in range(block_size):
            abs_pos = block_start + pos
            inp_type = token_type(inp[abs_pos].item())
            is_group = "GROUP" if inp[abs_pos].item() >= pure_vocab_size else "PURE"
            mask_val = mask[abs_pos].item() if mask is not None else "N/A"
            print(f"  pos {pos} (abs {abs_pos}): input={inp[abs_pos].item():>5} ({is_group:>5}), "
                  f"target={tgt[abs_pos].item():>5}, loss_mask={mask_val}")

    print()
    print("=== Summary: Position 0 of each block ===")
    print("Checking if position 0 within each block has group tokens and loss_mask=True:")
    for block_idx in range(min(10, max_seq_len // block_size)):
        abs_pos = block_idx * block_size
        is_group = inp[abs_pos].item() >= pure_vocab_size
        mask_val = mask[abs_pos].item() if mask is not None else None
        status = "OK" if is_group and mask_val else "ISSUE"
        print(f"  Block {block_idx}, pos 0 (abs {abs_pos}): is_group={is_group}, loss_mask={mask_val} [{status}]")


if __name__ == "__main__":
    main()
