#!/usr/bin/env python3
"""
Dump attention masks with prefix_ar_tokens to verify the implementation.

This script generates masks for various configurations of:
- seq_len (L)
- block_size
- prefix_ar_tokens

The output helps verify that:
1. AR prefix tokens use standard causal attention
2. Block tokens use block diffusion attention
3. Block tokens can attend to AR prefix tokens
"""
import os

import torch

from nanochat.attn_masks import gen_mask, block_diff_mask, block_diff_mask_causal


def _format_mask(mask: torch.Tensor) -> str:
    """Format mask as 0/1 grid."""
    rows = mask.to(torch.int).tolist()
    return "\n".join(" ".join(str(v) for v in row) for row in rows)


def _format_mask_labeled(mask: torch.Tensor, seq_len: int, block_size: int, prefix_ar_tokens: int) -> str:
    """Format mask with row/col labels showing position info."""
    total_len = seq_len * 2
    rows = mask.to(torch.int).tolist()

    # Build header
    lines = []

    # Column labels (position indices)
    col_header = "     " + " ".join(f"{i:2d}" for i in range(total_len))
    lines.append(col_header)

    # Separator showing xt | x0 boundary
    sep = "     " + "-" * (total_len * 3 - 1)
    lines.append(sep)

    # Add annotation for structure
    # xt half: [0, seq_len-1], x0 half: [seq_len, 2*seq_len-1]
    def get_label(pos):
        half = "xt" if pos < seq_len else "x0"
        pos_in_half = pos if pos < seq_len else pos - seq_len
        if pos_in_half < prefix_ar_tokens:
            return f"{half}:AR{pos_in_half}"
        else:
            block_idx = (pos_in_half - prefix_ar_tokens) // block_size
            pos_in_block = (pos_in_half - prefix_ar_tokens) % block_size
            return f"{half}:B{block_idx}p{pos_in_block}"

    for i, row in enumerate(rows):
        label = get_label(i)
        row_str = " ".join(f"{v:2d}" for v in row)
        lines.append(f"{label:>4s} {row_str}")

    return "\n".join(lines)


def main():
    # Test configurations: (seq_len, block_size, prefix_ar_tokens)
    configs = [
        # Basic: no prefix (should match old behavior)
        (8, 4, 0),

        # With prefix_ar_tokens = 1
        (8, 4, 1),

        # With prefix_ar_tokens = 2
        (8, 4, 2),

        # With prefix_ar_tokens = 3 (almost one full block as prefix)
        (8, 4, 3),

        # Smaller example for easier visualization
        (6, 2, 1),

        # Another small example
        (6, 2, 2),
    ]

    out_path = os.path.join(os.getcwd(), "block_mask_prefix_ar_dump.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("Block Diffusion Mask with prefix_ar_tokens\n")
        f.write("=" * 80 + "\n\n")

        f.write("Legend:\n")
        f.write("  - xt:AR{i} = AR prefix token at position i in xt half\n")
        f.write("  - x0:AR{i} = AR prefix token at position i in x0 half\n")
        f.write("  - xt:B{b}p{p} = Block b, position p in xt half\n")
        f.write("  - x0:B{b}p{p} = Block b, position p in x0 half\n")
        f.write("  - 1 = can attend, 0 = masked\n")
        f.write("\n")

        for seq_len, block_size, prefix_ar_tokens in configs:
            f.write("=" * 80 + "\n")
            f.write(f"L={seq_len}, block_size={block_size}, prefix_ar_tokens={prefix_ar_tokens}\n")
            f.write(f"Total 2L = {seq_len * 2}\n")
            f.write(f"Structure per half: [{prefix_ar_tokens} AR tokens] + [{(seq_len - prefix_ar_tokens) // block_size} blocks of {block_size}]\n")
            f.write("=" * 80 + "\n\n")

            q_idx = torch.arange(seq_len * 2)[:, None]
            kv_idx = torch.arange(seq_len * 2)[None, :]

            # Non-causal mask
            mask_default = block_diff_mask(
                b=None, h=None, q_idx=q_idx, kv_idx=kv_idx,
                block_size=block_size, n=seq_len, prefix_ar_tokens=prefix_ar_tokens
            )

            # Causal mask
            mask_causal = block_diff_mask_causal(
                b=None, h=None, q_idx=q_idx, kv_idx=kv_idx,
                block_size=block_size, n=seq_len, prefix_ar_tokens=prefix_ar_tokens
            )

            f.write("--- block_diff_mask (non-causal within block) ---\n")
            f.write(_format_mask_labeled(mask_default, seq_len, block_size, prefix_ar_tokens))
            f.write("\n\n")

            f.write("--- block_diff_mask_causal (causal within block) ---\n")
            f.write(_format_mask_labeled(mask_causal, seq_len, block_size, prefix_ar_tokens))
            f.write("\n\n")

            # Also dump simple format for diff comparison
            f.write("--- Simple format (block_diff_mask) ---\n")
            f.write(_format_mask(mask_default))
            f.write("\n\n")

            f.write("--- Simple format (block_diff_mask_causal) ---\n")
            f.write(_format_mask(mask_causal))
            f.write("\n\n")

    print(f"Wrote {out_path}")

    # Also test gen_mask API
    print("\nTesting gen_mask API:")
    for seq_len, block_size, prefix_ar_tokens in [(8, 4, 2)]:
        mask = gen_mask(seq_len, block_size, attn_backend="sdpa", is_causal=True, prefix_ar_tokens=prefix_ar_tokens)
        print(f"  gen_mask(L={seq_len}, bs={block_size}, prefix={prefix_ar_tokens}): shape={mask.shape}")


if __name__ == "__main__":
    main()
