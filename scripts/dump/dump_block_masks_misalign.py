#!/usr/bin/env python3
import os

import torch

from nanochat.attn_masks import block_diff_mask, block_diff_mask_causal


def _format_mask(mask: torch.Tensor) -> str:
  rows = mask.to(torch.int).tolist()
  return "\n".join(" ".join(str(v) for v in row) for row in rows)


def main():
  seq_len = 8
  block_size = 3
  q_idx = torch.arange(seq_len * 2)[:, None]
  kv_idx = torch.arange(seq_len * 2)[None, :]

  mask_default = block_diff_mask(
    b=None, h=None, q_idx=q_idx, kv_idx=kv_idx, block_size=block_size, n=seq_len)
  mask_causal = block_diff_mask_causal(
    b=None, h=None, q_idx=q_idx, kv_idx=kv_idx, block_size=block_size, n=seq_len)

  out_path = os.path.join(os.getcwd(), "block_mask_dump_misalign_L8_bs3.txt")
  with open(out_path, "w", encoding="utf-8") as f:
    f.write(f"L={seq_len} block_size={block_size}\n")
    f.write("mask=block_diff_mask\n")
    f.write(_format_mask(mask_default))
    f.write("\n\n")
    f.write("mask=block_diff_mask_causal\n")
    f.write(_format_mask(mask_causal))
    f.write("\n")

  print(f"Wrote {out_path}")


if __name__ == "__main__":
  main()
