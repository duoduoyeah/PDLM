#!/usr/bin/env python3
import os

import torch

from nanochat.sp_tokens.token_map import get_token_map


def main():
    batch_size = 64
    seq_len = 8
    token_map = get_token_map("model/tokenizer",device="cpu")
    ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
    levels = token_map.get_random_noisy_level(ids, step=None, total_steps=None)

    out_path = os.path.join(os.getcwd(), "temp", "random_noisy_levels_dump.txt")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(f"shape={tuple(levels.shape)}\n")
        handle.write(f"levels={levels.tolist()}\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
