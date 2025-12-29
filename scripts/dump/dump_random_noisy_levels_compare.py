#!/usr/bin/env python3
import os

import torch

from nanochat.sp_tokens.token_map import get_token_map


def main():
    batch_size = 4
    seq_len = 32
    vocab_size = 4096

    token_map = get_token_map("model/tokenizer",device="cpu")
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    noisy_levels = token_map.get_random_noisy_level(targets, step=None, total_steps=None)
    inputs = token_map.noise_tokens(targets, noisy_levels)

    same_mask = inputs.eq(targets)
    same_all = bool(same_mask.all().item())
    same_count = int(same_mask.sum().item())

    out_dir = os.path.join(os.getcwd(), "temp")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "random_noisy_levels_compare_dump.txt")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(f"shape={tuple(targets.shape)}\n")
        handle.write(f"inputs_targets_all_equal={same_all}\n")
        handle.write(f"inputs_targets_equal_count={same_count}\n")
        handle.write(f"targets={targets.tolist()}\n")
        handle.write(f"noisy_levels={noisy_levels.tolist()}\n")
        handle.write(f"inputs={inputs.tolist()}\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
