#!/usr/bin/env python3
"""
Dump the three maps produced by inject_tokens.py:
- pure_to_noisy_map
- tokens_to_pure_map
- noisy_level_map

Usage examples:
  python nanochat/sp_tokens/dump_maps.py --tokenizer-dir path/to/tokenizer --output-dir out_dir
Defaults to base_dir/tokenizer if tokenizer-dir is not provided.
"""
import os
import argparse
import torch

from nanochat.common import get_base_dir


def main():
    parser = argparse.ArgumentParser(description="Dump token maps to text files.")
    parser.add_argument(
        "--tokenizer-dir",
        default=None,
        help="Path to tokenizer directory (defaults to <base_dir>/tokenizer)",
    )
    parser.add_argument(
        "--output-dir",
        default="maps_dump",
        help="Directory to write dump files (default: maps_dump)",
    )
    args = parser.parse_args()

    tokenizer_dir = args.tokenizer_dir or os.path.join(get_base_dir(), "tokenizer")
    map_path = os.path.join(tokenizer_dir, "token_maps.pt")
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"Token maps not found at {map_path}")

    maps = torch.load(map_path, map_location="cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    pure_to_noisy = maps["pure_to_noisy_map"]  # shape: (vocab_size, num_levels, fanout)
    tokens_to_pure = maps["tokens_to_pure_map"]  # shape: (new_vocab,)
    noisy_level = maps["noisy_level_map"]  # shape: (new_vocab,)

    # Dump pure_to_noisy_map as tab-separated: token_id \t level \t fanout_idx \t target_id
    ptn_path = os.path.join(args.output_dir, "pure_to_noisy_map.txt")
    with open(ptn_path, "w", encoding="utf-8") as f:
        vocab_size, num_levels, fanout = pure_to_noisy.shape
        for tid in range(vocab_size):
            for lvl in range(num_levels):
                for fi in range(fanout):
                    f.write(f"{tid}\t{lvl}\t{fi}\t{int(pure_to_noisy[tid, lvl, fi])}\n")

    # Dump tokens_to_pure_map: token_id \t pure_id
    ttp_path = os.path.join(args.output_dir, "tokens_to_pure_map.txt")
    with open(ttp_path, "w", encoding="utf-8") as f:
        for tid, pid in enumerate(tokens_to_pure.tolist()):
            f.write(f"{tid}\t{pid}\n")

    # Dump noisy_level_map: token_id \t level
    nl_path = os.path.join(args.output_dir, "noisy_level_map.txt")
    with open(nl_path, "w", encoding="utf-8") as f:
        for tid, lvl in enumerate(noisy_level.tolist()):
            f.write(f"{tid}\t{lvl}\n")

    print(f"Wrote pure_to_noisy_map to {ptn_path}")
    print(f"Wrote tokens_to_pure_map to {ttp_path}")
    print(f"Wrote noisy_level_map to {nl_path}")


if __name__ == "__main__":
    main()
