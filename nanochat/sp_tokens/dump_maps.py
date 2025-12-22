#!/usr/bin/env python3
"""
Dump the maps produced by inject_tokens.py:
- pure_to_noisy_map
- noisy_level_map

Additional inverted view: for each noisy token, list the pure token IDs that map to it.

Usage examples:
  python nanochat/sp_tokens/dump_maps.py --tokenizer-dir path/to/tokenizer --output-dir out_dir
Defaults to base_dir/tokenizer if tokenizer-dir is not provided.
"""
import os
import argparse
import torch

from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer


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
    noisy_level = maps["noisy_level_map"]  # shape: (new_vocab, 2)

    # Load tokenizer to decode tokens
    tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)
    enc = tokenizer.enc

    # Dump pure_to_noisy_map as tab-separated: token_id \t level \t fanout_idx \t target_id
    ptn_path = os.path.join(args.output_dir, "pure_to_noisy_map.txt")
    with open(ptn_path, "w", encoding="utf-8") as f:
        vocab_size, num_levels, fanout = pure_to_noisy.shape
        for tid in range(vocab_size):
            for lvl in range(num_levels):
                for fi in range(fanout):
                    f.write(f"{tid}\t{lvl}\t{fi}\t{int(pure_to_noisy[tid, lvl, fi])}\n")

    # Dump noisy_level_map: token_id \t low_level \t high_level
    nl_path = os.path.join(args.output_dir, "noisy_level_map.txt")
    with open(nl_path, "w", encoding="utf-8") as f:
        for tid, lvls in enumerate(noisy_level.tolist()):
            f.write(f"{tid}\t{lvls[0]}\t{lvls[1]}\n")

    # Invert pure_to_noisy: noisy_token_id -> list of pure token ids (deduped)
    inverted = {}
    vocab_size = pure_to_noisy.shape[0]
    for pure_id in range(vocab_size):
        targets = set(int(t) for t in pure_to_noisy[pure_id].reshape(-1).tolist())
        for t in targets:
            inverted.setdefault(t, []).append(pure_id)

    inv_path = os.path.join(args.output_dir, "noisy_to_pure.txt")
    with open(inv_path, "w", encoding="utf-8") as f:
        for noisy_id, pure_ids in sorted(inverted.items()):
            token_str = enc.decode([noisy_id]) if 0 <= noisy_id < enc.n_vocab else "<out_of_range>"
            f.write(f"{token_str}\t{noisy_id}\t{pure_ids}\n")

    print(f"Wrote pure_to_noisy_map to {ptn_path}")
    print(f"Wrote noisy_level_map to {nl_path}")
    print(f"Wrote inverted map to {inv_path}")


if __name__ == "__main__":
    main()
