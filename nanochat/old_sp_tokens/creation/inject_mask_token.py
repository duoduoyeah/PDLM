#!/usr/bin/env python3
"""
Inject a single <|MASK|> special token into a RustBPE tokenizer.

Usage:
  uv run python nanochat/sp_tokens/inject_mask_token.py \
    --tokenizer-dir path/to/tokenizer \
    --output-tokenizer-dir path/to/output
"""

import os
import argparse
import pickle

import tiktoken
import torch

from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer

repo_root = os.getcwd()
os.environ["NANOCHAT_BASE_DIR"] = os.path.join(repo_root, "model")


def _get_special_tokens(enc):
    if hasattr(enc, "_special_tokens"):
        return dict(enc._special_tokens)
    return {tok: enc.encode_single_token(tok) for tok in enc.special_tokens_set}


def _extend_token_bytes(input_dir, output_dir, new_vocab_size):
    token_bytes_path = os.path.join(input_dir, "token_bytes.pt")
    if not os.path.exists(token_bytes_path):
        return False
    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location="cpu")
    if token_bytes.numel() > new_vocab_size:
        raise ValueError(
            f"token_bytes has {token_bytes.numel()} entries, expected <= {new_vocab_size}"
        )
    if token_bytes.numel() < new_vocab_size:
        pad = torch.zeros(new_vocab_size - token_bytes.numel(), dtype=token_bytes.dtype)
        token_bytes = torch.cat([token_bytes, pad], dim=0)
    output_path = os.path.join(output_dir, "token_bytes.pt")
    with open(output_path, "wb") as f:
        torch.save(token_bytes, f)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Inject <|MASK|> special token into a tokenizer."
    )
    parser.add_argument(
        "--tokenizer-dir",
        default=None,
        help="Path to tokenizer directory (defaults to <base_dir>/tokenizer)",
    )
    parser.add_argument(
        "--output-tokenizer-dir",
        required=True,
        help="Path to write the new tokenizer directory",
    )
    parser.add_argument(
        "--token",
        default="<|MASK|>",
        help="Special token string to inject (default: <|MASK|>)",
    )
    args = parser.parse_args()

    tokenizer_dir = args.tokenizer_dir or os.path.join(get_base_dir(), "tokenizer")
    output_dir = args.output_tokenizer_dir

    tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)
    enc = tokenizer.enc

    special_tokens = _get_special_tokens(enc)
    if args.token in special_tokens:
        raise ValueError(f"Special token already present: {args.token}")

    mergeable_ranks = enc._mergeable_ranks
    expected_vocab = len(mergeable_ranks) + len(special_tokens)
    if enc.n_vocab != expected_vocab:
        raise ValueError(
            f"Unexpected vocab size: enc.n_vocab={enc.n_vocab}, expected={expected_vocab}"
        )

    new_id = expected_vocab
    special_tokens[args.token] = new_id

    new_enc = tiktoken.Encoding(
        name="rustbpe_with_mask",
        pat_str=enc._pat_str,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    os.makedirs(output_dir, exist_ok=True)
    pickle_path = os.path.join(output_dir, "tokenizer.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(new_enc, f)

    wrote_token_bytes = _extend_token_bytes(tokenizer_dir, output_dir, new_enc.n_vocab)

    print(f"Added {args.token} as id {new_id}")
    print(f"Saved tokenizer encoding to {pickle_path}")
    if wrote_token_bytes:
        print(f"Updated token_bytes.pt in {output_dir}")


if __name__ == "__main__":
    main()
