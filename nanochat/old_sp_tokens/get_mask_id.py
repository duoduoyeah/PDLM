#!/usr/bin/env python3
"""Print the token id for the <|MASK|> special token."""

import argparse
import os

from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer

repo_root = os.getcwd()
os.environ["NANOCHAT_BASE_DIR"] = os.path.join(repo_root, "model")

def main() -> None:
    parser = argparse.ArgumentParser(description="Get the token id for <|MASK|>.")
    parser.add_argument(
        "--tokenizer-dir",
        default=None,
        help="Path to tokenizer directory (defaults to <base_dir>/tokenizer)",
    )
    parser.add_argument(
        "--token",
        default="<|MASK|>",
        help="Special token string to look up (default: <|MASK|>)",
    )
    args = parser.parse_args()

    tokenizer_dir = args.tokenizer_dir or os.path.join(get_base_dir(), "tokenizer")
    tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)

    try:
        token_id = tokenizer.encode_special(args.token)
    except KeyError as exc:
        raise ValueError(f"Special token not found: {args.token}") from exc
    if token_id is None:
        raise ValueError(f"Special token not found: {args.token}")
    print(token_id)


if __name__ == "__main__":
    main()
