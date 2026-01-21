#!/usr/bin/env python3
# python scripts/dump_token_ids.py --tokenizer-dir path/to/tokenizer --output /temp/token_dump.txt

import os
import argparse

from nanochat.tokenizer import RustBPETokenizer
from nanochat.common import get_base_dir

repo_root = os.getcwd()
os.environ["NANOCHAT_BASE_DIR"] = os.path.join(repo_root, "model")

def main():
    parser = argparse.ArgumentParser(
        description="Dump tokenizer tokens with their ids to a text file."
    )
    parser.add_argument(
        "--tokenizer-dir",
        default=None,
        help="Path to tokenizer directory (defaults to <base_dir>/tokenizer)",
    )
    parser.add_argument(
        "--output",
        default="token_dump.txt",
        help="Output file path (default: token_dump.txt)",
    )
    args = parser.parse_args()

    tokenizer_dir = args.tokenizer_dir or os.path.join(get_base_dir(), "tokenizer")
    tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)
    enc = tokenizer.enc

    output_path = args.output
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for tid in range(enc.n_vocab):
            token_str = enc.decode([tid])
            escaped = token_str.encode("unicode_escape").decode("ascii")
            f.write(f"{tid}\t{escaped}\n")

    print(f"Wrote {enc.n_vocab} tokens to {output_path}")


if __name__ == "__main__":
    main()
