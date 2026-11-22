"""
Dump group tokenizer contents: for each group, list the clean string representation
of every member token.

Usage:
    uv run python scripts/dump_group_members.py /path/to/tokenizer_dir
    uv run python scripts/dump_group_members.py /path/to/tokenizer_dir -o output.txt

Supports both local paths and rclone remote paths (e.g. gdrive:nanochat/group_tokenizers/n256_k1_g16_mask).
"""
import argparse
import os
import pickle
import shutil
import subprocess
import sys
import tempfile

import torch


def resolve_tokenizer_dir(path: str) -> str:
    """If path looks like an rclone remote, copy to a temp dir and return that."""
    if ":" in path and not os.path.exists(path):
        tmp = tempfile.mkdtemp(prefix="group_tok_")
        print(f"Downloading from {path} to {tmp} ...")
        subprocess.run(["rclone", "copy", path, tmp], check=True)
        return tmp
    return path


def clean_token_str(s: str) -> str:
    """Make token string safe for a single-line text dump."""
    # Replace common whitespace with visible representations
    s = s.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    # Strip surrounding whitespace but keep internal
    return s


def main():
    parser = argparse.ArgumentParser(description="Dump group members as clean token strings")
    parser.add_argument("tokenizer_dir", help="Path to tokenizer dir (local or rclone remote)")
    parser.add_argument("-o", "--output", default=None, help="Output file (default: stdout)")
    parser.add_argument("--all-dirs", action="store_true",
                        help="Process all tokenizer dirs under the given rclone path")
    args = parser.parse_args()

    if args.all_dirs:
        # List subdirs and process each
        result = subprocess.run(
            ["rclone", "lsf", args.tokenizer_dir, "--dirs-only"],
            capture_output=True, text=True, check=True,
        )
        subdirs = [d.rstrip("/") for d in result.stdout.strip().split("\n") if d.strip()]
        for subdir in sorted(subdirs):
            remote_path = f"{args.tokenizer_dir.rstrip('/')}/{subdir}"
            out_path = f"{subdir}_group_members.txt"
            print(f"\n{'='*60}")
            print(f"Processing {subdir} -> {out_path}")
            print(f"{'='*60}")
            try:
                process_one(remote_path, out_path)
            except Exception as e:
                print(f"  ERROR: {e}")
    else:
        process_one(args.tokenizer_dir, args.output)


def process_one(tokenizer_dir: str, output: str | None):
    local_dir = resolve_tokenizer_dir(tokenizer_dir)

    # Load token maps
    map_path = os.path.join(local_dir, "token_maps.pt")
    maps = torch.load(map_path, map_location="cpu", weights_only=True)

    # Load tokenizer
    tok_path = os.path.join(local_dir, "tokenizer.pkl")
    with open(tok_path, "rb") as f:
        tokenizer = pickle.load(f)

    pure_vocab_size = maps["pure_vocab_size"]
    num_groups = maps["num_groups"]
    group_to_pure_mask = maps["group_to_pure_mask"]  # (num_groups, pure_vocab_size)
    overlap_k = maps["overlap_k"]
    mask_token_id = maps.get("mask_token_id", -1)

    lines = []
    lines.append(f"# Group Tokenizer Members Dump")
    lines.append(f"# pure_vocab_size={pure_vocab_size}, num_groups={num_groups}, overlap_k={overlap_k}, mask_token_id={mask_token_id}")
    lines.append(f"# Format: group_id: token_str1, token_str2, ...")
    lines.append("")

    for g in range(num_groups):
        mask = group_to_pure_mask[g]
        member_ids = torch.where(mask)[0].tolist()

        # Decode each member token
        token_strs = []
        for tid in member_ids:
            try:
                raw = tokenizer.decode([tid])
                token_strs.append(clean_token_str(raw))
            except Exception:
                token_strs.append(f"<id:{tid}>")

        line = f"group_{g}: {', '.join(token_strs)}"
        lines.append(line)

    text = "\n".join(lines) + "\n"

    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Written to {output} ({num_groups} groups)")
    else:
        print(text)

    # Clean up temp dir if we created one
    if local_dir != tokenizer_dir and local_dir.startswith(tempfile.gettempdir()):
        shutil.rmtree(local_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
