"""
List all files in a HuggingFace repo without downloading anything.

Usage:
    python utils/hf_repo_tree.py duoduoyeah/bd3lm_d8
    python utils/hf_repo_tree.py duoduoyeah/gpt_d8 --token=hf_xxx
"""

import argparse
from huggingface_hub import list_repo_files


def main():
    parser = argparse.ArgumentParser(description="List HF repo files without downloading")
    parser.add_argument("repo_id", help="HuggingFace repo id, e.g. duoduoyeah/bd3lm_d8")
    parser.add_argument("--token", default=None, help="HF token for private repos")
    args = parser.parse_args()

    print(f"Repo: {args.repo_id}\n")
    files = sorted(list_repo_files(args.repo_id, repo_type="model", token=args.token))
    for f in files:
        print(f)


if __name__ == "__main__":
    main()
