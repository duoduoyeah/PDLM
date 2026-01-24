"""
Dump/inspect token map contents for debugging and analysis.

Usage:
    python -m nanochat.group_tokenizer.dump /path/to/tokenizer_dir
    python -m nanochat.group_tokenizer.dump /path/to/tokenizer_dir --group 5
    python -m nanochat.group_tokenizer.dump /path/to/tokenizer_dir --token 123
"""
import argparse
import os
import pickle
import torch
from typing import Optional


def load_token_maps(tokenizer_dir: str) -> dict:
    """Load token_maps.pt from directory."""
    map_path = os.path.join(tokenizer_dir, "token_maps.pt")
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"token_maps.pt not found at {map_path}")
    return torch.load(map_path, map_location="cpu", weights_only=True)


def load_tokenizer(tokenizer_dir: str):
    """Load tokenizer.pkl from directory."""
    tok_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
    if not os.path.exists(tok_path):
        return None
    with open(tok_path, "rb") as f:
        return pickle.load(f)


def dump_overview(maps: dict):
    """Print overview of token maps."""
    print("=" * 60)
    print("TOKEN MAP OVERVIEW")
    print("=" * 60)
    print(f"pure_vocab_size:  {maps['pure_vocab_size']}")
    print(f"num_groups:       {maps['num_groups']}")
    print(f"overlap_k:        {maps['overlap_k']}")
    print(f"mask_token_id:    {maps['mask_token_id']}")
    print()

    # Matrix shapes
    print("Matrices:")
    print(f"  pure_to_group:      {tuple(maps['pure_to_group'].shape)}")
    print(f"  group_to_pure_mask: {tuple(maps['group_to_pure_mask'].shape)}")
    print()

    # Group size stats
    group_sizes = maps["group_to_pure_mask"].sum(dim=1)
    print("Group size stats:")
    print(f"  min:  {group_sizes.min().item()}")
    print(f"  max:  {group_sizes.max().item()}")
    print(f"  mean: {group_sizes.float().mean().item():.1f}")
    print(f"  std:  {group_sizes.float().std().item():.1f}")
    print()


def dump_group(maps: dict, group_id: int, tokenizer=None, max_tokens: int = 50):
    """Dump details of a specific group."""
    num_groups = maps["num_groups"]
    pure_vocab_size = maps["pure_vocab_size"]

    if group_id < 0 or group_id >= num_groups:
        print(f"Error: group_id must be in [0, {num_groups})")
        return

    print("=" * 60)
    print(f"GROUP {group_id} DETAILS")
    print("=" * 60)

    # Get members
    mask = maps["group_to_pure_mask"][group_id]
    member_ids = torch.where(mask)[0].tolist()

    print(f"Group token id: {pure_vocab_size + group_id}")
    print(f"Group token:    <|G_{group_id}|>")
    print(f"Member count:   {len(member_ids)}")
    print()

    # Show members
    print(f"Members (showing up to {max_tokens}):")
    for i, tid in enumerate(member_ids[:max_tokens]):
        if tokenizer:
            try:
                token_str = tokenizer.decode([tid])
                escaped = repr(token_str)
            except Exception:
                escaped = "<decode error>"
        else:
            escaped = ""
        print(f"  {tid:5d}  {escaped}")

    if len(member_ids) > max_tokens:
        print(f"  ... and {len(member_ids) - max_tokens} more")
    print()


def dump_token(maps: dict, token_id: int, tokenizer=None):
    """Dump details of a specific pure token."""
    pure_vocab_size = maps["pure_vocab_size"]
    overlap_k = maps["overlap_k"]

    if token_id < 0 or token_id >= pure_vocab_size:
        print(f"Error: token_id must be in [0, {pure_vocab_size})")
        return

    print("=" * 60)
    print(f"TOKEN {token_id} DETAILS")
    print("=" * 60)

    # Token text
    if tokenizer:
        try:
            token_str = tokenizer.decode([token_id])
            print(f"Token text: {repr(token_str)}")
        except Exception:
            print("Token text: <decode error>")
    print()

    # Group assignments
    groups = maps["pure_to_group"][token_id].tolist()
    print(f"Group assignments (overlap_k={overlap_k}):")
    for i, g in enumerate(groups):
        group_size = maps["group_to_pure_mask"][g].sum().item()
        print(f"  [{i}] Group {g} (<|G_{g}|>) - {group_size} members")
    print()


def dump_all_groups(maps: dict, tokenizer=None):
    """Dump summary of all groups."""
    num_groups = maps["num_groups"]
    pure_vocab_size = maps["pure_vocab_size"]
    group_sizes = maps["group_to_pure_mask"].sum(dim=1)

    print("=" * 60)
    print("ALL GROUPS SUMMARY")
    print("=" * 60)
    print(f"{'Group':>6} {'TokenID':>8} {'Size':>6}")
    print("-" * 24)

    for g in range(num_groups):
        token_id = pure_vocab_size + g
        size = group_sizes[g].item()
        print(f"{g:>6} {token_id:>8} {size:>6}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Dump token map contents")
    parser.add_argument(
        "tokenizer_dir",
        type=str,
        help="Path to tokenizer directory containing token_maps.pt",
    )
    parser.add_argument(
        "--group",
        type=int,
        default=None,
        help="Dump details of specific group",
    )
    parser.add_argument(
        "--token",
        type=int,
        default=None,
        help="Dump details of specific pure token",
    )
    parser.add_argument(
        "--all-groups",
        action="store_true",
        help="Dump summary of all groups",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Max tokens to show per group (default: 50)",
    )
    args = parser.parse_args()

    # Load
    print(f"Loading from {args.tokenizer_dir}...")
    maps = load_token_maps(args.tokenizer_dir)
    tokenizer = load_tokenizer(args.tokenizer_dir)
    if tokenizer:
        print(f"Loaded tokenizer (vocab={tokenizer.n_vocab})")
    print()

    # Dispatch
    if args.group is not None:
        dump_group(maps, args.group, tokenizer, args.max_tokens)
    elif args.token is not None:
        dump_token(maps, args.token, tokenizer)
    elif args.all_groups:
        dump_all_groups(maps, tokenizer)
    else:
        dump_overview(maps)


if __name__ == "__main__":
    main()
