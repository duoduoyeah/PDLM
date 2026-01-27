"""
Dump/inspect token map contents for debugging and analysis.

Usage:
    python -m nanochat.group_tokenizer.dump /path/to/tokenizer_dir
    python -m nanochat.group_tokenizer.dump /path/to/tokenizer_dir --group 5
    python -m nanochat.group_tokenizer.dump /path/to/tokenizer_dir --token 123
    python -m nanochat.group_tokenizer.dump /path/to/tokenizer_dir --overlap-quality
    python -m nanochat.group_tokenizer.dump /path/to/tokenizer_dir --overlap-quality --sample-tokens 10
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


def dump_overlap_quality(maps: dict, tokenizer=None, sample_tokens: int = 5):
    """Analyze overlap quality - are groupmates diverse across k groups?"""
    pure_vocab_size = maps["pure_vocab_size"]
    num_groups = maps["num_groups"]
    overlap_k = maps["overlap_k"]
    pure_to_group = maps["pure_to_group"]  # (vocab, k)
    group_to_pure_mask = maps["group_to_pure_mask"]  # (groups, vocab)

    print("=" * 60)
    print("OVERLAP QUALITY ANALYSIS")
    print("=" * 60)
    print(f"Vocab size: {pure_vocab_size}, Groups: {num_groups}, Overlap k: {overlap_k}")
    print()

    if overlap_k == 1:
        print("Overlap k=1, no overlap analysis needed.")
        return

    # Compute metrics across all tokens
    total_unique_groupmates = 0
    total_shared_groupmates = 0
    total_pairwise_jaccard = 0.0
    num_pairs = 0

    for tid in range(pure_vocab_size):
        groups = pure_to_group[tid].tolist()

        # Get member sets for each group (excluding self)
        member_sets = []
        for g in groups:
            members = set(torch.where(group_to_pure_mask[g])[0].tolist())
            members.discard(tid)  # Exclude self
            member_sets.append(members)

        # Unique groupmates = union of all member sets
        all_groupmates = set().union(*member_sets)
        total_unique_groupmates += len(all_groupmates)

        # Shared groupmates = intersection of all member sets
        shared_groupmates = member_sets[0].intersection(*member_sets[1:])
        total_shared_groupmates += len(shared_groupmates)

        # Pairwise Jaccard similarity
        for i in range(len(member_sets)):
            for j in range(i + 1, len(member_sets)):
                intersection = len(member_sets[i] & member_sets[j])
                union = len(member_sets[i] | member_sets[j])
                if union > 0:
                    total_pairwise_jaccard += intersection / union
                    num_pairs += 1

    avg_unique = total_unique_groupmates / pure_vocab_size
    avg_shared = total_shared_groupmates / pure_vocab_size
    avg_jaccard = total_pairwise_jaccard / num_pairs if num_pairs > 0 else 0

    # Theoretical values for all_combos mode with sub_per_final=2
    # Each token's k groups share only the same sub-group members
    # Jaccard = (sub_size) / (2*sub_size - sub_size) = 1/(2-1) = for perfect case
    # With num_sub=8, sub_per_final=2: shared = 1/8 of group

    print("Aggregate Statistics:")
    print(f"  Avg pairwise Jaccard similarity: {avg_jaccard:.4f}  (lower = more diverse)")
    print(f"  Avg unique groupmates per token: {avg_unique:.1f} / {pure_vocab_size}")
    print(f"  Avg shared groupmates (in ALL k groups): {avg_shared:.1f}")
    print(f"  Shared ratio: {avg_shared / avg_unique * 100:.1f}%  (lower = more diverse)")
    print()

    # Diversity score: what fraction of possible tokens does each token see?
    diversity_score = avg_unique / (pure_vocab_size - 1) * 100
    print(f"  Diversity score: {diversity_score:.1f}%  (higher = better coverage)")
    print()

    # Sample token analysis
    print("-" * 60)
    print(f"Sample Token Analysis (first {sample_tokens} tokens):")
    print("-" * 60)

    for tid in range(min(sample_tokens, pure_vocab_size)):
        groups = pure_to_group[tid].tolist()

        # Token name
        if tokenizer:
            try:
                token_str = repr(tokenizer.decode([tid]))
            except Exception:
                token_str = f"token_{tid}"
        else:
            token_str = f"token_{tid}"

        print(f"\nToken {tid} {token_str}:")
        print(f"  Groups: {groups}")

        # Member sets
        member_sets = []
        for g in groups:
            members = set(torch.where(group_to_pure_mask[g])[0].tolist())
            members.discard(tid)
            member_sets.append(members)

        all_groupmates = set().union(*member_sets)
        shared_groupmates = member_sets[0].intersection(*member_sets[1:])

        print(f"  Total unique groupmates: {len(all_groupmates)}")
        print(f"  Shared across ALL {overlap_k} groups: {len(shared_groupmates)}")

        # Per-group breakdown
        for i, g in enumerate(groups):
            unique_to_this = member_sets[i] - set().union(*[member_sets[j] for j in range(len(member_sets)) if j != i])
            print(f"    Group {g}: {len(member_sets[i])} members, {len(unique_to_this)} unique to this group only")

    print()


def dump_all_groups(maps: dict, tokenizer=None, mask_rows: int = 5):
    """Dump summary of all groups."""
    num_groups = maps["num_groups"]
    pure_vocab_size = maps["pure_vocab_size"]
    overlap_k = maps["overlap_k"]
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

    # Dump pure_to_group: (pure_vocab_size, overlap_k)
    print("=" * 60)
    print(f"PURE_TO_GROUP ({pure_vocab_size}, {overlap_k})")
    print("=" * 60)
    print(f"{'Token':>6} {'Groups':>20}")
    print("-" * 30)
    pure_to_group = maps["pure_to_group"]
    for tid in range(pure_vocab_size):
        groups = pure_to_group[tid].tolist()
        groups_str = ", ".join(str(g) for g in groups)
        print(f"{tid:>6} [{groups_str:>16}]")
    print()

    # Dump group_to_pure_mask: (num_groups, pure_vocab_size) - first N rows
    rows_to_show = min(mask_rows, num_groups)
    print("=" * 60)
    print(f"GROUP_TO_PURE_MASK (first {rows_to_show} of {num_groups} rows)")
    print("=" * 60)
    group_to_pure_mask = maps["group_to_pure_mask"]
    for g in range(rows_to_show):
        mask = group_to_pure_mask[g]
        member_ids = torch.where(mask)[0].tolist()
        print(f"Group {g:>3}: {member_ids}")
    if rows_to_show < num_groups:
        print(f"... ({num_groups - rows_to_show} more groups)")
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
        "--overlap-quality",
        action="store_true",
        help="Analyze overlap quality (groupmate diversity)",
    )
    parser.add_argument(
        "--sample-tokens",
        type=int,
        default=5,
        help="Number of sample tokens for overlap quality analysis (default: 5)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Max tokens to show per group (default: 50)",
    )
    parser.add_argument(
        "--mask-rows",
        type=int,
        default=5,
        help="Number of group_to_pure_mask rows to show (default: 5, use -1 for all)",
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
        mask_rows = args.mask_rows if args.mask_rows >= 0 else maps["num_groups"]
        dump_all_groups(maps, tokenizer, mask_rows=mask_rows)
    elif args.overlap_quality:
        dump_overlap_quality(maps, tokenizer, args.sample_tokens)
    else:
        dump_overview(maps)


if __name__ == "__main__":
    main()
