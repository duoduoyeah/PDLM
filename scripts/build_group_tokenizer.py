"""
Build a group tokenizer from a base tokenizer and model embeddings.

Usage:
    # All combos mode:
    uv run -m scripts.build_group_tokenizer \
        --checkpoint-dir /path/to/base_checkpoints \
        --output-dir /path/to/output \
        --mode all_combos --num-sub 8 --sub-per-final 2

    # Flexible mode:
    uv run -m scripts.build_group_tokenizer \
        --checkpoint-dir /path/to/base_checkpoints \
        --output-dir /path/to/output \
        --mode flexible --num-sub 128 --sub-per-final 2 --overlap-k 4

    # With mask (default: no mask):
    uv run -m scripts.build_group_tokenizer ... --mask
"""
import argparse

from nanochat.checkpoint_manager import load_model_from_dir
from nanochat.group_tokenizer import TokenizerBuilder, GroupTokenizerConfig


def main():
    parser = argparse.ArgumentParser(description="Build group tokenizer")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Path to base_checkpoints directory containing model",
    )
    parser.add_argument(
        "--model-tag",
        type=str,
        default=None,
        help="Model tag (e.g., 'd20'). If not specified, uses largest model.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for the new tokenizer",
    )

    # New sub-group combination parameters
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all_combos", "flexible"],
        default="all_combos",
        help="Overlap mode: 'all_combos' for C(num_sub, sub_per_final), 'flexible' for custom overlap_k",
    )
    parser.add_argument(
        "--num-sub",
        type=int,
        required=True,
        help="Number of sub-groups from clustering",
    )
    parser.add_argument(
        "--sub-per-final",
        type=int,
        default=1,
        help="Number of sub-groups per final group (default: 1)",
    )
    parser.add_argument(
        "--overlap-k",
        type=int,
        default=None,
        help="Number of groups per token (only for flexible mode)",
    )

    parser.add_argument(
        "--mask",
        action="store_true",
        help="Include MASK token (default: no mask)",
    )
    parser.add_argument(
        "--clustering-method",
        type=str,
        default="kmeans",
        choices=["kmeans", "random"],
        help="Clustering method (default: kmeans)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for clustering (default: 42)",
    )
    args = parser.parse_args()

    # Validate args
    if args.mode == "flexible" and args.overlap_k is None:
        parser.error("--overlap-k is required for flexible mode")

    # Load model and tokenizer
    print(f"Loading model from {args.checkpoint_dir}...")
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model, base_tokenizer, meta_data = load_model_from_dir(
        args.checkpoint_dir,
        device=device,
        phase="eval",
        model_tag=args.model_tag,
    )

    # Extract lm_head embeddings
    embeddings = model.lm_head.weight.detach().clone()
    print(f"Extracted embeddings: {embeddings.shape}")

    # Verify vocab sizes match
    base_vocab_size = base_tokenizer.get_vocab_size()
    emb_vocab_size = embeddings.shape[0]
    print(f"Base tokenizer vocab size: {base_vocab_size}")
    print(f"Embedding vocab size: {emb_vocab_size}")

    if emb_vocab_size != base_vocab_size:
        print(f"Note: Embedding size ({emb_vocab_size}) differs from tokenizer ({base_vocab_size})")
        print(f"Using embedding size as pure_vocab_size")

    # Create config
    config = GroupTokenizerConfig(
        num_sub=args.num_sub,
        sub_per_final=args.sub_per_final,
        overlap_mode=args.mode,
        overlap_k=args.overlap_k,
        include_mask=args.mask,
        clustering_method=args.clustering_method,
        random_seed=args.seed,
    )
    print(f"\nConfig:")
    print(f"  num_sub: {config.num_sub}")
    print(f"  sub_per_final: {config.sub_per_final}")
    print(f"  overlap_mode: {config.overlap_mode}")
    print(f"  num_groups: {config.num_groups}")
    print(f"  effective_overlap_k: {config.effective_overlap_k}")
    print(f"  include_mask: {config.include_mask}")
    print(f"  output_name: {config.get_output_name_with_vocab(emb_vocab_size)}")

    # Build tokenizer
    print(f"\nBuilding group tokenizer...")
    builder = TokenizerBuilder(base_tokenizer, embeddings)
    builder.build(config)

    # Print stats
    stats = builder.get_stats()
    print(f"\nTokenizer stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Verify tensor shapes
    token_maps = builder.token_maps
    print(f"\nTensor shapes:")
    print(f"  pure_to_group: {token_maps['pure_to_group'].shape}")
    print(f"  group_to_pure_mask: {token_maps['group_to_pure_mask'].shape}")

    # Verify each token appears in exactly k groups
    k = config.effective_overlap_k
    pure_to_group = token_maps['pure_to_group']
    group_to_pure_mask = token_maps['group_to_pure_mask']

    # Count how many groups each token is in via mask
    tokens_per_group = group_to_pure_mask.sum(dim=0)
    assert (tokens_per_group == k).all(), f"Token membership mismatch: expected {k}, got {tokens_per_group.unique()}"
    print(f"  Verified: each token in exactly {k} groups")

    # Save
    print(f"\nSaving to {args.output_dir}...")
    builder.save(args.output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
