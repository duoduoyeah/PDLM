"""
Build a group tokenizer from a base tokenizer and model embeddings.

Usage:
    uv run -m scripts.build_group_tokenizer \
        --checkpoint-dir /path/to/base_checkpoints \
        --output-dir /path/to/output \
        --num-groups 64 \
        --overlap-k 1
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
    parser.add_argument(
        "--num-groups",
        type=int,
        required=True,
        help="Number of group tokens to create",
    )
    parser.add_argument(
        "--overlap-k",
        type=int,
        default=1,
        help="Number of groups each pure token belongs to (default: 1)",
    )
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="Don't include MASK token",
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
        num_groups=args.num_groups,
        overlap_k=args.overlap_k,
        include_mask=not args.no_mask,
        clustering_method=args.clustering_method,
        random_seed=args.seed,
    )
    print(f"\nConfig: {config}")

    # Build tokenizer
    print(f"\nBuilding group tokenizer...")
    builder = TokenizerBuilder(base_tokenizer, embeddings)
    builder.build(config)

    # Print stats
    stats = builder.get_stats()
    print(f"\nTokenizer stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Save
    print(f"\nSaving to {args.output_dir}...")
    builder.save(args.output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
