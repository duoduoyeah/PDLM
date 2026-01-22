"""
TokenizerBuilder - builds group tokenizer variants from a base tokenizer.
"""
import os
import torch
from typing import Optional, Dict, Any

from .config import GroupTokenizerConfig
from .clustering import kmeans_clustering, random_clustering, compute_overlap_assignments


class TokenizerBuilder:
    """
    Builds group tokenizer variants from a base tokenizer and model embeddings.

    Usage:
        builder = TokenizerBuilder(base_tokenizer, lm_head_weight)
        builder.build(config)
        builder.save(output_dir)
    """

    def __init__(
        self,
        base_tokenizer,
        embeddings: torch.Tensor,
    ):
        """
        Args:
            base_tokenizer: base tiktoken-style tokenizer (RustBPETokenizer)
            embeddings: (pure_vocab_size, dim) tensor, typically lm_head.weight
        """
        self.base_tokenizer = base_tokenizer
        self.embeddings = embeddings
        self.pure_vocab_size = embeddings.shape[0]

        # Built artifacts (populated by build())
        self.config: Optional[GroupTokenizerConfig] = None
        self.group_assignments: Optional[torch.Tensor] = None  # (pure_vocab, overlap_k)
        self.tokenizer = None
        self.token_maps: Optional[Dict[str, Any]] = None

    def build(self, config: GroupTokenizerConfig) -> "TokenizerBuilder":
        """
        Build the group tokenizer variant.

        Args:
            config: GroupTokenizerConfig specifying num_groups, overlap_k, etc.

        Returns:
            self for chaining
        """
        self.config = config

        # Validate lower-triangle constraint
        tokens_per_group = self.pure_vocab_size / config.num_groups
        max_reasonable_k = int(tokens_per_group ** 0.5)
        if config.overlap_k > max_reasonable_k:
            print(f"Warning: overlap_k={config.overlap_k} may be too high for "
                  f"{tokens_per_group:.0f} tokens/group (sqrt={max_reasonable_k})")

        # Step 1: Cluster pure tokens into groups
        if config.clustering_method == "kmeans":
            base_assignments = kmeans_clustering(
                self.embeddings,
                config.num_groups,
                seed=config.random_seed,
            )
        elif config.clustering_method == "random":
            base_assignments = random_clustering(
                self.pure_vocab_size,
                config.num_groups,
                seed=config.random_seed,
            )
        else:
            raise ValueError(f"Unknown clustering method: {config.clustering_method}")

        # Step 2: Compute overlap assignments if overlap_k > 1
        self.group_assignments = compute_overlap_assignments(
            self.embeddings,
            base_assignments,
            config.num_groups,
            config.overlap_k,
        )

        # Step 3: Build token maps
        self._build_token_maps()

        # Step 4: Build extended tokenizer
        self._build_tokenizer()

        return self

    def _build_token_maps(self):
        """Build the runtime token maps."""
        config = self.config
        pure_vocab = self.pure_vocab_size
        num_groups = config.num_groups

        # pure_to_group: (pure_vocab, overlap_k)
        pure_to_group = self.group_assignments.clone()

        # group_to_pure_mask: (num_groups, pure_vocab) bool
        group_to_pure_mask = torch.zeros(num_groups, pure_vocab, dtype=torch.bool)
        for g in range(num_groups):
            # A token belongs to group g if g is in any of its overlap_k assignments
            members = (pure_to_group == g).any(dim=1)
            group_to_pure_mask[g] = members

        # MASK token id
        mask_token_id = -1
        if config.include_mask:
            mask_token_id = pure_vocab + num_groups  # at the very end

        self.token_maps = {
            "pure_to_group": pure_to_group,
            "group_to_pure_mask": group_to_pure_mask,
            "pure_vocab_size": pure_vocab,
            "num_groups": num_groups,
            "overlap_k": config.overlap_k,
            "mask_token_id": mask_token_id,
        }

    def _build_tokenizer(self):
        """Build the extended tokenizer with group tokens and MASK."""
        import tiktoken

        config = self.config
        enc = self.base_tokenizer.enc  # tiktoken.Encoding

        # Extract components from base encoding
        mergeable_ranks = enc._mergeable_ranks
        pat_str = enc._pat_str

        # Get existing special tokens
        if hasattr(enc, "_special_tokens"):
            special_tokens = dict(enc._special_tokens)
        else:
            special_tokens = {tok: enc.encode_single_token(tok) for tok in enc.special_tokens_set}

        # Add group tokens: <|G_0|>, <|G_1|>, ..., <|G_{num_groups-1}|>
        self.group_tokens = {}
        for g in range(config.num_groups):
            token_name = f"<|G_{g}|>"
            token_id = self.pure_vocab_size + g
            special_tokens[token_name] = token_id
            self.group_tokens[token_name] = token_id

        # Add MASK token at the end
        if config.include_mask:
            mask_id = self.pure_vocab_size + config.num_groups
            special_tokens["<|MASK|>"] = mask_id
            self.group_tokens["<|MASK|>"] = mask_id

        # Create new tiktoken.Encoding with extended special tokens
        self.extended_encoding = tiktoken.Encoding(
            name="rustbpe_with_groups",
            pat_str=pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )

        self.all_vocab_size = self.extended_encoding.n_vocab

    def save(self, output_dir: str):
        """Save tokenizer and token maps to directory."""
        import pickle

        os.makedirs(output_dir, exist_ok=True)

        # Save extended tokenizer as tokenizer.pkl
        tokenizer_path = os.path.join(output_dir, "tokenizer.pkl")
        with open(tokenizer_path, "wb") as f:
            pickle.dump(self.extended_encoding, f)

        # Save token maps
        map_path = os.path.join(output_dir, "token_maps.pt")
        torch.save(self.token_maps, map_path)

        # Save config
        config_path = os.path.join(output_dir, "config.txt")
        with open(config_path, "w") as f:
            f.write(f"num_groups: {self.config.num_groups}\n")
            f.write(f"overlap_k: {self.config.overlap_k}\n")
            f.write(f"include_mask: {self.config.include_mask}\n")
            f.write(f"clustering_method: {self.config.clustering_method}\n")
            f.write(f"pure_vocab_size: {self.pure_vocab_size}\n")
            f.write(f"all_vocab_size: {self.all_vocab_size}\n")
            if self.config.include_mask:
                f.write(f"mask_token_id: {self.token_maps['mask_token_id']}\n")

        # Save group token mapping
        tokens_path = os.path.join(output_dir, "group_tokens.txt")
        with open(tokens_path, "w") as f:
            for name, idx in self.group_tokens.items():
                f.write(f"{idx}\t{name}\n")

        # Save group stats
        stats_path = os.path.join(output_dir, "group_stats.txt")
        group_sizes = self.token_maps["group_to_pure_mask"].sum(dim=1)
        with open(stats_path, "w") as f:
            f.write(f"Group size stats:\n")
            f.write(f"  min: {group_sizes.min().item()}\n")
            f.write(f"  max: {group_sizes.max().item()}\n")
            f.write(f"  mean: {group_sizes.float().mean().item():.1f}\n")
            f.write(f"  std: {group_sizes.float().std().item():.1f}\n")

        print(f"Saved to {output_dir}:")
        print(f"  - tokenizer.pkl (vocab_size={self.all_vocab_size})")
        print(f"  - token_maps.pt")
        print(f"  - config.txt, group_tokens.txt, group_stats.txt")

    def get_stats(self) -> Dict[str, Any]:
        """Return stats about the built tokenizer."""
        group_sizes = self.token_maps["group_to_pure_mask"].sum(dim=1)
        return {
            "pure_vocab_size": self.pure_vocab_size,
            "num_groups": self.config.num_groups,
            "overlap_k": self.config.overlap_k,
            "all_vocab_size": self.all_vocab_size,
            "group_size_min": group_sizes.min().item(),
            "group_size_max": group_sizes.max().item(),
            "group_size_mean": group_sizes.float().mean().item(),
            "group_size_std": group_sizes.float().std().item(),
        }
