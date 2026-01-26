"""
TokenizerBuilder - builds group tokenizer variants from a base tokenizer.
"""
import os
import torch
from typing import Optional, Dict, Any

from .config import GroupTokenizerConfig
from .clustering import (
    kmeans_clustering,
    random_clustering,
    build_all_combos_assignment,
    build_flexible_assignment,
    derive_pure_to_group,
)


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
        self.sub_assignments: Optional[torch.Tensor] = None  # (vocab_size,) sub-group assignments
        self.sub_to_final: Optional[torch.Tensor] = None  # (num_final, num_sub) assignment matrix
        self.tokenizer = None
        self.token_maps: Optional[Dict[str, Any]] = None

    def build(self, config: GroupTokenizerConfig) -> "TokenizerBuilder":
        """
        Build the group tokenizer variant using sub-group combination approach.

        Args:
            config: GroupTokenizerConfig specifying num_sub, sub_per_final, overlap_mode, etc.

        Returns:
            self for chaining
        """
        self.config = config

        # Step 1: Cluster pure tokens into SUB-GROUPS
        print(f"Step 1: Clustering into {config.num_sub} sub-groups...")
        if config.clustering_method == "kmeans":
            self.sub_assignments = kmeans_clustering(
                self.embeddings,
                config.num_sub,
                seed=config.random_seed,
            )
        elif config.clustering_method == "random":
            self.sub_assignments = random_clustering(
                self.pure_vocab_size,
                config.num_sub,
                seed=config.random_seed,
            )
        else:
            raise ValueError(f"Unknown clustering method: {config.clustering_method}")

        # Step 2: Build assignment matrix based on mode
        print(f"Step 2: Building assignment matrix (mode={config.overlap_mode})...")
        if config.overlap_mode == "all_combos":
            self.sub_to_final = build_all_combos_assignment(
                config.num_sub,
                config.sub_per_final,
            )
        else:  # flexible
            self.sub_to_final = build_flexible_assignment(
                config.num_sub,
                config.sub_per_final,
                config.overlap_k,
                seed=config.random_seed,
            )

        # Step 3: Derive final tensors
        print(f"Step 3: Deriving final tensors...")
        pure_to_group, group_to_pure_mask = derive_pure_to_group(
            self.sub_assignments,
            self.sub_to_final,
        )

        # Step 4: Build token maps
        print(f"Step 4: Building token maps...")
        self._build_token_maps(pure_to_group, group_to_pure_mask)

        # Step 5: Build extended tokenizer
        print(f"Step 5: Building extended tokenizer...")
        self._build_tokenizer()

        return self

    def _build_token_maps(self, pure_to_group: torch.Tensor, group_to_pure_mask: torch.Tensor):
        """Build the runtime token maps."""
        config = self.config
        pure_vocab = self.pure_vocab_size
        num_groups = config.num_groups
        overlap_k = config.effective_overlap_k

        # MASK token id
        mask_token_id = -1
        if config.include_mask:
            mask_token_id = pure_vocab + num_groups  # at the very end

        self.token_maps = {
            "pure_to_group": pure_to_group,
            "group_to_pure_mask": group_to_pure_mask,
            "pure_vocab_size": pure_vocab,
            "num_groups": num_groups,
            "overlap_k": overlap_k,
            "mask_token_id": mask_token_id,
        }

    def _build_tokenizer(self):
        """Build the extended tokenizer with group tokens and MASK."""
        import tiktoken

        config = self.config
        num_groups = config.num_groups
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
        for g in range(num_groups):
            token_name = f"<|G_{g}|>"
            token_id = self.pure_vocab_size + g
            special_tokens[token_name] = token_id
            self.group_tokens[token_name] = token_id

        # Add MASK token at the end
        if config.include_mask:
            mask_id = self.pure_vocab_size + num_groups
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
        config = self.config
        config_path = os.path.join(output_dir, "config.txt")
        with open(config_path, "w") as f:
            f.write(f"num_sub: {config.num_sub}\n")
            f.write(f"sub_per_final: {config.sub_per_final}\n")
            f.write(f"overlap_mode: {config.overlap_mode}\n")
            f.write(f"num_groups: {config.num_groups}\n")
            f.write(f"overlap_k: {config.effective_overlap_k}\n")
            f.write(f"include_mask: {config.include_mask}\n")
            f.write(f"clustering_method: {config.clustering_method}\n")
            f.write(f"pure_vocab_size: {self.pure_vocab_size}\n")
            f.write(f"all_vocab_size: {self.all_vocab_size}\n")
            if config.include_mask:
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

        # Save all tokens dump (id + text representation)
        all_tokens_path = os.path.join(output_dir, "all_tokens.txt")
        enc = self.extended_encoding
        with open(all_tokens_path, "w", encoding="utf-8") as f:
            for tid in range(enc.n_vocab):
                token_str = enc.decode([tid])
                escaped = token_str.encode("unicode_escape").decode("ascii")
                f.write(f"{tid}\t{escaped}\n")

        print(f"Saved to {output_dir}:")
        print(f"  - tokenizer.pkl (vocab_size={self.all_vocab_size})")
        print(f"  - token_maps.pt")
        print(f"  - config.txt, group_tokens.txt, group_stats.txt, all_tokens.txt")

    def get_stats(self) -> Dict[str, Any]:
        """Return stats about the built tokenizer."""
        group_sizes = self.token_maps["group_to_pure_mask"].sum(dim=1)
        return {
            "pure_vocab_size": self.pure_vocab_size,
            "num_sub": self.config.num_sub,
            "sub_per_final": self.config.sub_per_final,
            "overlap_mode": self.config.overlap_mode,
            "num_groups": self.config.num_groups,
            "overlap_k": self.config.effective_overlap_k,
            "all_vocab_size": self.all_vocab_size,
            "group_size_min": group_sizes.min().item(),
            "group_size_max": group_sizes.max().item(),
            "group_size_mean": group_sizes.float().mean().item(),
            "group_size_std": group_sizes.float().std().item(),
        }
