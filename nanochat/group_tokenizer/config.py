"""
Configuration for group tokenizer generation.
"""
from dataclasses import dataclass


@dataclass
class GroupTokenizerConfig:
    """
    Config for building a group tokenizer variant.

    Parameters:
        num_groups: Number of group tokens. Fewer groups = higher noise.
                    E.g., 1024 (4 tokens/group) to 4 (1024 tokens/group)
        overlap_k: Each pure token belongs to k groups. k=1 is hard clustering.
                   Constraint: higher overlap_k only useful at higher noise (fewer groups).
        include_mask: Whether to add MASK token (always at end of vocab).
    """
    num_groups: int
    overlap_k: int = 1
    include_mask: bool = True

    # Clustering params
    clustering_method: str = "kmeans"  # or "random" for baseline
    random_seed: int = 42

    def __post_init__(self):
        assert self.num_groups > 0, "num_groups must be positive"
        assert self.overlap_k >= 1, "overlap_k must be >= 1"
        # Lower triangle constraint: overlap_k <= sqrt(tokens_per_group)
        # This is checked at build time when we know vocab_size

    def get_output_name(self) -> str:
        """Generate a descriptive name for this config."""
        parts = [f"g{self.num_groups}"]
        if self.overlap_k > 1:
            parts.append(f"k{self.overlap_k}")
        if not self.include_mask:
            parts.append("nomask")
        return "_".join(parts)
