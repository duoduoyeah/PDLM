"""
Configuration for group tokenizer generation.
"""
from dataclasses import dataclass
from math import comb
from typing import Literal, Optional


@dataclass
class GroupTokenizerConfig:
    """
    Config for building a group tokenizer variant using sub-group combination approach.

    Parameters:
        num_sub: Number of sub-groups from clustering
        sub_per_final: Number of sub-groups combined per final group
        overlap_mode: "all_combos" for C(num_sub, sub_per_final) combinations,
                      "flexible" for custom overlap_k with balanced design
        overlap_k: Only for flexible mode - number of final groups each token belongs to
        include_mask: Whether to add MASK token (always at end of vocab)
        clustering_method: "kmeans" or "random" for baseline
        random_seed: Random seed for clustering and assignment
    """
    num_sub: int
    sub_per_final: int = 1
    overlap_mode: Literal["all_combos", "flexible"] = "all_combos"
    overlap_k: Optional[int] = None  # Only for flexible mode
    include_mask: bool = True
    clustering_method: str = "kmeans"
    random_seed: int = 42

    def __post_init__(self):
        assert self.num_sub > 0, "num_sub must be positive"
        assert self.sub_per_final >= 1, "sub_per_final must be >= 1"
        assert self.sub_per_final <= self.num_sub, "sub_per_final must be <= num_sub"

        if self.overlap_mode == "all_combos":
            # overlap_k is derived, ignore any provided value
            pass
        elif self.overlap_mode == "flexible":
            assert self.overlap_k is not None, "overlap_k required for flexible mode"
            assert self.overlap_k >= 1, "overlap_k must be >= 1"
            # Check that num_groups is an integer
            num_groups = (self.num_sub * self.overlap_k) // self.sub_per_final
            remainder = (self.num_sub * self.overlap_k) % self.sub_per_final
            assert remainder == 0, (
                f"Invalid flexible config: num_sub * overlap_k ({self.num_sub * self.overlap_k}) "
                f"must be divisible by sub_per_final ({self.sub_per_final})"
            )
        else:
            raise ValueError(f"Unknown overlap_mode: {self.overlap_mode}")

    @property
    def num_groups(self) -> int:
        """Number of final group tokens."""
        if self.overlap_mode == "all_combos":
            return comb(self.num_sub, self.sub_per_final)
        else:  # flexible
            return (self.num_sub * self.overlap_k) // self.sub_per_final

    @property
    def effective_overlap_k(self) -> int:
        """Number of final groups each token belongs to."""
        if self.overlap_mode == "all_combos":
            # Each sub-group appears in C(num_sub-1, sub_per_final-1) final groups
            return comb(self.num_sub - 1, self.sub_per_final - 1)
        else:  # flexible
            return self.overlap_k

    def get_output_name(self) -> str:
        """
        Generate a descriptive name for this config.
        Format: n{noise_level}_k{overlap}_g{groups}(_mask)

        Note: noise_level is based on vocab_size=4096. For actual noise level,
        use get_output_name_with_vocab().
        """
        return self.get_output_name_with_vocab(4096)

    def get_output_name_with_vocab(self, vocab_size: int) -> str:
        """
        Generate a descriptive name with actual noise level.
        Format: n{noise_level}_k{overlap}_g{groups}(_mask)
        """
        tokens_per_sub = vocab_size / self.num_sub
        tokens_per_final = int(tokens_per_sub * self.sub_per_final)
        k = self.effective_overlap_k
        g = self.num_groups

        name = f"n{tokens_per_final}_k{k}_g{g}"
        if self.include_mask:
            name += "_mask"
        return name
