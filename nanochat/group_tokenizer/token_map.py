"""
Runtime token map for noising/denoising operations.

Simplified for single-layer group tokens (Stage 1 design).
"""
import os
import torch
from typing import Optional


class TokenMap:
    """
    Maps between pure tokens and group tokens.

    Vocab layout:
        [0, pure_vocab_size)           - pure tokens
        [pure_vocab_size, pure_vocab_size + num_groups)  - group tokens
        [pure_vocab_size + num_groups] - MASK token (if present)

    Attributes:
        pure_to_group: (pure_vocab_size, overlap_k) - group assignments per pure token
        group_to_pure_mask: (num_groups, pure_vocab_size) - bool mask of members
        pure_vocab_size: number of pure tokens
        num_groups: number of group tokens
        mask_token_id: MASK token id (or -1 if not present)
    """

    def __init__(self, maps: dict, device: str = "cpu"):
        self.device = device

        # Core maps
        self.pure_to_group = maps["pure_to_group"].to(device)  # (pure_vocab, overlap_k)
        self.group_to_pure_mask = maps["group_to_pure_mask"].to(device)  # (num_groups, pure_vocab)

        # Sizes
        self.pure_vocab_size = maps["pure_vocab_size"]
        self.num_groups = maps["num_groups"]
        self.overlap_k = self.pure_to_group.shape[1]
        self.mask_token_id = maps.get("mask_token_id", -1)

        # Derived
        self.group_start_id = self.pure_vocab_size
        self.group_end_id = self.pure_vocab_size + self.num_groups

    def is_pure(self, ids: torch.Tensor) -> torch.Tensor:
        """Check which tokens are pure (not group, not MASK)."""
        return ids < self.pure_vocab_size

    def is_group(self, ids: torch.Tensor) -> torch.Tensor:
        """Check which tokens are group tokens."""
        return (ids >= self.group_start_id) & (ids < self.group_end_id)

    def is_mask(self, ids: torch.Tensor) -> torch.Tensor:
        """Check which tokens are MASK."""
        if self.mask_token_id < 0:
            return torch.zeros_like(ids, dtype=torch.bool)
        return ids == self.mask_token_id

    def noise_to_group(self, pure_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert pure tokens to group tokens.
        If overlap_k > 1, randomly samples one of the k groups.

        Args:
            pure_ids: (*, ) tensor of pure token ids

        Returns:
            group_ids: (*, ) tensor of group token ids
        """
        # Get group assignments
        groups = self.pure_to_group[pure_ids]  # (*, overlap_k)

        if self.overlap_k == 1:
            group_indices = groups.squeeze(-1)
        else:
            # Random sample from overlap_k options
            k_idx = torch.randint(0, self.overlap_k, pure_ids.shape, device=pure_ids.device)
            group_indices = torch.gather(groups, -1, k_idx.unsqueeze(-1)).squeeze(-1)

        # Convert to group token ids
        return group_indices + self.group_start_id

    def noise_to_mask(self, ids: torch.Tensor) -> torch.Tensor:
        """Replace all tokens with MASK."""
        assert self.mask_token_id >= 0, "MASK token not present"
        return torch.full_like(ids, self.mask_token_id)

    def get_group_members_mask(self, group_ids: torch.Tensor) -> torch.Tensor:
        """
        Get mask of which pure tokens belong to each group.

        Args:
            group_ids: (*, ) tensor of group token ids

        Returns:
            mask: (*, pure_vocab_size) bool tensor
        """
        group_indices = group_ids - self.group_start_id
        return self.group_to_pure_mask[group_indices]

    def get_group_sizes(self) -> torch.Tensor:
        """Return size of each group."""
        return self.group_to_pure_mask.sum(dim=1)


# Cache for loaded token maps
_TOKEN_MAP_CACHE = {}


def get_token_map(tokenizer_dir: Optional[str] = None, device: str = "cpu") -> TokenMap:
    """
    Load token map from directory.

    Args:
        tokenizer_dir: path containing token_maps.pt
        device: device to load to
    """
    if tokenizer_dir is None:
        from nanochat.common import get_base_dir
        tokenizer_dir = os.path.join(get_base_dir(), "tokenizer")

    device = device if isinstance(device, torch.device) else torch.device(device)
    key = (tokenizer_dir, str(device))

    if key not in _TOKEN_MAP_CACHE:
        map_path = os.path.join(tokenizer_dir, "token_maps.pt")
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Token maps not found at {map_path}")
        maps = torch.load(map_path, map_location=device, weights_only=True)
        _TOKEN_MAP_CACHE[key] = TokenMap(maps=maps, device=device)

    return _TOKEN_MAP_CACHE[key]
