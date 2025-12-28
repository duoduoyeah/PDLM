import os
from typing import Optional, Union
import torch
from nanochat.common import get_base_dir

class TokenMap:

    def __init__(self, maps, device="cpu"):
        if maps is None:
            raise ValueError("maps is required")
        
        # shape vocab_size, noisy_depth, fanout
        self.pure_to_noisy_map = maps["pure_to_noisy_map"].to(device)
        # shape all_tokens, 2 (low_level, high_level)
        self.noisy_level_map = maps["noisy_level_map"].to(device)
        
        self.device = device
        self.max_level = self.pure_to_noisy_map.shape[1] - 1
        self.fanout = self.pure_to_noisy_map.shape[2]

    def _sample_fanout(self, options: torch.Tensor, reference: torch.Tensor):
        """
        options: (..., fanout) tensor containing possible targets
        reference: tensor used to mirror dtype/pinning
        """
        fanout = options.shape[-1]
        if fanout == 1:
            sampled = options[..., 0]
        else:
            idx = torch.randint(0, fanout, options.shape[:-1], device=options.device)
            sampled = torch.gather(options, -1, idx.unsqueeze(-1)).squeeze(-1)
        # make sure the return tensor is the right dtype
        sampled = sampled.to(reference.dtype)
        if reference.is_pinned():
            sampled = sampled.pin_memory()
        return sampled

    def noise_tokens(self,
                     pure_ids: torch.tensor,
                     noisy_levels: torch.tensor):
        assert pure_ids.shape == noisy_levels.shape
        options = self.pure_to_noisy_map[pure_ids, noisy_levels]
        return self._sample_fanout(options, pure_ids)
        
    def transit_noisy_tokens(
        self,
        pure_ids: torch.tensor,
        noisy_ids: torch.tensor,
        enforce_monotonic: bool = False,
    ):
        """
        enforce_monotonic: if True, noisy levels are forced to be non-decreasing along the last dimension.
        """
        # noisy_ids -> noisy_levels, use the lowest level
        noisy_levels = self.noisy_level_map[noisy_ids, 0]
        noisy_levels = torch.where(noisy_levels > 0, noisy_levels - 1, noisy_levels)
        if enforce_monotonic:
            noisy_levels = torch.cummax(noisy_levels, dim=-1).values
        assert torch.all(noisy_levels >= 0).item(), "Expected all noisy levels to be >= 0"
        # pure ids, noisy_levels -> output_scratch_ids
        options = self.pure_to_noisy_map[pure_ids, noisy_levels]
        return self._sample_fanout(options, noisy_ids)
    
    def is_all_pure_tokens(self, ids:torch.tensor) -> bool:
        # All tokens are pure if their noisy level is 0
        return torch.all(self.noisy_level_map[ids, 0] == 0).item()

    def get_random_noisy_level(
        self,
        ids: torch.tensor,
        step: Optional[Union[int, torch.Tensor]] = None,
        total_steps: int = None,
        prefix_pure_tokens: int = 0,
    ) -> torch.tensor:
        """
        Sample noisy levels per token.
        - If step/total_steps are provided, draw from Binomial(max_level, p=step/total_steps) per element.
        - Otherwise, uniform random in [1, max_level].
        - Optionally force the first prefix_pure_tokens to be level 0.
        """
        if step is None or total_steps is None:
            fixed_level = torch.randint(1, self.max_level + 1, ()).item()
            levels = torch.full(ids.shape, fixed_level, device=ids.device, dtype=ids.dtype)
        else:
            assert total_steps > 0, "total_steps must be positive"
            if torch.is_tensor(step):
                prob = step.to(device=ids.device, dtype=torch.float32) / float(total_steps)
                prob = torch.clamp(prob, 0.0, 1.0)
                if prob.ndim == 0:
                    prob = prob.expand(ids.shape)
                else:
                    while prob.ndim < ids.ndim:
                        prob = prob.unsqueeze(-1)
                    prob = prob.expand(ids.shape)
            else:
                prob = max(0.0, min(1.0, float(step) / float(total_steps)))
                prob = torch.full(ids.shape, prob, device=ids.device)
            dist = torch.distributions.Binomial(total_count=self.max_level, probs=prob)
            levels = dist.sample().to(device=ids.device, dtype=ids.dtype)
            levels = torch.clamp(levels, min=1) # at least add one noisy

        if ids.is_pinned():
            levels = levels.pin_memory()
        if prefix_pure_tokens > 0:
            if ids.ndim == 1:
                levels[:prefix_pure_tokens] = 0
            else:
                levels[..., :prefix_pure_tokens] = 0
        return levels


_TOKEN_MAP_CACHE = {}


def get_token_map(tokenizer_dir=None, device="cpu"):
    if tokenizer_dir is None:
        base_dir = get_base_dir()
        tokenizer_dir = os.path.join(base_dir, "tokenizer")
    device = device if isinstance(device, torch.device) else torch.device(device)
    key = (tokenizer_dir, str(device))
    token_map = _TOKEN_MAP_CACHE.get(key)
    if token_map is None or str(token_map.device) != str(device):
        map_path = os.path.join(tokenizer_dir, "token_maps.pt")
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Token maps not found at {map_path}")
        maps = torch.load(map_path, map_location=device)
        token_map = TokenMap(maps=maps, device=device)
        _TOKEN_MAP_CACHE[key] = token_map
    return token_map
