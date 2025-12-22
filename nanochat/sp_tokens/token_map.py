import os
import torch
from nanochat.common import get_base_dir

class TokenMap:
    
    def __init__(self, tokenizer_dir=None, device="cpu"):
        if tokenizer_dir is None:
            base_dir = get_base_dir()
            tokenizer_dir = os.path.join(base_dir, "tokenizer")
        
        map_path = os.path.join(tokenizer_dir, "token_maps.pt")
        if not os.path.exists(map_path):
             raise FileNotFoundError(f"Token maps not found at {map_path}")
             
        maps = torch.load(map_path, map_location=device)
        
        # shape vocab_size, noisy_depth, fanout
        pure_to_noisy_map = maps["pure_to_noisy_map"]
        self.pure_to_noisy_map = pure_to_noisy_map.to(device)
        # shape all_tokens
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
        
    def transit_noisy_tokens(self, pure_ids: torch.tensor, noisy_ids: torch.tensor):
        # noisy_ids -> noisy_levels
        noisy_levels = self.noisy_level_map[noisy_ids]
        noisy_levels[noisy_levels >= 1] = -1
        # pure ids, noisy_levels -> output_scratch_ids
        options = self.pure_to_noisy_map[pure_ids, noisy_levels]
        return self._sample_fanout(options, noisy_ids)
    
    def is_all_pure_tokens(self, ids:torch.tensor) -> bool:
        # All tokens are pure if their noisy level is 0
        return torch.all(self.noisy_level_map[ids] == 0).item()

    def get_random_noisy_level(self, ids: torch.tensor, step: int = None, total_steps: int = None) -> torch.tensor:
        """
        Sample noisy levels per token.
        - If step/total_steps are provided, draw from Binomial(max_level, p=step/total_steps) per element.
        - Otherwise, uniform random in [0, max_level].
        """
        if step is None or total_steps is None:
            fixed_level = torch.randint(1, self.max_level + 1, ()).item()
            levels = torch.full(ids.shape, fixed_level, device=ids.device, dtype=ids.dtype)
        else:
            assert total_steps > 0, "total_steps must be positive"
            prob = max(0.0, min(1.0, float(step) / float(total_steps)))
            dist = torch.distributions.Binomial(total_count=self.max_level, probs=prob)
            levels = dist.sample(ids.shape).to(device=ids.device, dtype=ids.dtype)
        if ids.is_pinned():
            levels = levels.pin_memory()
        return levels
