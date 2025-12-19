import torch

class TokenMap:
    
    def __init__(self):
        # shape vocab_size, noisy_depth
        self.pure_to_noisy_map = torch.empty(N, depth, dtype=torch.long)
        # shape all_tokens, 
        self.tokens_to_pure_map = torch.empty(M, dtype=torch.long)
        # shape all_tokens, 
        self.noisy_level_map = torch.empty(M, dtype=torch.long)

    def noise_tokens(self,
                     pure_ids: torch.tensor,
                     noisy_levels: torch.tensor):
        assert pure_ids.shape == noisy_levels.shape
        noisy_ids = self.pure_to_noisy_map[pure_ids, noisy_levels]

        # Make sure the return tensor has the same dtype and pin_memory 
        noisy_ids = noisy_ids.to(pure_ids.dtype)
        if pure_ids.is_pinned():
            return noisy_ids.pin_memory()

        return noisy_ids
        
    def transit_noisy_tokens(self, pure_ids: torch.tensor, noisy_ids: torch.tensor):
        # noisy_ids -> noisy_levels
        noisy_levels = self.noisy_level_map[noisy_ids]
        noisy_levels[noisy_levels >= 1] = -1
        # pure ids, noisy_levels -> output_scratch_ids
        denoised_ids = self.pure_to_noisy_map[pure_ids, noisy_levels]

        # Make sure the return tensor has the same dtype and pin_memory 
        denoised_ids = denoised_ids.to(noisy_ids.dtype)
        if noisy_ids.is_pinned():
            return denoised_ids.pin_memory()

        return denoised_ids
    
    def is_all_pure_tokens(self, ids:torch.tensor) -> bool:
        pure_ids = self.tokens_to_pure_map[ids]
        # Compare pure_ids with ids, if every item is the same, then all tokens are pure.
        return torch.all(torch.eq(pure_ids, ids)).item()