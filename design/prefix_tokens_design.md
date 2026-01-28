# Prefix Tokens Design: prefix_pure_tokens vs prefix_sliding_tokens

## Overview

Two distinct prefix concepts in BD3LM training:

| Concept | Purpose | When Set | Scope |
|---------|---------|----------|-------|
| `prefix_pure_tokens` | AR-style prefix, never masked, no loss | Static config | Input masking + loss exclusion |
| `prefix_sliding_tokens` | Shifts block boundaries for data efficiency | `epoch % block_size` | Attention pattern + block alignment |

## prefix_sliding_tokens

**Problem it solves**: In target_shift mode, only 1 position per block computes loss. Without shifting, the same positions (0, 4, 8, ...) are trained every epoch, wasting 75% of data (for block_size=4).

**Solution**: Each epoch, shift block boundaries by 1:
- Epoch 0: blocks `[0-3], [4-7]` → loss at 0, 4
- Epoch 1: blocks `[1-4], [5-8]` → loss at 1, 5  (position 0 is sliding prefix)
- Epoch 2: blocks `[2-5], [6-9]` → loss at 2, 6
- Epoch 3: blocks `[3-6], [7-10]` → loss at 3, 7

After `block_size` epochs, all positions have been trained.

**Applies to**: Both target_shift AND normal mode (for consistency).

## No Conflict

Sliding prefix positions are inherently "pure" (outside blocks, bidirectional attention). `prefix_pure_tokens` is an additional static constraint applied last.

```
T=8, block_size=4, prefix_pure_tokens=2, epoch=1, target_shift=1

prefix_sliding_tokens = 1

pos 0: sliding prefix (no block, no loss)
pos 1: block 0 pos 0 → would have loss, but < prefix_pure_tokens → no loss
pos 2: block 0 pos 1
pos 3: block 0 pos 2
pos 4: block 0 pos 3
pos 5: block 1 pos 0 → loss computed here
pos 6: block 1 pos 1
pos 7: block 1 pos 2
```

## Key Formula

```python
prefix_sliding_tokens = epoch % block_size
num_blocks = (T - prefix_sliding_tokens) // block_size
block_start_positions = [prefix_sliding_tokens + i * block_size for i in range(num_blocks)]
```

## Files Affected

- `nanochat/attn_masks.py`: Already correct (uses prefix_sliding_tokens for attention)
- `nanochat/bd3lm_utils/bd3lm_mask.py`: Needs prefix_sliding_tokens in `q_xt`
- `nanochat/dataloader.py`: Needs to compute and pass prefix_sliding_tokens, adjust loss_mask
