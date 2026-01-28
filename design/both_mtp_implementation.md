# Combined PDLM (both_mtp) Implementation

## Overview

Implemented a unified model that performs both Stage 1 (MTP: Pure → K Group tokens) and Stage 2 (Group → Pure denoising) in a single forward pass using the 2L input structure.

## Architecture

```
Input (2L): [xt | x0]
             │         │
        first L    second L
             │         │
             ▼         ▼
     ┌───────────────────────────┐
     │   Shared Transformer      │
     │   + Block Diff Mask       │
     └───────────────────────────┘
             │         │
             ▼         ▼
        Stage 2     Stage 1
    (Group→Pure)  (Pure→Group MTP)
             │         │
             ▼         ▼
      L_stage2      L_stage1
             │         │
             └────┬────┘
                  ▼
           Combined Loss
```

- **First L (xt)**: Stage 2 - predict pure tokens from group tokens using `lm_head[:pure_vocab_size]`
- **Second L (x0)**: Stage 1 - predict K future group tokens using MTP head with `lm_head[pure_vocab_size:]`

## Files Modified

| File | Changes |
|------|---------|
| `nanochat/pdlm.py` | Added MTP components, implemented `_forward_both_mtp()` |
| `nanochat/dataloader_pdlm.py` | Added `both_mtp` branch with MTP targets |

## Detailed Changes

### 1. PDLMConfig (pdlm.py:37-40)

Added three new config fields:

```python
n_future_tokens: int = 4       # K: number of group tokens to predict for Stage 1
mtp_loss_beta: float = 0.8     # Exponential decay factor for MTP loss weights
mtp_loss_weight: float = 1.0   # Stage 1 MTP loss weight relative to Stage 2 (which is 1.0)
```

### 2. PDLM Constructor (pdlm.py:206-234)

For `both_mtp` stage, added:

- `group_wte`: `nn.Embedding(num_groups, n_embd)` for MTP teacher forcing
- `_GroupLogitHead`: Wrapper class that slices `lm_head` output to only group logits `[:, :, pure_vocab_size:]`
- `mtp_head`: `MTPHead` instance for predicting K-1 additional group tokens

```python
if config.stage == "both_mtp":
    from nanochat.mtp_head import MTPHead

    self.group_wte = nn.Embedding(config.num_groups, config.n_embd)

    class _GroupLogitHead(nn.Module):
        def __init__(self, lm_head, pure_vocab_size):
            super().__init__()
            self.lm_head = lm_head
            self.offset = pure_vocab_size

        def forward(self, x):
            return self.lm_head(x)[:, :, self.offset:]

    self._group_logit_head = _GroupLogitHead(self.lm_head, config.pure_vocab_size)

    self.mtp_head = MTPHead(
        n_embd=config.n_embd,
        n_head=config.n_head,
        n_future_tokens=config.n_future_tokens,
        group_wte=self.group_wte,
        lm_head=self._group_logit_head,
    )
```

### 3. init_weights (pdlm.py:253-258)

Added initialization for MTP components:

```python
if self.config.stage == "both_mtp":
    self.mtp_head.init_weights()
    if self.group_wte.weight.device.type == "cuda":
        self.group_wte.to(dtype=torch.bfloat16)
```

### 4. setup_optimizers (pdlm.py:305-318)

For `both_mtp` stage, added MTP parameters to optimizer groups:

- MTP block and projection parameters → Muon optimizer (matrix params)
- Group embedding parameters → AdamW optimizer (embedding params)

```python
if self.config.stage == "both_mtp":
    mtp_matrix_params = list(self.mtp_head.block.parameters()) + list(self.mtp_head.proj.parameters())
    matrix_params = matrix_params + mtp_matrix_params
    group_wte_params = list(self.group_wte.parameters())
    embedding_params = embedding_params + group_wte_params
```

### 5. forward method (pdlm.py:349-352)

Added dispatch to `_forward_both_mtp`:

```python
if self.config.stage == "both_mtp" and targets is not None:
    return self._forward_both_mtp(idx, targets, attn_mask, loss_extras)
```

### 6. _forward_both_mtp (pdlm.py:420-509)

New method implementing joint forward pass:

1. Concatenates `[idx | targets]` to form (B, 2L) input
2. Forwards through transformer with block diffusion mask
3. **Stage 2 Loss**: CE on `logits[:, :L, :pure_vocab_size]` with `loss_mask`
4. **Stage 1 Loss**:
   - Main group logits from `logits[:, L:, pure_vocab_size:]`
   - MTP head predicts k=1..K-1 group tokens from x0 hidden states
   - Uses `compute_mtp_loss()` with any-correct loss and exponential decay weights
5. Returns combined loss: `mtp_loss_weight * stage1_loss + stage2_loss`

### 7. Dataloader (dataloader_pdlm.py:136-196)

Added `both_mtp` branch:

1. Fetches `B * T + K` tokens (extra K for MTP future targets)
2. Creates inputs with group tokens at block positions (same as stage2)
3. Creates loss_mask for Stage 2 block positions
4. Creates `mtp_targets: (B, T, K, overlap_k)` - future group tokens for each position

```python
mtp_targets = torch.zeros(B, T, K, overlap_k, dtype=torch.long)
for k in range(K):
    shift = k + 1
    future_pure = scratch[shift : B * T + shift].view(B, T)
    mtp_targets[:, :, k, :] = pure_to_group[future_pure]
```

## Loss Computation

### Stage 2 Loss (Group → Pure)
Standard CE on block positions:
```python
stage2_logits = logits[:, :T, :pure_vocab_size]
log_probs = F.log_softmax(stage2_logits, dim=-1)
nll = -torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
stage2_loss = (nll * loss_mask).sum() / loss_mask.sum()
```

### Stage 1 Loss (Pure → Group MTP)
Uses `compute_mtp_loss()` with exponential decay:
```python
L_stage1 = Σ w_k * any_correct_loss(logits_k, valid_groups_k)
where w_k = β^k / Σβ^k
```

### Combined Loss
```python
combined_loss = mtp_loss_weight * stage1_loss + stage2_loss
```

## Testing

All tests pass:

1. **Import and config test**: Config creates correctly with new fields
2. **Model construction**: MTP components created for both_mtp stage
3. **Init weights**: Weights initialize without errors
4. **Parameter groups**: All parameters accounted for
5. **Forward pass**: Loss computed successfully
6. **Gradient flow**: All parameters receive gradients
7. **MTP target creation**: Shift logic verified
8. **End-to-end test**: Forward + backward pass works
9. **Training loop**: Loss decreases over steps
10. **Optimizer setup**: Both optimizers created with correct parameter counts

## Usage

```python
config = PDLMConfig(
    sequence_len=1024,
    pure_vocab_size=50257,
    num_groups=8192,
    stage='both_mtp',
    n_layer=12,
    n_head=6,
    n_kv_head=6,
    n_embd=768,
    bucket_size=64,
    n_future_tokens=4,
    mtp_loss_beta=0.8,
    mtp_loss_weight=1.0,
)

model = PDLM(config)
model.init_weights()
optimizers = model.setup_optimizers()

# Training loop uses dataloader with both_mtp stage
```

## Placeholder

`both_mask` stage remains as `NotImplementedError` for future implementation:
- Similar to both_mtp but Stage 1 uses MASK tokens instead of MTP
- No MTP head needed; direct group prediction from MASK positions
