# MTP-PDLM Evaluation Design

## Inference Function Design

### Overview

For `both_mtp` model, generation happens in **two steps per block**:

```
Step 1 (MTP): [prompt...] → predict K group tokens [G1, G2, G3, G4]
Step 2 (Denoise): [prompt..., G1, G2, G3, G4] → predict K pure tokens [P1, P2, P3, P4]
```

### Algorithm

```python
def generate_both_mtp(model, prompt_tokens, max_new_tokens, block_size=4):
    """
    Generate tokens using both_mtp model.

    Each block is generated in two steps:
    1. MTP (Pure → Group): predict K group tokens from pure context
    2. Denoise (Group → Pure): predict K pure tokens from group tokens

    Args:
        prompt_tokens: list of pure token ids
        max_new_tokens: number of new pure tokens to generate
        block_size: K, tokens per block (default 4)

    Returns:
        generated_tokens: list of pure token ids
        debug_info: per-block generation details
    """
    tokens = prompt_tokens.copy()
    debug_blocks = []

    while len(tokens) < len(prompt_tokens) + max_new_tokens:
        # === Step 1: MTP (Pure → Group) ===
        # Input: [pure tokens so far]
        # Use x0 path (second L) to predict K future group tokens

        # Main model predicts 1st group token
        # MTP head predicts 2nd..Kth group tokens (autoregressive within MTP)
        group_tokens = mtp_predict(model, tokens, K=block_size)  # [G1, G2, G3, G4]

        # === Step 2: Denoise (Group → Pure) ===
        # Input xt: [pure prefix..., G1, G2, G3, G4]
        # Input x0: [pure prefix..., placeholder...]
        # Use xt path (first L) to predict pure tokens at group positions

        pure_tokens = denoise_predict(model, tokens, group_tokens)  # [P1, P2, P3, P4]

        # Append to context
        tokens.extend(pure_tokens)

        debug_blocks.append({
            "step": len(debug_blocks),
            "group_tokens": group_tokens,
            "pure_tokens": pure_tokens,
        })

    return tokens[:len(prompt_tokens) + max_new_tokens], debug_blocks
```

### Implementation Details

#### Step 1: MTP Prediction

```python
def mtp_predict(model, context_tokens, K):
    """
    Predict K group tokens using MTP.

    Uses the x0 path (second L positions) of the model.
    - Main lm_head[pure_vocab:] predicts 1st group token
    - MTP head predicts 2nd..Kth group tokens autoregressively
    """
    # Forward through transformer (x0 path only, no xt)
    hidden = model.forward_x0_only(context_tokens)  # (1, T, D)

    # Main model: predict 1st group token
    group_logits = model.lm_head(hidden)[:, -1, pure_vocab:]  # (1, num_groups)
    g1 = group_logits.argmax(dim=-1)  # (1,)

    # MTP head: predict 2nd..Kth group tokens
    # Autoregressive: each prediction uses previous group token
    group_tokens = [g1]
    prev_h = hidden[:, -1:, :]
    prev_tok = g1

    for k in range(K - 1):
        tok_emb = model.group_wte(prev_tok)
        combined = cat([norm(prev_h), norm(tok_emb)], dim=-1)
        x = model.mtp_head.proj(combined)
        prev_h = model.mtp_head.block(x, cos, sin)
        logits = model.mtp_head.lm_head(prev_h)  # (1, 1, num_groups)
        g_next = logits.argmax(dim=-1)
        group_tokens.append(g_next)
        prev_tok = g_next

    return group_tokens  # [G1, G2, ..., GK]
```

#### Step 2: Denoise Prediction

```python
def denoise_predict(model, context_tokens, group_tokens, K):
    """
    Predict K pure tokens from group tokens using Stage 2.

    Uses the xt path (first L positions) of the model.
    Input structure: [xt | x0] where:
    - xt = [pure prefix..., G1, G2, G3, G4]
    - x0 = [pure prefix..., placeholder...] (or pure prefix only)
    """
    # Build xt: context + group tokens
    xt = context_tokens + [g + group_offset for g in group_tokens]

    # Build x0: context + placeholders (or zeros)
    x0 = context_tokens + [0] * K  # placeholder, won't affect xt predictions

    # Forward with block diffusion mask
    logits = model.forward_for_eval(xt, x0, attn_mask)  # (1, T, pure_vocab)

    # Extract predictions at group positions
    pure_logits = logits[:, -K:, :pure_vocab]  # (1, K, pure_vocab)

    # Optionally: mask logits to only tokens in each group
    for i, g in enumerate(group_tokens):
        mask = model.token_map.get_group_members_mask(g)
        pure_logits[:, i, ~mask] = -inf

    pure_tokens = pure_logits.argmax(dim=-1)  # (1, K)
    return pure_tokens.tolist()
```

### Key Design Decisions

1. **Two-step per block**: MTP first, then denoise. This matches the training objective.

2. **MTP is autoregressive within block**: The MTP head predicts g2 given g1, g3 given g2, etc.

3. **Denoise is parallel**: All K pure tokens are predicted in parallel from K group tokens.

4. **Group masking optional**: Can constrain pure predictions to tokens within the predicted group.

5. **No teacher forcing at inference**: Unlike training, we use model's own predictions.

### Attention Mask for Inference

For Step 2 (denoise), need a modified attention mask:
- xt positions (with group tokens) should attend to previous pure context
- x0 positions provide clean context via block diffusion mask

```
For sequence [P1, P2, P3, G1, G2, G3, G4]:
        P1  P2  P3  G1  G2  G3  G4
    P1   1   0   0   0   0   0   0
    P2   1   1   0   0   0   0   0
    P3   1   1   1   0   0   0   0
    G1   1   1   1   1   1   1   1  <- attend to all (block diffusion)
    G2   1   1   1   1   1   1   1
    G3   1   1   1   1   1   1   1
    G4   1   1   1   1   1   1   1
```

---

## Overview

The MTP-PDLM model (`both_mtp` stage) combines two tasks in a single forward pass:
- **Stage 1 (MTP)**: Pure → K Group tokens (second L positions)
- **Stage 2**: Group → Pure tokens (first L positions)

This document defines the evaluation metrics for the combined model.

---

## Architecture Recap

```
Input (2L): [xt | x0]
             │       │
        first L   second L
             │       │
             ▼       ▼
    ┌─────────────────────────┐
    │   Shared Transformer    │
    │   + Block Diff Mask     │
    └─────────────────────────┘
             │       │
             ▼       ▼
        Stage 2   Stage 1
    (Group→Pure) (Pure→Group MTP)
```

---

## Metrics Summary

| Stage | Task | Metrics |
|-------|------|---------|
| Stage 1 | Pure → Group (MTP) | Loss, PPL, Accuracy (any-correct), per-position (k=0..K-1) |
| Stage 2 | Group → Pure | Loss, PPL, Accuracy, per-position within block |
| Combined | Both | Weighted combined loss, individual stage metrics |

---

## Stage 1 Metrics (MTP: Pure → Group)

Inherited from `mtp_eval.py`.

### 1.1 Overall Metrics
- **loss**: Mean NLL across all positions and K predictions
- **ppl**: exp(loss)
- **accuracy**: Any-correct accuracy (prediction matches ANY valid group)

### 1.2 Per-Position Metrics (k=0..K-1)
For each prediction step k:
- **loss_k**: NLL at position k
- **ppl_k**: exp(loss_k)
- **accuracy_k**: Any-correct accuracy at position k

Where:
- k=0: Main model prediction (from `lm_head[pure_vocab:]`)
- k=1..K-1: MTP head predictions

### 1.3 Any-Correct Loss Formula
When `overlap_k > 1`, multiple groups are valid:
```
loss = -log(Σ P(g) for g in valid_groups)
     = -logsumexp(log_probs[valid_targets])
```

### 1.4 Any-Correct Accuracy
```
correct = (argmax(logits) in valid_targets)
```

---

## Stage 2 Metrics (Group → Pure)

Inherited from `pdlm_eval.py`.

### 2.1 Overall Metrics
- **loss**: CE loss on pure token prediction at block positions
- **ppl**: exp(loss)
- **accuracy**: Token-level accuracy

### 2.2 Per-Position Metrics (pos=0..block_size-1)
For each position within a block:
- **loss_pos_i**: Loss at position i within block
- **ppl_pos_i**: exp(loss_pos_i)
- **accuracy_pos_i**: Accuracy at position i

Note: Block 0 is skipped (no previous context).

### 2.3 Compatibility Eval (Self-Consistency)
Tests if parallel predictions are mutually consistent:
1. Generate x1, x2, x3, x4 in parallel from G1, G2, G3, G4
2. For each position i, re-predict x_i given OTHER model predictions
3. Check if x_i == x_i'

```
compatibility_score = % positions where prediction unchanged
```

### 2.4 Oracle Accuracy Eval (Capability)
Tests prediction given ground truth context:
1. Create input with ground truth pure tokens at OTHER positions
2. Predict target position
3. Check if prediction matches ground truth

```
oracle_accuracy = % positions where prediction matches ground truth
```

**Interpretation:**
- High oracle + low compatibility → Model explores valid alternatives
- Low oracle → Model struggles even with perfect context

---

## Combined Metrics

### 3.1 Individual Stage Losses
```python
stage1_loss = mtp_any_correct_loss(...)  # Pure → Group
stage2_loss = ce_loss(...)               # Group → Pure
```

### 3.2 Combined Loss (matches training)
```python
combined_loss = mtp_loss_weight * stage1_loss + stage2_loss
```

### 3.3 Eval Output Structure
```python
{
    "combined": {
        "loss": float,  # weighted combination
    },
    "stage1_mtp": {
        "overall_loss": float,
        "overall_ppl": float,
        "overall_accuracy": float,
        "positions": {
            0: {"loss": X, "ppl": Y, "accuracy": Z},
            1: {"loss": X, "ppl": Y, "accuracy": Z},
            ...  # up to K-1
        },
    },
    "stage2_denoise": {
        "overall_loss": float,
        "overall_ppl": float,
        "overall_accuracy": float,
        "positions": {
            0: {"loss": X, "ppl": Y, "accuracy": Z},
            1: {"loss": X, "ppl": Y, "accuracy": Z},
            ...  # up to block_size-1
        },
        "compatibility": {  # optional
            "overall": float,
            "by_position": {...},
        },
        "oracle_accuracy": {  # optional
            "overall": float,
            "by_position": {...},
        },
    },
    "num_batches": int,
    "config": {
        "block_size": int,
        "n_future_tokens": int,
        "mtp_loss_weight": float,
    },
}
```

---

## Eval Modes

### Mode 1: Quick Eval (During Training)
Fast eval for training loop callbacks.

**Metrics computed:**
- Stage 1: overall loss, ppl, accuracy + per-k breakdown
- Stage 2: overall loss, ppl, accuracy + per-position breakdown

**Output format (condensed):**
```
[mtp_pdlm] combined_loss: 2.15
  Stage 1 (MTP): loss=0.44, ppl=1.55, accuracy=81.95%
    k=0: loss=0.18, ppl=1.19, acc=92.90%
    k=1: loss=0.42, ppl=1.52, acc=82.85%
    k=2: loss=0.56, ppl=1.75, acc=77.20%
    k=3: loss=0.61, ppl=1.84, acc=74.85%
  Stage 2 (Denoise): loss=1.71, ppl=5.53, accuracy=62.30%
    pos=0: loss=1.65, ppl=5.21, acc=64.10%
    pos=1: loss=1.70, ppl=5.47, acc=62.80%
    pos=2: loss=1.73, ppl=5.64, acc=61.90%
    pos=3: loss=1.76, ppl=5.81, acc=60.40%
```

### Mode 2: Full Eval (Standalone)
Comprehensive evaluation with all metrics.

**Additional metrics:**
- Stage 2 compatibility eval
- Stage 2 oracle accuracy eval
- Per-group accuracy (Stage 1)

**Command:**
```bash
python -m scripts.mtp_pdlm_eval --ckpt_dir=/path/to/ckpt [--run_compatibility] [--run_oracle]
```

---

## Implementation Plan

### Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `nanochat/mtp_pdlm_eval.py` | Create | Combined eval module |
| `scripts/mtp_pdlm_eval.py` | Create | Standalone eval script |
| `launch/eval_mtp_pdlm.sh` | Create | Shell script launcher |

### Key Functions

```python
# nanochat/mtp_pdlm_eval.py

def eval_mtp_pdlm(
    model,
    val_loader,
    num_batches,
    device,
    autocast_ctx,
    attn_mask,
    run_compatibility=False,
    run_oracle=False,
) -> dict:
    """
    Evaluate MTP-PDLM model on both stages.

    Returns combined metrics for Stage 1 (MTP) and Stage 2 (Denoise).
    """
    pass

def eval_mtp_pdlm_quick(
    model,
    val_loader,
    num_batches,
    device,
    autocast_ctx,
    attn_mask,
) -> dict:
    """
    Quick eval for training callbacks.
    Only computes loss/ppl/accuracy metrics.
    """
    pass
```

### Dataloader Requirements

The eval dataloader must provide:
```python
inputs:      (B, T)           # xt: group tokens at block positions
targets:     (B, T)           # x0: pure tokens everywhere
loss_extras: {
    "loss_mask":    (B, T),           # Stage 2 loss mask
    "mtp_targets":  (B, T, K, overlap_k),  # Stage 1 MTP targets
}
```

---

## Priority

1. [TODO] Create `nanochat/mtp_pdlm_eval.py` with quick eval
2. [TODO] Add `forward_for_eval` to PDLM for `both_mtp` stage
3. [TODO] Create `scripts/mtp_pdlm_eval.py` standalone script
4. [TODO] Add compatibility/oracle eval support
5. [TODO] Create `launch/eval_mtp_pdlm.sh`
6. [TODO] Test on trained model
