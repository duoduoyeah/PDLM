# PDLM Eval Design (Two-Stage)

## Context

The new PDLM is a two-stage denoising model:
- **Stage 1**: MASK → Group (predict which group)
- **Stage 2**: Group → Pure (predict exact token given group)

The eval system needs to support both stages and the new Experiment C metrics.

---

## Eval Modes

### 1. Standard Loss Eval
Same as before - compute loss on validation set.

```python
# Stage 2 (Group → Pure) - Experiment C
input:  [pure_prefix..., group_tokens...]
target: [ignore..., pure_tokens...]
metric: CE loss on pure token prediction
```

### 2. Per-Position Loss
Break down loss by position within the block.

```python
# For block_size=4:
loss_pos_1: loss at position i+1
loss_pos_2: loss at position i+2
loss_pos_3: loss at position i+3
loss_pos_4: loss at position i+4
```

**Question**: Does loss degrade with distance even when group hints are given?

### 3. Compatibility Eval (New for Experiment C)

Test if parallel predictions are mutually consistent (self-consistency).

```python
# Step 1: Generate x1, x2, x3, x4 in parallel from G1, G2, G3, G4
# Step 2: For each position i, re-predict x_i given OTHER model predictions
#         e.g., predict x4' given [prefix, x1, x2, x3, G4]
# Step 3: Check if x_i == x_i'

compatibility_score = % positions where prediction unchanged
```

**Note**: This measures self-consistency, NOT accuracy. A model could be consistently wrong.

### 3b. Oracle Accuracy Eval (New)

Test if model can predict correctly given **ground truth** context.

```python
# For each position i in block:
# Step 1: Create input with ground truth pure tokens at OTHER positions
#         e.g., [prefix, p1, p2, p3, G4] where p1-p3 are from validation set
# Step 2: Predict x4'
# Step 3: Check if x4' == p4 (ground truth)

oracle_accuracy = % positions where prediction matches ground truth
```

**Why this matters**: Compatibility eval has a limitation - when `x4 ≠ x4'`, we don't know if:
1. **True incompatibility**: x4' conflicts with [x1, x2, x3] (bad)
2. **Valid alternative**: x4' is perfectly fine with [x1, x2, x3], just a different valid choice

Oracle accuracy helps interpret compatibility results:
- **High oracle accuracy + low compatibility** → Model explores valid alternatives, not broken
- **Low oracle accuracy** → Model fundamentally struggles even with perfect context

**Relationship**:
- Compatibility = self-consistency (does model agree with itself?)
- Oracle accuracy = capability (can model get correct answer given perfect context?)

### 4. Per-Group Accuracy (Analysis)

For each group G_i, compute accuracy when input contains G_i.

```python
group_accuracy = {
    0: 0.85,  # Group 0 has 85% accuracy
    1: 0.72,
    ...
}
```

Correlate with:
- Group size (num tokens in group)
- Embedding spread (std of embeddings within group)

### 5. Logit Leakage (Analysis)

When input is group G_i, how much probability mass goes to tokens OUTSIDE G_i?

```python
# For input with group G_i at position j:
logits = model(input)[:, j, :]  # (B, pure_vocab)
probs = softmax(logits)

# Probability mass within group
in_group_prob = probs[:, tokens_in_group_i].sum()
out_group_prob = 1 - in_group_prob  # This is "leakage"
```

Low leakage = model learned group structure well.

---

## Eval Scripts Structure

```
scripts/pdlm_eval/
├── __init__.py
├── loss_eval.py         # Standard loss + per-position breakdown
├── compatibility.py     # Compatibility score
├── per_group.py         # Per-group accuracy analysis
├── logit_leakage.py     # Probability mass outside group
├── run_eval.py          # Main entry point
└── utils.py             # Shared utilities
```

---

## Command Line Interface

```bash
# Basic loss eval
python -m scripts.pdlm_eval --ckpt_dir path/to/ckpt

# With compatibility eval (self-consistency)
python -m scripts.pdlm_eval --ckpt_dir path/to/ckpt --run_compatibility

# With oracle accuracy eval (capability)
python -m scripts.pdlm_eval --ckpt_dir path/to/ckpt --run_oracle_accuracy

# Full eval (both compatibility and oracle accuracy)
python -m scripts.pdlm_eval --ckpt_dir path/to/ckpt --run_compatibility --run_oracle_accuracy

# Via shell script (defaults: both compatibility and oracle accuracy enabled)
bash launch/eval_pdlm.sh --ckpt_dir=/path/to/ckpt
```

---

## Priority

For Experiment C first model:
1. [Done] Standard loss eval with per-position breakdown
2. [Done] Compatibility eval (self-consistency)
3. [Done] Oracle accuracy eval (capability given perfect context)
4. [Later] Per-group accuracy
5. [Later] Logit leakage

---

## TODO

- [x] Create basic loss_eval.py with per-position support
- [x] Create compatibility.py
- [x] Create oracle accuracy eval
- [x] Create run_eval.py entry point
- [ ] Test on first Experiment C model
- [ ] Per-group accuracy analysis
- [ ] Logit leakage analysis
