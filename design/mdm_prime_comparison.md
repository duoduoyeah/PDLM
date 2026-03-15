# MDM-Prime (Beyond Masked and Unmasked) Comparison

## Paper Info
- Title: Beyond Masked and Unmasked: Discrete Diffusion Models via Partial Masking
- Authors: Chen-Hao Chao et al. (U of Toronto, Vector Institute, NVIDIA, NTU)
- Core idea: decompose each token into sub-tokens via base-b encoding, mask at sub-token level to create intermediate states

## What both papers share
Both identify the same core problem: mask tokens are uninformative intermediate states in masked diffusion models, and both propose adding intermediate states between mask and clean to improve performance.

## Key Differences

| Aspect | PDLM (ours) | Prime |
|---|---|---|
| Intermediate state | Group tokens — semantic clusters via k-means on embeddings. Each group = a subset of the vocabulary | Sub-tokens — base-b encoding of token IDs. Each partial reveal = a digit of the base-b representation |
| Information type | Semantic — group captures meaning (similar words clustered together) | Arbitrary/positional — sub-token digits have no semantic meaning in text (the most-significant digit of a token ID says nothing about meaning) |
| Architecture | Block-level AR (like BD3-LM), block-causal attention. Group tokens extend the embedding table | Pure MDM (like MDLM), order-agnostic. Sub-token embeddings concatenated into token embeddings |
| Denoising | Two explicit stages: mask->group->clean | Continuous diffusion over sub-tokens; sub-tokens unmask independently at different timesteps |
| Training | Cross-entropy on pure token prediction; 3-state noise sampling per block | Variational bound; standard MDM loss on sub-token sequences |
| Softness control | Explicit p_within parameter controlling how deterministic group assignment is | No equivalent — noise level controlled only by diffusion schedule |
| Overlap | Group overlap (each token in multiple groups) adds discriminative power | No analog — base-b encoding is a fixed bijection |

## Advantages of PDLM

1. **Semantic grouping** — Group tokens carry structured, meaningful information (semantically similar tokens share a group). Prime's sub-tokens are arbitrary digit decompositions of token IDs with no semantic content in text.

2. **Controllable noise level** — The p_within parameter and number of groups give fine-grained control over how much information intermediate tokens carry. Prime's only knob is l (sub-token length), which is coarse.

3. **Mechanistic interpretability** — PDLM demonstrates the model learns to use group tokens as a prior (mechanistic validation experiment). Prime has no equivalent analysis.

4. **Direct confidence improvement** — PDLM explicitly shows group tokens improve per-position confidence (argmax prob, Ent-PPL) enabling threshold-based parallel decoding. Prime focuses on reducing idle steps, a different efficiency angle.

5. **Simpler architecture change** — PDLM only extends the embedding table. Prime requires sub-token encoding/decoding, joint probability parameterization with carry-over filtering, and a modified embedding layer with concatenation.

## LM Metrics Reported in Prime

| Metric | Where | Details |
|---|---|---|
| PPL on OpenWebText | Table 2 | Main result: MDLM-Prime l=6 achieves 15.36, vs MDLM 22.98, ARM 17.54 |
| Zero-shot PPL on 7 datasets | Table 1 | LAMBADA, WikiText, PTB, LM1B, AG News, PubMed, ArXiv |
| Idle Step Ratio (ISR) | Table 2 | Fraction of sampling steps where no token changes |

No accuracy, no argmax probability, no Ent-PPL, no parallel decoding throughput reported.

## Repo Exploration (github.com/chen-hao-chao/mdm-prime)

### Structure
- `text/` — language experiments (main.py, diffusion.py, dataloader.py, dit.py, configs/)
- `image/` — CIFAR-10, ImageNet-32
- `toy/` — 2D synthetic demos

### Implementation Notes
- **Base-b encoding**: `diffusion.py` has `encoder()`/`decoder()` for token ID <-> base-b digit conversion
- **Carry-over constraint**: `convert_to_marginal_filter_logit()` uses precomputed filters to zero out invalid sub-token combos
- **Joint probability decoder**: outputs C logits (not b^l), only over valid token mappings via `extended_masks`
- **Sub-token embedding**: each sub-token embedded into D/l dims, concatenated to form D-dim token embedding
- **Training**: Hydra + PyTorch Lightning, variational bound loss, DDPM-style reverse
- **Eval metrics in code**: PPL (main), MAUVE, Self-BLEU, repetition rate, entropy, generative perplexity

### Scale Differences
| | PDLM (ours) | Prime |
|---|---|---|
| Params | 30M | 130M (non-embedding) |
| Dataset | TinyStories | OpenWebText |
| Baseline | BD3-LM (block-AR) | MDLM (pure MDM) |
| PPL type | Exact teacher-forced | Variational upper bound (<=) |
| Seq length | block_size=4 | L=1024 |

## Experiment Design

### Position of BD3-LM-Prime in the paper
BD3-LM-Prime is a **secondary comparison**, not co-equal with BD3-LM. BD3-LM is our structural baseline (the "before" to PDLM's "after"). BD3-LM-Prime answers the reviewer question "how does this compare to Prime?" — that's an ablation-level concern, not a main-result concern. It appears in two experiments, not every table.

### Models
| Model | Intermediate State | Architecture |
|---|---|---|
| BD3-LM | mask only | block-AR (baseline) |
| BD3-LM-Prime | sub-tokens (base-b digits) | block-AR |
| PDLM | group tokens (semantic, k-means) | block-AR |

Same backbone, only intermediate state differs. Clean comparison.

### Experiment 1: End-to-end PPL (add row to Table 4)
- Add BD3-LM-Prime as one row alongside AR, BD3-LM, PDLM
- Just "does the model work" — establishing baseline numbers

### Experiment 2: Threshold-based parallel decoding (extend Table 6 / Figure)
- Add BD3-LM-Prime curve to the existing figure
- Same setup: for each threshold tau, measure avg tokens decoded per step

### Predicted Results

**PPL (Table 4)**: BD3-LM-Prime lands between BD3-LM and PDLM, probably closer to BD3-LM. Sub-token digits give some information (narrowing from ~4k tokens to ~224 candidates with base-b), but the information is arbitrary — knowing a token ID digit doesn't help predict meaning. Could also roughly match PDLM on PPL; if so, the story becomes "both help PPL, but differ on confidence."

**Parallel Decoding (Table 6)**: PDLM clearly wins. Confidence depends on how much the intermediate state narrows the output distribution in a meaningful way:
- Group token says "animal noun" → mass concentrates on ~64 semantically related tokens → high argmax prob
- Sub-token digit says "first base-15 digit is 3" → mass spreads across ~224 semantically unrelated tokens → lower argmax prob

At low thresholds (tau 0.4-0.5): both similar, bar is low.
At high thresholds (tau 0.7-0.9): PDLM pulls ahead clearly.
BD3-LM-Prime might even look similar to BD3-LM on this metric if arbitrary digits don't meaningfully sharpen the distribution.

**Summary of predictions:**

| | PPL | Parallel Decode |
|---|---|---|
| BD3-LM-Prime vs BD3-LM | better | slightly better or similar |
| BD3-LM-Prime vs PDLM | similar or slightly worse | clearly worse |

**Punchline**: Intermediate states help PPL broadly, but only semantically structured intermediate states help confidence/parallel decoding.

## BD3-LM-Prime Implementation

### Settings Rationale

**target_length (ℓ) = 2**: The only free design choice. Constrained by:
1. `base^ℓ >= vocab_size` must hold exactly (no wasted sub-token combos)
2. `n_embd % ℓ == 0` (embedding dimension must split evenly)

For vocab=4096, n_embd=512: ℓ=2 gives base=64 (64²=4096 exact, 512/2=256 clean). ℓ=3 gives base=16 (16³=4096 exact, but 512/3≈170.67 — breaks). ℓ=2 also matches MDM-Prime's own text experiments.

**base=64, sub_token_vocab=65**: Mechanically derived, no design choices:
- base = ceil(4096^(1/2)) = 64 (only valid value)
- mask sub-token ID = 64 (next available after valid range 0–63)
- embedding size = 64+1 = 65

Analogous to BD3-LM: valid tokens 0–4095, mask token 4096, vocab 4097.

### Files Created
- `nanochat/bd3lm_utils/prime_encoding.py` — base-b encode/decode (arithmetic, no learning)
- `nanochat/bd3lm_prime.py` — BD3LMPrime model + BD3LMPrimeConfig
- `nanochat/dataloader_bd3lm_prime.py` — sub-token-level masking dataloader
- `slurms/slurm_bd3lm_prime.sh` — SLURM training script

### Files Modified
- `nanochat/dataloader.py` — added BD3LMPrimeConfig dispatch
- `scripts/base_train.py` — added `bd3lm_prime` model type

### Model Architecture
- Input embedding: `nn.Embedding(65, 256)` — embeds sub-tokens, pairs concatenated to 512-dim
- Output head: `nn.Linear(512, 4096)` — predicts full tokens (same as BD3-LM)
- Transformer: identical to BD3-LM (shared Block/Attention/MLP code)
- Params: ~27.3M (vs ~29M BD3-LM, due to smaller embedding table)

## TODO
- [x] Explore their GitHub repo
- [x] Design comparison experiments
- [x] Implement BD3-LM-Prime model
- [ ] Run experiments
