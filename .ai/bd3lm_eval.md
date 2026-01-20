# BD3LM Evaluation

Evaluates BD3LM models on validation data by computing loss and perplexity.

## Setup

- Input: `[xt | x0]` of length 2L
- `xt` (first L): ALL tokens masked (or partially masked with suffix clear)
- `x0` (second L): ALL tokens clean
- Skip block 0 for loss computation (no previous context)
- Compute loss from blocks 1+ (they have clean context via cross-attention)

## Evaluation Modes

### Normal Mode (`target_shift < 0`)

Computes metrics for ALL positions in the block.

Output format (position-centric):
```
[normal mode] overall_loss: 4.19, overall_ppl: 66.11
    pos 0: loss=2.64, ppl=14.0 | 1s=2.30 | 2s=1.86 | 3s=1.45
    pos 1: loss=4.02, ppl=55.7 | 1s=3.50 | 2s=3.00
    pos 2: loss=4.84, ppl=126  | 1s=4.20
    pos 3: loss=5.27, ppl=194
```

- Each position shows base metrics + suffix metrics on one line
- `Ns=X` means "N suffix tokens revealed, loss=X"

### Target Shift Mode (`target_shift >= 1`)

Computes metrics only for position `(target_shift - 1)`.

```
[target_shift=1] loss: 2.64, ppl: 14.02
    1 suffix clear: loss=2.30, ppl=9.94
    2 suffix clear: loss=1.86, ppl=6.43
    3 suffix clear: loss=1.45, ppl=4.27
```

## Suffix Metrics

Suffix metrics reveal clean tokens AFTER the prediction target to measure how much the model benefits from future context.

For prediction position P with N suffix clear:
- Positions 0..P remain MASKED
- Positions P+1..P+N are CLEAN (revealed)
- Remaining positions stay MASKED

Example (block_size=4):
| pred_pos | suffix | Pattern |
|----------|--------|---------|
| 0 | 0 | `[MASK MASK MASK MASK]` |
| 0 | 1 | `[MASK clean MASK MASK]` |
| 0 | 2 | `[MASK clean clean MASK]` |
| 0 | 3 | `[MASK clean clean clean]` |
| 1 | 1 | `[MASK MASK clean MASK]` |
| 1 | 2 | `[MASK MASK clean clean]` |

## Usage

```bash
# Standalone evaluation
python -m scripts.bd3lm_eval --model_tag=d4 --num_batches=20

# With specific step and target_shift
python -m scripts.bd3lm_eval --model_tag=d4 --step=1000 --target_shift=1
```

## Return Format

```python
{
    "overall_loss": float,
    "overall_ppl": float,
    "positions": {
        0: {"loss": X, "ppl": Y, "loss_1suffix": A, "ppl_1suffix": B, ...},
        1: {"loss": X, "ppl": Y, "loss_1suffix": A, ...},
        ...
    },
    "suffix_overall": {
        1: {"positions": [0, 1, 2], "overall_loss": float, "overall_ppl": float},
        ...
    }
}
```
