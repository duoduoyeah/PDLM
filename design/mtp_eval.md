# MTP Stage 1 Evaluation

## Metrics

**Overall:**
- `loss`: mean NLL across all positions and K predictions
- `ppl`: exp(loss)
- `accuracy`: transition accuracy (argmax matches target)

**Per-position (k=0..K-1):**
- Same metrics broken down by prediction step
- k=0 is main model, k=1..K-1 are MTP head predictions

**Per-group:**
- `accuracy` (recall): when group G's tokens are targets, how often is the model correct? Distributes 1/num_valid across all valid target groups per position.
- `precision`: when the model predicts group G, how often is it correct?
- `predicted`: raw count of times the model chose this group
- `predicted_correct`: raw count of correct predictions for this group
- `correct` / `total`: weighted counts for the recall metric

## Overlap Handling (Any-Correct Loss)

When `overlap_k > 1`, each position has multiple valid target groups (due to overlapping group assignments in the tokenizer).

**Target shape:** `(B, T, K, overlap_k)`
- Invalid entries padded with -1

**Loss computation:**
```
loss = -log(Σ P(g) for g in valid_groups)
     = -logsumexp(log_probs[valid_targets])
```

This "any-correct" formulation means predicting ANY valid group is equally correct.

**Accuracy computation:**
```
correct = (argmax(logits) in valid_targets)
```

Prediction is correct if it matches any of the valid groups.

## Usage

**During training:** Called periodically, outputs condensed format:
```
[mtp] overall_loss: 0.44, overall_ppl: 1.55, overall_accuracy: 81.95%
  k=0: loss=0.18, ppl=1.19, accuracy=92.90%
  k=1: loss=0.42, ppl=1.52, accuracy=82.85%
  ...
```

**Standalone evaluation:**
```bash
uv run -m scripts.mtp_eval --model_tag=mtp_d8 --step=1000
```
Includes per-group breakdown and statistics.

**Dump predictions:**
```bash
uv run -m scripts.mtp_eval --ckpt_dir=/path/to/ckpt --dump_mtp=/tmp/dump.txt --dump_sequences=5
```
Shows K future predictions per position with PURE/GROUP/WRONG status.

**Shell launcher (eval + optional dump):**
```bash
bash launch/eval_mtp.sh --ckpt_path=/path/to/model --dump_path=/tmp/dump.txt
```

**Plot results:**
```bash
uv run -m scripts.dump.plot_mtp_eval eval_result.json --output_dir /tmp/plots
```
Generates position metrics, group accuracy distribution, and top/bottom group charts.

## Pure vs Group Target Mode

Controlled by `stage1_target_mode` in model config. Old checkpoints without this field default to `"group"`.

- **Pure mode**: `lm_head` outputs over pure vocab, collapsed to groups via `group_to_pure_mask` at eval. Accuracy uses the collapsed group argmax.
- **Group mode**: `lm_head` outputs over group vocab directly. Loss uses any-correct CE.
