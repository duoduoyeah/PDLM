#!/bin/bash
# Build group tokenizer(s) from base tokenizer + model embeddings
# For Experiment C: stage2 (Group -> Pure), no MASK token needed

set -e

# Paths - adjust these for your environment
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/content/drive/MyDrive/nanochat/gpt_d8_next1_r40_v4096_implicit_simple/base_checkpoints}"
OUTPUT_BASE="${OUTPUT_BASE:-/content/drive/MyDrive/nanochat/group_tokenizers}"

# Model tag (optional, uses largest if not specified)
MODEL_TAG="${MODEL_TAG:-}"

# Configurations for Experiment C
# Format: num_groups overlap_k
CONFIGS=(
    "64 1"    # g64_k1 - main experiment baseline
    "16 1"    # g16_k1 - high noise
    "256 1"   # g256_k1 - low noise
)

echo "============================================"
echo "Building Group Tokenizers for Experiment C"
echo "============================================"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Output base: $OUTPUT_BASE"
echo "Model tag: ${MODEL_TAG:-auto}"
echo ""

mkdir -p "$OUTPUT_BASE"

for config in "${CONFIGS[@]}"; do
    read -r num_groups overlap_k <<< "$config"

    output_dir="${OUTPUT_BASE}/g${num_groups}_k${overlap_k}"

    echo "--------------------------------------------"
    echo "Building: num-groups=$num_groups, overlap-k=$overlap_k"
    echo "Output: $output_dir"
    echo "--------------------------------------------"

    cmd="python -m scripts.build_group_tokenizer \
        --checkpoint-dir \"$CHECKPOINT_DIR\" \
        --output-dir \"$output_dir\" \
        --num-groups \"$num_groups\" \
        --overlap-k \"$overlap_k\" \
        --no-mask"

    if [ -n "$MODEL_TAG" ]; then
        cmd="$cmd --model-tag \"$MODEL_TAG\""
    fi

    eval $cmd

    echo ""
done

echo "============================================"
echo "All builds completed!"
echo "Outputs saved to: $OUTPUT_BASE"
echo ""
echo "Created tokenizers:"
ls -la "$OUTPUT_BASE"
echo "============================================"
