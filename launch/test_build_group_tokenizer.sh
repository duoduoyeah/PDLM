#!/bin/bash
# Test script for build_group_tokenizer
# Tests various num-groups and overlap-k combinations

set -e

# Source paths (Google Drive)
DRIVE_CHECKPOINT="/content/drive/MyDrive/nanochat/gpt_d8_next1_r40_v4096_implicit_simple/base_checkpoints"
DRIVE_TOKENIZER="/content/drive/MyDrive/nanochat/tokenizer/simplestory_tokenizer/4096/tokenizer"

# Local paths (VM) - structured for NANOCHAT_BASE_DIR
LOCAL_BASE="${LOCAL_BASE:-$HOME/temp/group_tokenizer_test}"
LOCAL_OUTPUT="${LOCAL_BASE}/output"

# Test configurations: num_groups overlap_k
CONFIGS=(
    "32 1"
    "64 1"
    "128 1"
    "64 2"
    "64 3"
    "128 2"
)

echo "============================================"
echo "Testing build_group_tokenizer"
echo "============================================"

# Setup: copy from Drive to local with correct structure for NANOCHAT_BASE_DIR
# Structure: LOCAL_BASE/
#            ├── d8/model_*.pt, meta_*.json  (checkpoint)
#            └── tokenizer/...
echo "Setting up local directories..."
mkdir -p "$LOCAL_BASE"
mkdir -p "$LOCAL_BASE/tokenizer"
mkdir -p "$LOCAL_OUTPUT"

# Copy checkpoint if not exists
if [ -d "$LOCAL_BASE/d8" ] && [ -n "$(ls -A $LOCAL_BASE/d8/*.pt 2>/dev/null)" ]; then
    echo "Checkpoint already exists, skipping copy..."
else
    echo "Copying checkpoint from Drive to local..."
    cp -r "$DRIVE_CHECKPOINT"/* "$LOCAL_BASE/"
fi

# Copy tokenizer if not exists
if [ -f "$LOCAL_BASE/tokenizer/tokenizer.pkl" ]; then
    echo "Tokenizer already exists, skipping copy..."
else
    echo "Copying base tokenizer from Drive to local..."
    cp -r "$DRIVE_TOKENIZER"/* "$LOCAL_BASE/tokenizer/"
fi

# Set NANOCHAT_BASE_DIR so get_tokenizer() finds the tokenizer
export NANOCHAT_BASE_DIR="$LOCAL_BASE"

echo ""
echo "NANOCHAT_BASE_DIR: $NANOCHAT_BASE_DIR"
echo "Local output: $LOCAL_OUTPUT"
echo ""

for config in "${CONFIGS[@]}"; do
    read -r num_groups overlap_k <<< "$config"

    output_dir="${LOCAL_OUTPUT}/g${num_groups}_k${overlap_k}"

    echo "--------------------------------------------"
    echo "Building: num-groups=$num_groups, overlap-k=$overlap_k"
    echo "Output: $output_dir"
    echo "--------------------------------------------"

    python -m scripts.build_group_tokenizer \
        --checkpoint-dir "$LOCAL_BASE" \
        --output-dir "$output_dir" \
        --num-groups "$num_groups" \
        --overlap-k "$overlap_k"

    echo ""
done

echo "============================================"
echo "All builds completed!"
echo "Outputs saved to: $LOCAL_OUTPUT"
echo ""
echo "Created tokenizers:"
ls -la "$LOCAL_OUTPUT"
echo "============================================"
