#!/bin/bash
# Test script for build_group_tokenizer
# Tests various num-groups and overlap-k combinations

set -e

# Source paths (Google Drive)
DRIVE_CHECKPOINT="/content/drive/MyDrive/nanochat/gpt_d8_next1_r40_v4096_implicit_simple/base_checkpoints"
DRIVE_TOKENIZER="/content/drive/MyDrive/nanochat/tokenizer/simplestory_tokenizer/4096/tokenizer"

# Local paths (VM) - structured for NANOCHAT_BASE_DIR
LOCAL_BASE="${LOCAL_BASE:-/content/temp/group_tokenizer_test}"
LOCAL_OUTPUT="${LOCAL_BASE}/output"

# Test configurations: num_groups overlap_k
# Output naming: n{noise}_k{overlap_k}_g{num_groups} where noise = 4096 / num_groups
CONFIGS=(
    "32 1"    # n128_k1_g32
    "64 1"    # n64_k1_g64
    "128 1"   # n32_k1_g128
    "64 2"    # n64_k2_g64
    "64 3"    # n64_k3_g64
    "128 2"   # n32_k2_g128
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

    # Calculate noise_level = 4096 / num_groups (assuming vocab_size=4096)
    noise_level=$((4096 / num_groups))
    output_dir="${LOCAL_OUTPUT}/n${noise_level}_k${overlap_k}_g${num_groups}"

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
