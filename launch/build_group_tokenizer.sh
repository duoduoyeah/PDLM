#!/bin/bash
# Build group tokenizer(s) from base tokenizer + model embeddings
# For Experiment C: stage2 (Group -> Pure), no MASK token needed

set -e

# Source paths (Google Drive)
DRIVE_CHECKPOINT="${DRIVE_CHECKPOINT:-/content/drive/MyDrive/nanochat/gpt_d8_next1_r40_v4096_implicit_simple/base_checkpoints}"
DRIVE_TOKENIZER="${DRIVE_TOKENIZER:-/content/drive/MyDrive/nanochat/tokenizer/simplestory_tokenizer/4096/tokenizer}"

# Local paths (VM) - structured for NANOCHAT_BASE_DIR
LOCAL_BASE="${LOCAL_BASE:-/content/temp/group_tokenizer_build}"
LOCAL_OUTPUT="${LOCAL_OUTPUT:-/content/drive/MyDrive/nanochat/group_tokenizers}"

# Model tag (optional, uses largest if not specified)
MODEL_TAG="${MODEL_TAG:-}"

# Configurations for Experiment C
# Format: num_groups overlap_k
# Output naming: n{noise}_k{overlap_k}_g{num_groups} where noise = 4096 / num_groups
CONFIGS=(
    "64 1"    # n64_k1_g64 - main experiment baseline (noise=4096/64=64)
    "16 1"    # n256_k1_g16 - high noise (noise=4096/16=256)
    "256 1"   # n16_k1_g256 - low noise (noise=4096/256=16)
)

echo "============================================"
echo "Building Group Tokenizers for Experiment C"
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
echo "Local base: $LOCAL_BASE"
echo "Output dir: $LOCAL_OUTPUT"
echo "Model tag: ${MODEL_TAG:-auto}"
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

    # Check if tokenizer already exists
    if [ -f "${output_dir}/token_maps.pt" ]; then
        echo "Tokenizer already exists, skipping build..."
    else
        cmd="python -m scripts.build_group_tokenizer \
            --checkpoint-dir \"$LOCAL_BASE\" \
            --output-dir \"$output_dir\" \
            --num-groups \"$num_groups\" \
            --overlap-k \"$overlap_k\" \
            --no-mask"

        if [ -n "$MODEL_TAG" ]; then
            cmd="$cmd --model-tag \"$MODEL_TAG\""
        fi

        eval $cmd
    fi

    # Dump token map info for inspection
    echo "Dumping token map info..."
    python -m nanochat.group_tokenizer.dump "$output_dir" > "${output_dir}/dump_overview.txt"
    python -m nanochat.group_tokenizer.dump "$output_dir" --all-groups > "${output_dir}/dump_all_groups.txt"
    echo "  -> dump_overview.txt"
    echo "  -> dump_all_groups.txt"

    echo ""
done

echo "============================================"
echo "All builds completed!"
echo "Outputs saved to: $LOCAL_OUTPUT"
echo ""
echo "Created tokenizers:"
ls -la "$LOCAL_OUTPUT"
echo "============================================"
