#!/bin/bash
# Build group tokenizer(s) from base tokenizer + model embeddings
# For Experiment C: stage2 (Group -> Pure), no MASK token needed
#
# Usage: LOCAL_OUTPUT=/path/to/output ./launch/build_group_tokenizer.sh --mode all_combos --num-sub 8 --sub-per-final 2

set -e

# Source paths (Google Drive)
DRIVE_CHECKPOINT="${DRIVE_CHECKPOINT:-/content/drive/MyDrive/nanochat/gpt_d8_next1_r40_v4096_implicit_simple/base_checkpoints}"
DRIVE_TOKENIZER="${DRIVE_TOKENIZER:-/content/drive/MyDrive/nanochat/tokenizer/simplestory_tokenizer/4096/tokenizer}"

# Local paths (VM) - structured for NANOCHAT_BASE_DIR
LOCAL_BASE="${LOCAL_BASE:-/content/temp/group_tokenizer_build}"
LOCAL_OUTPUT="${LOCAL_OUTPUT:-/content/drive/MyDrive/nanochat/group_tokenizers}"

# Model tag (optional, uses largest if not specified)
MODEL_TAG="${MODEL_TAG:-}"

echo "============================================"
echo "Building Group Tokenizer (Sub-group Combination)"
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
echo "Extra args: $@"
echo ""

echo "--------------------------------------------"
echo "Building with args: $@"
echo "Output: $LOCAL_OUTPUT"
echo "--------------------------------------------"

# Check if tokenizer already exists
if [ -f "${LOCAL_OUTPUT}/token_maps.pt" ]; then
    echo "Tokenizer already exists, skipping build..."
else
    cmd="python -m scripts.build_group_tokenizer \
        --checkpoint-dir \"$LOCAL_BASE\" \
        --output-dir \"$LOCAL_OUTPUT\" \
        $@"

    if [ -n "$MODEL_TAG" ]; then
        cmd="$cmd --model-tag \"$MODEL_TAG\""
    fi

    eval $cmd
fi

# Dump token map info for inspection
echo "Dumping token map info..."
python -m nanochat.group_tokenizer.dump "$LOCAL_OUTPUT" > "${LOCAL_OUTPUT}/dump_overview.txt"
python -m nanochat.group_tokenizer.dump "$LOCAL_OUTPUT" --all-groups > "${LOCAL_OUTPUT}/dump_all_groups.txt"
echo "  -> dump_overview.txt"
echo "  -> dump_all_groups.txt"

echo ""
echo "============================================"
echo "Build completed!"
echo "Output saved to: $LOCAL_OUTPUT"
echo "============================================"
