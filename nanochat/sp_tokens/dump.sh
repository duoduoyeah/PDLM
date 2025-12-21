#!/usr/bin/env bash
set -euo pipefail

TOKENIZER_DIR=${1:-$(pwd)/model/tokenizer}
OUT_DIR=${2:-temp}

mkdir -p "$OUT_DIR"

# Dump token IDs
uv run -m nanochat.sp_tokens.dump_token_ids --tokenizer-dir "$TOKENIZER_DIR" --output "$OUT_DIR/token_dump.txt"

# Dump maps
uv run -m nanochat.sp_tokens.dump_maps --tokenizer-dir "$TOKENIZER_DIR" --output-dir "$OUT_DIR/maps"

echo "Dumps written under $OUT_DIR"
