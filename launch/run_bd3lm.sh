#!/bin/bash

## BD3LM Dry Run Configurations
## depth=4, block_size=4, target_param_data_ratio=10 for quick testing

# Run 1: Normal BD3LM (random masking, ~50% tokens masked)
python -m scripts.base_train \
    --run=bd3lm_d4_b4_normal \
    --depth=4 \
    --block_size=4 \
    --prefix_pure_tokens=1 \
    --is_causal=False \
    --max_seq_len=512 \
    --device_batch_size=128 \
    --target_param_data_ratio=10 \
    --target_shift=-1 \
    --bd3lm_effective_ratio=0.5

# Run 2: BD3LM with target_shift=1 (predict 1st position in each block)
python -m scripts.base_train \
    --run=bd3lm_d4_b4_ts1 \
    --depth=4 \
    --block_size=4 \
    --prefix_pure_tokens=1 \
    --is_causal=False \
    --max_seq_len=512 \
    --device_batch_size=128 \
    --target_param_data_ratio=10 \
    --target_shift=1 \
    --bd3lm_effective_ratio=0.25

# Run 3: BD3LM with target_shift=2 (predict 2nd position in each block)
python -m scripts.base_train \
    --run=bd3lm_d4_b4_ts2 \
    --depth=4 \
    --block_size=4 \
    --prefix_pure_tokens=1 \
    --is_causal=False \
    --max_seq_len=512 \
    --device_batch_size=128 \
    --target_param_data_ratio=10 \
    --target_shift=2 \
    --bd3lm_effective_ratio=0.25

# Run 4: BD3LM with target_shift=4 (predict 4th/last position in each block)
python -m scripts.base_train \
    --run=bd3lm_d4_b4_ts4 \
    --depth=4 \
    --block_size=4 \
    --prefix_pure_tokens=1 \
    --is_causal=False \
    --max_seq_len=512 \
    --device_batch_size=128 \
    --target_param_data_ratio=10 \
    --target_shift=4 \
    --bd3lm_effective_ratio=0.25
