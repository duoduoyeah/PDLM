






## pre-train

python -m scripts.base_train \
    --run=bd3lm_d8_b4_target2_r40 \
    --depth=4 \
    --block_size=4 \
    --prefix_pure_tokens=1 \
    --is_causal=False \
    --max_seq_len=512 \
    --device_batch_size=64 \
    --target_param_data_ratio=40
