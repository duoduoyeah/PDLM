## colab version of training gpt2 arch next token prediction model

##-- set env--------------------------

# uv
uv pip uninstall torch torchvision torchaudio
uv pip install -e ".[gpu]" --force-reinstall




## prepare the data
python -m nanochat.dataset -n 8

##--- pre-train --------------------------

python -m scripts.base_train \
    --run=gpt_d8_next1_r40 \
    --depth=8 \
    --max_seq_len=512 \
    --device_batch_size=64 \
    --total_batch_size=131072 \
    --save_every=6000 \
    --target_shift=1 \
    --mask_token=0 \
    --target_param_data_ratio=40