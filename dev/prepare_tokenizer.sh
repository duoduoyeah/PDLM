
# download repo
branch_name=tokenizer

git clone https://github.com/duoduoyeah/nanochat.git
cd nanochat
git fetch origin
git checkout $branch_name

# set up in colab
cd nanochat
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
uv pip install -e . --config-settings="compile-args=--release"


# download 
python -m nanochat.dataset -n 8

python -m nanochat.report reset

# train and eval tokenizer
# we train a tokenizer with vocab size 4096
python -m scripts.tok_train --vocab_size 4096 --max_chars=2000000000
# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval