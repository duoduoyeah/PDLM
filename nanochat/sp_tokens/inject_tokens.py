import os
import torch
import logging

# --------------------------------------- 
# Configuration for loading the model checkpoint
# --------------------------------------- 

# Set the base directory to "model" in the current folder.
# This folder should contain 'tokenizer/' and 'base_checkpoints/'
repo_root = os.getcwd()
os.environ["NANOCHAT_BASE_DIR"] = os.path.join(repo_root, "model")

from nanochat.checkpoint_manager import load_model_from_dir
from nanochat.common import autodetect_device_type

def get_model_and_tokenizer():
    # Detect the best available device (cuda, mps, or cpu)
    device_type = autodetect_device_type()
    device = torch.device(device_type)
    
    # We follow the convention: model/base_checkpoints/<model_tag>/model_*.pt
    base_dir = os.environ["NANOCHAT_BASE_DIR"]
    checkpoints_dir = os.path.join(base_dir, "base_checkpoints")
    
    if not os.path.exists(checkpoints_dir):
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")
    
    print(f"Loading model and tokenizer using base dir: {base_dir}")
    
    # load_model_from_dir returns (model, tokenizer, meta_data)
    # The tokenizer is the second return value, loaded from <base_dir>/tokenizer/
    model, tokenizer, _ = load_model_from_dir(checkpoints_dir, device, phase="eval")
    
    return model, tokenizer

# --------------------------------------- 
# 2. Extract the lm_head embedding tensor
# --------------------------------------- 


model, tokenizer = get_model_and_tokenizer()

# The lm_head weights are the embeddings we want to cluster
lm_head_embeddings = model.lm_head.weight.detach().clone()

print("\nSuccessfully extracted lm_head embeddings!")
print(f"Shape of embedding tensor: {lm_head_embeddings.shape}")

# You can add further checks or save the embeddings here if needed.
# For instance, to ensure vocab_size matches what the model was trained with:
expected_vocab_size = tokenizer.get_vocab_size()
assert lm_head_embeddings.shape[0] == expected_vocab_size, \
    f"Embedding vocab size {lm_head_embeddings.shape[0]} does not match tokenizer vocab size {expected_vocab_size}"
print(f"Embedding vocab size matches tokenizer vocab size ({expected_vocab_size}).")
    
    
if __name__ == "__main__":
    pass