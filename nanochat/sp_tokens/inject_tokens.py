import os
import torch
import logging
import pickle
import tiktoken
from nanochat.checkpoint_manager import load_model_from_dir
from nanochat.common import autodetect_device_type
from nanochat.tokenizer import RustBPETokenizer
from nanochat.sp_tokens.kmeans import kmeans

# --------------------------------------- 
# Configuration for loading the model checkpoint
# --------------------------------------- 

# Set the base directory to "model" in the current folder.
# This folder should contain 'tokenizer/' and 'base_checkpoints/'
repo_root = os.getcwd()
os.environ["NANOCHAT_BASE_DIR"] = os.path.join(repo_root, "model")

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
# Hierarchical Clustering Logic
# --------------------------------------- 

def perform_hierarchical_kmeans(X, k=4, max_depth=6):
    """
    Performs hierarchical k-means clustering.
    Returns:
        ancestry: (N, max_depth) LongTensor containing the cluster index (0..k-1)
                  at each depth for every token.
    """
    N = X.shape[0]
    ancestry = torch.full((N, max_depth), -1, dtype=torch.long, device=X.device)
    
    # Track groups as (data_subset, original_indices)
    # Start with the global group containing all tokens
    current_groups = [(X, torch.arange(N, device=X.device))]
    
    print(f"Starting hierarchical k-means (k={k}, depth={max_depth}) on {N} tokens...")

    for depth in range(max_depth):
        print(f"Processing Depth {depth}...")
        next_level_groups = []
        
        for data_subset, original_indices in current_groups:
            # If group is too small (<= k), we can't meaningfully split it into k clusters
            # For consistency, we just assign them to cluster 0 repeatedly or handle gracefully.
            # Here we assign them to '0' and pass them down.
            if len(data_subset) < k:
                ancestry[original_indices, depth] = 0
                next_level_groups.append((data_subset, original_indices))
                continue
            
            # Run k-means on this subset
            # kmeans returns labels (M,) and centroids (k, D)
            labels, _ = kmeans(data_subset, k, iters=20, restarts=5)
            
            # Record ancestry
            ancestry[original_indices, depth] = labels
            
            # Prepare groups for next level
            for i in range(k):
                mask = (labels == i)
                if mask.sum() > 0:
                    sub_indices = original_indices[mask]
                    sub_data = data_subset[mask]
                    next_level_groups.append((sub_data, sub_indices))
        
        current_groups = next_level_groups
        print(f"  > Depth {depth} complete. Found {len(current_groups)} active nodes.")

    return ancestry

# --------------------------------------- 
# Token Generation & Saving
# --------------------------------------- 

def identify_new_tokens(ancestry, base_tokenizer):
    """
    Identifies new group tokens from ancestry and assigns IDs.
    Returns:
        new_tokens_list: List of new token strings.
        path_to_id: Dict mapping ancestry path tuple -> token ID.
        final_special_tokens: Dict of all special tokens (old + new) -> token ID.
    """
    print("Generating new special tokens...")
    
    # 1. Collect all unique paths present in the ancestry
    unique_paths = set()
    # Add Root explicitly (empty tuple)
    unique_paths.add(())
    
    ancestry_cpu = ancestry.cpu()
    N, D = ancestry_cpu.shape
    
    for i in range(N):
        path = []
        for d in range(D):
            cluster_id = ancestry_cpu[i, d].item()
            if cluster_id == -1: break 
            path.append(cluster_id)
            unique_paths.add(tuple(path))
            
    # 2. Sort paths and create token strings
    # Scheme: Root -> "<|MASK|>", Path (0, 3) -> "<|G_03|>"
    sorted_paths = sorted(list(unique_paths), key=lambda x: (len(x), x))
    
    # Get existing special tokens and IDs
    enc = base_tokenizer.enc
    existing_special_tokens = {}
    for name in enc.special_tokens_set:
        existing_special_tokens[name] = enc.encode_single_token(name)

    def path_to_token_str(path):
        if len(path) == 0:
            return "<|MASK|>"
        # Use delimiter to avoid ambiguity between [1, 23] vs [12, 3]
        path_str = "_".join(str(x) for x in path)
        return f"<|G_{path_str}|>"
        
    next_id = enc.n_vocab
    new_tokens_list = []
    path_to_id = {}
    
    for path in sorted_paths:
        token_str = path_to_token_str(path)
        
        if token_str in existing_special_tokens:
            raise ValueError(f"Generated token name {token_str} already exists in base tokenizer.")

        token_id = next_id
        new_tokens_list.append(token_str)
        next_id += 1
            
        path_to_id[path] = token_id

    print(f"Identified {len(new_tokens_list)} new group tokens to add.")
    
    # Combine special tokens
    final_special_tokens = existing_special_tokens.copy()
    for path, tid in path_to_id.items():
        t_str = path_to_token_str(path)
        final_special_tokens[t_str] = tid
            
    return new_tokens_list, path_to_id, final_special_tokens

def create_new_tokenizer_encoding(base_tokenizer, final_special_tokens):
    """Creates the new Tiktoken Encoding object."""
    print(f"Creating new tiktoken encoding with {len(final_special_tokens)} special tokens...")
    enc = base_tokenizer.enc
    mergeable_ranks = enc._mergeable_ranks
    
    new_enc = tiktoken.Encoding(
        name="rustbpe_with_groups",
        pat_str=enc._pat_str,
        mergeable_ranks=mergeable_ranks,
        special_tokens=final_special_tokens
    )
    return new_enc

def build_token_maps(ancestry, path_to_id, vocab_size, new_total_vocab, new_tokens_list, final_special_tokens, max_fanout=1):
    """Builds the pure_to_noisy, tokens_to_pure, and noisy_level maps.
    
    pure_to_noisy_map shape: (vocab_size, num_levels, max_fanout)
    tokens_to_pure_map shape: (new_total_vocab,)
    noisy_level_map shape: (new_total_vocab,)
    """
    print("Building token maps...")
    
    ancestry_cpu = ancestry.cpu()
    N, D = ancestry_cpu.shape
    num_levels = D + 2 # 0 (pure) + D (groups) + 1 (root)
    
    # 1. Pure to Noisy Map (3D for fanout)
    pure_to_noisy_map = torch.full((vocab_size, num_levels, max_fanout), -1, dtype=torch.long)
    # Level 0 is the token itself (fill all fanout slots so sampling is deterministic)
    base_ids = torch.arange(vocab_size).unsqueeze(-1)
    pure_to_noisy_map[:vocab_size, 0] = base_ids
    
    for i in range(N):
        full_path = tuple(ancestry_cpu[i].tolist())
        path_len = len(full_path)
        
        # Level 1 (Finest group) -> Full path
        if path_len > 0 and full_path in path_to_id:
            pure_to_noisy_map[i, 1] = path_to_id[full_path]
            
        # Intermediate/coarser levels: peel off suffixes until length 1
        for level in range(2, path_len + 1):
            sub_path = full_path[: path_len - (level - 1)]
            if sub_path in path_to_id:
                pure_to_noisy_map[i, level] = path_to_id[sub_path]
                
        # Root (Level D+1). We keep this at the final column regardless of actual path length.
        root_path = ()
        if root_path in path_to_id:
             pure_to_noisy_map[i, D + 1] = path_to_id[root_path]

    # For any populated slot, replicate into remaining fanout entries to avoid random -1 picks.
    # This keeps behavior deterministic when only one option exists.
    filled_mask = pure_to_noisy_map != -1
    if max_fanout > 1:
        first_values = pure_to_noisy_map[..., 0].unsqueeze(-1)
        pure_to_noisy_map = torch.where(filled_mask, pure_to_noisy_map, first_values)

    # 2. Tokens to Pure Map
    tokens_to_pure_map = torch.arange(new_total_vocab, dtype=torch.long)
    # Mark new group tokens as -1
    for t_str in new_tokens_list:
        tid = final_special_tokens[t_str]
        tokens_to_pure_map[tid] = -1

    # 3. Noisy Level Map
    noisy_level_map = torch.zeros(new_total_vocab, dtype=torch.long)
    for path, tid in path_to_id.items():
        # level = D + 1 - length
        level = D + 1 - len(path)
        noisy_level_map[tid] = level
        
    return {
        "pure_to_noisy_map": pure_to_noisy_map,
        "tokens_to_pure_map": tokens_to_pure_map,
        "noisy_level_map": noisy_level_map
    }

def compute_max_fanout(overlap_sizes):
    """
    Compute the max fanout (e.g., least common multiple of overlap sizes).
    Placeholder for future overlapping-level support.
    """
    pass

def save_artifacts(output_dir, new_enc, maps):
    """Saves the tokenizer and maps to disk."""
    print("Saving tokenizer and maps...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save Tokenizer
    pickle_path = os.path.join(output_dir, "tokenizer.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(new_enc, f)
    
    # Save Maps
    map_path = os.path.join(output_dir, "token_maps.pt")
    torch.save(maps, map_path)
    
    print(f"Saved tokenizer to {pickle_path}")
    print(f"Saved token maps to {map_path}")
    print(f"New vocab size: {new_enc.n_vocab}")

def generate_and_save_tokenizer(base_tokenizer, ancestry, k, output_dir, max_fanout=1):
    """
    Orchestrates the generation and saving of the new tokenizer and maps.
    """
    # 1. Identify new tokens
    new_tokens_list, path_to_id, final_special_tokens = identify_new_tokens(ancestry, base_tokenizer)
    
    # 2. Create Encoding
    new_enc = create_new_tokenizer_encoding(base_tokenizer, final_special_tokens)
    
    # 3. Build Maps
    maps = build_token_maps(
        ancestry, 
        path_to_id, 
        base_tokenizer.get_vocab_size(), 
        new_enc.n_vocab, 
        new_tokens_list, 
        final_special_tokens,
        max_fanout=max_fanout
    )
    
    # 4. Save
    save_artifacts(output_dir, new_enc, maps)
    
    return new_enc

# --------------------------------------- 
# Main Execution
# --------------------------------------- 

if __name__ == "__main__":
    torch.manual_seed(42)
    
    # 1. Load Resources
    model, tokenizer = get_model_and_tokenizer()
    
    # 2. Extract Embeddings
    lm_head_embeddings = model.lm_head.weight.detach().clone()
    print(f"\nExtracted embeddings: {lm_head_embeddings.shape}")
    
    # 3. Run Hierarchical Clustering
    # We use k=4. Vocab is ~4096. 4^6 = 4096. So depth 6 is appropriate.
    k = 4
    depth = 6
    ancestry = perform_hierarchical_kmeans(lm_head_embeddings, k=k, max_depth=depth)
    
    # 4. Generate Group Tokens & Save New Tokenizer
    new_tokenizer_dir = os.path.join(os.environ["NANOCHAT_BASE_DIR"], "tokenizer")
    
    generate_and_save_tokenizer(tokenizer, ancestry, k, new_tokenizer_dir)
    
    print("\nDone!")
    
