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

def generate_and_save_tokenizer(base_tokenizer, ancestry, k, output_dir):
    """
    Generates special tokens based on ancestry and creates a new tokenizer.
    """
    print("Generating new special tokens...")
    
    # 1. Collect all unique paths present in the ancestry
    # A path is a tuple of indices (c_0, c_1, ..., c_d)
    unique_paths = set()
    
    # Add Root explicitly
    # We will represent root as empty tuple ()
    unique_paths.add(())
    
    # Iterate over all tokens and all depths
    # ancestry shape: (N, D)
    ancestry_cpu = ancestry.cpu()
    N, D = ancestry_cpu.shape
    
    for i in range(N):
        path = []
        for d in range(D):
            cluster_id = ancestry_cpu[i, d].item()
            if cluster_id == -1: break # Should not happen with our logic
            path.append(cluster_id)
            unique_paths.add(tuple(path))
            
    # 2. Create Token Strings
    # Scheme: 
    # Root -> "<|MASK|>"
    # Path (0, 3) -> "<|G_03|>"
    # Path (1, 2, 0) -> "<|G_120|>"
    
    sorted_paths = sorted(list(unique_paths), key=lambda x: (len(x), x))
    
    new_special_token_map = {} # str -> int (ID)
    
    # Start IDs after the current vocabulary?
    # Actually, tiktoken manages IDs. We just need to provide the mapping.
    # But for compatibility, we usually want to append them.
    # Let's inspect the base tokenizer's special tokens to find the next available ID
    
    # Access the inner tiktoken encoding
    enc = base_tokenizer.enc
    
    # Get existing special tokens
    # enc.special_tokens_set is a set of strings
    # We need the values (IDs) to find the max ID.
    existing_special_tokens = {}
    for name in enc.special_tokens_set:
        existing_special_tokens[name] = enc.encode_single_token(name)
        
    # Also consider regular tokens
    vocab_size = enc.n_vocab
    # usually n_vocab includes everything, but let's be safe.
    # The max ID in use is vocab_size - 1 usually.
    
    next_id = vocab_size
    
    # 3. Define new tokens
    new_tokens_list = []
    
    # We need to map path -> new_token_id
    path_to_id = {}
    
    for path in sorted_paths:
        if len(path) == 0:
            token_str = "<|MASK|>"
        else:
            path_str = "".join(str(x) for x in path)
            token_str = f"<|G_{path_str}|>"
        
        # Check if it already exists (unlikely for new G_ tokens but possible for MASK?)
        if token_str in existing_special_tokens:
            token_id = existing_special_tokens[token_str]
        else:
            token_id = next_id
            new_tokens_list.append(token_str)
            next_id += 1
            
        path_to_id[path] = token_id

    print(f"Identified {len(new_tokens_list)} new group tokens to add.")
    
    # 4. Merge and Create New Tokenizer
    # We need mergeable_ranks from the old tokenizer
    mergeable_ranks = enc._mergeable_ranks
    
    # Combine special tokens
    final_special_tokens = existing_special_tokens.copy()
    current_special_id = vocab_size # Start assigning new IDs from here
    
    # Re-assign IDs to be contiguous just in case, though next_id logic above was fine.
    # Actually, let's stick to the IDs we generated in step 3 to match the maps we build below.
    for t_str in new_tokens_list:
        final_special_tokens[t_str] = existing_special_tokens.get(t_str, current_special_id)
        if t_str not in existing_special_tokens:
            current_special_id += 1
        
    print(f"Creating new tiktoken encoding with {len(final_special_tokens)} special tokens...")
    
    new_enc = tiktoken.Encoding(
        name="rustbpe_with_groups",
        pat_str=enc._pat_str,
        mergeable_ranks=mergeable_ranks,
        special_tokens=final_special_tokens
    )
    
    # 5. Build the Maps for TokenMap class
    # pure_to_noisy_map: shape (vocab_size, max_depth)
    # tokens_to_pure_map: shape (new_vocab_size)
    # noisy_level_map: shape (new_vocab_size)
    
    # We need max_depth from ancestry
    # ancestry was (N, D). Note that the 'D' in ancestry is how many splits we did.
    # But our paths can be length 0 to D.
    # Actually, the depth of the tree is D.
    # Let's say depth 0 is root (MASK), depth 1 is 1st split, ..., depth D is leaves (pure tokens).
    # Wait, the TokenMap likely expects:
    # level 0 = pure token
    # level 1 = parent group
    # ...
    # level D = root
    # OR the reverse?
    # Usually "noisy level" implies 0 is clean, higher is more noise.
    # Let's assume Level 0 = Pure Token. Level 1 = Fine Group... Level Max = Root.
    
    # In ancestry, we have N tokens, D cols.
    # ancestry[i] = [c0, c1, c2, ..., c_{D-1}]
    # Token i corresponds to path (c0, c1, ..., c_{D-1}).
    
    # Let's align with the levels:
    # Level 0: The original token ID.
    # Level 1: The finest group (full path in ancestry).
    # ...
    # Level D: The coarsest group (ancestry[:, 0]).
    # Level D+1: The Root (empty path).
    
    # So max_level = D + 1.
    # pure_to_noisy_map shape: (vocab_size, D + 2) -> columns 0..(D+1)
    
    N, D = ancestry_cpu.shape
    num_levels = D + 2 # 0 (pure) + D (groups) + 1 (root)
    
    pure_to_noisy_map = torch.full((vocab_size, num_levels), -1, dtype=torch.long)
    
    # Initialize with identity for level 0
    # Note: vocab_size might be larger than N if there are special tokens in the base vocab?
    # ancestry is for 0..N-1.
    # For special tokens outside 0..N-1, they might not have a group? Or map to themselves?
    # Let's assume they map to themselves or Root?
    # Let's stick to 0..N-1 (the text tokens) having ancestry.
    
    pure_to_noisy_map[:vocab_size, 0] = torch.arange(vocab_size)
    
    for i in range(N):
        # The full path for token i
        full_path = tuple(ancestry_cpu[i].tolist())
        
        # Level 1 (Finest group) -> Full path
        if full_path in path_to_id:
            pure_to_noisy_map[i, 1] = path_to_id[full_path]
            
        # Intermediate levels
        # If D=6. Path has 6 elements.
        # Level 1: path[:] (length 6)
        # Level 2: path[:-1] (length 5)
        # ...
        # Level 6: path[:1] (length 1)
        # Level 7: path[:0] (Root)
        
        for d in range(D):
            # length of path for this level
            path_len = D - d
            sub_path = full_path[:path_len]
            level = d + 1
            if sub_path in path_to_id:
                pure_to_noisy_map[i, level] = path_to_id[sub_path]
                
        # Root (Level D+1)
        root_path = ()
        if root_path in path_to_id:
             pure_to_noisy_map[i, D + 1] = path_to_id[root_path]

    # tokens_to_pure_map
    # Maps ANY token ID (pure or group) to its "pure" representative?
    # Wait, a group token maps to WHAT pure token?
    # Usually it maps to a "representative" or is just invalid?
    # OR:
    # If the input is a pure token, return it.
    # If the input is a group token, return... it? or -1?
    # "is_all_pure_tokens" checks if tokens_to_pure_map[ids] == ids.
    # So for group tokens, tokens_to_pure_map[g] should NOT be g.
    # Maybe it maps to a canonical pure token (e.g. the centroid)?
    # Or maybe it just maps to -1?
    # Let's look at is_all_pure_tokens implementation:
    # return torch.all(torch.eq(pure_ids, ids)).item()
    # If I pass a group token G, and map[G] != G, then it returns False. Correct.
    # So we can map group tokens to -1 or 0 or anything distinct.
    
    new_total_vocab = new_enc.n_vocab
    tokens_to_pure_map = torch.arange(new_total_vocab, dtype=torch.long)
    
    # We want tokens_to_pure_map[group_id] != group_id.
    # Let's set them to -1 (or 0 if unsigned, but long is signed).
    # Actually, to be safe, let's map them to 0 (usually <|bos|>) or -1.
    # Let's use -1.
    
    # Initialize all new tokens (groups) to -1
    # The first 'vocab_size' tokens are pure (mostly), except original special tokens?
    # Original special tokens are "pure" in the sense they aren't our noisy groups.
    # So we only mark our NEW group tokens as "not pure".
    
    for t_str in new_tokens_list:
        tid = final_special_tokens[t_str]
        tokens_to_pure_map[tid] = -1

    # noisy_level_map
    # Maps token ID -> level index (0..D+1)
    noisy_level_map = torch.zeros(new_total_vocab, dtype=torch.long)
    
    # Default is 0 (Pure)
    # Set levels for group tokens
    for path, tid in path_to_id.items():
        # What level is this path?
        # path length L corresponds to level...
        # In our loop above:
        # level 1 has length D
        # level D+1 has length 0
        # So: level = D + 1 - len(path)
        level = D + 1 - len(path)
        noisy_level_map[tid] = level

    # 6. Save Everything
    print("Saving tokenizer and maps...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save Tokenizer
    pickle_path = os.path.join(output_dir, "tokenizer.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(new_enc, f)
    
    # Save Maps
    maps = {
        "pure_to_noisy_map": pure_to_noisy_map,
        "tokens_to_pure_map": tokens_to_pure_map,
        "noisy_level_map": noisy_level_map
    }
    map_path = os.path.join(output_dir, "token_maps.pt")
    torch.save(maps, map_path)
    
    print(f"Saved tokenizer to {pickle_path}")
    print(f"Saved token maps to {map_path}")
    print(f"New vocab size: {new_enc.n_vocab}")
    
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
