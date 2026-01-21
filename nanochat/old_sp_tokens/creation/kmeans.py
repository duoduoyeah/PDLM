import torch
import torch.nn.functional as F


@torch.no_grad()
def kmeans(X, k, iters=30, restarts=10):
    """
    Performs K-Means clustering using cosine distance.
    X: (N, D) float tensor
    k: number of clusters
    returns: labels (N,), centroids (k, D)
    """
    # To use cosine distance, we normalize all vectors to unit length.
    # K-means on L2-normalized vectors is equivalent to spherical K-means.
    X = F.normalize(X, p=2, dim=1)

    N, D = X.shape
    best_inertia = float("inf")
    best_labels = None
    best_C = None

    for _ in range(restarts):
        # init: pick k random points as centroids. They are already normalized as they come from X.
        idx = torch.randperm(N)[:k]
        C = X[idx].clone()  # (k, D)

        for _ in range(iters):
            # Using Euclidean distance on normalized vectors is equivalent to maximizing cosine similarity.
            # dist = 2 - 2 * cos_sim. Minimizing dist is maximizing cos_sim.
            d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(dim=2)
            labels = d2.argmin(dim=1)  # (N,)

            # recompute centroids; guard against empty cluster
            newC = C.clone()
            for i in range(k):
                mask = labels == i
                if mask.any():
                    # The new centroid is the mean of the points, re-normalized to unit length.
                    newC[i] = F.normalize(X[mask].mean(dim=0), p=2, dim=0)
                else:
                    # re-seed empty cluster with a random normalized point
                    newC[i] = X[torch.randint(0, N, (1,))]

            # stop if converged
            if torch.allclose(newC, C, atol=1e-6):
                C = newC
                break
            C = newC

        # inertia (SSE): sum of squared distances to assigned centroid
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(dim=2)
        inertia = d2[torch.arange(N, device=X.device), labels].sum().item()

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.clone()
            best_C = C.clone()

    return best_labels, best_C

def split_into_n_groups(X, k, iters=40, restarts=20):
    """
    Splits the tensor X into n groups using k-means.
    Returns a list of index tensors, one for each group.
    """
    labels, C = kmeans(X, k, iters=iters, restarts=restarts)
    groups = []
    for i in range(k):
        groups.append((labels == i).nonzero(as_tuple=True)[0])
    return groups, C  # list of indices for the n groups + centroids

if __name__ == "__main__":
    # --- Device Setup ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_default_device("cuda") # Set default device for newly created tensors
        print("CUDA is available. Setting default device to 'cuda'.")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using 'cpu'.")

    # --- Setup ---
    X = torch.randn(4096, 768)
    k = 4
    hierarchy_levels = [1, 4, 16, 64, 256, 1024]
    max_depth = len(hierarchy_levels) - 1

    print(f"Building a {k}-way hierarchical clustering for data of shape {X.shape} on {X.device}")
    print(f"Hierarchy levels (num clusters): {hierarchy_levels}")
    print("-" * 40)

    # --- Hierarchical Clustering & Ancestry Tracking ---
    # This matrix will store the path for each token through the hierarchy.
    # Shape: (num_tokens, num_levels_of_splits).
    # ancestry[i, d] will be the cluster index (0..k-1) for token i at depth d.
    ancestry = torch.full((X.shape[0], max_depth), -1, dtype=torch.long)

    # Start with level 0, which is the entire dataset
    current_groups = [(X, torch.arange(X.shape[0]))]

    for depth in range(max_depth):
        print(f"Processing Level {depth} -> {depth+1} (Target: {hierarchy_levels[depth+1]} clusters)")
        
        next_level_groups = []
        for data_subset, original_indices in current_groups:
            # If a group is too small to be split further, we stop.
            if len(data_subset) <= k:
                # Assign all items in this small leaf group to a default path '0' for subsequent levels
                ancestry[original_indices, depth:] = 0
                next_level_groups.append((data_subset, original_indices))
                continue

            # This is the core of the ancestry tracking.
            # We get the labels (0..k-1) for the items *within* the current subset.
            labels, _ = kmeans(data_subset, k)

            # We then assign these local labels to the correct tokens and depth in our global ancestry matrix.
            ancestry[original_indices, depth] = labels

            # Re-create the groups for the next level's loop, just as before
            for i in range(k):
                local_mask = (labels == i)
                if local_mask.sum() == 0:
                    continue
                
                local_indices = local_mask.nonzero(as_tuple=True)[0]
                new_global_indices = original_indices[local_indices]
                new_data_subset = X[new_global_indices]
                
                next_level_groups.append((new_data_subset, new_global_indices))
        
        current_groups = next_level_groups
        print(f"  > Completed Level {depth+1}, found {len(current_groups)} clusters.")

    # --- Summary & Example Usage ---
    print("-" * 40)
    print("Ancestry tracking complete.")
    
    # Example: Show the ancestry for the first 5 tokens
    print("\nAncestry path for first 5 tokens:")
    print("Each path shows the sequence of cluster choices (0-3) at each split.")
    for i in range(5):
        path = ancestry[i].tolist()
        print(f"Token {i}: Path = {path}")

    # Example: Reconstruct a token's final cluster ID from its path
    token_idx = 123
    token_path = ancestry[token_idx].tolist()
    
    # The path [c0, c1, c2, c3, c4] can be converted to a unique final ID
    # This is like converting a base-4 number to base-10
    global_cluster_id = 0
    for i, path_val in enumerate(token_path):
        global_cluster_id += path_val * (k ** (max_depth - 1 - i))

    print(f"\nExample: Token {token_idx} with path {token_path} belongs to final global cluster ID: {global_cluster_id}")
    print(f"(This final ID will be a number between 0 and {k**max_depth - 1})")

