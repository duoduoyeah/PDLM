"""
Clustering methods for grouping pure tokens.
"""
import torch


def kmeans_clustering(
    embeddings: torch.Tensor,
    num_groups: int,
    max_iters: int = 100,
    seed: int = 42,
) -> torch.Tensor:
    """
    K-means clustering on embeddings (cosine similarity).

    Args:
        embeddings: (vocab_size, dim) tensor, typically lm_head.weight
        num_groups: number of clusters
        max_iters: max iterations
        seed: random seed for init

    Returns:
        assignments: (vocab_size,) tensor of group indices [0, num_groups)
    """
    torch.manual_seed(seed)
    vocab_size, dim = embeddings.shape

    # L2 normalize for cosine similarity
    X = torch.nn.functional.normalize(embeddings.float(), dim=1)

    # Random init: pick num_groups random points as centroids
    perm = torch.randperm(vocab_size)[:num_groups]
    centroids = X[perm].clone()  # (num_groups, dim)

    assignments = torch.zeros(vocab_size, dtype=torch.long, device=X.device)

    for _ in range(max_iters):
        # Assign each point to nearest centroid (cosine = dot product after L2 norm)
        sims = X @ centroids.T  # (vocab_size, num_groups)
        new_assignments = sims.argmax(dim=1)

        # Check convergence
        if torch.equal(assignments, new_assignments):
            break
        assignments = new_assignments

        # Update centroids
        for g in range(num_groups):
            mask = assignments == g
            if mask.sum() > 0:
                centroids[g] = X[mask].mean(dim=0)
                centroids[g] = torch.nn.functional.normalize(centroids[g], dim=0)

    return assignments


def random_clustering(
    vocab_size: int,
    num_groups: int,
    seed: int = 42,
) -> torch.Tensor:
    """
    Random assignment baseline.

    Returns:
        assignments: (vocab_size,) tensor of group indices [0, num_groups)
    """
    torch.manual_seed(seed)
    return torch.randint(0, num_groups, (vocab_size,))


def compute_overlap_assignments(
    embeddings: torch.Tensor,
    base_assignments: torch.Tensor,
    num_groups: int,
    overlap_k: int,
) -> torch.Tensor:
    """
    Extend hard assignments to overlap_k assignments per token.
    Each token gets assigned to its k nearest group centroids.

    Args:
        embeddings: (vocab_size, dim)
        base_assignments: (vocab_size,) from kmeans
        num_groups: number of groups
        overlap_k: number of groups per token

    Returns:
        assignments: (vocab_size, overlap_k) tensor of group indices
    """
    if overlap_k == 1:
        return base_assignments.unsqueeze(1)

    vocab_size, dim = embeddings.shape
    X = torch.nn.functional.normalize(embeddings.float(), dim=1)

    # Compute centroids from base assignments
    centroids = torch.zeros(num_groups, dim, device=X.device)
    for g in range(num_groups):
        mask = base_assignments == g
        if mask.sum() > 0:
            centroids[g] = X[mask].mean(dim=0)
            centroids[g] = torch.nn.functional.normalize(centroids[g], dim=0)

    # For each token, find top-k nearest centroids
    sims = X @ centroids.T  # (vocab_size, num_groups)
    _, topk_groups = torch.topk(sims, k=overlap_k, dim=1)  # (vocab_size, overlap_k)

    return topk_groups
