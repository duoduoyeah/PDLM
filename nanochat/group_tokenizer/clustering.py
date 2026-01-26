"""
Clustering methods for grouping pure tokens and building assignment matrices.
"""
import torch
from itertools import combinations
from typing import Tuple


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


def build_all_combos_assignment(num_sub: int, sub_per_final: int) -> torch.Tensor:
    """
    Generate all C(num_sub, sub_per_final) combinations as assignment matrix.

    Args:
        num_sub: number of sub-groups
        sub_per_final: sub-groups per final group

    Returns:
        assignment: (num_final, num_sub) boolean tensor
                   assignment[f, s] = True means sub-group s is in final group f
    """
    combos = list(combinations(range(num_sub), sub_per_final))
    num_final = len(combos)

    assignment = torch.zeros(num_final, num_sub, dtype=torch.bool)
    for f, combo in enumerate(combos):
        for s in combo:
            assignment[f, s] = True

    return assignment


def build_flexible_assignment(
    num_sub: int,
    sub_per_final: int,
    overlap_k: int,
    seed: int = 42,
) -> torch.Tensor:
    """
    Build assignment matrix using greedy balanced design.
    Each row has exactly sub_per_final True values.
    Each column has exactly overlap_k True values.

    Args:
        num_sub: number of sub-groups (columns)
        sub_per_final: sub-groups per final group (True values per row)
        overlap_k: final groups per sub-group (True values per column)
        seed: random seed for tie-breaking

    Returns:
        assignment: (num_final, num_sub) boolean tensor
    """
    torch.manual_seed(seed)

    num_final = (num_sub * overlap_k) // sub_per_final
    assignment = torch.zeros(num_final, num_sub, dtype=torch.bool)
    col_counts = torch.zeros(num_sub, dtype=torch.long)  # Track appearances per sub-group

    for f in range(num_final):
        # Find sub-groups with room (col_counts < overlap_k)
        available_mask = col_counts < overlap_k
        available_indices = torch.where(available_mask)[0]

        # Get counts for available sub-groups
        available_counts = col_counts[available_indices]

        # Find the minimum count among available
        min_count = available_counts.min()

        # Get indices with minimum count (for tie-breaking)
        min_mask = available_counts == min_count
        min_indices = available_indices[min_mask]

        # If we have enough at minimum count, sample from them
        if len(min_indices) >= sub_per_final:
            # Random permutation for tie-breaking
            perm = torch.randperm(len(min_indices))
            selected = min_indices[perm[:sub_per_final]]
        else:
            # Need to take all at min count and some from next level
            selected_list = min_indices.tolist()
            remaining = sub_per_final - len(min_indices)

            # Get next level candidates
            other_mask = available_mask & (col_counts > min_count)
            other_indices = torch.where(other_mask)[0]
            other_counts = col_counts[other_indices]

            # Sort by count (ascending) and take remaining
            sorted_idx = torch.argsort(other_counts)
            for i in range(remaining):
                selected_list.append(other_indices[sorted_idx[i]].item())

            selected = torch.tensor(selected_list, dtype=torch.long)

        # Update assignment matrix
        assignment[f, selected] = True
        col_counts[selected] += 1

    # Verify constraints
    row_sums = assignment.sum(dim=1)
    col_sums = assignment.sum(dim=0)
    assert (row_sums == sub_per_final).all(), f"Row constraint violated: {row_sums}"
    assert (col_sums == overlap_k).all(), f"Column constraint violated: {col_sums}"

    return assignment


def derive_pure_to_group(
    sub_assignments: torch.Tensor,
    sub_to_final: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert sub-group assignments + assignment matrix to final output tensors.

    Args:
        sub_assignments: (vocab_size,) - which sub-group each token belongs to
        sub_to_final: (num_final, num_sub) - boolean matrix of which sub-groups in each final group

    Returns:
        pure_to_group: (vocab_size, overlap_k) - final group IDs for each token
        group_to_pure_mask: (num_final, vocab_size) - boolean mask of member tokens per group
    """
    vocab_size = sub_assignments.shape[0]
    num_final, num_sub = sub_to_final.shape

    # Each sub-group appears in overlap_k final groups
    overlap_k = sub_to_final.sum(dim=0)[0].item()  # Should be same for all columns

    # For each token, find which final groups it belongs to
    # Token t is in sub-group s, and s appears in final groups where sub_to_final[:, s] is True
    pure_to_group = torch.zeros(vocab_size, overlap_k, dtype=torch.long)

    for t in range(vocab_size):
        s = sub_assignments[t].item()
        final_groups = torch.where(sub_to_final[:, s])[0]
        pure_to_group[t] = final_groups

    # Build group_to_pure_mask: (num_final, vocab_size)
    # Token t is in final group f if t's sub-group s is in f
    group_to_pure_mask = torch.zeros(num_final, vocab_size, dtype=torch.bool)

    for f in range(num_final):
        # Which sub-groups are in this final group?
        sub_groups_in_f = torch.where(sub_to_final[f])[0]
        # Which tokens are in those sub-groups?
        for s in sub_groups_in_f:
            tokens_in_s = sub_assignments == s
            group_to_pure_mask[f] |= tokens_in_s

    return pure_to_group, group_to_pure_mask
