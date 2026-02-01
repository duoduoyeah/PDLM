"""
Learnable group assignment matrix for PDLM.

The only trainable component: a (V_pure, G) matrix that maps pure token logits
to group-level predictions. Trained with BCE loss against soft membership targets,
with regularization penalties for group size, overlap, and sharpening.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AssignmentMatrix(nn.Module):
    """
    Trainable assignment matrix A of shape (pure_vocab_size, num_groups).

    sigmoid(A) gives soft group membership: how strongly each pure token
    belongs to each group. During training, everything is continuous.
    After training, binarize() converts to a TokenMap-compatible dict.

    Args:
        pure_vocab_size: Number of pure tokens (V).
        num_groups: Number of groups (G).
        max_size_soft: Soft upper bound on group size (column sum of soft_assign).
        min_overlap_soft: Soft lower bound on token overlap (row sum of soft_assign).
    """

    def __init__(self, pure_vocab_size, num_groups, max_size_soft, min_overlap_soft):
        super().__init__()
        self.pure_vocab_size = pure_vocab_size
        self.num_groups = num_groups
        self.max_size_soft = max_size_soft
        self.min_overlap_soft = min_overlap_soft

        # The only trainable parameter
        self.A = nn.Parameter(torch.randn(pure_vocab_size, num_groups))

    def get_soft_assign(self):
        """Returns soft assignment matrix: sigmoid(A), shape (V, G)."""
        return torch.sigmoid(self.A)

    def forward(self, pure_logits, pure_targets, loss_mask, lambda_noise, lambda_overlap, sharp_weight):
        """
        Compute all losses for the assignment matrix.

        Args:
            pure_logits: (B, T, V) logits from frozen model (softcap already applied).
            pure_targets: (B, T) target pure token ids.
            loss_mask: (B, T) bool mask (True = include in loss).
            lambda_noise: Weight for group size penalty.
            lambda_overlap: Weight for overlap penalty.
            sharp_weight: Weight for sharpening penalty (annealed).

        Returns:
            total_loss: Scalar loss for backward().
            loss_dict: Dict of individual loss components for logging.
        """
        soft_assign = self.get_soft_assign()  # (V, G)

        # Group logits: project pure logits to group space
        group_logits = pure_logits @ soft_assign  # (B, T, G)

        # Target membership: soft assignment values for the target tokens
        target_membership = soft_assign[pure_targets]  # (B, T, G)

        # Task loss: BCE with logits on group predictions vs soft membership
        # group_logits are raw (not sigmoided), target_membership is in [0, 1]
        bce_unreduced = F.binary_cross_entropy_with_logits(
            group_logits, target_membership, reduction="none"
        )  # (B, T, G)

        # Average over groups, then mask
        bce_per_pos = bce_unreduced.mean(dim=-1)  # (B, T)
        mask_float = loss_mask.float()
        num_valid = mask_float.sum().clamp(min=1.0)
        loss_task = (bce_per_pos * mask_float).sum() / num_valid

        # Noise penalty: discourage groups from being too large
        col_sums = soft_assign.sum(dim=0)  # (G,)
        loss_noise = F.relu(col_sums - self.max_size_soft).mean()

        # Overlap penalty: encourage tokens to belong to multiple groups
        row_sums = soft_assign.sum(dim=1)  # (V,)
        loss_overlap = F.relu(self.min_overlap_soft - row_sums).mean()

        # Sharpening penalty: push soft assignments toward 0 or 1
        loss_sharp = (soft_assign * (1 - soft_assign)).mean()

        # Total loss
        total_loss = (
            loss_task
            + lambda_noise * loss_noise
            + lambda_overlap * loss_overlap
            + sharp_weight * loss_sharp
        )

        loss_dict = {
            "loss_total": total_loss.item(),
            "loss_task": loss_task.item(),
            "loss_noise": loss_noise.item(),
            "loss_overlap": loss_overlap.item(),
            "loss_sharp": loss_sharp.item(),
            "sharp_weight": sharp_weight,
            "col_sum_mean": col_sums.mean().item(),
            "col_sum_max": col_sums.max().item(),
            "row_sum_mean": row_sums.mean().item(),
            "row_sum_min": row_sums.min().item(),
        }

        return total_loss, loss_dict

    @torch.no_grad()
    def binarize(self, threshold=0.5):
        """
        Convert soft assignment to binary and produce a TokenMap-compatible dict.

        Args:
            threshold: Binarization threshold for soft assignments.

        Returns:
            Dict with keys: pure_to_group, group_to_pure_mask, pure_vocab_size,
            num_groups, overlap_k, mask_token_id. Compatible with TokenMap.__init__.
        """
        soft = self.get_soft_assign()  # (V, G)
        binary = (soft >= threshold).bool()  # (V, G)

        # Stats
        row_sums = binary.sum(dim=1)  # how many groups per token
        col_sums = binary.sum(dim=0)  # how many tokens per group

        # Handle tokens with no group assignment (below threshold everywhere)
        orphan_mask = row_sums == 0
        num_orphans = orphan_mask.sum().item()
        if num_orphans > 0:
            # Assign orphans to their highest-scoring group
            orphan_groups = soft[orphan_mask].argmax(dim=-1)
            binary[orphan_mask, :] = False
            binary[orphan_mask, orphan_groups] = True
            row_sums = binary.sum(dim=1)

        # Build pure_to_group: (V, overlap_k) where overlap_k = max groups per token
        overlap_k = row_sums.max().item()
        pure_to_group = torch.full(
            (self.pure_vocab_size, overlap_k), -1, dtype=torch.long
        )
        for v in range(self.pure_vocab_size):
            groups = binary[v].nonzero(as_tuple=True)[0]
            pure_to_group[v, : len(groups)] = groups

        # Build group_to_pure_mask: (G, V) bool
        group_to_pure_mask = binary.T.contiguous()  # (G, V)

        # Report stats
        col_sums = binary.sum(dim=0)
        print(f"Binarize stats (threshold={threshold}):")
        print(f"  orphans rescued: {num_orphans}")
        print(f"  overlap_k: {overlap_k}")
        print(f"  row_sums (groups/token): min={row_sums.min().item()}, "
              f"mean={row_sums.float().mean().item():.1f}, max={row_sums.max().item()}")
        print(f"  col_sums (tokens/group): min={col_sums.min().item()}, "
              f"mean={col_sums.float().mean().item():.1f}, max={col_sums.max().item()}")

        # Ambiguous entries: soft values close to threshold
        ambiguous = ((soft - threshold).abs() < 0.1).sum().item()
        total = soft.numel()
        print(f"  ambiguous entries (|s - {threshold}| < 0.1): {ambiguous}/{total} "
              f"({ambiguous/total*100:.1f}%)")

        return {
            "pure_to_group": pure_to_group,
            "group_to_pure_mask": group_to_pure_mask,
            "pure_vocab_size": self.pure_vocab_size,
            "num_groups": self.num_groups,
            "overlap_k": overlap_k,
            "mask_token_id": -1,
        }
