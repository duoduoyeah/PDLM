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
        self.A = nn.Parameter(torch.randn(pure_vocab_size, num_groups) * 0.01 + 0.5)

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
    def binarize(self, threshold=0.5, topk=0):
        """
        Convert soft assignment to binary and produce a TokenMap-compatible dict.

        Args:
            threshold: Binarization threshold for soft assignments (used when topk=0).
            topk: If > 0, each token is assigned to its top-k groups by soft value,
                  ignoring the threshold. This guarantees overlap_k = topk.

        Returns:
            Dict with keys: pure_to_group, group_to_pure_mask, pure_vocab_size,
            num_groups, overlap_k, mask_token_id. Compatible with TokenMap.__init__.
        """
        soft = self.get_soft_assign()  # (V, G)

        if topk > 0:
            # Top-k mode: each token gets assigned to its top-k groups
            _, top_indices = soft.topk(topk, dim=1)  # (V, topk)
            binary = torch.zeros_like(soft, dtype=torch.bool)
            binary.scatter_(1, top_indices, True)
            num_orphans = 0
            print(f"Binarize mode: top-k (k={topk})")
        else:
            binary = (soft >= threshold).bool()  # (V, G)

            # Handle tokens with no group assignment (below threshold everywhere)
            row_sums = binary.sum(dim=1)
            orphan_mask = row_sums == 0
            num_orphans = orphan_mask.sum().item()
            if num_orphans > 0:
                # Assign orphans to their highest-scoring group
                orphan_groups = soft[orphan_mask].argmax(dim=-1)
                binary[orphan_mask, :] = False
                binary[orphan_mask, orphan_groups] = True
            print(f"Binarize mode: threshold ({threshold})")

        # Stats
        row_sums = binary.sum(dim=1)
        col_sums = binary.sum(dim=0)

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
        q = torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0], device=soft.device)
        row_pct = torch.quantile(row_sums.float(), q)
        col_pct = torch.quantile(col_sums.float(), q)
        mode_str = f"topk={topk}" if topk > 0 else f"threshold={threshold}"
        print(f"Binarize stats ({mode_str}):")
        print(f"  orphans rescued: {num_orphans}")
        print(f"  overlap_k: {overlap_k}")
        print(f"  row_sums (groups/token): "
              f"p0={row_pct[0].item():.0f}, p10={row_pct[1].item():.0f}, "
              f"p25={row_pct[2].item():.0f}, p50={row_pct[3].item():.0f}, "
              f"p75={row_pct[4].item():.0f}, p90={row_pct[5].item():.0f}, "
              f"p100={row_pct[6].item():.0f}")
        print(f"  col_sums (tokens/group): "
              f"p0={col_pct[0].item():.0f}, p10={col_pct[1].item():.0f}, "
              f"p25={col_pct[2].item():.0f}, p50={col_pct[3].item():.0f}, "
              f"p75={col_pct[4].item():.0f}, p90={col_pct[5].item():.0f}, "
              f"p100={col_pct[6].item():.0f}")

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
