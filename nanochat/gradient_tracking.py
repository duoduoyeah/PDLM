"""
Gradient tracking utility for per-position gradient diagnostics.

Decomposes training loss by block position, performs per-position backward passes,
and computes gradient norms and pairwise cosine similarities. Works for both
PDLM stage1_block (real block positions) and GPT (virtual index % block_size positions).
"""

import torch
import torch.nn.functional as F


def _get_param_groups(model):
    """Extract named parameter groups from model for per-layer gradient tracking.

    Works for both GPT and PDLM since they share the same structure:
    model.transformer.wte, model.transformer.h[i], model.lm_head
    """
    groups = {}
    groups["wte"] = list(model.transformer.wte.parameters())
    for i, block in enumerate(model.transformer.h):
        groups[f"block_{i}"] = list(block.parameters())
    groups["lm_head"] = list(model.lm_head.parameters())
    return groups


def compute_gradient_metrics(model, per_token_nll, loss_mask, block_size, prefix_sliding_tokens=0):
    """Compute per-position gradient metrics for wandb logging.

    Args:
        model: nn.Module (PDLM or GPT)
        per_token_nll: (B, T) per-token negative log-likelihood (must be in computation graph)
        loss_mask: (B, T) bool — which tokens contribute to loss
        block_size: K — number of positions per block
        prefix_sliding_tokens: offset for block position computation

    Returns:
        dict of metrics ready for wandb logging
    """
    B, T = per_token_nll.shape
    device = per_token_nll.device
    K = block_size

    # Build position masks: for each k in 0..K-1
    pos_indices = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
    block_pos = (pos_indices - prefix_sliding_tokens) % K  # (1, T)

    pos_masks = []
    pos_losses = []
    for k in range(K):
        mask_k = loss_mask & (block_pos == k)  # (B, T)
        pos_masks.append(mask_k)
        count_k = mask_k.sum().clamp(min=1)
        loss_k = (per_token_nll * mask_k).sum() / count_k
        pos_losses.append(loss_k)

    # Get parameter groups for per-layer norms
    param_groups = _get_param_groups(model)
    all_params = [p for group in param_groups.values() for p in group]

    # For each position: backward, snapshot gradients
    grad_vectors = []  # list of K flat gradient vectors
    per_layer_norms = []  # list of K dicts {layer_name: norm}
    total_norms = []

    for k in range(K):
        model.zero_grad()
        pos_losses[k].backward(retain_graph=True)

        # Snapshot: flatten all grads into one vector for cosine sim
        flat_grads = []
        for p in all_params:
            if p.grad is not None:
                flat_grads.append(p.grad.detach().float().flatten())
            else:
                flat_grads.append(torch.zeros(p.numel(), device=device))
        grad_vec = torch.cat(flat_grads)
        grad_vectors.append(grad_vec)

        # Total grad norm
        total_norm = grad_vec.norm().item()
        total_norms.append(total_norm)

        # Per-layer grad norms
        layer_norms = {}
        for name, params in param_groups.items():
            layer_grad_parts = []
            for p in params:
                if p.grad is not None:
                    layer_grad_parts.append(p.grad.detach().float().flatten())
                else:
                    layer_grad_parts.append(torch.zeros(p.numel(), device=device))
            layer_vec = torch.cat(layer_grad_parts)
            layer_norms[name] = layer_vec.norm().item()
        per_layer_norms.append(layer_norms)

    # Clean up gradients before actual training backward
    model.zero_grad()

    # Build metrics dict
    metrics = {}

    # Per-position losses and grad norms
    total_grad_mass = sum(total_norms)
    for k in range(K):
        metrics[f"gradient/pos_{k}_loss"] = pos_losses[k].item()
        metrics[f"gradient/pos_{k}_grad_norm"] = total_norms[k]
        metrics[f"gradient/pos_{k}_grad_pct"] = (
            100.0 * total_norms[k] / total_grad_mass if total_grad_mass > 0 else 0.0
        )

    # Per-layer per-position grad norms
    for k in range(K):
        for layer_name, norm_val in per_layer_norms[k].items():
            metrics[f"gradient/layer_{layer_name}_pos_{k}_grad_norm"] = norm_val

    # Pairwise cosine similarity
    for i in range(K):
        for j in range(i + 1, K):
            cos_sim = F.cosine_similarity(
                grad_vectors[i].unsqueeze(0),
                grad_vectors[j].unsqueeze(0),
            ).item()
            metrics[f"gradient/cos_pos{i}_pos{j}"] = cos_sim

    return metrics
