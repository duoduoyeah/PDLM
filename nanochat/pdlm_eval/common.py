"""Shared utilities for PDLM evaluation."""

import torch
from contextlib import contextmanager


@contextmanager
def model_eval_context(model):
    """Context manager that sets model to eval mode and restores after."""
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        if was_training:
            model.train()


def build_result_dict(metrics_by_pos, block_size, include_accuracy=False):
    """Build standard result dict from per-position metrics."""
    total_nll = sum(metrics_by_pos[p]["nll"] for p in range(block_size))
    total_tokens = sum(metrics_by_pos[p]["tokens"] for p in range(block_size))

    result = {
        "overall_loss": total_nll / total_tokens if total_tokens > 0 else 0.0,
        "overall_ppl": torch.exp(torch.tensor(total_nll / total_tokens if total_tokens > 0 else 0.0)).item(),
        "num_tokens_evaluated": total_tokens,
        "positions": {},
    }

    if include_accuracy:
        total_correct = sum(metrics_by_pos[p]["correct"] for p in range(block_size))
        result["overall_accuracy"] = total_correct / total_tokens if total_tokens > 0 else 0.0

    for pos in range(block_size):
        result["positions"][pos] = build_position_result(metrics_by_pos[pos], include_accuracy)

    return result


def build_position_result(pos_metrics, include_accuracy=False):
    """Build result dict for a single position."""
    nll = pos_metrics["nll"]
    tokens = pos_metrics["tokens"]
    loss = nll / tokens if tokens > 0 else 0.0

    result = {
        "loss": loss,
        "ppl": torch.exp(torch.tensor(loss)).item(),
        "tokens": tokens,
    }

    if include_accuracy:
        correct = pos_metrics["correct"]
        result["accuracy"] = correct / tokens if tokens > 0 else 0.0

    return result
