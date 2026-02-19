"""
GPT Evaluation Module.

Computes loss, perplexity, entropy-based perplexity, and accuracy metrics
for autoregressive GPT models (both standard next-token and next-k prediction).

Metrics:
- loss: average negative log-likelihood per token
- ppl: true perplexity = exp(loss)
- entropy_ppl: exp(average entropy of predicted distributions)
- accuracy: fraction of tokens where argmax prediction matches target
"""

import torch
import torch.nn.functional as F


def eval_gpt(
    model,
    val_loader,
    num_batches,
    device,
    autocast_ctx,
):
    """
    Evaluate GPT model on validation set.

    Args:
        model: GPT model
        val_loader: validation data loader yielding (inputs, targets, loss_extras, state_dict)
                    inputs: (B, T) token ids
                    targets: (B, T) target token ids (shifted by target_shift)
        num_batches: number of batches to evaluate
        device: device to run on
        autocast_ctx: autocast context for mixed precision

    Returns:
        dict with evaluation results:
        {
            "overall_loss": float,
            "overall_ppl": float,
            "overall_entropy_ppl": float,
            "overall_accuracy": float,
            "num_tokens_evaluated": int,
        }
    """
    was_training = model.training
    model.eval()

    total_nll = 0.0
    total_entropy = 0.0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for batch_idx in range(num_batches):
            inputs, targets, loss_extras, state = next(val_loader)
            # inputs: (B, T), targets: (B, T)

            B, T = inputs.shape

            with autocast_ctx:
                # Forward without targets to get logits
                logits = model(inputs)  # (B, T, vocab_size)

                # Compute log probabilities
                log_probs = F.log_softmax(logits.float(), dim=-1)  # (B, T, V)

                # NLL: gather log prob of each target token
                target_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B, T)
                nll = -target_log_probs  # (B, T)

                # Entropy: -sum(p * log_p) over vocab dimension
                probs = log_probs.exp()
                entropy = -(probs * log_probs).sum(dim=-1)  # (B, T)

                # Accuracy: argmax prediction matches target
                preds = logits.argmax(dim=-1)  # (B, T)
                correct = (preds == targets)  # (B, T)

            total_nll += nll.sum().item()
            total_entropy += entropy.sum().item()
            total_correct += correct.sum().item()
            total_tokens += B * T

    if was_training:
        model.train()

    # Compute final metrics
    avg_loss = total_nll / total_tokens if total_tokens > 0 else 0.0
    avg_entropy = total_entropy / total_tokens if total_tokens > 0 else 0.0

    return {
        "overall_loss": avg_loss,
        "overall_ppl": torch.exp(torch.tensor(avg_loss)).item(),
        "overall_entropy_ppl": torch.exp(torch.tensor(avg_entropy)).item(),
        "overall_accuracy": total_correct / total_tokens if total_tokens > 0 else 0.0,
        "num_tokens_evaluated": total_tokens,
    }
