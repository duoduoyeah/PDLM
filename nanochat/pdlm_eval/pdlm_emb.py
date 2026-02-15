"""
PDLM Embedding evaluation.

Unified loss (no stage1/stage2 split) — same structure as mask_pdlm eval
but uses averaged noise embeddings instead of discrete group/mask tokens.
"""

import torch
import torch.nn.functional as F

from nanochat.pdlm_eval.common import model_eval_context, build_result_dict


def eval_pdlm_emb(
    model,
    val_loader,
    block_size,
    num_batches,
    attn_mask,
    device,
    autocast_ctx,
    prefix_pure_tokens=0,
):
    """
    Evaluate pdlm_emb model.

    Returns a dict with per-position loss/ppl breakdown.
    """
    nll_by_pos = {p: {"nll": 0.0, "entropy": 0.0, "set_recall": 0.0, "tokens": 0} for p in range(block_size)}

    with model_eval_context(model):
        with torch.no_grad():
            for _ in range(num_batches):
                inputs, targets, loss_extras, _ = next(val_loader)
                B, T = inputs.shape
                noise_tokens = loss_extras["noise_tokens"]  # (B, block_region_len, noise_count)

                with autocast_ctx:
                    logits = model.forward_for_eval_pdlm_emb(
                        inputs, targets, attn_mask=attn_mask,
                        noise_tokens=noise_tokens,
                    )

                    log_probs = F.log_softmax(logits.float(), dim=-1)
                    target_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)  # (B, T)
                    probs = log_probs.exp()  # (B, T, vocab_size)

                    num_blocks = T // block_size

                    for pos in range(block_size):
                        for block_idx in range(1, num_blocks):  # skip block 0
                            pos_in_seq = block_idx * block_size + pos
                            nll = -target_log_probs[:, pos_in_seq]  # (B,)
                            nll_by_pos[pos]["nll"] += nll.sum().item()
                            nll_by_pos[pos]["entropy"] += entropy[:, pos_in_seq].sum().item()

                            # Set recall: sum of probs on noise set tokens
                            noise_idx = (block_idx - 1) * block_size + pos
                            noise_tok_ids = noise_tokens[:, noise_idx, :]  # (B, noise_count)
                            pos_probs = probs[:, pos_in_seq, :]  # (B, vocab_size)
                            noise_probs = pos_probs.gather(-1, noise_tok_ids)  # (B, noise_count)
                            set_recall = noise_probs.sum(dim=-1)  # (B,)
                            nll_by_pos[pos]["set_recall"] += set_recall.sum().item()

                            nll_by_pos[pos]["tokens"] += B

    return build_result_dict(nll_by_pos, block_size, include_accuracy=False)
