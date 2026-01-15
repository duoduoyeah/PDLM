#!/usr/bin/env python3
"""
Dump BD3LM loss computation to verify the implementation.

This script tests:
1. Basic NLL computation with simple logits
2. Attention mask: only compute loss where mask == 1
3. Loss scale weighting
4. Batch size > 1

Output goes to ./temp/bd3lm_loss_dump.txt
"""
import os
import torch
import torch.nn.functional as F

from nanochat.bd3lm_utils.bd3lm_loss import (
    compute_token_nll,
    apply_loss_scale,
    aggregate_loss,
    compute_bd3lm_loss,
)


def dump_tensor(name: str, t: torch.Tensor) -> str:
    """Format a tensor for display."""
    if t.dim() == 0:
        return f"{name}: {t.item():.6f}"
    return f"{name}: shape={list(t.shape)}\n{t}"


def test_basic_nll(f):
    """Test 1: Basic NLL computation with simple logits."""
    f.write("=" * 80 + "\n")
    f.write("TEST 1: Basic NLL computation\n")
    f.write("=" * 80 + "\n\n")

    # Simple case: B=1, L=2, vocab=3
    # logits such that softmax gives easy-to-verify probs
    logits = torch.tensor([
        [[1.0, 0.0, 0.0],   # pos 0: softmax -> [0.576, 0.212, 0.212]
         [0.0, 2.0, 0.0]],  # pos 1: softmax -> [0.106, 0.788, 0.106]
    ])  # shape (1, 2, 3)

    targets = torch.tensor([[0, 1]])  # shape (1, 2)

    f.write("Input:\n")
    f.write(dump_tensor("  logits", logits) + "\n")
    f.write(dump_tensor("  targets", targets) + "\n\n")

    # Compute softmax manually for verification
    probs = F.softmax(logits, dim=-1)
    f.write("Softmax probabilities:\n")
    f.write(dump_tensor("  probs", probs) + "\n\n")

    # Compute NLL
    nll = compute_token_nll(logits, targets)
    f.write("NLL (computed):\n")
    f.write(dump_tensor("  nll", nll) + "\n\n")

    # Manual verification
    # NLL[0,0] = -log(probs[0,0,0]) = -log(0.576...)
    # NLL[0,1] = -log(probs[0,1,1]) = -log(0.788...)
    manual_nll_0 = -torch.log(probs[0, 0, targets[0, 0]])
    manual_nll_1 = -torch.log(probs[0, 1, targets[0, 1]])
    f.write("Manual verification:\n")
    f.write(f"  NLL[0,0] = -log(probs[0,0,{targets[0,0].item()}]) = -log({probs[0,0,targets[0,0]].item():.6f}) = {manual_nll_0.item():.6f}\n")
    f.write(f"  NLL[0,1] = -log(probs[0,1,{targets[0,1].item()}]) = -log({probs[0,1,targets[0,1]].item():.6f}) = {manual_nll_1.item():.6f}\n")
    f.write(f"  Match: NLL[0,0]={nll[0,0].item():.6f}, NLL[0,1]={nll[0,1].item():.6f}\n\n")


def test_attention_mask(f):
    """Test 2: Attention mask - only compute loss where mask == 1."""
    f.write("=" * 80 + "\n")
    f.write("TEST 2: Attention mask (only compute for mask == 1)\n")
    f.write("=" * 80 + "\n\n")

    # B=1, L=4, vocab=3
    logits = torch.tensor([
        [[1.0, 0.0, 0.0],   # pos 0 (will be masked out)
         [0.0, 1.0, 0.0],   # pos 1 (will be masked out)
         [0.0, 0.0, 1.0],   # pos 2 (included)
         [1.0, 1.0, 0.0]],  # pos 3 (included)
    ])  # shape (1, 4, 3)

    targets = torch.tensor([[0, 1, 2, 0]])  # shape (1, 4)

    # Mask: first 2 tokens are masked (prefix), last 2 are valid
    attention_mask = torch.tensor([[0.0, 0.0, 1.0, 1.0]])

    f.write("Input:\n")
    f.write(dump_tensor("  logits", logits) + "\n")
    f.write(dump_tensor("  targets", targets) + "\n")
    f.write(dump_tensor("  attention_mask", attention_mask) + "\n\n")

    # Compute NLL
    nll = compute_token_nll(logits, targets)
    f.write("Per-token NLL (all positions):\n")
    f.write(dump_tensor("  nll", nll) + "\n\n")

    # Without loss_scale (set to -1 so weighted = nll)
    loss_scale = torch.ones_like(nll) * -1.0  # -1 * nll * -1 = nll

    # Test with mask
    weighted_nll = apply_loss_scale(nll, loss_scale)
    loss_with_mask = aggregate_loss(weighted_nll, attention_mask)
    loss_without_mask = aggregate_loss(weighted_nll, None)

    f.write("Loss computation:\n")
    f.write(f"  weighted_nll (loss_scale=-1): {weighted_nll}\n")
    f.write(f"  Loss WITH mask (mean of pos 2,3): {loss_with_mask.item():.6f}\n")
    f.write(f"  Loss WITHOUT mask (mean of all): {loss_without_mask.item():.6f}\n\n")

    # Manual verification
    manual_loss_with_mask = (weighted_nll[0, 2] + weighted_nll[0, 3]) / 2
    f.write("Manual verification:\n")
    f.write(f"  Mean of pos 2,3: ({weighted_nll[0,2].item():.6f} + {weighted_nll[0,3].item():.6f}) / 2 = {manual_loss_with_mask.item():.6f}\n")
    f.write(f"  Match: {torch.isclose(loss_with_mask, manual_loss_with_mask).item()}\n\n")


def test_loss_scale(f):
    """Test 3: Loss scale weighting."""
    f.write("=" * 80 + "\n")
    f.write("TEST 3: Loss scale weighting\n")
    f.write("=" * 80 + "\n\n")

    # B=1, L=3, vocab=2
    # Use uniform logits so NLL is the same for all positions
    logits = torch.zeros(1, 3, 2)  # uniform distribution
    targets = torch.tensor([[0, 0, 0]])

    f.write("Input:\n")
    f.write(dump_tensor("  logits", logits) + "\n")
    f.write(dump_tensor("  targets", targets) + "\n\n")

    nll = compute_token_nll(logits, targets)
    f.write("NLL (should be log(2) = 0.693 for all positions):\n")
    f.write(dump_tensor("  nll", nll) + "\n\n")

    # Test different loss_scale values
    # In BD3LM, loss_scale = -1/t where t is noise level
    # Higher t (more noise) -> smaller |loss_scale| -> smaller weight
    loss_scale_cases = [
        ("t=1.0 -> scale=-1.0", torch.tensor([[-1.0, -1.0, -1.0]])),
        ("t=0.5 -> scale=-2.0", torch.tensor([[-2.0, -2.0, -2.0]])),
        ("t=0.25 -> scale=-4.0", torch.tensor([[-4.0, -4.0, -4.0]])),
        ("varied: t=[1.0, 0.5, 0.25]", torch.tensor([[-1.0, -2.0, -4.0]])),
    ]

    f.write("Loss scale effect:\n")
    f.write("  Formula: weighted_nll = -loss_scale * nll = (1/t) * nll\n\n")

    for name, loss_scale in loss_scale_cases:
        weighted_nll = apply_loss_scale(nll, loss_scale)
        loss = aggregate_loss(weighted_nll, None)
        f.write(f"  {name}:\n")
        f.write(f"    loss_scale = {loss_scale.tolist()}\n")
        f.write(f"    weighted_nll = {weighted_nll.tolist()}\n")
        f.write(f"    mean loss = {loss.item():.6f}\n\n")


def test_batch_size(f):
    """Test 4: Batch size > 1."""
    f.write("=" * 80 + "\n")
    f.write("TEST 4: Batch size > 1 (B=2)\n")
    f.write("=" * 80 + "\n\n")

    # B=2, L=3, vocab=3
    logits = torch.tensor([
        # Batch 0
        [[2.0, 0.0, 0.0],   # strong preference for token 0
         [0.0, 2.0, 0.0],   # strong preference for token 1
         [0.0, 0.0, 2.0]],  # strong preference for token 2
        # Batch 1
        [[0.0, 0.0, 0.0],   # uniform
         [0.0, 0.0, 0.0],   # uniform
         [0.0, 0.0, 0.0]],  # uniform
    ])  # shape (2, 3, 3)

    targets = torch.tensor([
        [0, 1, 2],  # batch 0: correct predictions
        [0, 1, 2],  # batch 1: same targets, but uniform logits
    ])  # shape (2, 3)

    # Different loss_scale per batch
    loss_scale = torch.tensor([
        [-1.0, -1.0, -1.0],  # batch 0
        [-2.0, -2.0, -2.0],  # batch 1: higher weight
    ])

    # Mask: mask first token for both batches
    attention_mask = torch.tensor([
        [0.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ])

    f.write("Input:\n")
    f.write(dump_tensor("  logits", logits) + "\n")
    f.write(dump_tensor("  targets", targets) + "\n")
    f.write(dump_tensor("  loss_scale", loss_scale) + "\n")
    f.write(dump_tensor("  attention_mask", attention_mask) + "\n\n")

    # Full pipeline
    loss, per_token_nll = compute_bd3lm_loss(logits, targets, loss_scale, attention_mask)

    f.write("Results:\n")
    f.write(dump_tensor("  per_token_nll", per_token_nll) + "\n\n")

    probs = F.softmax(logits, dim=-1)
    f.write("Softmax probs (for verification):\n")
    f.write(f"  Batch 0, pos 0, token 0: {probs[0,0,0].item():.4f} (high)\n")
    f.write(f"  Batch 1, pos 0, token 0: {probs[1,0,0].item():.4f} (uniform=0.333)\n\n")

    f.write(f"  Final loss: {loss.item():.6f}\n\n")

    # Manual verification
    weighted = apply_loss_scale(per_token_nll, loss_scale)
    f.write("Manual breakdown:\n")
    f.write(f"  weighted_nll:\n{weighted}\n\n")
    f.write(f"  After masking (pos 0 ignored):\n")
    valid_weighted = weighted * attention_mask
    f.write(f"  valid_weighted:\n{valid_weighted}\n\n")
    f.write(f"  Sum of valid: {valid_weighted.sum().item():.6f}\n")
    f.write(f"  Count of valid: {attention_mask.sum().item():.0f}\n")
    f.write(f"  Mean: {valid_weighted.sum().item() / attention_mask.sum().item():.6f}\n")


def test_full_pipeline(f):
    """Test 5: Full compute_bd3lm_loss pipeline."""
    f.write("=" * 80 + "\n")
    f.write("TEST 5: Full compute_bd3lm_loss pipeline\n")
    f.write("=" * 80 + "\n\n")

    # Realistic-ish example: B=2, L=4, vocab=5
    torch.manual_seed(42)
    logits = torch.randn(2, 4, 5)
    targets = torch.randint(0, 5, (2, 4))

    # Noise levels: t in (0, 1]
    # loss_scale = -1/t
    noise_levels = torch.tensor([
        [1.0, 0.8, 0.5, 0.2],  # batch 0: decreasing noise
        [0.3, 0.3, 0.3, 0.3],  # batch 1: constant noise
    ])
    loss_scale = -1.0 / noise_levels

    # Mask first position (prefix pure token)
    attention_mask = torch.ones(2, 4)
    attention_mask[:, 0] = 0

    f.write("Input:\n")
    f.write(f"  logits shape: {list(logits.shape)}\n")
    f.write(f"  targets: {targets.tolist()}\n")
    f.write(f"  noise_levels: {noise_levels.tolist()}\n")
    f.write(f"  loss_scale: {loss_scale.tolist()}\n")
    f.write(f"  attention_mask: {attention_mask.tolist()}\n\n")

    loss, per_token_nll = compute_bd3lm_loss(logits, targets, loss_scale, attention_mask)

    f.write("Results:\n")
    f.write(f"  per_token_nll:\n{per_token_nll}\n\n")
    f.write(f"  Final loss: {loss.item():.6f}\n\n")

    # Show weighted breakdown
    weighted = apply_loss_scale(per_token_nll, loss_scale)
    f.write("Weighted NLL breakdown:\n")
    f.write(f"  weighted_nll:\n{weighted}\n\n")
    f.write("Note: Lower noise (t) -> higher |loss_scale| -> higher weight\n")
    f.write("      This makes sense: model should be more confident when noise is low\n")


def main():
    out_dir = os.path.join(os.getcwd(), "temp")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "bd3lm_loss_dump.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("BD3LM Loss Computation Verification\n")
        f.write("=" * 80 + "\n\n")

        f.write("This dump verifies:\n")
        f.write("1. Basic NLL computation is correct\n")
        f.write("2. Attention mask: only positions with mask==1 contribute to loss\n")
        f.write("3. Loss scale weighting: weighted_nll = -loss_scale * nll = (1/t) * nll\n")
        f.write("4. Batch size > 1 works correctly\n")
        f.write("5. Full pipeline works end-to-end\n")
        f.write("\n")

        test_basic_nll(f)
        test_attention_mask(f)
        test_loss_scale(f)
        test_batch_size(f)
        test_full_pipeline(f)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
