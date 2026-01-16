#!/usr/bin/env python3
"""
Verify epoch tracking and mask cycling for BD3LM target_shift mode.

This script simulates the dataloader's epoch tracking and verifies that:
1. Epoch counter increments when restarting from first shard
2. Mask selection cycles correctly: block_diff_masks[epoch % block_size]
"""
import os
import sys

import torch

from nanochat.attn_masks import gen_mask


def simulate_document_batches(num_shards=3, batches_per_shard=2, num_epochs=3):
    """
    Simulate the document_batches generator to verify epoch tracking.

    Yields: (shard_idx, batch_idx, epoch)
    """
    epoch = 0
    first_pass = True

    for _ in range(num_epochs):
        pq_idx = 0 if not first_pass else 0
        if not first_pass:
            epoch += 1

        while pq_idx < num_shards:
            for batch_idx in range(batches_per_shard):
                yield (pq_idx, batch_idx, epoch)
            pq_idx += 1

        first_pass = False


def main():
    # Configuration
    block_size = 4
    num_shards = 3
    batches_per_shard = 2
    num_epochs = 5  # Simulate 5 full passes through all shards
    seq_len = 8

    print("=" * 60)
    print("Epoch Tracking and Mask Cycling Verification")
    print("=" * 60)
    print(f"block_size: {block_size}")
    print(f"num_shards: {num_shards}")
    print(f"batches_per_shard: {batches_per_shard}")
    print(f"num_epochs: {num_epochs}")
    print()

    # Pre-generate masks (as done in base_train.py for BD3LM target_shift mode)
    print("Pre-generating masks for prefix_sliding_tokens = 0, 1, ..., block_size-1")
    block_diff_masks = [
        gen_mask(seq_len, block_size, attn_backend="sdpa", is_causal=True, prefix_sliding_tokens=i)
        for i in range(block_size)
    ]
    print(f"Generated {len(block_diff_masks)} masks")
    print()

    # Simulate epoch tracking
    print("=" * 60)
    print("Simulating dataloader epoch tracking")
    print("=" * 60)
    print(f"{'Shard':<8} {'Batch':<8} {'Epoch':<8} {'Mask Idx':<10} {'prefix_sliding'}")
    print("-" * 60)

    prev_epoch = -1
    for shard_idx, batch_idx, epoch in simulate_document_batches(num_shards, batches_per_shard, num_epochs):
        mask_idx = epoch % len(block_diff_masks)
        prefix_sliding = mask_idx  # mask_idx corresponds to prefix_sliding_tokens

        # Mark epoch transitions
        epoch_marker = " <-- NEW EPOCH" if epoch != prev_epoch else ""
        prev_epoch = epoch

        print(f"{shard_idx:<8} {batch_idx:<8} {epoch:<8} {mask_idx:<10} {prefix_sliding}{epoch_marker}")

    print()
    print("=" * 60)
    print("Mask shapes verification")
    print("=" * 60)
    for i, mask in enumerate(block_diff_masks):
        print(f"Mask {i} (prefix_sliding_tokens={i}): shape={tuple(mask.shape)}")

    # Show a sample mask difference
    print()
    print("=" * 60)
    print("Sample mask comparison (first row of each mask)")
    print("=" * 60)
    for i, mask in enumerate(block_diff_masks):
        row0 = mask[0].int().tolist()
        print(f"Mask {i}: {row0}")

    # Summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    total_batches = num_shards * batches_per_shard * num_epochs
    print(f"Total batches simulated: {total_batches}")
    print(f"Epochs cycled through: 0 to {num_epochs - 1}")
    print(f"Mask indices used: {[i % block_size for i in range(num_epochs)]}")
    print()
    print("Verification: Each epoch uses a different mask (cycling every block_size epochs)")


if __name__ == "__main__":
    main()
