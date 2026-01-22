#!/usr/bin/env python3
"""
Verify target_shift masking behavior for BD3LM.

Tests target_shift = 1, 2, 4 with block_size = 4 and prefix_pure_tokens = 1.
Shows which positions get masked and how prefix_pure_tokens overrides target_shift.
"""
import torch

from nanochat.bd3lm_utils.bd3lm_mask import sample_t, q_xt


def visualize_masking(
    seq_len: int,
    block_size: int,
    target_shift: int,
    prefix_pure_tokens: int,
    mask_token_id: int = 99999,
):
    """Visualize masking for a single sequence."""
    B = 1
    T = seq_len
    num_blocks = T // block_size

    # Create dummy clean tokens (just use position indices for clarity)
    targets = torch.arange(T).unsqueeze(0)  # (1, T)

    # Sample t (use fixed seed for reproducibility)
    torch.manual_seed(42)
    t = sample_t(
        batch_size=B,
        num_blocks=num_blocks,
        sampling_eps_min=1e-3,
        sampling_eps_max=1.0,
        device="cpu",
        antithetic_sampling=True,
    )

    # Apply q_xt (random masking with prefix protection)
    inputs, mask = q_xt(
        x0=targets,
        t=t,
        mask_token_id=mask_token_id,
        block_size=block_size,
        prefix_pure_tokens=prefix_pure_tokens,
    )

    # Apply target_shift masking (same logic as dataloader.py)
    if target_shift >= 1:
        positions_to_mask = torch.arange(target_shift - 1, T, block_size)
        inputs[:, positions_to_mask] = mask_token_id
        mask[:, positions_to_mask] = True

        # prefix_pure_tokens overrides target_shift
        if prefix_pure_tokens > 0:
            inputs[:, :prefix_pure_tokens] = targets[:, :prefix_pure_tokens]
            mask[:, :prefix_pure_tokens] = False

    return targets[0], inputs[0], mask[0], t[0]


def format_sequence(targets, inputs, mask, block_size, mask_token_id):
    """Format sequence for display."""
    lines = []
    T = len(targets)

    for block_idx in range(T // block_size):
        start = block_idx * block_size
        end = start + block_size

        block_targets = targets[start:end].tolist()
        block_inputs = inputs[start:end].tolist()
        block_mask = mask[start:end].tolist()

        # Format each position
        pos_strs = []
        for tgt, inp, m in zip(block_targets, block_inputs, block_mask):
            if inp == mask_token_id:
                pos_strs.append(f"[M]")  # Masked
            else:
                pos_strs.append(f"{inp:2d}")  # Original token

        mask_strs = ["T" if m else "F" for m in block_mask]

        lines.append(f"  Block {block_idx}: positions {start:2d}-{end-1:2d} | inputs: {' '.join(pos_strs)} | mask: {' '.join(mask_strs)}")

    return "\n".join(lines)


def main():
    seq_len = 16
    block_size = 4
    prefix_pure_tokens = 1
    mask_token_id = 99999

    print("=" * 70)
    print("BD3LM Target Shift Masking Verification")
    print("=" * 70)
    print(f"seq_len: {seq_len}")
    print(f"block_size: {block_size}")
    print(f"prefix_pure_tokens: {prefix_pure_tokens}")
    print(f"[M] = masked position")
    print()

    for target_shift in [1, 2, 4]:
        print("=" * 70)
        print(f"target_shift = {target_shift} (masks position {target_shift - 1} in each block, 0-indexed)")
        print("=" * 70)

        # Calculate expected masked positions
        expected_positions = list(range(target_shift - 1, seq_len, block_size))
        # Filter out prefix positions
        expected_after_prefix = [p for p in expected_positions if p >= prefix_pure_tokens]

        print(f"Expected target_shift positions (before prefix override): {expected_positions}")
        print(f"Expected target_shift positions (after prefix override): {expected_after_prefix}")
        print()

        targets, inputs, mask, t = visualize_masking(
            seq_len=seq_len,
            block_size=block_size,
            target_shift=target_shift,
            prefix_pure_tokens=prefix_pure_tokens,
            mask_token_id=mask_token_id,
        )

        print(format_sequence(targets, inputs, mask, block_size, mask_token_id))
        print()

        # Verify target_shift positions
        actual_target_shift_masked = []
        for pos in expected_positions:
            if inputs[pos] == mask_token_id:
                actual_target_shift_masked.append(pos)

        print(f"Actual target_shift positions masked: {actual_target_shift_masked}")

        # Check prefix protection
        prefix_protected = all(inputs[i] != mask_token_id for i in range(prefix_pure_tokens))
        prefix_mask_false = all(not mask[i] for i in range(prefix_pure_tokens))
        print(f"Prefix positions ({list(range(prefix_pure_tokens))}) protected: {prefix_protected and prefix_mask_false}")

        # Verify correctness
        if actual_target_shift_masked == expected_after_prefix:
            print("✓ CORRECT: target_shift masking matches expected (with prefix override)")
        else:
            print("✗ ERROR: target_shift masking does not match expected")
        print()

    # Additional test: prefix_pure_tokens = 0 (no prefix protection)
    print("=" * 70)
    print("Additional test: prefix_pure_tokens = 0 (no prefix protection)")
    print("=" * 70)

    for target_shift in [1]:
        print(f"\ntarget_shift = {target_shift}, prefix_pure_tokens = 0")

        targets, inputs, mask, t = visualize_masking(
            seq_len=seq_len,
            block_size=block_size,
            target_shift=target_shift,
            prefix_pure_tokens=0,  # No prefix protection
            mask_token_id=mask_token_id,
        )

        print(format_sequence(targets, inputs, mask, block_size, mask_token_id))

        expected_positions = list(range(target_shift - 1, seq_len, block_size))
        actual_masked = [p for p in expected_positions if inputs[p] == mask_token_id]
        print(f"Expected: {expected_positions}")
        print(f"Actual: {actual_masked}")
        print(f"✓ Position 0 IS masked (no prefix protection)" if inputs[0] == mask_token_id else "✗ Position 0 should be masked")


if __name__ == "__main__":
    main()
