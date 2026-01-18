"""
Test the new BD3LM masking logic:
1. sample_t now uses [1/block_size, 1] range
2. get_adjusted_mask_prob computes p' = (block_size * t - 1) / (block_size - 1)
3. q_xt guarantees at least 1 mask per block via forced_mask_position
4. prefix_pure_tokens has highest priority (never masked)
5. Both normal BD3LM and target_shift use same adjusted probability logic
"""

import torch
import sys
sys.path.insert(0, "/home/shiyuan/workspace/interestRepo/nanochat")

from nanochat.bd3lm_utils.bd3lm_mask import (
    sample_t,
    get_adjusted_mask_prob,
    get_loss_scale,
    expand_block_to_seq,
    q_xt,
)

def test_sample_t():
    """Test that sample_t produces t in [1/block_size, 1]"""
    print("=" * 60)
    print("TEST: sample_t range")
    print("=" * 60)

    for block_size in [4, 8, 16]:
        t = sample_t(
            batch_size=32,
            num_blocks=128,
            block_size=block_size,
            sampling_eps_max=1.0,
            device="cpu",
            antithetic_sampling=True,
        )
        t_min_expected = 1.0 / block_size
        t_min_actual = t.min().item()
        t_max_actual = t.max().item()

        print(f"block_size={block_size}:")
        print(f"  Expected t_min: {t_min_expected:.4f}")
        print(f"  Actual t range: [{t_min_actual:.4f}, {t_max_actual:.4f}]")
        assert t_min_actual >= t_min_expected - 1e-6, f"t_min too small: {t_min_actual} < {t_min_expected}"
        assert t_max_actual <= 1.0 + 1e-6, f"t_max too large: {t_max_actual}"
        print("  PASS")
    print()


def test_adjusted_mask_prob():
    """Test get_adjusted_mask_prob formula"""
    print("=" * 60)
    print("TEST: get_adjusted_mask_prob formula")
    print("=" * 60)

    block_size = 8

    # Test specific t values
    test_cases = [
        (1.0 / block_size, 0.0),  # t = 1/8 -> p' = 0
        (0.5, (8 * 0.5 - 1) / 7),  # t = 0.5 -> p' = 3/7 ≈ 0.4286
        (1.0, 1.0),  # t = 1 -> p' = 1
    ]

    for t_val, expected_p in test_cases:
        t = torch.tensor([[t_val]])
        p_adjusted = get_adjusted_mask_prob(t, block_size)
        actual_p = p_adjusted.item()
        print(f"t={t_val:.4f} -> p'={actual_p:.4f} (expected {expected_p:.4f})")
        assert abs(actual_p - expected_p) < 1e-6, f"Mismatch: {actual_p} != {expected_p}"

    print("PASS")
    print()


def test_expected_mask_count():
    """Test that expected mask count = block_size * t"""
    print("=" * 60)
    print("TEST: Expected mask count = block_size * t")
    print("=" * 60)

    block_size = 8
    B, L = 1000, 1024  # Large batch for statistical test
    num_blocks = L // block_size
    mask_token_id = 99999

    for t_val in [0.125, 0.25, 0.5, 0.75, 1.0]:
        t = torch.full((B, num_blocks), t_val)
        x0 = torch.randint(0, 1000, (B, L))

        # Normal BD3LM (random forced position)
        xt, mask = q_xt(
            x0=x0,
            t=t,
            mask_token_id=mask_token_id,
            block_size=block_size,
            prefix_pure_tokens=0,
            forced_mask_position=None,
        )

        # Count masks per block
        mask_reshaped = mask.view(B, num_blocks, block_size)
        masks_per_block = mask_reshaped.sum(dim=-1).float()  # (B, num_blocks)
        avg_masks = masks_per_block.mean().item()
        expected_masks = block_size * t_val

        print(f"t={t_val:.3f}: avg masks/block = {avg_masks:.3f} (expected {expected_masks:.3f})")

        # Allow some statistical variance
        assert abs(avg_masks - expected_masks) < 0.1, f"Mismatch: {avg_masks} vs {expected_masks}"

    print("PASS")
    print()


def test_at_least_one_mask_per_block():
    """Test that every block has at least 1 mask"""
    print("=" * 60)
    print("TEST: At least 1 mask per block")
    print("=" * 60)

    block_size = 8
    B, L = 100, 1024
    num_blocks = L // block_size
    mask_token_id = 99999

    # Sample t from the valid range
    t = sample_t(
        batch_size=B,
        num_blocks=num_blocks,
        block_size=block_size,
        device="cpu",
    )

    x0 = torch.randint(0, 1000, (B, L))

    # Test both normal and target_shift modes
    for mode, forced_pos in [("normal (random)", None), ("target_shift=1", 0), ("target_shift=4", 3)]:
        xt, mask = q_xt(
            x0=x0,
            t=t,
            mask_token_id=mask_token_id,
            block_size=block_size,
            prefix_pure_tokens=0,
            forced_mask_position=forced_pos,
        )

        # Check each block has at least 1 mask
        mask_reshaped = mask.view(B, num_blocks, block_size)
        masks_per_block = mask_reshaped.sum(dim=-1)  # (B, num_blocks)
        min_masks = masks_per_block.min().item()

        print(f"{mode}: min masks/block = {min_masks}")
        assert min_masks >= 1, f"Found block with 0 masks in {mode} mode"

    print("PASS")
    print()


def test_target_shift_forced_position():
    """Test that target_shift mode forces the correct position"""
    print("=" * 60)
    print("TEST: target_shift forces correct position")
    print("=" * 60)

    block_size = 8
    B, L = 10, 64
    num_blocks = L // block_size
    mask_token_id = 99999

    # Use t = 1/block_size so only forced position is masked
    t = torch.full((B, num_blocks), 1.0 / block_size)
    x0 = torch.randint(0, 1000, (B, L))

    for target_shift in [1, 4, 8]:  # 1-indexed
        forced_pos = target_shift - 1  # 0-indexed

        xt, mask = q_xt(
            x0=x0,
            t=t,
            mask_token_id=mask_token_id,
            block_size=block_size,
            prefix_pure_tokens=0,
            forced_mask_position=forced_pos,
        )

        # Check that only the forced position is masked in each block
        mask_reshaped = mask.view(B, num_blocks, block_size)

        # Forced position should be True for all blocks
        forced_all_true = mask_reshaped[:, :, forced_pos].all().item()

        # Other positions should be False (when t = 1/block_size, p' = 0)
        other_mask = mask_reshaped.clone()
        other_mask[:, :, forced_pos] = False
        other_all_false = (~other_mask.any()).item()

        print(f"target_shift={target_shift} (pos {forced_pos}): forced_all_true={forced_all_true}, other_all_false={other_all_false}")
        assert forced_all_true, f"Forced position {forced_pos} not always masked"
        assert other_all_false, f"Other positions should not be masked when t=1/block_size"

    print("PASS")
    print()


def test_prefix_pure_tokens_priority():
    """Test that prefix_pure_tokens has highest priority (never masked)"""
    print("=" * 60)
    print("TEST: prefix_pure_tokens has highest priority")
    print("=" * 60)

    block_size = 8
    B, L = 100, 64
    num_blocks = L // block_size
    mask_token_id = 99999
    prefix_pure_tokens = 8  # First block is pure

    # Use high t value so many positions would be masked
    t = torch.full((B, num_blocks), 0.9)
    x0 = torch.randint(0, 1000, (B, L))

    # Test both normal and target_shift modes
    for mode, forced_pos in [("normal", None), ("target_shift=1", 0), ("target_shift=8", 7)]:
        xt, mask = q_xt(
            x0=x0,
            t=t,
            mask_token_id=mask_token_id,
            block_size=block_size,
            prefix_pure_tokens=prefix_pure_tokens,
            forced_mask_position=forced_pos,
        )

        # Check prefix is never masked
        prefix_mask = mask[:, :prefix_pure_tokens]
        prefix_any_masked = prefix_mask.any().item()

        # Check prefix tokens are unchanged
        prefix_unchanged = (xt[:, :prefix_pure_tokens] == x0[:, :prefix_pure_tokens]).all().item()

        # Check suffix has masks
        suffix_mask = mask[:, prefix_pure_tokens:]
        suffix_has_masks = suffix_mask.any().item()

        print(f"{mode}: prefix_masked={prefix_any_masked}, prefix_unchanged={prefix_unchanged}, suffix_has_masks={suffix_has_masks}")
        assert not prefix_any_masked, f"Prefix should never be masked in {mode}"
        assert prefix_unchanged, f"Prefix tokens should be unchanged in {mode}"
        assert suffix_has_masks, f"Suffix should have masks in {mode}"

    print("PASS")
    print()


def test_loss_scale():
    """Test that loss_scale = -1/t"""
    print("=" * 60)
    print("TEST: loss_scale = -1/t")
    print("=" * 60)

    block_size = 8
    B, num_blocks = 4, 16

    t = sample_t(
        batch_size=B,
        num_blocks=num_blocks,
        block_size=block_size,
        device="cpu",
    )

    loss_scale = get_loss_scale(t)
    expected_loss_scale = -1.0 / t

    match = torch.allclose(loss_scale, expected_loss_scale)
    print(f"loss_scale = -1/t: {match}")
    assert match, "loss_scale should equal -1/t"

    # Test expansion to sequence level
    loss_scale_expanded = expand_block_to_seq(loss_scale, block_size)
    expected_shape = (B, num_blocks * block_size)
    print(f"Expanded shape: {loss_scale_expanded.shape} (expected {expected_shape})")
    assert loss_scale_expanded.shape == expected_shape

    # Check that each position in a block has the same loss_scale
    loss_scale_reshaped = loss_scale_expanded.view(B, num_blocks, block_size)
    for i in range(block_size):
        same_as_first = (loss_scale_reshaped[:, :, i] == loss_scale_reshaped[:, :, 0]).all().item()
        assert same_as_first, f"Position {i} should have same loss_scale as position 0"

    print("All positions in block have same loss_scale: PASS")
    print()


def test_visual_example():
    """Visual example of masking"""
    print("=" * 60)
    print("VISUAL EXAMPLE")
    print("=" * 60)

    block_size = 8
    B, L = 2, 32
    num_blocks = L // block_size
    mask_token_id = -1

    torch.manual_seed(42)

    t = sample_t(batch_size=B, num_blocks=num_blocks, block_size=block_size, device="cpu")
    x0 = torch.arange(L).unsqueeze(0).expand(B, -1)

    print(f"x0[0]: {x0[0].tolist()}")
    print(f"t[0]:  {t[0].tolist()} (per block)")
    print()

    # Normal BD3LM
    xt_normal, mask_normal = q_xt(x0, t, mask_token_id, block_size, prefix_pure_tokens=0, forced_mask_position=None)
    print("Normal BD3LM (random forced position):")
    print(f"  xt[0]:   {xt_normal[0].tolist()}")
    print(f"  mask[0]: {mask_normal[0].int().tolist()}")
    mask_reshaped = mask_normal[0].view(num_blocks, block_size)
    print(f"  masks per block: {mask_reshaped.sum(dim=-1).tolist()}")
    print()

    # Target shift = 1 (position 0)
    xt_ts1, mask_ts1 = q_xt(x0, t, mask_token_id, block_size, prefix_pure_tokens=0, forced_mask_position=0)
    print("Target shift = 1 (force position 0):")
    print(f"  xt[0]:   {xt_ts1[0].tolist()}")
    print(f"  mask[0]: {mask_ts1[0].int().tolist()}")
    mask_reshaped = mask_ts1[0].view(num_blocks, block_size)
    print(f"  masks per block: {mask_reshaped.sum(dim=-1).tolist()}")
    print()

    # With prefix_pure_tokens
    xt_prefix, mask_prefix = q_xt(x0, t, mask_token_id, block_size, prefix_pure_tokens=8, forced_mask_position=0)
    print("With prefix_pure_tokens=8 (first block unmasked):")
    print(f"  xt[0]:   {xt_prefix[0].tolist()}")
    print(f"  mask[0]: {mask_prefix[0].int().tolist()}")
    print()


if __name__ == "__main__":
    test_sample_t()
    test_adjusted_mask_prob()
    test_expected_mask_count()
    test_at_least_one_mask_per_block()
    test_target_shift_forced_position()
    test_prefix_pure_tokens_priority()
    test_loss_scale()
    test_visual_example()

    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
