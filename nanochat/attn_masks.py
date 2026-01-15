from functools import partial

import torch
import torch.nn.functional as F

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    FLEX_ATTN_AVAILABLE = True
except Exception:
    FLEX_ATTN_AVAILABLE = False


def block_diff_mask(b, h, q_idx, kv_idx, block_size=None, n=None, prefix_ar_tokens=0):
    """
    Constructs the specialized block diffusion attention mask for training
    composed of three masks:
    - **Block Diagonal Mask (M_BD)**: Self-attention within noised blocks
    - **Offset Block Causal Mask (M_OBC)**: Cross-attention for conditional context
    - **Block Causal Mask (M_BC)**: Attention to update x0

    When prefix_ar_tokens > 0, the first `prefix_ar_tokens` positions use standard
    causal AR attention (not part of any block). Blocks start after the prefix.

    Args:
        b, h: Batch and head indices (ignored for mask logic).
        q_idx, kv_idx: Query and Key indices.
        n: Sequence length (L), total is 2L for [xt | x0].
        block_size: Defines the block structure.
        prefix_ar_tokens: Number of AR prefix tokens before blocks start.

    Returns:
        A boolean attention mask.
    """
    # Position within each half (xt or x0)
    pos_in_half_q = torch.where(q_idx >= n, q_idx - n, q_idx)
    pos_in_half_kv = torch.where(kv_idx >= n, kv_idx - n, kv_idx)

    # Indicate whether token belongs to xt or x0
    x0_flag_q = q_idx >= n
    x0_flag_kv = kv_idx >= n

    # Indicate whether token is in AR prefix (not in any block)
    is_prefix_q = pos_in_half_q < prefix_ar_tokens
    is_prefix_kv = pos_in_half_kv < prefix_ar_tokens

    # Compute block indices (only meaningful for non-prefix tokens)
    # Blocks start after prefix_ar_tokens
    block_q = torch.where(
        x0_flag_q == 1,
        (q_idx - n - prefix_ar_tokens) // block_size,
        (q_idx - prefix_ar_tokens) // block_size
    )
    block_kv = torch.where(
        x0_flag_kv == 1,
        (kv_idx - n - prefix_ar_tokens) // block_size,
        (kv_idx - prefix_ar_tokens) // block_size
    )

    # =========================================================================
    # AR Prefix Attention (standard causal within prefix)
    # =========================================================================
    # Prefix in xt attends to prefix in xt (causally)
    prefix_xt_to_xt = is_prefix_q & ~x0_flag_q & is_prefix_kv & ~x0_flag_kv & (pos_in_half_q >= pos_in_half_kv)

    # Prefix in x0 attends to prefix in xt (causally)
    prefix_x0_to_xt = is_prefix_q & x0_flag_q & is_prefix_kv & ~x0_flag_kv & (pos_in_half_q >= pos_in_half_kv)

    # Prefix in x0 attends to prefix in x0 (causally)
    prefix_x0_to_x0 = is_prefix_q & x0_flag_q & is_prefix_kv & x0_flag_kv & (pos_in_half_q >= pos_in_half_kv)

    ar_prefix_mask = prefix_xt_to_xt | prefix_x0_to_xt | prefix_x0_to_x0

    # =========================================================================
    # Block tokens attend to AR prefix
    # =========================================================================
    # Block tokens in xt can attend to all prefix tokens in xt
    block_xt_to_prefix_xt = ~is_prefix_q & ~x0_flag_q & is_prefix_kv & ~x0_flag_kv

    # Block tokens in x0 can attend to all prefix tokens in xt
    block_x0_to_prefix_xt = ~is_prefix_q & x0_flag_q & is_prefix_kv & ~x0_flag_kv

    # Block tokens in x0 can attend to all prefix tokens in x0
    block_x0_to_prefix_x0 = ~is_prefix_q & x0_flag_q & is_prefix_kv & x0_flag_kv

    block_to_prefix_mask = block_xt_to_prefix_xt | block_x0_to_prefix_xt | block_x0_to_prefix_x0

    # =========================================================================
    # Block Diffusion Attention (only for non-prefix tokens)
    # =========================================================================
    not_prefix_q = ~is_prefix_q
    not_prefix_kv = ~is_prefix_kv

    # **1. Block Diagonal Mask (M_BD) **
    block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv) & not_prefix_q & not_prefix_kv

    # **2. Offset Block-Causal Mask (M_OBC) **
    offset_block_causal = (block_q > block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 0) & not_prefix_q & not_prefix_kv

    # **3. Block-Causal Mask (M_BC) **
    block_causal = (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1) & not_prefix_q & not_prefix_kv

    # **4. Combine All Masks **
    return ar_prefix_mask | block_to_prefix_mask | block_diagonal | offset_block_causal | block_causal


def block_diff_mask_causal(b, h, q_idx, kv_idx, block_size=None, n=None, prefix_ar_tokens=0):
    """
    Constructs the specialized block diffusion attention mask for training
    with within-block causality, composed of three masks:
    - **Block Diagonal Mask (M_BD)**: Self-attention within noised blocks (causal within block)
    - **Offset Block Causal Mask (M_OBC)**: Cross-attention for conditional context
    - **Block Causal Mask (M_BC)**: Attention to update x0

    When prefix_ar_tokens > 0, the first `prefix_ar_tokens` positions use standard
    causal AR attention (not part of any block). Blocks start after the prefix.

    Args:
        b, h: Batch and head indices (ignored for mask logic).
        q_idx, kv_idx: Query and Key indices.
        n: Sequence length (L), total is 2L for [xt | x0].
        block_size: Defines the block structure.
        prefix_ar_tokens: Number of AR prefix tokens before blocks start.

    Returns:
        A boolean attention mask.
    """
    # Position within each half (xt or x0)
    pos_in_half_q = torch.where(q_idx >= n, q_idx - n, q_idx)
    pos_in_half_kv = torch.where(kv_idx >= n, kv_idx - n, kv_idx)

    # Indicate whether token belongs to xt or x0
    x0_flag_q = q_idx >= n
    x0_flag_kv = kv_idx >= n

    # Indicate whether token is in AR prefix (not in any block)
    is_prefix_q = pos_in_half_q < prefix_ar_tokens
    is_prefix_kv = pos_in_half_kv < prefix_ar_tokens

    # Compute block indices (only meaningful for non-prefix tokens)
    # Blocks start after prefix_ar_tokens
    block_q = torch.where(
        x0_flag_q == 1,
        (q_idx - n - prefix_ar_tokens) // block_size,
        (q_idx - prefix_ar_tokens) // block_size
    )
    block_kv = torch.where(
        x0_flag_kv == 1,
        (kv_idx - n - prefix_ar_tokens) // block_size,
        (kv_idx - prefix_ar_tokens) // block_size
    )

    # Position within block (for within-block causality)
    pos_in_block_q = torch.where(
        x0_flag_q == 1,
        (q_idx - n - prefix_ar_tokens) % block_size,
        (q_idx - prefix_ar_tokens) % block_size
    )
    pos_in_block_kv = torch.where(
        x0_flag_kv == 1,
        (kv_idx - n - prefix_ar_tokens) % block_size,
        (kv_idx - prefix_ar_tokens) % block_size
    )

    # =========================================================================
    # AR Prefix Attention (standard causal within prefix)
    # =========================================================================
    # Prefix in xt attends to prefix in xt (causally)
    prefix_xt_to_xt = is_prefix_q & ~x0_flag_q & is_prefix_kv & ~x0_flag_kv & (pos_in_half_q >= pos_in_half_kv)

    # Prefix in x0 attends to prefix in xt (causally)
    prefix_x0_to_xt = is_prefix_q & x0_flag_q & is_prefix_kv & ~x0_flag_kv & (pos_in_half_q >= pos_in_half_kv)

    # Prefix in x0 attends to prefix in x0 (causally)
    prefix_x0_to_x0 = is_prefix_q & x0_flag_q & is_prefix_kv & x0_flag_kv & (pos_in_half_q >= pos_in_half_kv)

    ar_prefix_mask = prefix_xt_to_xt | prefix_x0_to_xt | prefix_x0_to_x0

    # =========================================================================
    # Block tokens attend to AR prefix
    # =========================================================================
    # Block tokens in xt can attend to all prefix tokens in xt
    block_xt_to_prefix_xt = ~is_prefix_q & ~x0_flag_q & is_prefix_kv & ~x0_flag_kv

    # Block tokens in x0 can attend to all prefix tokens in xt
    block_x0_to_prefix_xt = ~is_prefix_q & x0_flag_q & is_prefix_kv & ~x0_flag_kv

    # Block tokens in x0 can attend to all prefix tokens in x0
    block_x0_to_prefix_x0 = ~is_prefix_q & x0_flag_q & is_prefix_kv & x0_flag_kv

    block_to_prefix_mask = block_xt_to_prefix_xt | block_x0_to_prefix_xt | block_x0_to_prefix_x0

    # =========================================================================
    # Block Diffusion Attention with within-block causality (only for non-prefix tokens)
    # =========================================================================
    not_prefix_q = ~is_prefix_q
    not_prefix_kv = ~is_prefix_kv

    same_block = block_q == block_kv
    causal_within_block = pos_in_block_q >= pos_in_block_kv

    # **1. Offset Block-Causal Mask (M_OBC) **
    offset_block_causal = (block_q > block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 0) & not_prefix_q & not_prefix_kv

    # **2. Block Diagonal Mask (M_BD) with within-block causality **
    block_diagonal = same_block & (x0_flag_q == x0_flag_kv) & causal_within_block & not_prefix_q & not_prefix_kv

    # **3. Block-Causal Mask (M_BC) **
    block_causal = (
        ((block_q > block_kv) | (same_block & causal_within_block))
        & (x0_flag_kv == 1)
        & (x0_flag_q == 1)
        & not_prefix_q
        & not_prefix_kv
    )

    # **4. Combine All Masks **
    return ar_prefix_mask | block_to_prefix_mask | block_diagonal | offset_block_causal | block_causal


def gen_mask(seqlen, block_size, attn_backend="sdpa", is_causal=False, prefix_ar_tokens=0):
    """
    Builds a 2L x 2L mask for xt || x0.

    Args:
        seqlen: Sequence length L (total mask is 2L x 2L).
        block_size: Size of each block for block diffusion.
        attn_backend: "sdpa" or "flex".
        is_causal: If True, use within-block causality.
        prefix_ar_tokens: Number of AR prefix tokens before blocks start.
                         These tokens use standard causal attention.

    Returns:
        Attention mask of shape (2L, 2L) for sdpa or BlockMask for flex.
    """
    mask_fn = block_diff_mask_causal if is_causal else block_diff_mask
    if attn_backend == "flex" and FLEX_ATTN_AVAILABLE:
        return create_block_mask(
            partial(mask_fn, block_size=block_size, n=seqlen, prefix_ar_tokens=prefix_ar_tokens),
            B=None,
            H=None,
            Q_LEN=seqlen * 2,
            KV_LEN=seqlen * 2,
        )
    if attn_backend == "sdpa":
        return mask_fn(
            b=None,
            h=None,
            q_idx=torch.arange(seqlen * 2)[:, None],
            kv_idx=torch.arange(seqlen * 2)[None, :],
            block_size=block_size,
            n=seqlen,
            prefix_ar_tokens=prefix_ar_tokens,
        )
    raise ValueError("Unknown attention backend")


def forward_mask_use(mask):
    # Reference use inside forward:
    # mask = self.block_diff_mask
    # x = block(..., mask=mask)
    return mask


def demo_attention_usage():
    # Simple SDPA demo: mask is 2L x 2L and broadcasts over batch/heads.
    torch.manual_seed(0)
    seq_len = 6
    block_size = 2
    total_len = seq_len * 2
    batch = 1
    heads = 2
    head_dim = 4
    q = torch.randn(batch, heads, total_len, head_dim)
    k = torch.randn(batch, heads, total_len, head_dim)
    v = torch.randn(batch, heads, total_len, head_dim)
    mask = gen_mask(seq_len, block_size, attn_backend="sdpa")
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
    return out, mask


def demo_flex_attention_usage():
    if not FLEX_ATTN_AVAILABLE:
        return None, None
    torch.manual_seed(0)
    seq_len = 6
    block_size = 2
    total_len = seq_len * 2
    batch = 1
    heads = 2
    head_dim = 4
    q = torch.randn(batch, heads, total_len, head_dim)
    k = torch.randn(batch, heads, total_len, head_dim)
    v = torch.randn(batch, heads, total_len, head_dim)
    mask = gen_mask(seq_len, block_size, attn_backend="flex")
    out = flex_attention(q, k, v, block_mask=mask)
    return out, mask


if __name__ == "__main__":
    out, mask = demo_attention_usage()
    print(f"mask.shape={tuple(mask.shape)} out.shape={tuple(out.shape)}")
    flex_out, flex_mask = demo_flex_attention_usage()
    if flex_out is None:
        print("flex_attention not available")
    else:
        print(f"flex_out.shape={tuple(flex_out.shape)} flex_mask={type(flex_mask)}")
