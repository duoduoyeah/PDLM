from functools import partial

import torch
import torch.nn.functional as F

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    FLEX_ATTN_AVAILABLE = True
except Exception:
    FLEX_ATTN_AVAILABLE = False


def block_diff_mask(b, h, q_idx, kv_idx, block_size=None, n=None):
    """
    Constructs the specialized block diffusion attention mask for training
    composed of three masks:
    - **Block Diagonal Mask (M_BD)**: Self-attention within noised blocks
    - **Offset Block Causal Mask (M_OBC)**: Cross-attention for conditional context
    - **Block Causal Mask (M_BC)**: Attention to update x0

    Args:
        b, h: Batch and head indices (ignored for mask logic).
        q_idx, kv_idx: Query and Key indices.
        seq_len: Total sequence length.
        block_size: Defines the block structure.

    Returns:
        A boolean attention mask.
    """

    # Indicate whether token belongs to xt or x0
    x0_flag_q = q_idx >= n
    x0_flag_kv = kv_idx >= n

    # Compute block indices
    block_q = torch.where(
        x0_flag_q == 1, (q_idx - n) // block_size, q_idx // block_size
    )
    block_kv = torch.where(
        x0_flag_kv == 1, (kv_idx - n) // block_size, kv_idx // block_size
    )

    # **1. Block Diagonal Mask (M_BD) **
    block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)

    # **2. Offset Block-Causal Mask (M_OBC) **
    offset_block_causal = (block_q > block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 0)

    # **3. Block-Causal Mask (M_BC) **
    block_causal = (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)

    # **4. Combine Masks **
    return block_diagonal | offset_block_causal | block_causal


def block_diff_mask_causal(b, h, q_idx, kv_idx, block_size=None, n=None):
    """
    Constructs the specialized block diffusion attention mask for training
    composed of three masks:
    - **Block Diagonal Mask (M_BD)**: Self-attention within noised blocks
    - **Offset Block Causal Mask (M_OBC)**: Cross-attention for conditional context
    - **Block Causal Mask (M_BC)**: Attention to update x0

    Args:
        b, h: Batch and head indices (ignored for mask logic).
        q_idx, kv_idx: Query and Key indices.
        seq_len: Total sequence length.
        block_size: Defines the block structure.

    Returns:
        A boolean attention mask.
    """

    # Indicate whether token belongs to xt or x0
    x0_flag_q = q_idx >= n
    x0_flag_kv = kv_idx >= n

    # Compute block indices
    block_q = torch.where(
        x0_flag_q == 1, (q_idx - n) // block_size, q_idx // block_size
    )
    block_kv = torch.where(
        x0_flag_kv == 1, (kv_idx - n) // block_size, kv_idx // block_size
    )

    # **1. Offset Block-Causal Mask (M_OBC) **
    offset_block_causal = (block_q > block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 0)

    # Add within-block causality.
    pos_q = torch.where(x0_flag_q == 1, (q_idx - n) % block_size, q_idx % block_size)
    pos_kv = torch.where(
        x0_flag_kv == 1, (kv_idx - n) % block_size, kv_idx % block_size
    )
    same_block = block_q == block_kv
    causal_within_block = pos_q >= pos_kv

    # **2. Block Diagonal Mask (M_BD) **
    block_diagonal = same_block & (x0_flag_q == x0_flag_kv) & causal_within_block

    # **3. Block-Causal Mask (M_BC) **
    block_causal = (
        ((block_q > block_kv) | (same_block & causal_within_block))
        & (x0_flag_kv == 1)
        & (x0_flag_q == 1)
    )
    return block_diagonal | offset_block_causal | block_causal


def gen_mask(seqlen, block_size, attn_backend="sdpa", is_causal=False):
    # Builds a 2L x 2L mask for xt || x0.
    mask_fn = block_diff_mask_causal if is_causal else block_diff_mask
    if attn_backend == "flex" and FLEX_ATTN_AVAILABLE:
        return create_block_mask(
            partial(mask_fn, block_size=block_size, n=seqlen),
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
