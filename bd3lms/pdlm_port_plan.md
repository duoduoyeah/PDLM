PDLM Port Plan (BD3LM Block Mask)

One sentence:
Port BD3LM-style 2x sequence input and block diffusion attention mask into PDLM, then compute loss only on the first half using shared rotary positions.

Detailed:
- Add `block_size` and a `use_twice_seq` toggle to `PDLMConfig`.
- Precompute a `[2L, 2L]` block diffusion mask once (from `block_diff_mask`) and store as a buffer on the model.
- Extend `CausalSelfAttention` to accept `attn_mask` and call SDPA with `attn_mask=...`, `is_causal=False` when a mask is provided.
- In training, build `x_input = concat(xt, x0)` and pass it through the model; slice logits to `:L` for loss.
- Keep rotary positions shared across both halves (use positions `0..L-1` for `xt` and `x0`).
- Ensure compile/static shape uses `2L` (update token count math and any asserts).
