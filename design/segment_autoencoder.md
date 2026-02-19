# Segment Autoencoder

## Goal

Convert variable-length token segments into fixed-shape tensors that the main LLM consumes as its atomic units. The decoder reconstructs the original segment from the fixed-shape representation.

```
segment (x, emb)  →  Encoder  →  (f, emb) + length signal  →  Decoder  →  segment (x, emb)
   variable                        fixed                                     variable
```

## Desired Properties

1. **Fixed output shape**: Any segment of 1~K tokens produces the same shape `(f, emb)`. This is the entire reason the autoencoder exists.
2. **High reconstruction fidelity**: Input and output should be as close to identical as possible. This is not a lossy bottleneck — the main LLM needs the full content.
3. **Self-contained representation**: The fixed-shape output encodes everything needed to reconstruct, including the original length.

## Not goals

- Smooth/regularized latent space (no KL, no prior — this is AE, not VAE)
- Text generation or controlled generation
- Compression for its own sake
- Semantic summarization

## Open question: the variable→fixed transform

The encoder must map `(x, emb) → (f, emb)` where `x` varies per segment. A static weight matrix shaped `(f, x)` doesn't work because `x` isn't constant. The transform must adapt to input length.

Candidate mechanism: cross-attention with `f` learned queries (Perceiver-style). The queries are fixed parameters; keys/values come from the input. Attention scores form the dynamic `(f, x)` matrix that adapts per input.
