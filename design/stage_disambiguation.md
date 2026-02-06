# Stage Disambiguation

## Motivation

In Block-PDLM, the `[xt | x0]` training structure lets every token position contribute to loss: xt positions compute Stage 2 loss (group → pure), x0 positions compute Stage 1 loss (block → block). If the two stages were trained with separate forward passes, each pass would only use half the sequence for loss — halving training token efficiency.

Mixing stages in a single forward pass achieves full token utilization, but introduces a disambiguation problem: the same transformer weights, at the same RoPE positions, must produce different behaviors depending on which half of the input a token belongs to:

- **Stage 1** (x0 half): position i → predict target at i + B (next block)
- **Stage 2** (xt half): position i → predict target at i (denoise in-place)

Currently the model disambiguates purely from input embedding type (pure vs group tokens). This experiment verifies whether that implicit signal is sufficient.

## Theory: Saddle-Node Dynamics

The shared position can be viewed as a **saddle point** — stable along one direction (Stage 2), unstable along another (Stage 1). Relevant dynamical systems concepts:

- **Saddle-Node Bifurcation**: Two fixed points (stable + unstable) collide and annihilate. If the model conflates stages, the two distinct behaviors collapse into one degenerate mode.
- **Basin of Attraction**: The input embedding type (pure vs group) defines the basin. The question is whether these basins are separated enough for correct routing.
- **Self-Organized Criticality**: Systems naturally evolve toward states on the verge of collapse. The model may settle into a configuration where disambiguation appears to work but is fragile.
- **Crisis-Induced Intermittency**: Conflation may not appear at small scale but emerge at higher complexity — a stable-looking system hits a critical point and breaks.

## Experiment

### Hypothesis

The model can implicitly disambiguate Stage 1 and Stage 2 from input embedding type alone.

### Verification

TODO

### Fallback: Explicit Stage Conditioning

If implicit disambiguation fails:

- **Option A**: Learned stage embedding: `input_emb + stage_emb[stage_id]`
- **Option B (preferred)**: AdaLN — inject stage identity into every transformer block's layernorm (scale & shift modulation), as in diffusion model timestep conditioning.
