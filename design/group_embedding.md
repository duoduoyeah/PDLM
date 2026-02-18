# Group Embedding

## Goal

Test whether using the average of group member embeddings as the intermediate representation can match or beat discrete group tokens for Group → Pure denoising. Instead of assigning each group a learned embedding via a separate token ID, we construct the group representation directly from the pure token embeddings: `norm(mean(norm(wte(tok_i)) for tok_i in group))`. This removes the need for group tokenizer infrastructure and keeps the intermediate state tied to the actual token embeddings.

## Pipeline

```
pure_token_id → wte → norm → [per group: mean of member embeddings] → norm → transformer → predict pure token
```

At block positions, instead of embedding a discrete group token ID, we sum the normalized embeddings of all pure tokens belonging to that group and normalize the result. The transformer then denoises this continuous group representation back to the exact pure token.

## Overhead

Recomputing the group embeddings from wte adds a small overhead per training step; at inference it's just a precomputed table lookup.

## Difference from Discrete Group Tokens

In experiment_c, each group has its own learned token ID in wte — the group embedding is independent of the pure token embeddings. Here, the group representation is derived directly from its members, so it naturally encodes what tokens are in the group and stays coupled to the pure embedding space.

## Noising

A pure token is noised by averaging its embedding with 63 randomly sampled token embeddings. The original token's signal is diluted — still present in the mix, but buried among 63 other vectors. The noise level is controlled by how many random tokens we mix in.

## Why This Experiment

Unlike experiment_c where groups are fixed clusters, here the noising is random — each token gets a different set of 63 random companions every time. There's no fixed group structure to memorize, just a noise level. The question is whether the transformer can learn to extract the original token's signal from an arbitrary random average.

## Mixing Method Alternatives

Current: `norm(sum(norm(emb)))` — confirmed working, 92% set_recall with noise_count=128.

1. **`norm(sum(emb))`** — drop inner norm. Token magnitude affects contribution.
2. **`sum(norm(emb))`** — drop outer norm. Result magnitude encodes token alignment (random vectors ~sqrt(N), similar vectors ~N). Free signal.
3. **`norm(max(norm(emb)))`** — element-wise max pool. Preserves salient features instead of averaging them out.
4. **`norm(sum(norm(emb)) + noise)`** — Gaussian noise before outer norm. Regularization.
