# Perplexity

## Standard Perplexity (needs targets)

$$PPL = e^{-\frac{1}{N}\sum \log p(\text{correct token})}$$

"How surprised is the model by the right answer?"

## Entropy-based Perplexity (no targets needed)

$$PPL_{entropy} = e^{-\sum_v p(v) \log p(v)}$$

"How spread out is the model's probability distribution?"

## Perplexity on Generated Sequences

# Use a separate pre-trained LM to score your generations
ppl = evaluate_with_external_LM(generated_sequences)
# Lower = more fluent/natural


## Embedding-Based Metrics

