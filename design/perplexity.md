# Perplexity

## Standard Perplexity (needs targets)

$$PPL = e^{-\frac{1}{N}\sum \log p(\text{correct token})}$$

"How surprised is the model by the right answer?"

## Entropy-based Perplexity (no targets needed)

$$PPL_{entropy} = e^{-\sum_v p(v) \log p(v)}$$

"How spread out is the model's probability distribution?"
