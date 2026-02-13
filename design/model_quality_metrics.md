# Perplexity

## Research Background of the metrics that measure how confident the model's output distribution is

## Standard Perplexity (needs targets)

$$PPL = e^{-\frac{1}{N}\sum \log p(\text{correct token})}$$

"How surprised is the model by the right answer?"

## Entropy-based Perplexity (no targets needed)

$$PPL_{entropy} = e^{-\sum_v p(v) \log p(v)}$$

"How spread out is the model's probability distribution?"

## Average Log Probability (no targets needed)

$$\text{AvgLogProb} = \frac{1}{N}\sum_{i=1}^{N} \log p(s_i \mid s_{<i}, \mathbf{x})$$

Normalized log-likelihood of the generated sequence. Higher = model is more confident about the whole sequence. Directly related to perplexity: $PPL = e^{-\text{AvgLogProb}}$.

## Perplexity on Generated Sequences

# Use a separate pre-trained LM to score your generations
ppl = evaluate_with_external_LM(generated_sequences)
# Lower = more fluent/natural


## Embedding-Based Metrics

