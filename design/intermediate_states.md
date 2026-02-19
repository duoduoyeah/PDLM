
#

## Types of Intermediates

* Mask — Token becomes a special [MASK] symbol outside the vocabulary, giving zero information about the original token.

* Random Change — Token is replaced by another real token from the vocabulary. The corruption is the substitution itself. The replacement distribution can be uniform, ordinal-biased (discretized Gaussian), or embedding-biased (semantic proximity) — these are just different flavors of the transition matrix Q.

* Group Token — Token is replaced by a coarser-level label that represents a subset of the vocabulary (e.g., "this token is one of these 64"). Neither a real token nor a mask — it's a different level of representation that carries structured partial information.

* Group Embedding Average — Token is replaced by the mean of normalized embeddings of all tokens in its group: norm(mean(norm(wte(tok_i)) for tok_i in group)). A continuous-valued intermediate rather than a discrete symbol.

