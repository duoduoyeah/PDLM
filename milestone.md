## PDLM Project Milestones

-  Make a method that will, denoise in the steps we force it to.

-  Train group: predict the next n-th token


## Temporary 
-  Make `generate_with_blocks` in `nanochat/pdlm.py` compatible with batch processing (B > 1) to speed up evaluation.
-  Train a group of 8d models on the `SimpleStories` dataset (Group: `simple_stories_8d`).
    -> we need to retrain the vocab stuff for each of these dataset i guess
    -> use the simple story dataset
    -> 8d, with bs 1,2,4,8,16; 512 sequence len; both causal and non-causal; param_ratio_30 ; same noisy
    -> 
1. we will train the tokenizer again, each of these two SimpleStory and TinyStory get their tokenizer
1. we will get the tokenizer with group tokens, also the tokenizer will just one more mask token
2. we will train two gpt2 model for each dataset in 8b


## When spare
- Multi-machine for training multi model

