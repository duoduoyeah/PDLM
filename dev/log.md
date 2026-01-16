01/16
1. Resolve conflict between target_shift and prefix_pure_tokens [Done]
   - Fixed: `dataloader.py` target_shift is now 1-indexed, and `prefix_pure_tokens` correctly overrides it.
   - Verified with `scripts/dump/dump_bd3lm_target_shift.py`.

01/15
2. loss = model(x, y, attn_mask=block_diff_mask) #TODO: different model different branch here i guess [Done]
3. make sure the bd3lm forward okay [Checked]
4. make sure the loss chain(target_shift stuff) [Checked] (verified with scripts/dump/dump_bd3lm_loss.py)
5. bd3lm inference methods (generate, eval_specify_position) need implementation/fixes [TODO]
6. we will use the same data multiple times because of shift, but this will be in a whole loop way, i mean, after we loop through all shards, then we will shift 1, and use the new mask [TODO]
7. we tend to ignore the prefix_pure_tokens of bd3lm when there is target_shift [Done] (added comment in dataloader.py)
8. BD3LM loss now only computed for MASKED positions [Done]
9. Adjusted BD3LM iterations using `bd3lm_effective_ratio` to ensure fair comparison of total effective (masked) supervision tokens [Done].
10. Refactored attention mask to use bidirectional sliding prefix for target shift [Done].
11. Implemented BD3LM inference `generate` method (block-wise decoding) and refactored model init in `base_train.py` [Done].
12. we need to verify the dataloader work good with the mask change, A dump script or a dry run of the training loop would be appropriate here.[Done]

01/14
1. bd3lm file:
    * config [Done]
    * the lm_head and wte shape -> all_vocab_size [Done] (wte uses all_vocab, lm_head uses pure_vocab)
    * the inference method [Done] (implemented block-wise generate)
    * check the attention part, should be similar to pdlm, not sure
2. pdlm_base_train
    * only pdlm has the pure_to_noisy_map stuff, while ar and bd3lm do not have
    * ...
3.dataloader
    * bd3lm also will use the target_shift, when target_shift is -1, bd3lm will just train a model that is the normal bd3lm
    * when shift, we also need to train four times longer since the loss only use 1 positions instead of 4 position? so we need to adjust this part by make sure we use the same amount of tokens to update the model weight.

but later, when we do the Experiment C, I guess we still need to do a target_shift pdlm? thus we also need to use such special mask and compute loss within a specific position. 

