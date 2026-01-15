01/15
2. loss = model(x, y, attn_mask=block_diff_mask) #TODO: different model different branch here i guess [Done]
3. make sure the bd3lm forward okay [Checked]
4. make sure the loss chain(target_shift stuff) [Checked] (verified with scripts/dump/dump_bd3lm_loss.py)
5. bd3lm inference methods (generate, eval_specify_position) need implementation/fixes [TODO]
6. we will use the same data multiple times because of shift, but this will be in a whole loop way, i mean, after we loop through all shards, then we will shift 1, and use the new mask [TODO]
7. we tend to ignore the prefix_pure_tokens of bd3lm when there is target_shift [Done] (added comment in dataloader.py)
8. BD3LM loss now only computed for MASKED positions [Done]
   - Fixed: dataloader passes mask in loss_extras, bd3lm.py uses it as attention_mask
   - Previously computed loss on all positions (trivial prediction for unmasked tokens)
9. Fair comparison of effective training tokens [Done]:
   - AR/PDLM: effective_tokens ≈ batch_size * seq_len (loss on ~all tokens)
   - BD3LM (normal, no target_shift): effective_tokens ≈ batch_size * seq_len * avg_mask_rate
     * With LogLinear schedule t~Uniform(0,1), avg_mask_rate ≈ 0.5
     * So BD3LM uses ~50% effective tokens per step vs AR/PDLM
   - BD3LM (with target_shift): even fewer effective tokens (only 1/block_size guaranteed masked)
   - Solution implemented:
     * Added `bd3lm_effective_ratio` parameter (default 0.5 for normal bd3lm)
     * When model_type=="bd3lm", num_iterations is adjusted by 1/bd3lm_effective_ratio
     * Track `total_effective_tokens` (actual masked positions contributing to loss)
     * Log effective tokens to wandb and report
     * Print actual effective ratio at end of training for verification 


01/14
1. bd3lm file:
    * config [Done]
    * the lm_head and wte shape -> all_vocab_size [TODO]
    * the inference method
    * check the attention part, should be similar to pdlm, not sure
2. pdlm_base_train
    * only pdlm has the pure_to_noisy_map stuff, while ar and bd3lm do not have
    * ...
3.dataloader
    * bd3lm also will use the target_shift, when target_shift is -1, bd3lm will just train a model that is the normal bd3lm
    * when shift, we also need to train four times longer since the loss only use 1 positions instead of 4 position? so we need to adjust this part by make sure we use the same amount of tokens to update the model weight.

but later, when we do the Experiment C, I guess we still need to do a target_shift pdlm? thus we also need to use such special mask and compute loss within a specific position. 

