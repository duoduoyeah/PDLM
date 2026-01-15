01/15
1. we will need to make sure there will be 4 masks for bs==4
2. loss = model(x, y, attn_mask=block_diff_mask) #TODO: different model different branch here i guess

01/14
1. bd3lm file:
    * config [Done]
    * the lm_head and wte shape -> all_vocab_size
    * the inference method
    * check the attention part, should be similar to pdlm, not sure
    *  
2. pdlm_base_train
    * only pdlm has the pure_to_noisy_map stuff, while ar and bd3lm do not have
    * ...
3.dataloader
    * bd3lm also will use the target_shift, when target_shift is -1, bd3lm will just train a model that is the normal bd3lm
    * when shift, we also need to train four times longer since the loss only use 1 positions instead of 4 position? so we need to adjust this part by make sure we use the same amount of tokens to update the model weight.

but later, when we do the Experiment C, I guess we still need to do a target_shift pdlm? thus we also need to use such special mask and compute loss within a specific position. 

