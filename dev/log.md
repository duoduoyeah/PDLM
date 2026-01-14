
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