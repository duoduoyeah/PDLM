01/21
1. Research proper k-overlap clustering methods (current impl uses topk post-hoc, may not be optimal) [TODO]
2. Created `group_tokenizer/` folder with builder, config, clustering, token_map modules [Done]
2. Moved `sp_tokens/` to `old_sp_tokens/` for reference [Done]
3. Modify PDLM lm_head to output `pure + group` tokens (no MASK) [TODO]
4. Test TokenizerBuilder on real 4096 embeddings [TODO]
5. Implement two-stage training loss (MASK→Group, Group→Pure) [TODO]
6. Per-group accuracy evaluation script [TODO]
7. Logit leakage analysis (check prob mass outside group) [TODO]
8. 

01/19~01/20
1. Fixed `prefix_sliding_tokens` synchronization bug between attention mask and data masking. [Done]
2. continue working on the tokenizer, need some rewrite of previous code [In Progress]
    * Refactor tokenizer creation into `TokenizerBuilder` for modularity.
    * Support multiple variants (MASK-only, hierarchical).
3. Cleanup: Removed `bd3lms/` folder and `CLAUDE.md` as part of repository cleanup. [Done]
4. Implemented suffix metrics in `bd3lm_eval.py` to evaluate prediction with revealed context. [Done]
5. Created `scripts/bd3lm_eval.py` for standalone checkpoint evaluation. [Done]
6. Added `bd3lm_compute_matched` arg (default=True) to control iteration adjustment vs compute-matched. [Done]
7. Updated `run_bd3lm.sh` to expose `bd3lm_compute_matched` and handle naming. [Done]
8. test bd3lm_eval.sh [Done]
9. train new 8d bd3lm [Done]
10. Enabled `target_shift` passing to `BDLMConfig` and `base_train.py`, and enhanced `eval_bd3lm.sh` to auto-detect `target_shift` from checkpoints and report results more clearly. [Done]
11. Added `--ckpt_dir` argument to `bd3lm_eval.py` and `eval_bd3lm.sh` to support direct checkpoint path specification, overriding `model_tag`. [Done]
12. Documented Stage 1 plan for single-layer group tokens in `sp_tokens_rewrite_plan.md` and added PDLM computational overhead analysis in `computation_overhead.md`. [Done]


01/18
1. Verified train/val split separation with new test script `tests/test_dataloader.py`. [Done]
2. Auto-compute `bd3lm_effective_ratio` from `target_shift` and `block_size`; added `rl_tok/sec` (real loss tokens/sec) metric to logging. [Done]
3. Bug fix: Corrected `loss_mask` usage in `dataloader.py`, `bd3lm.py`, and `base_train.py`. [Done]
   - Renamed `mask` -> `loss_mask` in `loss_extras` to clarify distinction between input mask and loss positions.
   - For `target_shift` mode, `loss_mask` now strictly includes only the forced position (1/block_size), fixing an issue where all masked positions were counting towards loss.
   - Result: `rl_tok/sec` for target_shift mode dropped from ~145k (incorrect, similar to normal) to ~58k (correct, 1/4 of total).
4. Cleanup: Removed redundant `prefix_pure_tokens` masking in `bd3lm.py` as it is now handled upstream in `dataloader.py`. [Done]
5. Added `ts3` variant to `launch/run_bd3lm.sh` to support predicting the 3rd position in each block. [Done]

01/17
1. Update `dataset.py` and `dataloader` to support validation shards. [Done]
2. Implement `bd3lm.eval_specify_position` logic (implemented via `bd3lm_eval.py`). [Done]
3. Hook evaluation logic into `base_train.py` (BD3LM active). [Done]
4. Fixed BD3LM masking: now guarantees at least 1 mask per block, t sampled from [1/block_size, 1], adjusted p' for remaining positions; normal and target_shift use same logic [Done]

01/16
1. Resolve conflict between target_shift and prefix_pure_tokens [Done]
   - Fixed: `dataloader.py` target_shift is now 1-indexed, and `prefix_pure_tokens` correctly overrides it.
   - Verified with `scripts/dump/dump_bd3lm_target_shift.py`.
2. Dry run of full BD3LM training loop to verify end-to-end stability [In Progress]

3. Cleanup and update evaluation logic in `base_train.py` [Done]
4. make sure the colab has the hf token [Done]
5. we need to test the bd3lm target shift on on depth 8, ratio 10 and only 1 shard i guess to make sure our loop is okay to run [TODO]
6. we need to still keep mask ratio around 50% for the target_shift, current it is apparent that more than 505 since we just force one pos to be MASK [Done]
7. answer the question that why bd3lm target_shift has larger loss? could be eval method difference [TODO]

01/15
2. loss = model(x, y, attn_mask=block_diff_mask) #TODO: different model different branch here i guess [Done]
3. make sure the bd3lm forward okay [Checked]
4. make sure the loss chain(target_shift stuff) [Checked] (verified with scripts/dump/dump_bd3lm_loss.py)
5. bd3lm inference methods (generate, eval_specify_position) need implementation/fixes [TODO]
6. we will use the same data multiple times because of shift, but this will be in a whole loop way, i mean, after we loop through all shards, then we will shift 1, and use the new mask [Done]
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

