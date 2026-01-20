# PDLM Project Milestones

## Current Focus
1. **Tokenizer Refactor**:
   - Implement `TokenizerBuilder` for modular creation of MASK-only and hierarchical group tokenizers.
   - Standardize token map generation (`pure_to_noisy`, `noisy_level`).
2. **BD3LM Target Shift Experiments**:
   - Validate `target_shift` training logic (done).
   - Run depth-8 model training with varying `target_shift` (1-4).
   - Evaluate with new `eval_bd3lm.sh` script to compare `target_shift` variants against normal BD3LM.
3. **Evaluation**:
   - Ensure apples-to-apples comparison between AR, BD3LM (normal), and BD3LM (shifted).

---

## Historical / Archived Notes

### PDLM Project Milestones (Old)
-  Make a method that will, denoise in the steps we force it to.
-  Train group: predict the next n-th token

### Temporary
-  Make `generate_with_blocks` in `nanochat/pdlm.py` compatible with batch processing (B > 1) to speed up evaluation.
-  Train a group of 8d models on the `SimpleStories` dataset (Group: `simple_stories_8d`).
    -> we need to retrain the vocab stuff for each of these dataset i guess
    -> use the simple story dataset
    -> 8d, with bs 1,2,4,8,16; 512 sequence len; both causal and non-causal; param_ratio_30 ; same noisy
    -> 
1. we will train the tokenizer again, each of these two SimpleStory and TinyStory get their tokenizer
1. we will get the tokenizer with group tokens, also the tokenizer will just one more mask token
2. we will train two gpt2 model for each dataset in 8b

3. we are making norm_ar, bd3lm, pdlm all in pdlm_base_train 
-> inputs, targets are the most critical one here

for the bd3lm, the first model definitly be the normal bd3lm, then we will
try the bd3lm that will only predict one position, but when traiing, all other block position could be mask or unmask.

so there will be a new arg called `target\_shift' that this stuff is used for training the bd3lm right.

Target_shift should be within this range [1, block_size], and when the target_shift = k, when train: all other places could be mask or none_mask. When inference, all other places should be mask so that we controlled experiment against normal ar.

first impl bd3lm.py
then continue modify based on base_train file, here check both training and inference stuff.

### When spare
- Multi-machine for training multi model