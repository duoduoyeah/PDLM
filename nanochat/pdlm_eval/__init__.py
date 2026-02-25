"""PDLM Evaluation Module - supports stage1_mask, stage1_block, stage2, both_block, block_pdlm_inference, mask_pdlm."""

from nanochat.pdlm_eval.stage1_mask import eval_pdlm_stage1_mask
from nanochat.pdlm_eval.stage1_block import eval_pdlm_stage1_block, dump_stage1_block_batch
from nanochat.pdlm_eval.stage2 import (
    eval_pdlm,
    eval_pdlm_compatibility,
    eval_pdlm_oracle_accuracy,
    eval_pdlm_full,
)
from nanochat.pdlm_eval.both_block import eval_pdlm_both_block
from nanochat.pdlm_eval.block_pdlm_inference import eval_block_pdlm_inference
from nanochat.pdlm_eval.mask_pdlm import eval_mask_pdlm, eval_mask_pdlm_parallel, eval_mask_pdlm_refresh
from nanochat.pdlm_eval.pdlm_emb import eval_pdlm_emb
from nanochat.pdlm_eval.dump import dump_batch_to_file

__all__ = [
    "eval_pdlm_stage1_mask",
    "eval_pdlm_stage1_block",
    "eval_pdlm",
    "eval_pdlm_compatibility",
    "eval_pdlm_oracle_accuracy",
    "eval_pdlm_full",
    "eval_pdlm_both_block",
    "eval_block_pdlm_inference",
    "eval_mask_pdlm",
    "eval_mask_pdlm_parallel",
    "eval_mask_pdlm_refresh",
    "eval_pdlm_emb",
    "dump_batch_to_file",
    "dump_stage1_block_batch",
]
