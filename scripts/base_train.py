"""
Train model.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
from contextlib import nullcontext

import wandb
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.pdlm import PDLM, PDLMConfig
from nanochat.bd3lm import BDLM, BDLMConfig
from nanochat.gpt_mtp import GPTMTP, GPTMTPConfig
from nanochat.dataloader import get_data_loader
from nanochat.bd3lm_eval import eval_bd3lm
from nanochat.pdlm_eval import eval_pdlm, eval_pdlm_stage1_mask, eval_pdlm_stage1_block, eval_pdlm_both_block, eval_pdlm_full, eval_block_pdlm_inference, eval_mask_pdlm, eval_pdlm_emb
from nanochat.mtp_eval import eval_mtp
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type
from nanochat.tokenizer import get_tokenizer
from nanochat.group_tokenizer.token_map import get_token_map
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.attn_masks import gen_mask, gen_block_causal_mask
print_banner()

# -----------------------------------------------------------------------------
# User settings
run = "dummy" # wandb run name default ("dummy" is special - we won't log to wandb)
wandb_group = None # wandb group
# Runtime
device_type = "" # cuda|cpu|mps (empty => autodetect good device type default, in order: CUDA > MPS > CPU)

# Model architecture
model_architecture = "Karpathy_gpt2"
model_type = "bd3lm"
target_shift = -1 # only ar and bd3lm: predict token this many steps ahead (1 = next-token)
depth = 20 # the depth of the Transformer model to train, rest of the kwargs are derived
max_seq_len = 1024 # max context length
block_size = 8 # the training use block size
prefix_pure_tokens = 1 # pure prefix tokens (0 = disabled)
is_causal = True # the model' attn direction

noise_total_steps = 16 # Noisy for pdlm
pdlm_stage = "stage2" # pdlm stage: stage1_mtp, stage1_mask, stage2, both_mtp, both_mask
bd3lm_effective_ratio = None # For bd3lm: auto-computed if None, or override with explicit value
bd3lm_compute_matched = True # If True, don't adjust iterations for BD3LM (compute-matched). If False, adjust to match loss tokens (supervision-matched).
mtp_loss_beta = 0.8 # MTP: exponential decay factor for loss weighting (β^k)
n_future_tokens = 4 # MTP/both_mtp: number of future group tokens to predict (K)
mtp_loss_weight = 1.0 # both_mtp: Stage 1 MTP loss weight relative to Stage 2
soft_p_within = 1.0 # stage2: prob of correct group mapping (1.0 = hard, <1.0 = soft noise)
noise_count = 64 # pdlm_emb: total tokens in noise average (including target)
mask_pdlm_4state = False # mask_pdlm: use 4-state variant (k states instead of k+1)
loss_weight_mode = "manual" # "manual" or "fixed" - fixed computes weight from warmup batches
loss_weight_warmup_steps = 10 # number of batches for estimating loss ratio (used when loss_weight_mode="fixed")
stage1_target_mode = "pure" # MTP: "pure" (default) or "group" (legacy) - determines target format and loss
# Debug
debug = False
# Training horizon. Only one of these 3 will be used, in this order of precedence.
num_iterations = -1 # explicit number of steps of the optimization (-1 = disable)
target_flops = -1.0 # calculate num_iterations to reach target_flops. Useful for scaling laws experiments (-1 = disable)
target_param_data_ratio = 20 # calculate num_iterations to maintain fixed data:param ratio (Chinchilla=20) (-1 = disable)
# Optimization
device_batch_size = 32 # per-device batch size (set to not OOM)
total_batch_size = 131072 # total desired batch size, in #tokens
embedding_lr = 0.2 # learning rate for the embedding parameters (Adam)
unembedding_lr = 0.004 # learning rate for the unembedding parameters (Adam)
weight_decay = 0.0 # weight decay for the embedding/unembedding parameters (Adam)
matrix_lr = 0.02 # learning rate for the matrix parameters (Muon)
grad_clip = 1.0 # gradient clipping value (0.0 = disabled)
warmup_ratio = 0.0 # ratio of iterations for LR warmup
warmdown_ratio = 0.2 # ratio of iterations for LR warmdown
final_lr_frac = 0.0 # final LR is this fraction of the initial LR
resume_from_step = -1 # resume training from this step of the optimization (-1 = disable)
# Evaluation
eval_every = -1 # evaluate every N steps (-1 = disable)
eval_num_batches = 20 # number of batches for intermediate evaluation (quick)
eval_num_batches_final = 100 # number of batches for final evaluation (thorough)
save_every = -1 # every how many steps to save model checkpoints (-1 = disable, and save only at the end of the run)
# Gradient tracking
gradient_track_every = 0    # gradient tracking every N steps (0 = disabled)
gradient_block_size = 4     # virtual block size for GPT gradient tracking
# Output
model_tag = "" # optionally override the model tag for the output checkpoint directory name
# now allow CLI to override the settings via the configurator lol
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read()) # overrides from command line or config file
user_config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------
assert 0 <= prefix_pure_tokens <= block_size <= max_seq_len, "Expected prefix_pure_tokens <= block_size <= max_seq_len"
assert model_type in {"next_token_ar", "bd3lm", "pdlm", "mtp"}, f"Invalid model_type: {model_type}"


# Compute init
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# wandb logging init
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=run, group=wandb_group ,config=user_config)

# Tokenizer will be useful for evaluation, also we need the vocab size
tokenizer = get_tokenizer()
all_vocab_size = tokenizer.get_vocab_size()
if model_type == "next_token_ar":
    pure_vocab_size = all_vocab_size
    token_map = None
elif model_type == "bd3lm":
    pure_vocab_size = all_vocab_size - 1  # MASK is the only extra token
    token_map = None
elif model_type == "pdlm":
    if pdlm_stage == "pdlm_emb":
        token_map = None
        pure_vocab_size = all_vocab_size
        num_groups = 0
    else:
        token_map = get_token_map(device="cpu")
        pure_vocab_size = token_map.pure_vocab_size
        num_groups = token_map.num_groups
elif model_type == "mtp":
    token_map = get_token_map(device="cpu")
    pure_vocab_size = token_map.pure_vocab_size
    num_groups = token_map.num_groups
else:
    raise ValueError(f"Unknown model_type: {model_type}")
assert pure_vocab_size <= all_vocab_size, "pure_vocab_size should not exceed all_vocab_size"
mask_token_id = -1
try:
    maybe_mask_token_id = tokenizer.encode_special("<|MASK|>")
    if maybe_mask_token_id is not None:
        mask_token_id = maybe_mask_token_id
except KeyError:
    pass
print0(f"Vocab size: {all_vocab_size:,}")
print0(f"Pure vocab size: {pure_vocab_size:,}")
if model_type in {"pdlm", "mtp"} and num_groups > 0:
    print0(f"Num groups: {num_groups:,}")
if mask_token_id != -1:
    print0(f"Mask token id: {mask_token_id}")

# Model kwargs are derived from the desired depth of the model
num_layers = depth
model_dim = depth * 64 # aspect ratio 64 (usually this is varied from 64 -> 128 as model size increases)
num_heads = max(1, (model_dim + 127) // 128) # head dim 128 (the division here is ceil div)
num_kv_heads = num_heads # default is 1:1 GQA (Group Query Attention) ratio (i.e. GQA is disabled)
print0(f"num_layers: {num_layers}")
print0(f"model_dim: {model_dim}")
print0(f"num_heads: {num_heads}")
print0(f"num_kv_heads: {num_kv_heads}")

# Optimizer / data / training length related hyperparameters
# figure out the needed gradient accumulation to reach the desired total batch size
tokens_per_fwdbwd = device_batch_size * max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank (L): {device_batch_size} x {max_seq_len} = {tokens_per_fwdbwd:,}")
tokens_per_fwdbwd_twice = device_batch_size * max_seq_len * 2
print0(f"Tokens / micro-batch / rank (2L): {device_batch_size} x {max_seq_len * 2} = {tokens_per_fwdbwd_twice:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# -----------------------------------------------------------------------------
# Initialize the Model

# Create model config based on model_type
if model_type == "next_token_ar":
    ModelConfig, Model = GPTConfig, GPT
    model_config_kwargs = dict(
        sequence_len=max_seq_len,
        vocab_size=all_vocab_size,
        n_layer=num_layers,
        n_head=num_heads,
        n_kv_head=num_kv_heads,
        n_embd=model_dim,
    )
elif model_type == "bd3lm":
    ModelConfig, Model = BDLMConfig, BDLM
    model_config_kwargs = dict(
        sequence_len=max_seq_len,
        pure_vocab_size=pure_vocab_size,
        all_vocab_size=all_vocab_size,
        n_layer=num_layers,
        n_head=num_heads,
        n_kv_head=num_kv_heads,
        n_embd=model_dim,
        prefix_pure_tokens=prefix_pure_tokens,
        mask_token_id=mask_token_id,
        is_causal=is_causal,
        bucket_size=block_size,
        model_name=run,
        target_shift=target_shift,
    )
elif model_type == "pdlm":
    ModelConfig, Model = PDLMConfig, PDLM
    model_config_kwargs = dict(
        sequence_len=max_seq_len,
        pure_vocab_size=pure_vocab_size,
        num_groups=num_groups,
        stage=pdlm_stage,
        n_layer=num_layers,
        n_head=num_heads,
        n_kv_head=num_kv_heads,
        n_embd=model_dim,
        prefix_pure_tokens=prefix_pure_tokens,
        mask_token_id=mask_token_id,
        is_causal=is_causal,
        bucket_size=block_size,
        model_name=run,
        soft_p_within=soft_p_within,
        stage1_target_mode=stage1_target_mode,
    )
    # Add loss weight for combined stages
    if pdlm_stage in ("both_mtp", "both_block", "block_pdlm_inference"):
        model_config_kwargs["mtp_loss_weight"] = mtp_loss_weight
    # Add noise_count for pdlm_emb stage
    if pdlm_stage == "pdlm_emb":
        model_config_kwargs["noise_count"] = noise_count
    # Add MTP-specific params for both_mtp stage
    if pdlm_stage == "both_mtp":
        model_config_kwargs.update(
            n_future_tokens=n_future_tokens,
            mtp_loss_beta=mtp_loss_beta,
        )
    # Add 4-state flag for mask_pdlm stage
    if pdlm_stage == "mask_pdlm":
        model_config_kwargs["mask_pdlm_4state"] = mask_pdlm_4state
elif model_type == "mtp":
    ModelConfig, Model = GPTMTPConfig, GPTMTP
    model_config_kwargs = dict(
        sequence_len=max_seq_len,
        pure_vocab_size=pure_vocab_size,
        num_groups=num_groups,
        n_future_tokens=block_size,  # predict block_size tokens
        mtp_loss_beta=mtp_loss_beta,
        n_layer=num_layers,
        n_head=num_heads,
        n_kv_head=num_kv_heads,
        n_embd=model_dim,
        stage1_target_mode=stage1_target_mode,
    )
else:
    raise ValueError(f"Unknown model_type: {model_type}")
with torch.device("meta"):
    model_config = ModelConfig(**model_config_kwargs)
    model = Model(model_config)
model.to_empty(device=device)
model.init_weights()

# Register group mask for MTP pure mode (needed for inference collapse)
if model_type == "mtp" and stage1_target_mode == "pure":
    model.register_group_mask(token_map.group_to_pure_mask.to(device=device))
    print0(f"Registered group_to_pure_mask for MTP pure mode ({token_map.num_groups} groups × {token_map.pure_vocab_size} pure tokens)")

# Register group mask for stage1_block pure mode (needed for eval group collapse)
if model_type == "pdlm" and pdlm_stage == "stage1_block" and stage1_target_mode == "pure":
    model.register_group_mask(token_map.group_to_pure_mask.to(device=device))
    print0(f"Registered group_to_pure_mask for stage1_block ({token_map.num_groups} groups × {token_map.pure_vocab_size} pure tokens)")

# Register group mask for block_pdlm_inference (needed for inference collapse)
if model_type == "pdlm" and pdlm_stage == "block_pdlm_inference":
    model.register_group_mask(token_map.group_to_pure_mask.to(device=device))
    print0(f"Registered group_to_pure_mask for block_pdlm_inference ({token_map.num_groups} groups × {token_map.pure_vocab_size} pure tokens)")

# Register group mask for mask_pdlm (needed for inference collapse)
if model_type == "pdlm" and pdlm_stage == "mask_pdlm":
    model.register_group_mask(token_map.group_to_pure_mask.to(device=device))
    print0(f"Registered group_to_pure_mask for mask_pdlm ({token_map.num_groups} groups × {token_map.pure_vocab_size} pure tokens)")

# Generate attention masks
# For BD3LM: pre-generate block_size masks for prefix_sliding_tokens cycling (both normal and target_shift modes)
# For other model types: single mask with prefix_sliding_tokens = 0
if model_type == "bd3lm":
    # Pre-generate all masks for cycling prefix_sliding_tokens = 0, 1, ..., block_size-1
    # This ensures block boundaries shift each epoch for better data utilization
    block_diff_masks = [
        gen_mask(max_seq_len, block_size, attn_backend="sdpa", is_causal=is_causal, prefix_sliding_tokens=i).to(device=device)
        for i in range(block_size)
    ]
    print0(f"Pre-generated {block_size} attention masks for prefix_sliding_tokens cycling")
elif model_type == "pdlm" and pdlm_stage == "stage1_block":
    # stage1_block uses L×L block-causal mask (no 2L structure)
    block_diff_masks = [gen_block_causal_mask(max_seq_len, block_size, attn_backend="sdpa", is_causal=is_causal).to(device=device)]
else:
    # Single mask with prefix_sliding_tokens = 0
    block_diff_masks = [gen_mask(max_seq_len, block_size, attn_backend="sdpa", is_causal=is_causal, prefix_sliding_tokens=0).to(device=device)]

# If we are resuming, overwrite the model parameters with those of the checkpoint
base_dir = get_base_dir()
output_dirname = model_tag if model_tag else f"d{depth}" # e.g. d12
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = resume_from_step != -1
if resuming:
    print0(f"Resuming optimization from step {resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data # free up this memory after the copy

orig_model = model # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
model = torch.compile(model, dynamic=False) # the inputs to model will never change shape so dynamic=False is safe
num_params = sum(p.numel() for p in model.parameters())
print0(f"Number of parameters: {num_params:,}")
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# Calculate number of iterations. Either it is given, or from target flops, or from target data:param ratio (in that order)
assert num_iterations > 0 or target_param_data_ratio > 0 or target_flops > 0
if num_iterations > 0:
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif target_flops > 0:
    # calculate the number of iterations from the target flops
    num_iterations = round(target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif target_param_data_ratio > 0:
    # calculate the number of iterations from the target param data ratio
    target_tokens = target_param_data_ratio * num_params
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")

# For BD3LM: optionally adjust iterations to account for lower effective token ratio
# BD3LM only computes loss on masked positions
# - bd3lm_compute_matched=True (default): no adjustment, compare at equal compute/FLOPs
# - bd3lm_compute_matched=False: adjust iterations to match loss tokens (supervision-matched)
if model_type == "bd3lm":
    # Auto-compute bd3lm_effective_ratio if not specified
    if bd3lm_effective_ratio is None:
        if target_shift >= 1:
            # Target_shift mode: exactly 1 position per block is masked
            bd3lm_effective_ratio = 1.0 / block_size
        else:
            # Normal mode: t ~ Uniform[1/block_size, 1], E[t] = (1/block_size + 1) / 2
            bd3lm_effective_ratio = (1.0 / block_size + 1.0) / 2.0
        print0(f"BD3LM auto-computed effective_ratio={bd3lm_effective_ratio:.4f} (target_shift={target_shift}, block_size={block_size})")

    if bd3lm_compute_matched:
        print0(f"BD3LM compute_matched=True: no iteration adjustment (comparing at equal FLOPs)")
    elif bd3lm_effective_ratio < 1.0:
        original_iterations = num_iterations
        num_iterations = int(num_iterations / bd3lm_effective_ratio)
        print0(f"BD3LM compute_matched=False, effective_ratio={bd3lm_effective_ratio:.4f} => adjusted iterations: {original_iterations:,} -> {num_iterations:,}")
total_tokens = total_batch_size * num_iterations
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Params ratio: {total_batch_size * num_iterations / num_params:.2f}") # Chinchilla is ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# -----------------------------------------------------------------------------
# Initialize the Optimizer (Muon for Linear layers, AdamW for embedding and lm_head)
optimizers = model.setup_optimizers(unembedding_lr=unembedding_lr, embedding_lr=embedding_lr, matrix_lr=matrix_lr, weight_decay=weight_decay)
adamw_optimizer, muon_optimizer = optimizers

if resuming:
    for opt, dat in zip(optimizers, optimizer_data):
        opt.load_state_dict(dat)
    del optimizer_data # free up the memory

# -----------------------------------------------------------------------------
# Initialize the DataLoaders for train/val
tokens_dir = os.path.join(base_dir, "tokenized_data")
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]

train_loader = get_data_loader(
    device_batch_size,
    max_seq_len,
    split="train",
    device=device,
    model_config=model_config,
    resume_state_dict=dataloader_resume_state_dict,
)
x, y, loss_extras, dataloader_state_dict = next(train_loader) # kick off load of the very first batch of data

# Initialize validation dataloader and eval-specific resources (model-type specific)
val_loader = None
eval_attn_mask = None
if eval_every > 0:
    if model_type == "bd3lm":
        val_loader = get_data_loader(
            device_batch_size,
            max_seq_len,
            split="val",
            device=device,
            model_config=model_config,
            resume_state_dict=None,  # always start fresh for validation
        )
        # Eval uses prefix_sliding_tokens=0 (no sliding prefix for eval)
        eval_attn_mask = gen_mask(max_seq_len, block_size, attn_backend="sdpa", is_causal=is_causal, prefix_sliding_tokens=0).to(device=device)
        print0(f"Initialized validation dataloader and eval attention mask for BD3LM evaluation")
    elif model_type == "pdlm":
        val_loader = get_data_loader(
            device_batch_size,
            max_seq_len,
            split="val",
            device=device,
            model_config=model_config,
            resume_state_dict=None,  # always start fresh for validation
        )
        # Eval mask: L×L for stage1_block, 2L×2L for other stages
        if pdlm_stage == "stage1_block":
            eval_attn_mask = gen_block_causal_mask(max_seq_len, block_size, attn_backend="sdpa", is_causal=is_causal).to(device=device)
        else:
            eval_attn_mask = gen_mask(max_seq_len, block_size, attn_backend="sdpa", is_causal=is_causal, prefix_sliding_tokens=0).to(device=device)
        print0(f"Initialized validation dataloader and eval attention mask for PDLM evaluation")
    elif model_type == "next_token_ar":
        pass  # TODO: AR validation setup
    elif model_type == "mtp":
        val_loader = get_data_loader(
            device_batch_size,
            max_seq_len,
            split="val",
            device=device,
            model_config=model_config,
            resume_state_dict=None,  # always start fresh for validation
        )
        # MTP uses standard causal attention, no special eval mask needed
        print0(f"Initialized validation dataloader for MTP evaluation")

debug_dump_path = None
if debug:
    debug_dir = os.path.join(os.getcwd(), "temp")
    os.makedirs(debug_dir, exist_ok=True)
    debug_dump_path = os.path.join(debug_dir, "pdlm_debug_xy.txt")

# -----------------------------------------------------------------------------
# Compute fixed loss weight if requested (for both_block stage)
if model_type == "pdlm" and pdlm_stage in ("both_block", "block_pdlm_inference") and loss_weight_mode == "fixed":
    print0(f"Computing fixed loss weight from {loss_weight_warmup_steps} warmup batches...")
    total_s1, total_s2, count = 0.0, 0.0, 0

    with torch.no_grad():
        for _ in range(loss_weight_warmup_steps):
            with autocast_ctx:
                _, s1_loss, s2_loss = model(x, y, attn_mask=block_diff_masks[0],
                                            loss_extras=loss_extras, return_separate_losses=True)
            total_s1 += s1_loss.item()
            total_s2 += s2_loss.item()
            count += 1
            x, y, loss_extras, dataloader_state_dict = next(train_loader)

    avg_s1 = total_s1 / count
    avg_s2 = total_s2 / count
    computed_weight = avg_s2 / avg_s1
    orig_model.config.mtp_loss_weight = computed_weight
    print0(f"  Stage 1 avg loss: {avg_s1:.4f}")
    print0(f"  Stage 2 avg loss: {avg_s2:.4f}")
    print0(f"  Computed mtp_loss_weight: {computed_weight:.4f}")

# -----------------------------------------------------------------------------
# Set up hyperparameter schedulers

# Learning rate scheduler
def get_lr_multiplier(it):
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac

# Momentum scheduler for Muon optimizer
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# -----------------------------------------------------------------------------
# Loop state (variables updated by the training loop)

if not resuming:
    step = 0
    smooth_train_loss = 0 # EMA of training loss
    total_training_time = 0 # total wall-clock time of training
    total_effective_tokens = 0 # for bd3lm: actual masked tokens that contribute to loss
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]
    total_effective_tokens = loop_state.get("total_effective_tokens", 0)

# -----------------------------------------------------------------------------
# Training loop
while True:
    last_step = step == num_iterations # loop runs num_iterations+1 times so that we can eval/save at the end
    flops_so_far = num_flops_per_token * total_batch_size * step

    # Evaluation (model-type specific)
    if eval_every > 0 and (last_step or (step > 0 and step % eval_every == 0)):
        # Use more batches for final evaluation
        current_eval_batches = eval_num_batches_final if last_step else eval_num_batches
        if model_type == "bd3lm":
            print0(f"Running BD3LM evaluation at step {step} ({current_eval_batches} batches)...")
            eval_result = eval_bd3lm(
                model=orig_model,  # use uncompiled model
                val_loader=val_loader,
                block_size=block_size,
                target_shift=target_shift,
                num_batches=current_eval_batches,
                attn_mask=eval_attn_mask,
                device=device,
                autocast_ctx=autocast_ctx,
                mask_token_id=mask_token_id,
            )
            # Log eval results
            if target_shift >= 1:
                # Core metrics (all masked)
                print0(f"  [target_shift={target_shift}] loss: {eval_result['loss']:.4f}, ppl: {eval_result['ppl']:.2f}")
                log_data = {
                    "step": step,
                    "eval/loss": eval_result["loss"],
                    "eval/ppl": eval_result["ppl"],
                }
                # Suffix metrics
                max_suffix = block_size - target_shift
                for s in range(1, max_suffix + 1):
                    key_loss = f"loss_{s}suffix"
                    key_ppl = f"ppl_{s}suffix"
                    if key_loss in eval_result:
                        print0(f"    {s} suffix clear: loss={eval_result[key_loss]:.4f}, ppl={eval_result[key_ppl]:.2f}")
                        log_data[f"eval/loss_{s}suffix"] = eval_result[key_loss]
                        log_data[f"eval/ppl_{s}suffix"] = eval_result[key_ppl]
                wandb_run.log(log_data)
            else:
                # Core metrics (all masked)
                print0(f"  [normal mode] overall_loss: {eval_result['overall_loss']:.4f}, overall_ppl: {eval_result['overall_ppl']:.2f}")
                # Position-centric printing: each position on one line with all suffix metrics
                for pos in range(block_size):
                    pos_data = eval_result["positions"][pos]
                    line = f"    pos {pos}: loss={pos_data['loss']:.2f}, ppl={pos_data['ppl']:.1f}"
                    # Append suffix metrics for this position
                    max_suffix_for_pos = block_size - 1 - pos
                    for s in range(1, max_suffix_for_pos + 1):
                        if f"loss_{s}suffix" in pos_data:
                            line += f" | {s}s={pos_data[f'loss_{s}suffix']:.2f}/{pos_data[f'ppl_{s}suffix']:.1f}"
                    print0(line)
                # Build log data for wandb
                log_data = {
                    "step": step,
                    "eval/overall_loss": eval_result["overall_loss"],
                    "eval/overall_ppl": eval_result["overall_ppl"],
                }
                for pos in range(block_size):
                    pos_data = eval_result["positions"][pos]
                    log_data[f"eval/pos_{pos}_loss"] = pos_data["loss"]
                    log_data[f"eval/pos_{pos}_ppl"] = pos_data["ppl"]
                    # Log suffix metrics per position
                    max_suffix_for_pos = block_size - 1 - pos
                    for s in range(1, max_suffix_for_pos + 1):
                        if f"loss_{s}suffix" in pos_data:
                            log_data[f"eval/pos_{pos}_loss_{s}suffix"] = pos_data[f"loss_{s}suffix"]
                            log_data[f"eval/pos_{pos}_ppl_{s}suffix"] = pos_data[f"ppl_{s}suffix"]
                # Also log overall suffix metrics
                for s, suffix_data in eval_result.get("suffix_overall", {}).items():
                    log_data[f"eval/suffix_{s}_overall_loss"] = suffix_data["overall_loss"]
                    log_data[f"eval/suffix_{s}_overall_ppl"] = suffix_data["overall_ppl"]
                wandb_run.log(log_data)
        elif model_type == "pdlm":
            if pdlm_stage == "stage1_mask":
                print0(f"Running PDLM Stage 1 MASK evaluation at step {step} ({current_eval_batches} batches)...")
                eval_result = eval_pdlm_stage1_mask(
                    model=orig_model,  # use uncompiled model
                    val_loader=val_loader,
                    block_size=block_size,
                    num_batches=current_eval_batches,
                    attn_mask=eval_attn_mask,
                    device=device,
                    autocast_ctx=autocast_ctx,
                    prefix_pure_tokens=prefix_pure_tokens,
                )
                # Log eval results
                print0(f"  [pdlm stage1_mask] overall_loss: {eval_result['overall_loss']:.4f}, overall_ppl: {eval_result['overall_ppl']:.2f}, overall_accuracy: {eval_result['overall_accuracy']:.2%}")
                # Per-position metrics
                for pos in range(block_size):
                    pos_data = eval_result["positions"][pos]
                    print0(f"    pos {pos}: loss={pos_data['loss']:.4f}, ppl={pos_data['ppl']:.2f}, accuracy={pos_data['accuracy']:.2%}")
                # Build log data for wandb
                log_data = {
                    "step": step,
                    "eval/overall_loss": eval_result["overall_loss"],
                    "eval/overall_ppl": eval_result["overall_ppl"],
                    "eval/overall_accuracy": eval_result["overall_accuracy"],
                }
                for pos in range(block_size):
                    pos_data = eval_result["positions"][pos]
                    log_data[f"eval/pos_{pos}_loss"] = pos_data["loss"]
                    log_data[f"eval/pos_{pos}_ppl"] = pos_data["ppl"]
                    log_data[f"eval/pos_{pos}_accuracy"] = pos_data["accuracy"]
                wandb_run.log(log_data)
            elif pdlm_stage == "stage1_block":
                print0(f"Running PDLM Stage 1 Block evaluation at step {step} ({current_eval_batches} batches)...")
                eval_result = eval_pdlm_stage1_block(
                    model=orig_model,
                    val_loader=val_loader,
                    block_size=block_size,
                    num_batches=current_eval_batches,
                    attn_mask=eval_attn_mask,
                    device=device,
                    autocast_ctx=autocast_ctx,
                    prefix_pure_tokens=prefix_pure_tokens,
                )
                print0(f"  [pdlm stage1_block] overall_loss: {eval_result['overall_loss']:.4f}, overall_ppl: {eval_result['overall_ppl']:.2f}, overall_accuracy: {eval_result['overall_accuracy']:.2%}")
                for pos in range(block_size):
                    pos_data = eval_result["positions"][pos]
                    print0(f"    pos {pos}: loss={pos_data['loss']:.4f}, ppl={pos_data['ppl']:.2f}, accuracy={pos_data['accuracy']:.2%}")
                log_data = {
                    "step": step,
                    "eval/overall_loss": eval_result["overall_loss"],
                    "eval/overall_ppl": eval_result["overall_ppl"],
                    "eval/overall_accuracy": eval_result["overall_accuracy"],
                }
                for pos in range(block_size):
                    pos_data = eval_result["positions"][pos]
                    log_data[f"eval/pos_{pos}_loss"] = pos_data["loss"]
                    log_data[f"eval/pos_{pos}_ppl"] = pos_data["ppl"]
                    log_data[f"eval/pos_{pos}_accuracy"] = pos_data["accuracy"]
                wandb_run.log(log_data)
            elif pdlm_stage == "both_block":
                print0(f"Running PDLM both_block evaluation at step {step} ({current_eval_batches} batches)...")
                eval_result = eval_pdlm_both_block(
                    model=orig_model,
                    val_loader=val_loader,
                    block_size=block_size,
                    num_batches=current_eval_batches,
                    attn_mask=eval_attn_mask,
                    device=device,
                    autocast_ctx=autocast_ctx,
                    prefix_pure_tokens=prefix_pure_tokens,
                    mtp_loss_weight=mtp_loss_weight,
                    run_compatibility=False,
                    run_oracle_accuracy=False,
                )
                s1 = eval_result["stage1"]
                s2 = eval_result["stage2"]
                print0(f"  [block_pdlm] combined_loss: {eval_result['combined_loss']:.4f}, end2end_loss: {eval_result['end2end_loss']:.4f}")
                print0(f"    mtp_loss_weight: {eval_result['mtp_loss_weight']}, stage1_loss: {s1['overall_loss']:.4f}, stage2_loss: {s2['overall_loss']:.4f}")
                print0(f"    Stage 1 (Block→Block): loss={s1['overall_loss']:.4f}, ppl={s1['overall_ppl']:.2f}, accuracy={s1['overall_accuracy']:.2%}")
                for pos in range(block_size):
                    pos_data = s1["positions"][pos]
                    print0(f"      pos {pos}: loss={pos_data['loss']:.4f}, ppl={pos_data['ppl']:.2f}, accuracy={pos_data['accuracy']:.2%}")
                print0(f"    Stage 2 (Denoise): loss={s2['overall_loss']:.4f}, ppl={s2['overall_ppl']:.2f}")
                for pos in range(block_size):
                    pos_data = s2["positions"][pos]
                    print0(f"      pos {pos}: loss={pos_data['loss']:.4f}, ppl={pos_data['ppl']:.2f}")
                e2e = eval_result["end2end"]
                print0(f"    End-to-End (Stage1→Stage2): loss={e2e['overall_loss']:.4f}, ppl={e2e['overall_ppl']:.2f}")
                for pos in range(block_size):
                    pos_data = e2e["positions"][pos]
                    print0(f"      pos {pos}: loss={pos_data['loss']:.4f}, ppl={pos_data['ppl']:.2f}")
                log_data = {
                    "step": step,
                    "eval/combined_loss": eval_result["combined_loss"],
                    "eval/end2end_loss": eval_result["end2end_loss"],
                    "eval/end2end_ppl": e2e["overall_ppl"],
                    "eval/stage1_loss": s1["overall_loss"],
                    "eval/stage1_ppl": s1["overall_ppl"],
                    "eval/stage1_accuracy": s1["overall_accuracy"],
                    "eval/stage2_loss": s2["overall_loss"],
                    "eval/stage2_ppl": s2["overall_ppl"],
                }
                for pos in range(block_size):
                    s1p = s1["positions"][pos]
                    log_data[f"eval/stage1_pos_{pos}_loss"] = s1p["loss"]
                    log_data[f"eval/stage1_pos_{pos}_ppl"] = s1p["ppl"]
                    log_data[f"eval/stage1_pos_{pos}_accuracy"] = s1p["accuracy"]
                    s2p = s2["positions"][pos]
                    log_data[f"eval/stage2_pos_{pos}_loss"] = s2p["loss"]
                    log_data[f"eval/stage2_pos_{pos}_ppl"] = s2p["ppl"]
                    e2ep = e2e["positions"][pos]
                    log_data[f"eval/end2end_pos_{pos}_loss"] = e2ep["loss"]
                    log_data[f"eval/end2end_pos_{pos}_ppl"] = e2ep["ppl"]
                wandb_run.log(log_data)
            elif pdlm_stage == "block_pdlm_inference":
                print0(f"Running block_pdlm_inference evaluation at step {step} ({current_eval_batches} batches)...")
                eval_result = eval_block_pdlm_inference(
                    model=orig_model,
                    val_loader=val_loader,
                    block_size=block_size,
                    num_batches=current_eval_batches,
                    attn_mask=eval_attn_mask,
                    device=device,
                    autocast_ctx=autocast_ctx,
                    prefix_pure_tokens=prefix_pure_tokens,
                    mtp_loss_weight=mtp_loss_weight,
                )
                s1 = eval_result["stage1"]
                s2 = eval_result["stage2"]
                e2e = eval_result["end2end"]
                print0(f"  [block_pdlm_inference] combined_loss: {eval_result['combined_loss']:.4f}, end2end_loss: {eval_result['end2end_loss']:.4f}")
                print0(f"    mtp_loss_weight: {eval_result['mtp_loss_weight']}, stage1_loss: {s1['overall_loss']:.4f}, stage2_loss: {s2['overall_loss']:.4f}")
                print0(f"    Stage 1 (Pure→Pure): loss={s1['overall_loss']:.4f}, ppl={s1['overall_ppl']:.2f}")
                for pos in range(block_size):
                    pos_data = s1["positions"][pos]
                    print0(f"      pos {pos}: loss={pos_data['loss']:.4f}, ppl={pos_data['ppl']:.2f}")
                print0(f"    Stage 2 (Mixed→Pure): loss={s2['overall_loss']:.4f}, ppl={s2['overall_ppl']:.2f}")
                pure_bk = s2.get("pure_breakdown", {})
                group_bk = s2.get("group_breakdown", {})
                for pos in range(block_size):
                    pos_data = s2["positions"][pos]
                    parts = [f"pos {pos}: loss={pos_data['loss']:.4f}"]
                    if pos in pure_bk:
                        parts.append(f"pure={pure_bk[pos]['loss']:.4f}")
                    g_parts = []
                    for g in range(pos + 1):
                        if pos in group_bk and g in group_bk[pos]:
                            g_parts.append(f"g{g}={group_bk[pos][g]['loss']:.4f}")
                    if g_parts:
                        parts.append("group[" + " ".join(g_parts) + "]")
                    print0(f"      {' | '.join(parts)}")
                print0(f"    End-to-End (Iterative): loss={e2e['overall_loss']:.4f}, ppl={e2e['overall_ppl']:.2f}")
                for pos in range(block_size):
                    pos_data = e2e["positions"][pos]
                    print0(f"      pos {pos}: loss={pos_data['loss']:.4f}, ppl={pos_data['ppl']:.2f}")
                log_data = {
                    "step": step,
                    "eval/combined_loss": eval_result["combined_loss"],
                    "eval/end2end_loss": eval_result["end2end_loss"],
                    "eval/end2end_ppl": e2e["overall_ppl"],
                    "eval/stage1_loss": s1["overall_loss"],
                    "eval/stage1_ppl": s1["overall_ppl"],
                    "eval/stage2_loss": s2["overall_loss"],
                    "eval/stage2_ppl": s2["overall_ppl"],
                }
                for pos in range(block_size):
                    s1p = s1["positions"][pos]
                    log_data[f"eval/stage1_pos_{pos}_loss"] = s1p["loss"]
                    log_data[f"eval/stage1_pos_{pos}_ppl"] = s1p["ppl"]
                    s2p = s2["positions"][pos]
                    log_data[f"eval/stage2_pos_{pos}_loss"] = s2p["loss"]
                    log_data[f"eval/stage2_pos_{pos}_ppl"] = s2p["ppl"]
                    # Stage2 breakdown logging
                    if pos in pure_bk:
                        log_data[f"eval/stage2_pos_{pos}_pure_loss"] = pure_bk[pos]["loss"]
                    for g in range(pos + 1):
                        if pos in group_bk and g in group_bk[pos]:
                            log_data[f"eval/stage2_pos_{pos}_group_g{g}_loss"] = group_bk[pos][g]["loss"]
                    e2ep = e2e["positions"][pos]
                    log_data[f"eval/end2end_pos_{pos}_loss"] = e2ep["loss"]
                    log_data[f"eval/end2end_pos_{pos}_ppl"] = e2ep["ppl"]
                wandb_run.log(log_data)
            elif pdlm_stage == "pdlm_emb":
                print0(f"Running pdlm_emb evaluation at step {step} ({current_eval_batches} batches)...")
                eval_result = eval_pdlm_emb(
                    model=orig_model,
                    val_loader=val_loader,
                    block_size=block_size,
                    num_batches=current_eval_batches,
                    attn_mask=eval_attn_mask,
                    device=device,
                    autocast_ctx=autocast_ctx,
                    prefix_pure_tokens=prefix_pure_tokens,
                )
                print0(f"  [pdlm_emb] overall_loss: {eval_result['overall_loss']:.4f}, overall_ppl: {eval_result['overall_ppl']:.2f}")
                for pos in range(block_size):
                    pos_data = eval_result["positions"][pos]
                    print0(f"    pos {pos}: loss={pos_data['loss']:.4f}, ppl={pos_data['ppl']:.2f}")
                log_data = {
                    "step": step,
                    "eval/overall_loss": eval_result["overall_loss"],
                    "eval/overall_ppl": eval_result["overall_ppl"],
                }
                for pos in range(block_size):
                    pos_data = eval_result["positions"][pos]
                    log_data[f"eval/pos_{pos}_loss"] = pos_data["loss"]
                    log_data[f"eval/pos_{pos}_ppl"] = pos_data["ppl"]
                wandb_run.log(log_data)
            elif pdlm_stage == "mask_pdlm":
                print0(f"Running mask_pdlm evaluation at step {step} ({current_eval_batches} batches)...")
                eval_result = eval_mask_pdlm(
                    model=orig_model,
                    val_loader=val_loader,
                    block_size=block_size,
                    num_batches=current_eval_batches,
                    attn_mask=eval_attn_mask,
                    device=device,
                    autocast_ctx=autocast_ctx,
                    prefix_pure_tokens=prefix_pure_tokens,
                )
                unified = eval_result["unified"]
                e2e = eval_result["end2end"]
                print0(f"  [mask_pdlm] unified_loss: {eval_result['unified_loss']:.4f}, end2end_loss: {eval_result['end2end_loss']:.4f}")
                print0(f"    Unified: loss={unified['overall_loss']:.4f}, ppl={unified['overall_ppl']:.2f}")
                mask_bk = unified.get("mask_breakdown", {})
                group_bk = unified.get("group_breakdown", {})
                for pos in range(block_size):
                    pos_data = unified["positions"][pos]
                    parts = [f"pos {pos}: loss={pos_data['loss']:.4f}"]
                    if pos in mask_bk:
                        parts.append(f"mask={mask_bk[pos]['loss']:.4f}")
                    g_parts = []
                    for g in range(pos + 1):
                        if pos in group_bk and g in group_bk[pos]:
                            g_parts.append(f"g{g}={group_bk[pos][g]['loss']:.4f}")
                    if g_parts:
                        parts.append("group[" + " ".join(g_parts) + "]")
                    print0(f"      {' | '.join(parts)}")
                print0(f"    End-to-End: loss={e2e['overall_loss']:.4f}, ppl={e2e['overall_ppl']:.2f}")
                for pos in range(block_size):
                    pos_data = e2e["positions"][pos]
                    print0(f"      pos {pos}: loss={pos_data['loss']:.4f}, ppl={pos_data['ppl']:.2f}")
                log_data = {
                    "step": step,
                    "eval/unified_loss": eval_result["unified_loss"],
                    "eval/end2end_loss": eval_result["end2end_loss"],
                    "eval/unified_ppl": unified["overall_ppl"],
                    "eval/end2end_ppl": e2e["overall_ppl"],
                }
                for pos in range(block_size):
                    up = unified["positions"][pos]
                    log_data[f"eval/unified_pos_{pos}_loss"] = up["loss"]
                    log_data[f"eval/unified_pos_{pos}_ppl"] = up["ppl"]
                    ep = e2e["positions"][pos]
                    log_data[f"eval/end2end_pos_{pos}_loss"] = ep["loss"]
                    log_data[f"eval/end2end_pos_{pos}_ppl"] = ep["ppl"]
                    # Detailed breakdown
                    if pos in mask_bk:
                        log_data[f"eval/unified_pos_{pos}_mask_loss"] = mask_bk[pos]["loss"]
                    for g in range(pos + 1):
                        if pos in group_bk and g in group_bk[pos]:
                            log_data[f"eval/unified_pos_{pos}_group_g{g}_loss"] = group_bk[pos][g]["loss"]
                wandb_run.log(log_data)
            else:
                # Stage 2 evaluation
                # Use full eval (with compatibility + oracle accuracy) for final step
                run_full_eval = last_step
                if run_full_eval:
                    print0(f"Running PDLM Stage 2 full evaluation at step {step} ({current_eval_batches} batches)...")
                    eval_result = eval_pdlm_full(
                        model=orig_model,
                        val_loader=val_loader,
                        block_size=block_size,
                        num_batches=current_eval_batches,
                        attn_mask=eval_attn_mask,
                        device=device,
                        autocast_ctx=autocast_ctx,
                        prefix_pure_tokens=prefix_pure_tokens,
                        run_compatibility=True,
                        compatibility_batches=max(1, current_eval_batches // 4),
                        run_oracle_accuracy=True,
                        oracle_accuracy_batches=max(1, current_eval_batches // 4),
                    )
                else:
                    print0(f"Running PDLM Stage 2 evaluation at step {step} ({current_eval_batches} batches)...")
                    eval_result = eval_pdlm(
                        model=orig_model,  # use uncompiled model
                        val_loader=val_loader,
                        block_size=block_size,
                        num_batches=current_eval_batches,
                        attn_mask=eval_attn_mask,
                        device=device,
                        autocast_ctx=autocast_ctx,
                        prefix_pure_tokens=prefix_pure_tokens,
                    )
                # Log eval results
                print0(f"  [pdlm stage2] overall_loss: {eval_result['overall_loss']:.4f}, overall_ppl: {eval_result['overall_ppl']:.2f}")
                # Per-position metrics
                for pos in range(block_size):
                    pos_data = eval_result["positions"][pos]
                    print0(f"    pos {pos}: loss={pos_data['loss']:.4f}, ppl={pos_data['ppl']:.2f}")
                # Compatibility metrics (final eval only)
                if "compatibility" in eval_result:
                    compat = eval_result["compatibility"]
                    print0(f"  Compatibility: {compat['overall_compatibility']:.2%}")
                    for pos in range(block_size):
                        pos_data = compat["positions"][pos]
                        print0(f"    pos {pos}: {pos_data['compatibility']:.2%} ({pos_data['matched']}/{pos_data['total']})")
                # Oracle accuracy metrics (final eval only)
                if "oracle_accuracy" in eval_result:
                    oracle = eval_result["oracle_accuracy"]
                    print0(f"  Oracle accuracy: {oracle['overall_accuracy']:.2%}")
                    for pos in range(block_size):
                        pos_data = oracle["positions"][pos]
                        print0(f"    pos {pos}: {pos_data['accuracy']:.2%} ({pos_data['matched']}/{pos_data['total']})")
                # Build log data for wandb
                log_data = {
                    "step": step,
                    "eval/overall_loss": eval_result["overall_loss"],
                    "eval/overall_ppl": eval_result["overall_ppl"],
                }
                for pos in range(block_size):
                    pos_data = eval_result["positions"][pos]
                    log_data[f"eval/pos_{pos}_loss"] = pos_data["loss"]
                    log_data[f"eval/pos_{pos}_ppl"] = pos_data["ppl"]
                # Log compatibility and oracle accuracy (final eval only)
                if "compatibility" in eval_result:
                    log_data["eval/compatibility"] = eval_result["compatibility"]["overall_compatibility"]
                    for pos in range(block_size):
                        log_data[f"eval/pos_{pos}_compatibility"] = eval_result["compatibility"]["positions"][pos]["compatibility"]
                if "oracle_accuracy" in eval_result:
                    log_data["eval/oracle_accuracy"] = eval_result["oracle_accuracy"]["overall_accuracy"]
                    for pos in range(block_size):
                        log_data[f"eval/pos_{pos}_oracle_accuracy"] = eval_result["oracle_accuracy"]["positions"][pos]["accuracy"]
                wandb_run.log(log_data)
        elif model_type == "next_token_ar":
            pass  # TODO: AR evaluation
        elif model_type == "mtp":
            print0(f"Running MTP Stage 1 evaluation at step {step} ({current_eval_batches} batches)...")
            eval_result = eval_mtp(
                model=orig_model,  # use uncompiled model
                val_loader=val_loader,
                num_batches=current_eval_batches,
                device=device,
                autocast_ctx=autocast_ctx,
            )
            # Log eval results
            K = block_size  # n_future_tokens = block_size
            print0(f"  [mtp] overall_loss: {eval_result['overall_loss']:.4f}, overall_ppl: {eval_result['overall_ppl']:.2f}, overall_accuracy: {eval_result['overall_accuracy']:.2%}")
            # Per-position metrics
            for k in range(K):
                pos_data = eval_result["positions"][k]
                print0(f"    k={k}: loss={pos_data['loss']:.4f}, ppl={pos_data['ppl']:.2f}, accuracy={pos_data['accuracy']:.2%}")
            # Build log data for wandb
            log_data = {
                "step": step,
                "eval/overall_loss": eval_result["overall_loss"],
                "eval/overall_ppl": eval_result["overall_ppl"],
                "eval/overall_accuracy": eval_result["overall_accuracy"],
            }
            for k in range(K):
                pos_data = eval_result["positions"][k]
                log_data[f"eval/k{k}_loss"] = pos_data["loss"]
                log_data[f"eval/k{k}_ppl"] = pos_data["ppl"]
                log_data[f"eval/k{k}_accuracy"] = pos_data["accuracy"]
            wandb_run.log(log_data)

    # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
    if last_step or (step > 0 and step != resume_from_step and save_every > 0 and step % save_every == 0):
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(), # model parameters
            [opt.state_dict() for opt in optimizers], # optimizer states
            { # metadata saved as json
                "step": step,
                "model_config": model_config_kwargs,
                "user_config": user_config, # inputs to the training script
                "device_batch_size": device_batch_size,
                "max_seq_len": max_seq_len,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": { # all loop state (other than step) so that we can resume training
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                    "total_effective_tokens": total_effective_tokens,
                },
            },
            rank=ddp_rank,
        )

    # termination conditions (TODO: possibly also add loss explosions etc.)
    if last_step:
        break

    # -------------------------------------------------------------------------
    # Gradient tracking (periodic diagnostic)
    if gradient_track_every > 0 and step % gradient_track_every == 0 and step > 0:
        with autocast_ctx:
            if model_type == "pdlm" and pdlm_stage == "stage1_block":
                nll = orig_model._forward_stage1_block(
                    x, y, attn_mask=block_diff_masks[0],
                    loss_extras=loss_extras, return_nll=True)
                gt_block_size = block_size
                gt_loss_mask = loss_extras.get("loss_mask", loss_extras.get("block_loss_mask"))
            elif model_type == "next_token_ar":
                nll = orig_model(x, y, loss_reduction='none').view(x.size(0), x.size(1))
                gt_block_size = gradient_block_size
                gt_loss_mask = torch.ones_like(nll, dtype=torch.bool)
            elif model_type == "pdlm" and pdlm_stage == "both_block":
                s1_loss, s2_loss = orig_model._forward_both_block(
                    x, y, attn_mask=block_diff_masks[0],
                    loss_extras=loss_extras, return_nll=True)
                from nanochat.gradient_tracking import compute_stage_gradient_metrics
                grad_metrics = compute_stage_gradient_metrics(
                    orig_model, [s1_loss, s2_loss], ["s1", "s2"])
                wandb_run.log({"step": step, **grad_metrics})
                print0(f"[gradient_tracking] step {step}: logged {len(grad_metrics)} metrics")
                nll = None  # skip per-position path
            elif model_type == "pdlm" and pdlm_stage == "block_pdlm_inference":
                s1_loss, s2_loss = orig_model._forward_block_pdlm_inference(
                    x, y, attn_mask=block_diff_masks[0],
                    loss_extras=loss_extras, return_nll=True)
                from nanochat.gradient_tracking import compute_stage_gradient_metrics
                grad_metrics = compute_stage_gradient_metrics(
                    orig_model, [s1_loss, s2_loss], ["s1", "s2"])
                wandb_run.log({"step": step, **grad_metrics})
                print0(f"[gradient_tracking] step {step}: logged {len(grad_metrics)} metrics")
                nll = None  # skip per-position path
            elif model_type == "pdlm" and pdlm_stage == "mask_pdlm":
                nll_loss = orig_model._forward_mask_pdlm(
                    x, y, attn_mask=block_diff_masks[0],
                    loss_extras=loss_extras, return_nll=True)
                from nanochat.gradient_tracking import compute_stage_gradient_metrics
                grad_metrics = compute_stage_gradient_metrics(
                    orig_model, [nll_loss], ["unified"])
                wandb_run.log({"step": step, **grad_metrics})
                print0(f"[gradient_tracking] step {step}: logged {len(grad_metrics)} metrics")
                nll = None  # skip per-position path
            elif model_type == "pdlm" and pdlm_stage == "pdlm_emb":
                nll_loss = orig_model._forward_pdlm_emb(
                    x, y, attn_mask=block_diff_masks[0],
                    loss_extras=loss_extras, return_nll=True)
                from nanochat.gradient_tracking import compute_stage_gradient_metrics
                grad_metrics = compute_stage_gradient_metrics(
                    orig_model, [nll_loss], ["unified"])
                wandb_run.log({"step": step, **grad_metrics})
                print0(f"[gradient_tracking] step {step}: logged {len(grad_metrics)} metrics")
                nll = None  # skip per-position path
            else:
                nll = None  # gradient tracking not supported for this model type

        if nll is not None:
            from nanochat.gradient_tracking import compute_gradient_metrics
            grad_metrics = compute_gradient_metrics(
                orig_model, nll, gt_loss_mask, gt_block_size)
            wandb_run.log({"step": step, **grad_metrics})
            print0(f"[gradient_tracking] step {step}: logged {len(grad_metrics)} metrics")

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    step_effective_tokens = 0  # effective tokens this step (for rl_tok/sec)
    for micro_step in range(grad_accum_steps):
        if debug:
            x_cpu = x.detach().cpu()
            y_cpu = y.detach().cpu()
            print0(f"[debug] step {step} micro {micro_step} x={x_cpu.tolist()} y={y_cpu.tolist()}")
            if debug_dump_path is not None:
                with open(debug_dump_path, "a", encoding="utf-8") as handle:
                    handle.write(f"step={step} micro={micro_step}\n")
                    handle.write(f"x={x_cpu.tolist()}\n")
                    handle.write(f"y={y_cpu.tolist()}\n")
        with autocast_ctx:
            if model_type == "bd3lm":
                # Select attention mask based on epoch (for target_shift cycling)
                epoch = dataloader_state_dict.get("epoch", 0)
                block_diff_mask = block_diff_masks[epoch % len(block_diff_masks)]
                loss = model(x, y, attn_mask=block_diff_mask, loss_extras=loss_extras)
                # Count effective tokens (positions that contribute to loss)
                batch_effective_tokens = loss_extras["loss_mask"].sum().item() * ddp_world_size
                step_effective_tokens += batch_effective_tokens
                total_effective_tokens += batch_effective_tokens
            elif model_type == "pdlm":
                loss = model(x, y, attn_mask=block_diff_masks[0], loss_extras=loss_extras)
                # Count effective tokens (positions that contribute to loss)
                _eff_mask = loss_extras.get("loss_mask", loss_extras.get("block_loss_mask")) if loss_extras is not None else None
                if _eff_mask is not None:
                    batch_effective_tokens = _eff_mask.sum().item() * ddp_world_size
                else:
                    batch_effective_tokens = x.numel() * ddp_world_size
                total_effective_tokens += batch_effective_tokens
            elif model_type == "mtp":
                # MTP: x is (B, T) pure tokens, y is targets (format depends on stage1_target_mode)
                # For pure mode: loss_extras contains pure_to_group mapping
                pure_to_group = loss_extras.get("pure_to_group") if loss_extras else None
                loss = model(x, y, pure_to_group=pure_to_group)
                total_effective_tokens += x.numel() * ddp_world_size
            else:
                # next_token_ar: GPT forward doesn't take attn_mask
                loss = model(x, y)
                total_effective_tokens += x.numel() * ddp_world_size
        train_loss = loss.detach() # for logging
        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        loss.backward()
        x, y, loss_extras, dataloader_state_dict = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
    # gradient clipping
    grad_clip_enabled = grad_clip > 0.0
    if grad_clip_enabled:
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(orig_model.parameters(), grad_clip)
        grad_norm = grad_norm_tensor.item() # GPU tensor -> CPU float (note: cpu-gpu sync point)
    # step the optimizers
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    muon_momentum = get_muon_momentum(step)
    for group in muon_optimizer.param_groups:
        group["momentum"] = muon_momentum
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # logging
    ema_beta = 0.9 # EMA decay factor for some smoothing just for nicer logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item() # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    rl_tok_per_sec = int(step_effective_tokens / dt) if model_type == "bd3lm" else None
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    promised_flops_per_sec_h100 = 989e12 * ddp_world_size # bfloat16 H100 SXM and without 2:4 sparsity
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100 # in %
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    print_grad_norm = f" grad norm: {grad_norm:.4f} |" if grad_clip_enabled else ""
    print_rl_tok = f" rl_tok/sec: {rl_tok_per_sec:,} |" if rl_tok_per_sec is not None else ""
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} |{print_grad_norm} lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} |{print_rl_tok} mfu: {mfu:.2f} | total time: {total_training_time/60:.2f}m")
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "total_effective_tokens": total_effective_tokens,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
        }
        if grad_clip_enabled:
            log_data["train/grad_norm"] = grad_norm
        if rl_tok_per_sec is not None:
            log_data["train/rl_tok_per_sec"] = rl_tok_per_sec
        wandb_run.log(log_data)

    # state update
    step += 1

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Total effective tokens: {total_effective_tokens:,}")
effective_ratio_actual = total_effective_tokens / (total_batch_size * num_iterations) if num_iterations > 0 else 0
print0(f"Actual effective ratio: {effective_ratio_actual:.4f}")

# Log to report
from nanochat.report import get_report
get_report().log(section="Base model training", data=[
    user_config, # CLI args
    { # stats about the training setup
        "Number of parameters": num_params,
        "Number of FLOPs per token": f"{num_flops_per_token:e}",
        "Calculated number of iterations": num_iterations,
        "Number of training tokens": total_tokens,
        "Tokens : Params ratio": total_batch_size * num_iterations / num_params,
        "DDP world size": ddp_world_size,
        "warmup_ratio": warmup_ratio,
        "warmdown_ratio": warmdown_ratio,
        "final_lr_frac": final_lr_frac,
    },
    { # stats about training outcomes
        "MFU %": f"{mfu:.2f}%",
        "Total training flops": f"{flops_so_far:e}",
        "Total training time": f"{total_training_time/60:.2f}m",
        "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
        "Total effective tokens": total_effective_tokens,
        "Actual effective ratio": effective_ratio_actual,
    }
])

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()
