"""
Train a learned group assignment matrix for a frozen stage1_block model.

Uses chunked streaming: the frozen model stays loaded, and batches are cached
to GPU in chunks. Each chunk is trained on for several epochs, then freed.
This keeps GPU memory bounded while allowing training on more data.

Usage:
    python -m scripts.train_group_mapping --ckpt_dir=/path/to/ckpt --num_groups=256
    python -m scripts.train_group_mapping --cache_dir=/path/to/cached_logits --num_groups=256
"""

import os
import time

import wandb
import torch

from nanochat.common import print0, DummyWandb
from nanochat.learned_group_mapping import AssignmentMatrix
from nanochat.learned_group_mapping.cache_logits import cache_logits, LogitCacher, CachedLogitDataset

# -----------------------------------------------------------------------------
# Config defaults (Poor Man's Configurator style)
# -----------------------------------------------------------------------------

# Model / checkpoint
ckpt_dir = ""          # path to frozen stage1_block checkpoint
ckpt_step = -1         # checkpoint step (-1 = last)

# Logit cache
cache_dir = ""         # path to pre-cached logits (if empty, will use chunked streaming)
cache_split = "train"
skip_pos0 = False      # mask out pos 0 within each block

# Chunked training
chunk_train_batches = 200   # train batches per chunk
chunk_eval_batches = 25     # eval batches per chunk
epochs_per_chunk = 30       # epochs to train on each chunk
num_chunks = 3              # number of training chunks
final_eval_batches = 225    # batches for final eval (fresh data)

# Assignment matrix
num_groups = 256       # number of groups (G)
max_size_soft_multiplier = 2  # max_size_soft = expected_group_size * multiplier (overlap-aware)
min_overlap_soft = 4   # soft lower bound on groups per token

# Loss weights
lambda_noise = 0.1     # weight for group size penalty
lambda_overlap = 0.1   # weight for overlap penalty
lambda_sharp = 1.0     # final weight for sharpening penalty
sharp_ramp_start = 0.7 # fraction of total steps before sharpening starts ramping

# Binarization
binarize_topk = 0      # if > 0, use top-k per token instead of threshold (sets overlap_k = topk)

# Optimization
lr = 0.1               # learning rate (high is ok for single matrix)
eval_every_epoch = 5   # evaluate every N epochs

# Output
output_dir = ""        # where to save token_maps.pt and assignment_raw.pt
run = "dummy"          # wandb run name ("dummy" = no logging)
wandb_group = None     # wandb group
wandb_project = "nanochat"

# Device
device_type = ""       # cuda|cpu|mps (empty => autodetect)

# now allow CLI to override the settings via the configurator
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None)))]
exec(open(os.path.join('nanochat', 'configurator.py')).read())
user_config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

from nanochat.common import autodetect_device_type

device_type = autodetect_device_type() if device_type == "" else device_type
device = torch.device(device_type)

# Wandb
use_dummy_wandb = run == "dummy"
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(
    project=wandb_project, name=run, group=wandb_group, config=user_config
)

# ============================================================
# Step 1: Load frozen model (or prepare disk cache path)
# ============================================================
cacher = None
use_disk_cache = bool(cache_dir)

if not use_disk_cache:
    assert ckpt_dir, "Must provide --ckpt_dir or --cache_dir"
    step_arg = None if ckpt_step == -1 else ckpt_step
    print0(f"Loading frozen model from {ckpt_dir} (step={step_arg})...")
    cacher = LogitCacher(
        ckpt_dir=ckpt_dir,
        step=step_arg,
        split=cache_split,
        device_type=device_type,
        skip_pos0=skip_pos0,
    )
elif not os.path.exists(os.path.join(cache_dir, "cache_meta.json")):
    # cache_dir specified but doesn't exist yet: write to disk first
    assert ckpt_dir, "Cache does not exist and no --ckpt_dir provided to create it"
    step_arg = None if ckpt_step == -1 else ckpt_step
    print0(f"Caching logits from {ckpt_dir} (step={step_arg}) to {cache_dir}...")
    total_disk_batches = chunk_train_batches + chunk_eval_batches
    cache_logits(
        ckpt_dir=ckpt_dir,
        step=step_arg,
        output_dir=cache_dir,
        split=cache_split,
        num_batches=total_disk_batches,
        device_type=device_type,
        skip_pos0=skip_pos0,
    )

# ============================================================
# Helper: load gpu_batches from disk cache
# ============================================================
def load_disk_cache_to_gpu():
    """Load all batches from disk cache to GPU. Returns (gpu_batches, meta)."""
    print0(f"Loading logit cache from {cache_dir}")
    dataset = CachedLogitDataset(cache_dir)
    meta = dataset.meta
    gpu_batches = []
    for i in range(len(dataset)):
        pure_logits, pure_targets, loss_mask = dataset[i]
        gpu_batches.append((
            pure_logits.to(device=device),
            pure_targets.to(device=device),
            loss_mask.to(device=device),
        ))
    print0(f"  preloaded {len(gpu_batches)} batches to GPU")
    return gpu_batches, meta

# ============================================================
# Get pure_vocab_size
# ============================================================
if use_disk_cache:
    import json
    meta_path = os.path.join(cache_dir, "cache_meta.json")
    with open(meta_path, "r") as f:
        disk_meta = json.load(f)
    pure_vocab_size = disk_meta["pure_vocab_size"]
    meta = disk_meta

num_train = chunk_train_batches
num_eval = chunk_eval_batches
batches_per_chunk = num_train + num_eval

if output_dir:
    os.makedirs(output_dir, exist_ok=True)

# ============================================================
# Step 2: Chunked training loop
# ============================================================
# For streaming mode, we defer model creation until after the first chunk
# is cached (so we know pure_vocab_size). For disk cache mode, we create
# it immediately since we already have the metadata.
assign_model = None
optimizer = None
total_steps = num_chunks * epochs_per_chunk * num_train
global_step = 0

for chunk_idx in range(num_chunks):
    print0(f"\n{'='*60}")
    print0(f"CHUNK {chunk_idx+1}/{num_chunks}")
    print0(f"{'='*60}")

    # Cache batches for this chunk
    if use_disk_cache:
        gpu_batches, _ = load_disk_cache_to_gpu()
        total_chunk_batches = len(gpu_batches)
        actual_train = min(num_train, total_chunk_batches - 1)
        actual_eval = total_chunk_batches - actual_train
    else:
        gpu_batches = cacher.cache_batches(batches_per_chunk, device)
        total_chunk_batches = len(gpu_batches)
        actual_train = num_train
        actual_eval = num_eval

    # On first chunk in streaming mode, now we know pure_vocab_size
    if assign_model is None:
        if not use_disk_cache:
            pure_vocab_size = cacher.pure_vocab_size
            meta = cacher.get_meta()

        print0(f"Chunked training: {num_chunks} chunks, {num_train} train + {num_eval} eval per chunk")
        print0(f"  epochs_per_chunk={epochs_per_chunk}, final_eval_batches={final_eval_batches}")
        print0(f"  pure_vocab_size={pure_vocab_size}, B={meta['batch_size']}, T={meta['seq_len']}")

        # Overlap-aware expected group size: V * overlap_k / G
        # If binarize_topk not set, fall back to no-overlap estimate V / G
        effective_overlap = binarize_topk if binarize_topk > 0 else 1
        expected_group_size = pure_vocab_size * effective_overlap / num_groups
        max_size_soft = expected_group_size * max_size_soft_multiplier
        print0(f"  expected_group_size={expected_group_size:.1f} (V={pure_vocab_size}, overlap_k={effective_overlap}, G={num_groups})")
        assign_model = AssignmentMatrix(
            pure_vocab_size=pure_vocab_size,
            num_groups=num_groups,
            max_size_soft=max_size_soft,
            min_overlap_soft=min_overlap_soft,
        ).to(device)
        print0(f"AssignmentMatrix: V={pure_vocab_size}, G={num_groups}")
        print0(f"  max_size_soft={max_size_soft:.1f}, min_overlap_soft={min_overlap_soft}")
        print0(f"  lambda_noise={lambda_noise}, lambda_overlap={lambda_overlap}")
        print0(f"  lambda_sharp={lambda_sharp}, sharp_ramp_start={sharp_ramp_start}")

        optimizer = torch.optim.Adam(assign_model.parameters(), lr=lr)

        print0(f"\nStarting training: {num_chunks} chunks x {epochs_per_chunk} epochs x {num_train} batches = {total_steps} total steps")
        print0(f"  lr={lr}, eval_every_epoch={eval_every_epoch}")

    train_indices = list(range(actual_train))
    eval_indices = list(range(actual_train, total_chunk_batches))

    print0(f"  {total_chunk_batches} batches cached: {actual_train} train / {actual_eval} eval")

    # Train epochs_per_chunk epochs on this chunk
    for epoch in range(epochs_per_chunk):
        t0 = time.time()
        epoch_loss = 0.0
        assign_model.train()

        # Shuffle train indices each epoch
        perm = torch.randperm(actual_train).tolist()

        for i, batch_idx in enumerate(perm):
            global_step += 1

            # Compute sharp_weight via annealing schedule (global progress)
            frac = global_step / total_steps
            if frac < sharp_ramp_start:
                sharp_weight = 0.0
            else:
                sharp_weight = lambda_sharp * (frac - sharp_ramp_start) / (1.0 - sharp_ramp_start)

            # Load batch from GPU cache
            pure_logits, pure_targets, loss_mask = gpu_batches[train_indices[batch_idx]]
            pure_logits = pure_logits.float()

            # Forward + backward
            optimizer.zero_grad()
            total_loss, loss_dict = assign_model(
                pure_logits, pure_targets, loss_mask,
                lambda_noise, lambda_overlap, sharp_weight,
            )
            total_loss.backward()
            optimizer.step()

            epoch_loss += loss_dict["loss_task"]

            # Log to wandb every step
            wandb_run.log({
                f"train/{k}": v for k, v in loss_dict.items()
            }, step=global_step)

        epoch_loss /= actual_train
        dt = time.time() - t0
        global_epoch = chunk_idx * epochs_per_chunk + epoch + 1
        total_epochs = num_chunks * epochs_per_chunk
        print0(f"Epoch {global_epoch}/{total_epochs} [chunk {chunk_idx+1}/{num_chunks}] | task_loss={epoch_loss:.4f} | {dt:.1f}s")

        # Evaluation
        if (epoch + 1) % eval_every_epoch == 0 or epoch == epochs_per_chunk - 1:
            assign_model.eval()
            eval_loss_accum = {}
            eval_group_correct = 0
            eval_group_total = 0

            with torch.no_grad():
                soft_assign = assign_model.get_soft_assign()

                for eval_idx in eval_indices:
                    pure_logits, pure_targets, loss_mask = gpu_batches[eval_idx]
                    pure_logits = pure_logits.float()

                    _, loss_dict = assign_model(
                        pure_logits, pure_targets, loss_mask,
                        lambda_noise, lambda_overlap, sharp_weight,
                    )
                    for k, v in loss_dict.items():
                        eval_loss_accum[k] = eval_loss_accum.get(k, 0.0) + v

                    group_logits = pure_logits @ soft_assign
                    pred_groups = group_logits.argmax(dim=-1)

                    target_membership = soft_assign[pure_targets]
                    target_best_group = target_membership.argmax(dim=-1)
                    correct = (pred_groups == target_best_group)

                    mask_float = loss_mask.float()
                    eval_group_correct += (correct.float() * mask_float).sum().item()
                    eval_group_total += mask_float.sum().item()

            for k in eval_loss_accum:
                eval_loss_accum[k] /= actual_eval
            eval_acc = eval_group_correct / max(eval_group_total, 1)

            print0(f"  eval [chunk {chunk_idx+1}/{num_chunks}]: loss_total={eval_loss_accum.get('loss_total', 0):.4f}, "
                   f"task={eval_loss_accum.get('loss_task', 0):.4f}, "
                   f"noise={eval_loss_accum.get('loss_noise', 0):.4f}, "
                   f"overlap={eval_loss_accum.get('loss_overlap', 0):.4f}, "
                   f"sharp={eval_loss_accum.get('loss_sharp', 0):.4f}, "
                   f"group_accuracy={eval_acc:.2%}")
            wandb_run.log({
                **{f"eval/{k}": v for k, v in eval_loss_accum.items()},
                "eval/group_accuracy": eval_acc,
            }, step=global_step)

    # Free chunk GPU memory
    del gpu_batches
    torch.cuda.empty_cache()
    print0(f"Chunk {chunk_idx+1} done, GPU memory freed.")

# ============================================================
# Step 4: Save outputs
# ============================================================
if output_dir:
    raw_path = os.path.join(output_dir, "assignment_raw.pt")
    torch.save({
        "A": assign_model.A.data.cpu(),
        "pure_vocab_size": pure_vocab_size,
        "num_groups": num_groups,
        "config": user_config,
    }, raw_path)
    print0(f"Saved raw assignment to {raw_path}")

    token_maps = assign_model.binarize(threshold=0.5, topk=binarize_topk)
    maps_path = os.path.join(output_dir, "token_maps.pt")
    torch.save(token_maps, maps_path)
    print0(f"Saved token_maps.pt to {maps_path}")

    from nanochat.group_tokenizer.token_map import TokenMap
    tm = TokenMap(token_maps)
    print0(f"Verification: TokenMap loaded successfully")
    print0(f"  pure_vocab_size={tm.pure_vocab_size}, num_groups={tm.num_groups}, overlap_k={tm.overlap_k}")
else:
    token_maps = assign_model.binarize(threshold=0.5, topk=binarize_topk)
    print0("Warning: no --output_dir specified, results not saved")

# ============================================================
# Step 5: Final evaluation on fresh data
# ============================================================
print0("")
print0("=" * 60)
print0(f"FINAL EVALUATION ({final_eval_batches} fresh batches)")
print0("=" * 60)

# --- Mapping stats ---
pure_to_group = token_maps["pure_to_group"]  # (V, overlap_k)
group_to_pure_mask = token_maps["group_to_pure_mask"]  # (G, V)

groups_per_token = (pure_to_group >= 0).sum(dim=1).float()
tokens_per_group = group_to_pure_mask.sum(dim=1).float()

q = torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0], device=device)
gpt_pct = torch.quantile(groups_per_token.to(device), q)
tpg_pct = torch.quantile(tokens_per_group.to(device), q)
print0(f"\nMapping stats (V={pure_vocab_size}, G={num_groups}):")
print0(f"  overlap_k:        {token_maps['overlap_k']}")
print0(f"  groups/token:     "
       f"p0={gpt_pct[0].item():.0f}, p10={gpt_pct[1].item():.0f}, "
       f"p25={gpt_pct[2].item():.0f}, p50={gpt_pct[3].item():.0f}, "
       f"p75={gpt_pct[4].item():.0f}, p90={gpt_pct[5].item():.0f}, "
       f"p100={gpt_pct[6].item():.0f}")
print0(f"  tokens/group:     "
       f"p0={tpg_pct[0].item():.0f}, p10={tpg_pct[1].item():.0f}, "
       f"p25={tpg_pct[2].item():.0f}, p50={tpg_pct[3].item():.0f}, "
       f"p75={tpg_pct[4].item():.0f}, p90={tpg_pct[5].item():.0f}, "
       f"p100={tpg_pct[6].item():.0f}")
print0(f"  ideal tokens/grp: {pure_vocab_size / num_groups:.1f}")

empty_groups = (tokens_per_group == 0).sum().item()
if empty_groups > 0:
    print0(f"  empty groups:     {empty_groups}/{num_groups}")

# --- Accuracy on fresh data using binarized mapping ---
if use_disk_cache:
    final_gpu_batches, _ = load_disk_cache_to_gpu()
else:
    final_gpu_batches = cacher.cache_batches(final_eval_batches, device)

num_final = len(final_gpu_batches)
print0(f"\nFinal eval accuracy (binarized, {num_final} fresh batches):")

binary_assign = group_to_pure_mask.T.float().to(device)
eval_group_correct = 0
eval_group_total = 0

assign_model.eval()
with torch.no_grad():
    for eval_idx in range(num_final):
        pure_logits, pure_targets, loss_mask = final_gpu_batches[eval_idx]
        pure_logits = pure_logits.float()

        group_logits = pure_logits @ binary_assign
        pred_groups = group_logits.argmax(dim=-1)

        target_in_group = group_to_pure_mask.to(device)[pred_groups]
        B, T = pure_targets.shape
        correct = target_in_group.gather(-1, pure_targets.unsqueeze(-1)).squeeze(-1)

        mask_float = loss_mask.float()
        eval_group_correct += (correct.float() * mask_float).sum().item()
        eval_group_total += mask_float.sum().item()

eval_acc = eval_group_correct / max(eval_group_total, 1)
print0(f"  group_accuracy (binarized): {eval_acc:.2%} ({int(eval_group_correct)}/{int(eval_group_total)})")

wandb_run.log({
    "final/group_accuracy_binarized": eval_acc,
    "final/overlap_k": token_maps["overlap_k"],
    "final/groups_per_token_mean": groups_per_token.mean().item(),
    "final/tokens_per_group_mean": tokens_per_group.mean().item(),
    "final/tokens_per_group_max": tokens_per_group.max().item(),
    "final/tokens_per_group_min": tokens_per_group.min().item(),
}, step=global_step)

# Free final eval batches
del final_gpu_batches
torch.cuda.empty_cache()

# Clean up frozen model if still loaded
if cacher is not None:
    cacher.cleanup()

print0("")
print0("=" * 60)
print0("Done.")
