"""
Train a learned group assignment matrix for a frozen stage1_block model.

Uses cached pure logits (pre-computed from frozen model) to train only the
(V_pure, G) assignment matrix. Produces a token_maps.pt file compatible with
TokenMap for downstream use.

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
from nanochat.learned_group_mapping.cache_logits import cache_logits, cache_logits_to_gpu, CachedLogitDataset

# -----------------------------------------------------------------------------
# Config defaults (Poor Man's Configurator style)
# -----------------------------------------------------------------------------

# Model / checkpoint
ckpt_dir = ""          # path to frozen stage1_block checkpoint
ckpt_step = -1         # checkpoint step (-1 = last)

# Logit cache
cache_dir = ""         # path to pre-cached logits (if empty, will cache first)
cache_num_batches = 200
cache_split = "train"
skip_pos0 = False      # mask out pos 0 within each block

# Assignment matrix
num_groups = 256       # number of groups (G)
max_size_soft_multiplier = 4  # max_size_soft = (V / G) * multiplier
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
num_epochs = 50        # number of training epochs
eval_every_epoch = 5   # evaluate every N epochs

# Train/eval split
eval_frac = 0.1        # fraction of cached batches for eval

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
# Step 1: Cache logits (GPU-direct or disk)
# ============================================================
gpu_batches = None

if not cache_dir:
    # No cache_dir: cache directly to GPU memory (skip disk entirely)
    assert ckpt_dir, "Must provide --ckpt_dir or --cache_dir"
    step_arg = None if ckpt_step == -1 else ckpt_step
    print0(f"Caching logits from {ckpt_dir} (step={step_arg}) directly to GPU...")
    gpu_batches, meta = cache_logits_to_gpu(
        ckpt_dir=ckpt_dir,
        step=step_arg,
        split=cache_split,
        num_batches=cache_num_batches,
        device_type=device_type,
        skip_pos0=skip_pos0,
    )
    pure_vocab_size = meta["pure_vocab_size"]
    total_batches = len(gpu_batches)
elif not os.path.exists(os.path.join(cache_dir, "cache_meta.json")):
    # cache_dir specified but doesn't exist yet: write to disk first
    assert ckpt_dir, "Cache does not exist and no --ckpt_dir provided to create it"
    step_arg = None if ckpt_step == -1 else ckpt_step
    print0(f"Caching logits from {ckpt_dir} (step={step_arg}) to {cache_dir}...")
    cache_logits(
        ckpt_dir=ckpt_dir,
        step=step_arg,
        output_dir=cache_dir,
        split=cache_split,
        num_batches=cache_num_batches,
        device_type=device_type,
        skip_pos0=skip_pos0,
    )

# ============================================================
# Step 2: Load cached data (from disk if not already in GPU)
# ============================================================
if gpu_batches is None:
    print0(f"Loading logit cache from {cache_dir}")
    dataset = CachedLogitDataset(cache_dir)
    meta = dataset.meta
    pure_vocab_size = meta["pure_vocab_size"]
    total_batches = len(dataset)

    # Preload all batches to GPU (bf16) to avoid per-step disk I/O
    print0("Preloading all batches to GPU (bf16)...")
    gpu_batches = []
    for i in range(total_batches):
        pure_logits, pure_targets, loss_mask = dataset[i]
        gpu_batches.append((
            pure_logits.to(device=device),
            pure_targets.to(device=device),
            loss_mask.to(device=device),
        ))
    print0(f"  preloaded {total_batches} batches to GPU")

# Train/eval split
num_eval = max(1, int(total_batches * eval_frac))
num_train = total_batches - num_eval
train_indices = list(range(num_train))
eval_indices = list(range(num_train, total_batches))

print0(f"Dataset: {total_batches} batches, {num_train} train / {num_eval} eval")
print0(f"  pure_vocab_size={pure_vocab_size}, B={meta['batch_size']}, T={meta['seq_len']}")

# ============================================================
# Step 3: Create assignment matrix
# ============================================================
max_size_soft = (pure_vocab_size / num_groups) * max_size_soft_multiplier

model = AssignmentMatrix(
    pure_vocab_size=pure_vocab_size,
    num_groups=num_groups,
    max_size_soft=max_size_soft,
    min_overlap_soft=min_overlap_soft,
).to(device)

print0(f"AssignmentMatrix: V={pure_vocab_size}, G={num_groups}")
print0(f"  max_size_soft={max_size_soft:.1f}, min_overlap_soft={min_overlap_soft}")
print0(f"  lambda_noise={lambda_noise}, lambda_overlap={lambda_overlap}")
print0(f"  lambda_sharp={lambda_sharp}, sharp_ramp_start={sharp_ramp_start}")

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# ============================================================
# Step 4: Training loop
# ============================================================
total_steps = num_epochs * num_train
step = 0

if output_dir:
    os.makedirs(output_dir, exist_ok=True)

print0(f"\nStarting training: {num_epochs} epochs, {num_train} batches/epoch, {total_steps} total steps")
print0(f"  lr={lr}, eval_every_epoch={eval_every_epoch}")

for epoch in range(num_epochs):
    t0 = time.time()
    epoch_loss = 0.0
    model.train()

    # Shuffle train indices each epoch
    perm = torch.randperm(num_train).tolist()

    for i, batch_idx in enumerate(perm):
        step += 1

        # Compute sharp_weight via annealing schedule
        frac = step / total_steps
        if frac < sharp_ramp_start:
            sharp_weight = 0.0
        else:
            sharp_weight = lambda_sharp * (frac - sharp_ramp_start) / (1.0 - sharp_ramp_start)

        # Load batch from GPU cache (bf16 -> float32 cast is fast on GPU)
        pure_logits, pure_targets, loss_mask = gpu_batches[train_indices[batch_idx]]
        pure_logits = pure_logits.float()

        # Forward + backward
        optimizer.zero_grad()
        total_loss, loss_dict = model(
            pure_logits, pure_targets, loss_mask,
            lambda_noise, lambda_overlap, sharp_weight,
        )
        total_loss.backward()
        optimizer.step()

        epoch_loss += loss_dict["loss_task"]

        # Log to wandb every step
        wandb_run.log({f"train/{k}": v for k, v in loss_dict.items()}, step=step)

    epoch_loss /= num_train
    dt = time.time() - t0
    print0(f"Epoch {epoch+1}/{num_epochs} | task_loss={epoch_loss:.4f} | {dt:.1f}s")

    # ============================================================
    # Evaluation
    # ============================================================
    if (epoch + 1) % eval_every_epoch == 0 or epoch == num_epochs - 1:
        model.eval()
        eval_task_loss = 0.0
        eval_group_correct = 0
        eval_group_total = 0

        with torch.no_grad():
            soft_assign = model.get_soft_assign()  # (V, G)

            for eval_idx in eval_indices:
                pure_logits, pure_targets, loss_mask = gpu_batches[eval_idx]
                pure_logits = pure_logits.float()

                # Task loss
                _, loss_dict = model(
                    pure_logits, pure_targets, loss_mask,
                    lambda_noise, lambda_overlap, sharp_weight,
                )
                eval_task_loss += loss_dict["loss_task"]

                # Group accuracy: does argmax group match target's best group?
                group_logits = pure_logits @ soft_assign  # (B, T, G)
                pred_groups = group_logits.argmax(dim=-1)  # (B, T)

                # Use argmax-based check instead of binary threshold
                target_membership = soft_assign[pure_targets]  # (B, T, G)
                target_best_group = target_membership.argmax(dim=-1)  # (B, T)
                correct = (pred_groups == target_best_group)  # (B, T)

                mask_float = loss_mask.float()
                eval_group_correct += (correct.float() * mask_float).sum().item()
                eval_group_total += mask_float.sum().item()

        eval_task_loss /= num_eval
        eval_acc = eval_group_correct / max(eval_group_total, 1)

        print0(f"  eval: task_loss={eval_task_loss:.4f}, group_accuracy={eval_acc:.2%}")
        wandb_run.log({
            "eval/task_loss": eval_task_loss,
            "eval/group_accuracy": eval_acc,
        }, step=step)

# ============================================================
# Step 5: Save outputs
# ============================================================
if output_dir:
    # Save raw soft weights
    raw_path = os.path.join(output_dir, "assignment_raw.pt")
    torch.save({
        "A": model.A.data.cpu(),
        "pure_vocab_size": pure_vocab_size,
        "num_groups": num_groups,
        "config": user_config,
    }, raw_path)
    print0(f"Saved raw assignment to {raw_path}")

    # Binarize and save TokenMap-compatible dict
    token_maps = model.binarize(threshold=0.5, topk=binarize_topk)
    maps_path = os.path.join(output_dir, "token_maps.pt")
    torch.save(token_maps, maps_path)
    print0(f"Saved token_maps.pt to {maps_path}")

    # Verify it loads
    from nanochat.group_tokenizer.token_map import TokenMap
    tm = TokenMap(token_maps)
    print0(f"Verification: TokenMap loaded successfully")
    print0(f"  pure_vocab_size={tm.pure_vocab_size}, num_groups={tm.num_groups}, overlap_k={tm.overlap_k}")
else:
    token_maps = model.binarize(threshold=0.5, topk=binarize_topk)
    print0("Warning: no --output_dir specified, results not saved")

# ============================================================
# Step 6: Final evaluation summary
# ============================================================
print0("")
print0("=" * 60)
print0("FINAL EVALUATION")
print0("=" * 60)

# --- Mapping stats ---
pure_to_group = token_maps["pure_to_group"]  # (V, overlap_k)
group_to_pure_mask = token_maps["group_to_pure_mask"]  # (G, V)

# Overlap: how many groups each token belongs to
groups_per_token = (pure_to_group >= 0).sum(dim=1).float()  # (V,)
# Group sizes: how many tokens in each group
tokens_per_group = group_to_pure_mask.sum(dim=1).float()  # (G,)

q = torch.tensor([0.1, 0.25, 0.75, 0.9], device=device)
gpt_pct = torch.quantile(groups_per_token.to(device), q)
tpg_pct = torch.quantile(tokens_per_group.to(device), q)
print0(f"\nMapping stats (V={pure_vocab_size}, G={num_groups}):")
print0(f"  overlap_k:        {token_maps['overlap_k']}")
print0(f"  groups/token:     min={groups_per_token.min().item():.0f}, "
       f"p10={gpt_pct[0].item():.0f}, p25={gpt_pct[1].item():.0f}, "
       f"mean={groups_per_token.mean().item():.2f}, "
       f"p75={gpt_pct[2].item():.0f}, p90={gpt_pct[3].item():.0f}, "
       f"max={groups_per_token.max().item():.0f}")
print0(f"  tokens/group:     min={tokens_per_group.min().item():.0f}, "
       f"p10={tpg_pct[0].item():.0f}, p25={tpg_pct[1].item():.0f}, "
       f"mean={tokens_per_group.mean().item():.2f}, "
       f"p75={tpg_pct[2].item():.0f}, p90={tpg_pct[3].item():.0f}, "
       f"max={tokens_per_group.max().item():.0f}")
print0(f"  ideal tokens/grp: {pure_vocab_size / num_groups:.1f}")

# Empty groups
empty_groups = (tokens_per_group == 0).sum().item()
if empty_groups > 0:
    print0(f"  empty groups:     {empty_groups}/{num_groups}")

# --- Accuracy on eval set using binarized mapping ---
print0(f"\nEval accuracy (binarized, {num_eval} batches):")

binary_assign = group_to_pure_mask.T.float().to(device)  # (V, G)
eval_group_correct = 0
eval_group_total = 0

model.eval()
with torch.no_grad():
    for eval_idx in eval_indices:
        pure_logits, pure_targets, loss_mask = gpu_batches[eval_idx]
        pure_logits = pure_logits.float()

        # Group logits using binarized assignment
        group_logits = pure_logits @ binary_assign  # (B, T, G)
        pred_groups = group_logits.argmax(dim=-1)  # (B, T)

        # Check if target token is in predicted group
        target_in_group = group_to_pure_mask.to(device)[pred_groups]  # (B, T, V)
        B, T = pure_targets.shape
        correct = target_in_group.gather(-1, pure_targets.unsqueeze(-1)).squeeze(-1)  # (B, T)

        mask_float = loss_mask.float()
        eval_group_correct += (correct.float() * mask_float).sum().item()
        eval_group_total += mask_float.sum().item()

eval_acc = eval_group_correct / max(eval_group_total, 1)
print0(f"  group_accuracy:   {eval_acc:.2%} ({int(eval_group_correct)}/{int(eval_group_total)})")

wandb_run.log({
    "final/group_accuracy_binarized": eval_acc,
    "final/overlap_k": token_maps["overlap_k"],
    "final/groups_per_token_mean": groups_per_token.mean().item(),
    "final/tokens_per_group_mean": tokens_per_group.mean().item(),
    "final/tokens_per_group_max": tokens_per_group.max().item(),
    "final/tokens_per_group_min": tokens_per_group.min().item(),
}, step=step)

print0("")
print0("=" * 60)
print0("Done.")
