"""
Cache pure logits from a frozen stage1_block model to disk.

Pre-computing logits avoids repeated forward passes during assignment matrix
training. Each batch is saved as a separate .pt file for lazy loading.
"""

import os
import json

import torch


def cache_logits(
    ckpt_dir,
    step,
    output_dir,
    split="val",
    num_batches=200,
    device_type="auto",
    skip_pos0=False,
):
    """
    Run frozen stage1_block model and save pure logits to disk.

    Args:
        ckpt_dir: Path to model checkpoint directory.
        step: Checkpoint step to load (None = last).
        output_dir: Directory to write cached batch files.
        split: Data split ("train" or "val").
        num_batches: Number of batches to cache.
        device_type: Device type ("cuda", "cpu", "auto").
        skip_pos0: If True, also mask out pos 0 within each block.
    """
    from scripts.pdlm_eval import load_pdlm_model
    from nanochat.dataloader import get_data_loader
    from nanochat.attn_masks import gen_block_causal_mask

    # Load frozen model
    model, meta_data, device, autocast_ctx, model_config = load_pdlm_model(
        model_tag=None, step=step, device_type=device_type, ckpt_dir=ckpt_dir
    )

    # Extract config
    model_config_dict = meta_data["model_config"]
    user_config = meta_data.get("user_config", {})
    max_seq_len = model_config_dict["sequence_len"]
    block_size = model_config_dict.get("bucket_size", user_config.get("block_size", 4))
    is_causal = model_config_dict.get("is_causal", True)
    device_batch_size = user_config.get("device_batch_size", 32)

    assert model_config_dict.get("stage") == "stage1_block", (
        f"Expected stage1_block model, got {model_config_dict.get('stage')}"
    )

    # Create dataloader
    val_loader = get_data_loader(
        device_batch_size, max_seq_len, split=split, device=device,
        model_config=model_config, resume_state_dict=None,
    )

    # Generate block-causal attention mask
    attn_mask = gen_block_causal_mask(
        max_seq_len, block_size, attn_backend="sdpa", is_causal=is_causal
    ).to(device=device)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Caching {num_batches} batches to {output_dir}")
    print(f"  B={device_batch_size}, T={max_seq_len}, block_size={block_size}")

    pure_vocab_size = None

    with torch.no_grad():
        for batch_idx in range(num_batches):
            inputs, targets, loss_extras, state = next(val_loader)
            # inputs: (B, T) pure tokens
            loss_mask = loss_extras["loss_mask"]  # (B, T)
            pure_targets = loss_extras["pure_targets"]  # (B, T)

            # Optionally mask out pos 0 within each block
            if skip_pos0:
                B, T = inputs.shape
                num_blocks = T // block_size
                for blk in range(num_blocks):
                    loss_mask[:, blk * block_size] = False

            with autocast_ctx:
                logits = model(inputs, attn_mask=attn_mask)  # (B, T, V_pure)

            if pure_vocab_size is None:
                pure_vocab_size = logits.shape[-1]

            # Save as bf16 to reduce disk usage
            batch_data = {
                "pure_logits": logits.to(torch.bfloat16).cpu(),
                "pure_targets": pure_targets.cpu(),
                "loss_mask": loss_mask.cpu(),
            }
            batch_path = os.path.join(output_dir, f"batch_{batch_idx:04d}.pt")
            torch.save(batch_data, batch_path)

            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                print(f"  cached batch {batch_idx + 1}/{num_batches}")

    # Save metadata
    meta = {
        "num_batches": num_batches,
        "batch_size": device_batch_size,
        "seq_len": max_seq_len,
        "block_size": block_size,
        "pure_vocab_size": pure_vocab_size,
        "split": split,
        "skip_pos0": skip_pos0,
        "ckpt_dir": ckpt_dir,
        "step": step,
    }
    meta_path = os.path.join(output_dir, "cache_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done. Metadata saved to {meta_path}")
    return meta


class CachedLogitDataset(torch.utils.data.Dataset):
    """
    Lazily loads cached logit batches from disk.

    Each item is one pre-computed batch: (pure_logits, pure_targets, loss_mask).
    Files are loaded on demand (not held in memory).

    Args:
        cache_dir: Directory containing batch_NNNN.pt files and cache_meta.json.
    """

    def __init__(self, cache_dir):
        self.cache_dir = cache_dir

        meta_path = os.path.join(cache_dir, "cache_meta.json")
        with open(meta_path, "r") as f:
            self.meta = json.load(f)

        self.num_batches = self.meta["num_batches"]

        # Verify files exist
        for i in range(self.num_batches):
            path = os.path.join(cache_dir, f"batch_{i:04d}.pt")
            assert os.path.exists(path), f"Missing cached batch: {path}"

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        path = os.path.join(self.cache_dir, f"batch_{idx:04d}.pt")
        data = torch.load(path, map_location="cpu", weights_only=True)
        return data["pure_logits"], data["pure_targets"], data["loss_mask"]
