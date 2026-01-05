
import os
import torch
from contextlib import nullcontext

from nanochat.common import compute_init, autodetect_device_type, get_base_dir
from nanochat.checkpoint_manager import load_checkpoint, find_last_step, find_largest_model
from nanochat.pdlm import PDLM, PDLMConfig
from nanochat.attn_masks import gen_mask

def load_pdlm_model(model_tag=None, step=None, device_type="auto"):
    """
    Loads the PDLM model and returns (model, device, autocast_context)
    """
    device_type = autodetect_device_type() if device_type == "auto" else device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    
    base_dir = get_base_dir()
    checkpoint_dir = os.path.join(base_dir, "base_checkpoints")
    
    if model_tag is None:
        model_tag = find_largest_model(checkpoint_dir)
        
    ckpt_dir = os.path.join(checkpoint_dir, model_tag)
    
    if step is None:
        step = find_last_step(ckpt_dir)
        
    print(f"Loading model: {model_tag} at step {step} on {device}")

    model_data, _, meta_data = load_checkpoint(ckpt_dir, step, device, load_optimizer=False)
    
    # Handle float32 conversion for CPU/MPS if needed
    if device.type in {"cpu", "mps"}:
        model_data = {k: v.float() if v.dtype == torch.bfloat16 else v for k, v in model_data.items()}
    
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    
    model_config_kwargs = meta_data["model_config"]
    model_config = PDLMConfig(**model_config_kwargs)
    
    with torch.device("meta"):
        model = PDLM(model_config)
    
    model.to_empty(device=device)
    model.init_weights()
    model.load_state_dict(model_data, strict=True, assign=True)
    model.eval()
    
    # Prepare autocast
    ptdtype = torch.bfloat16 if device_type == "cuda" else torch.float32
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()
    
    return model, device, autocast_ctx


def generate_single_sample(model, device, autocast_ctx, prompt_ids, max_new_tokens=64, bucket_size=8, topk=5, temperature=0, transit_topk=10):
    """
    Runs generation for a single sample and returns the full debug info.
    """
    
    # Prepare Attention Mask if non-causal
    attn_mask = None
    if not model.config.is_causal:
        L = model.config.sequence_len
        mask_gen_len = L + bucket_size
        full_mask = gen_mask(mask_gen_len, bucket_size, attn_backend="sdpa", is_causal=False)
        attn_mask = full_mask[mask_gen_len:, mask_gen_len:].to(device)

    # Ensure length alignment with bucket size
    prompt_len = len(prompt_ids)
    max_total_tokens = prompt_len + max_new_tokens
    
    if max_total_tokens % bucket_size != 0:
        max_total_tokens = ((max_total_tokens + bucket_size - 1) // bucket_size) * bucket_size

    with autocast_ctx:
        ids, block_debug = model.generate_with_blocks(
            prompt_ids, 
            max_tokens=max_total_tokens,
            attn_mask=attn_mask,
            bucket_size=bucket_size,
            topk=topk,
            temperature=temperature,
            transit_topk=transit_topk
        )
        
    generated_ids = ids[0].tolist()
    new_tokens = generated_ids[prompt_len:]
    
    return new_tokens, block_debug
