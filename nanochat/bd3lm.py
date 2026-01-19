

"""
Block Discrete Denoising Diffusion Language Model.
"""

import math
from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW
from nanochat.bd3lm_utils.bd3lm_loss import compute_bd3lm_loss

@dataclass
class BDLMConfig:
    sequence_len: int = 512
    pure_vocab_size: int = 50304
    all_vocab_size: int = -1
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    
    bucket_size: int = -1
    is_causal: bool = False

    # need for training
    model_name: str = "bd3lm"
    prefix_pure_tokens: int = 1 
    mask_token_id: int = -1

def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last time into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3) # re-assemble
    out = out.to(x.dtype) # ensure input/output dtypes match
    return out

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache, attn_mask=None):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin) # QK rotary embedding
        q, k = norm(q), norm(k) # QK norm
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) # make head be batch dim, i.e. (B, T, H, D) -> (B, H, T, D)

        # Apply KV cache: insert current k,v into cache, get the full view so far
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2) # number of queries in this forward pass
        Tk = k.size(2) # number of keys/values in total (in the cache + current forward pass)

        # Attention: queries attend to keys/values autoregressively. A few cases to handle:
        enable_gqa = self.n_head != self.n_kv_head # Group Query Attention (GQA): duplicate key/value heads to match query heads if desired
        if attn_mask is not None:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False, enable_gqa=enable_gqa)
        elif kv_cache is None or Tq == Tk:
            # During training (no KV cache), attend as usual with causal attention
            # And even if there is KV cache, we can still use this simple version when Tq == Tk
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
        elif Tq == 1:
            # During inference but with a single query in this forward pass:
            # The query has to attend to all the keys/values in the cache
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
        else:
            # During inference AND we have a chunk of queries in this forward pass:
            # First, each query attends to all the cached keys/values (i.e. full prefix)
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device) # True = keep, False = mask
            prefix_len = Tk - Tq
            attn_mask[:, :prefix_len] = True
            # Then, causal attention within this chunk
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)

        # Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache, attn_mask=None):
        x = x + self.attn(norm(x), cos_sin, kv_cache, attn_mask=attn_mask)
        x = x + self.mlp(norm(x))
        return x


class BDLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.all_vocab_size, config.n_embd), # wte changed by all_vocab_size
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.pure_vocab_size, bias=False)
        self.rotary_seq_len = max(config.sequence_len, 1024) * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)
        self._is_causal = self.config.is_causal
        self.inference_mask = None
        self.bucket_size = config.bucket_size

    def init_weights(self):
        self.apply(self._init_weights)
        # zero out classifier weights
        torch.nn.init.zeros_(self.lm_head.weight)
        # zero out c_proj weights in all blocks
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
        # init the rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast the embeddings from fp32 to bf16: optim can tolerate it and it saves memory: both in the model and the activations
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    # TODO: bump base theta more, e.g. 100K is more common more recently
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """ 
        This may be not accurate for our model
        Return the estimated FLOPs per token for the model. Ref: https://arxiv.org/abs/2204.02311 """
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        l, h, q, t = self.config.n_layer, self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return num_flops_per_token

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        # Separate out all parameters into 3 groups (matrix, embedding, lm_head)
        matrix_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params)
        # Create the AdamW optimizer for the embedding and lm_head
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if rank == 0:
            print(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
        ]
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        # Create the Muon optimizer for the linear layers
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        # Combine them the two optimizers into one list
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def forward(self, idx, targets=None, kv_cache=None, attn_mask=None, loss_extras=None):
        """Training: idx/targets are length L; we concat to 2L inside this and apply block mask."""
        if targets is not None:
            B, T = idx.size()
            assert attn_mask is not None, "Train should has attn mask"
            assert self.config.sequence_len == T, "use double seq length when train"
            assert targets.size(1) == T, "Targets should match the base sequence length"
            idx = torch.cat((idx, targets), dim=1)
        else:
            B, T = idx.size()
        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"


        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        if targets is not None:
            cos = self.cos[:, T0:T0+T]
            sin = self.sin[:, T0:T0+T]
            cos_sin = (torch.cat((cos, cos), dim=1), torch.cat((sin, sin), dim=1)) # truncate cache to current sequence length
        else:
            cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx)
        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache, attn_mask=attn_mask)
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x) # (B, T, pure_vocab_size) <- very big tensor, large amount of memory
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        if targets is not None:
            logits = logits[:, :T, :]

            # Get loss_scale and loss_mask from loss_extras
            assert loss_extras is not None and "loss_scale" in loss_extras, "BD3LM requires loss_extras with loss_scale"
            assert "loss_mask" in loss_extras, "BD3LM requires loss_extras with loss_mask"
            loss_scale = loss_extras["loss_scale"]
            loss_mask = loss_extras["loss_mask"]  # True = position to compute loss (not same as input mask for target_shift)

            # Build attention_mask for loss: only compute loss at loss_mask positions
            # loss_mask is bool (True=compute loss), convert to float for attention_mask (1=compute loss)
            attention_mask = loss_mask.float()

            # Also exclude prefix_pure_tokens from loss
            prefix_pure_tokens = self.config.prefix_pure_tokens
            if prefix_pure_tokens > 0:
                attention_mask[:, :prefix_pure_tokens] = 0

            loss, _ = compute_bd3lm_loss(logits, targets, loss_scale, attention_mask=attention_mask)
            return loss
        else:
            # inference: just return the logits directly
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, bucket_size=None, tokens_per_step=1):
        """
        Block diffusion generation: decode one block at a time, unmasking tokens_per_step tokens per forward pass.

        Args:
            tokens: List of token ids (prompt). Batch size is 1.
            max_tokens: Maximum total tokens (prompt + generated).
            bucket_size: Block size for generation. If None, uses config.bucket_size.
            tokens_per_step: Number of tokens to unmask per step within a block (default=1).

        Returns:
            Generated token ids as a 1D tensor.
        """
        assert isinstance(tokens, list), "tokens must be a list"
        assert self.config.mask_token_id != -1, "mask_token_id must be set for generate"

        device = self.get_device()
        mask_id = self.config.mask_token_id

        if bucket_size is None:
            bucket_size = self.bucket_size
        assert bucket_size > 0, "bucket_size must be set in config or passed as arg"

        # Convert to tensor and add batch dim
        ids = torch.tensor([tokens], dtype=torch.long, device=device)  # (1, prompt_len)

        # Pad to next bucket_size boundary with masks
        # If prompt is exactly at boundary, add a full block
        prompt_len = ids.size(1)
        remainder = prompt_len % bucket_size
        pad_len = bucket_size if remainder == 0 else (bucket_size - remainder)
        ids = F.pad(ids, (0, pad_len), value=mask_id)

        while True:
            # Forward pass to get logits
            logits = self.forward(ids)  # (1, T, pure_vocab_size)
            block_logits = logits[:, -bucket_size:, :]  # (1, bucket_size, pure_vocab_size)

            # Get current block tokens
            block_ids = ids[:, -bucket_size:]  # (1, bucket_size)

            # Find masked positions in the block
            is_masked = (block_ids == mask_id)  # (1, bucket_size)
            num_masked = is_masked.sum().item()

            # Get max prob for each position (confidence)
            probs = F.softmax(block_logits.float(), dim=-1)  # (1, bucket_size, vocab)
            max_probs, best_tokens = probs.max(dim=-1)  # (1, bucket_size)

            # Only consider masked positions for unmasking
            # Set confidence of non-masked positions to -inf so they won't be selected
            masked_confidence = max_probs.clone()
            masked_confidence[~is_masked] = -float('inf')

            # Select top tokens_per_step positions to unmask
            num_to_unmask = min(tokens_per_step, num_masked)
            _, top_positions = masked_confidence.topk(num_to_unmask, dim=-1)  # (1, num_to_unmask)

            # Unmask selected positions using argmax tokens
            new_block = block_ids.clone()
            for i in range(num_to_unmask):
                pos = top_positions[0, i].item()
                new_block[0, pos] = best_tokens[0, pos]

            # Update ids with the new block
            ids = torch.cat([ids[:, :-bucket_size], new_block], dim=1)

            # Check if block is fully decoded
            num_masked_after = (ids[:, -bucket_size:] == mask_id).sum().item()
            if num_masked_after == 0:
                if ids.size(1) >= max_tokens:
                    ids = ids[:, :max_tokens]
                    break
                else:
                    # Add a new masked block
                    ids = F.pad(ids, (0, bucket_size), value=mask_id)

        return ids[0]  # Return 1D tensor (remove batch dim)
    

    def forward_for_eval(self, idx, targets, attn_mask):
        """
        Forward pass for evaluation that returns logits instead of loss.

        Args:
            idx: (B, L) input tokens (should be all MASK for eval)
            targets: (B, L) clean target tokens
            attn_mask: attention mask for block diffusion

        Returns:
            logits: (B, L, vocab_size) logits for the xt (input) positions
        """
        B, T = idx.size()
        assert targets.size(1) == T, "Targets should match input length"

        # Concatenate [xt | x0] = [idx | targets]
        idx = torch.cat((idx, targets), dim=1)  # (B, 2L)

        # Get rotary embeddings for 2L sequence
        cos = self.cos[:, :T]
        sin = self.sin[:, :T]
        cos_sin = (torch.cat((cos, cos), dim=1), torch.cat((sin, sin), dim=1))

        # Forward through transformer
        x = self.transformer.wte(idx)
        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache=None, attn_mask=attn_mask)
        x = norm(x)

        # Compute logits
        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        # Return logits for xt part (first L positions)
        return logits[:, :T, :]
        


