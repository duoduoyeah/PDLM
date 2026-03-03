
"""
Parallel Denosing Language Model
"""

import math
from functools import partial
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW
from nanochat.group_tokenizer.token_map import get_token_map


def any_correct_ce_loss(logits, valid_targets, loss_mask=None, reduction="mean"):
    """
    Cross-entropy loss where ANY of the valid targets is considered correct.

    Loss = -log(Σ P(g) for g in valid_groups)

    This allows the model to predict any of the valid group assignments
    when overlap_k > 1.

    Args:
        logits: (N, num_groups) - logits over group vocabulary (flattened)
        valid_targets: (N, overlap_k) - ALL valid group indices for each position
                       Use -1 for padding (will be masked out)
        loss_mask: optional (N,) bool - True for positions to include in loss
        reduction: "mean" (default) or "none". When "none", return per-token NLL (N,)

    Returns:
        loss: scalar mean loss (reduction="mean") or (N,) per-token NLL (reduction="none")
    """
    # Compute log probabilities
    log_probs = F.log_softmax(logits, dim=-1)

    # Mask invalid targets (where valid_targets == -1)
    valid_mask = valid_targets >= 0  # (N, overlap_k)

    # Clamp to valid indices for gather (masked positions will be ignored)
    safe_targets = valid_targets.clamp(min=0)

    # Gather log probs for valid targets
    valid_log_probs = torch.gather(log_probs, dim=-1, index=safe_targets)  # (N, overlap_k)

    # Mask out invalid positions with -inf before logsumexp
    valid_log_probs = valid_log_probs.masked_fill(~valid_mask, float('-inf'))

    # Log-sum-exp over valid groups: log(Σ P(valid_g))
    log_valid_prob = torch.logsumexp(valid_log_probs, dim=-1)  # (N,)

    # Negative log likelihood
    nll = -log_valid_prob

    if reduction == "none":
        return nll

    # Apply loss_mask if provided
    if loss_mask is not None:
        nll = nll * loss_mask.float()
        return nll.sum() / loss_mask.sum().clamp(min=1)
    else:
        return nll.mean()

# Stage types for PDLM
PDLMStage = Literal["stage1_mtp", "stage1_mask", "stage1_block", "stage2", "both_mtp", "both_mask", "both_block", "block_pdlm_inference", "mask_pdlm", "pdlm_emb"]

@dataclass
class PDLMConfig:
    sequence_len: int = 1024
    pure_vocab_size: int = -1
    num_groups: int = 0
    # Stage types: see PDLMStage
    stage: PDLMStage = "stage2"
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768

    bucket_size: int = -1
    is_causal: bool = True
    # need for training
    model_name: str = "pdlm"
    prefix_pure_tokens: int = 0
    mask_token_id: int = -1  # only needed for stage1_mask and both_mask
    soft_p_within: float = 1.0  # 1.0 = hard mapping, <1.0 = soft (prob of correct group)
    stage1_target_mode: str = "pure"  # "pure" (CE over pure_vocab) or "group" (any_correct_ce over num_groups)
    noise_count: int = 64  # pdlm_emb: total tokens in noise average (including target)
    mask_pdlm_4state: bool = False  # mask_pdlm: use 4-state variant (k states instead of k+1)

    # MTP (Multi-Token Prediction) config for both_mtp stage
    n_future_tokens: int = 4       # K: number of group tokens to predict for Stage 1
    mtp_loss_beta: float = 0.8     # Exponential decay factor for MTP loss weights
    mtp_loss_weight: float = 1.0   # Stage 1 MTP loss weight relative to Stage 2 (which is 1.0)

    def __post_init__(self):
        valid_stages = {"stage1_mtp", "stage1_mask", "stage1_block", "stage2", "both_mtp", "both_mask", "both_block", "block_pdlm_inference", "mask_pdlm", "pdlm_emb"}
        if self.stage not in valid_stages:
            raise ValueError(f"Invalid stage: {self.stage}. Must be one of {valid_stages}")

    def get_vocab_sizes(self):
        """Compute wte and lm_head sizes based on stage."""
        if self.stage == "stage1_mtp":
            wte_size = self.pure_vocab_size
            lm_head_size = self.num_groups
        elif self.stage == "stage1_mask":
            assert self.mask_token_id != -1, "stage1_mask requires mask_token_id"
            wte_size = self.pure_vocab_size + 1  # pure + MASK
            lm_head_size = self.num_groups
        elif self.stage == "stage1_block":
            wte_size = self.pure_vocab_size
            if self.stage1_target_mode == "group":
                lm_head_size = self.num_groups
            else:
                lm_head_size = self.pure_vocab_size  # Full vocab for pure-target mode
        elif self.stage == "stage2":
            wte_size = self.pure_vocab_size + self.num_groups
            lm_head_size = self.pure_vocab_size
        elif self.stage == "both_mtp":
            wte_size = self.pure_vocab_size + self.num_groups
            lm_head_size = self.pure_vocab_size + self.num_groups
        elif self.stage == "both_block":
            wte_size = self.pure_vocab_size + self.num_groups
            lm_head_size = self.pure_vocab_size + self.num_groups
        elif self.stage == "block_pdlm_inference":
            wte_size = self.pure_vocab_size + self.num_groups
            lm_head_size = self.pure_vocab_size
        elif self.stage == "both_mask":
            assert self.mask_token_id != -1, "both_mask requires mask_token_id"
            wte_size = self.pure_vocab_size + self.num_groups + 1
            lm_head_size = self.pure_vocab_size + self.num_groups
        elif self.stage == "mask_pdlm":
            wte_size = self.pure_vocab_size + self.num_groups + 1  # pure + groups + mask
            lm_head_size = self.pure_vocab_size
        elif self.stage == "pdlm_emb":
            wte_size = self.pure_vocab_size  # no group tokens in wte
            lm_head_size = self.pure_vocab_size
        else:
            raise ValueError(f"Unknown stage: {self.stage}")
        return wte_size, lm_head_size

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


class PDLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        wte_size, lm_head_size = config.get_vocab_sizes()
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(wte_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, lm_head_size, bias=False)
        self.rotary_seq_len = max(config.sequence_len, 1024) * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)
        self._token_map = None
        self._is_causal = self.config.is_causal
        self.register_buffer("group_to_pure_mask", None, persistent=False)
        self.inference_mask = None
        self.bucket_size = config.bucket_size

        # MTP components for both_mtp stage
        if config.stage == "both_mtp":
            from nanochat.mtp_head import MTPHead

            # Group embeddings for MTP teacher forcing
            self.group_wte = nn.Embedding(config.num_groups, config.n_embd)

            # Wrapper to slice lm_head output to only group logits
            # MTPHead expects lm_head to output num_groups logits
            class _GroupLogitHead(nn.Module):
                def __init__(self, lm_head, pure_vocab_size):
                    super().__init__()
                    self.lm_head = lm_head
                    self.offset = pure_vocab_size

                def forward(self, x):
                    return self.lm_head(x)[:, :, self.offset:]

            self._group_logit_head = _GroupLogitHead(self.lm_head, config.pure_vocab_size)

            # MTP head (uses wrapper that slices to group logits)
            self.mtp_head = MTPHead(
                n_embd=config.n_embd,
                n_head=config.n_head,
                n_future_tokens=config.n_future_tokens,
                group_wte=self.group_wte,
                lm_head=self._group_logit_head,
            )

    def register_group_mask(self, group_to_pure_mask):
        """Register group mask for stage1_block inference collapse."""
        self.register_buffer("group_to_pure_mask", group_to_pure_mask.float())

    def get_group_head_weight(self):
        """Get collapsed group head: (num_groups, n_embd)"""
        return self.group_to_pure_mask @ self.lm_head.weight

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

        # Init MTP components for both_mtp stage
        if self.config.stage == "both_mtp":
            self.mtp_head.init_weights()
            # Cast group_wte to bf16
            if self.group_wte.weight.device.type == "cuda":
                self.group_wte.to(dtype=torch.bfloat16)

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

        # For both_mtp stage, also include MTP head parameters
        if self.config.stage == "both_mtp":
            # MTP block parameters go to matrix optimizer (Muon)
            mtp_matrix_params = list(self.mtp_head.block.parameters()) + list(self.mtp_head.proj.parameters())
            matrix_params = matrix_params + mtp_matrix_params
            # Group embeddings go to AdamW like other embeddings
            group_wte_params = list(self.group_wte.parameters())
            embedding_params = embedding_params + group_wte_params
            assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params)
        else:
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

    def forward(self, idx, targets=None, kv_cache=None, attn_mask=None, loss_extras=None, return_separate_losses=False, return_nll=False):
        """Training: idx/targets are length L; we concat to 2L inside this and apply block mask."""
        # Dispatch to specialized forward methods for certain stages
        if self.config.stage == "stage1_block" and targets is not None:
            return self._forward_stage1_block(idx, targets, attn_mask, loss_extras, return_nll=return_nll)
        if self.config.stage == "both_block" and targets is not None:
            return self._forward_both_block(idx, targets, attn_mask, loss_extras, return_separate_losses)
        if self.config.stage == "block_pdlm_inference" and targets is not None:
            return self._forward_block_pdlm_inference(idx, targets, attn_mask, loss_extras, return_separate_losses)
        if self.config.stage == "mask_pdlm" and targets is not None:
            return self._forward_mask_pdlm(idx, targets, attn_mask, loss_extras, return_nll=return_nll)
        if self.config.stage == "pdlm_emb" and targets is not None:
            return self._forward_pdlm_emb(idx, targets, attn_mask, loss_extras, return_nll=return_nll)
        if self.config.stage == "both_mtp" and targets is not None:
            return self._forward_both_mtp(idx, targets, attn_mask, loss_extras)

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
            # training: given the targets, compute and return the loss
            logits = logits[:, :T, :]  # first T positions (xt half)
            V = logits.size(-1)

            if self.config.stage == "stage1_mask":
                # Stage 1 MASK: predict group tokens from MASK positions
                # logits: (B, T, num_groups)
                # group_targets: (B, T, overlap_k) from loss_extras
                # loss_mask: (B, T) from loss_extras
                assert loss_extras is not None and "group_targets" in loss_extras
                group_targets = loss_extras["group_targets"]
                loss_mask = loss_extras["loss_mask"]
                overlap_k = group_targets.size(-1)

                # Compute any-correct loss at MASK positions
                loss = any_correct_ce_loss(
                    logits.reshape(-1, V),
                    group_targets.reshape(-1, overlap_k),
                    loss_mask.reshape(-1)
                )
            elif loss_extras is not None and "loss_mask" in loss_extras:
                # Stage2: use loss_mask to compute loss only on block positions
                loss_mask = loss_extras["loss_mask"]
                # Compute per-token cross-entropy
                log_probs = F.log_softmax(logits, dim=-1)
                target_log_probs = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1))
                nll = -target_log_probs.squeeze(-1)  # (B, T)
                # Apply mask and compute mean
                loss = (nll * loss_mask).sum() / loss_mask.sum().clamp(min=1)
            else:
                # Legacy mode: use ignore_index for prefix_pure_tokens
                loss_targets = targets
                prefix_pure_tokens = self.config.prefix_pure_tokens
                if prefix_pure_tokens > 0:
                    loss_targets = loss_targets.clone()
                    loss_targets[:, :prefix_pure_tokens] = -1
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    loss_targets.reshape(-1),
                    ignore_index=-1,
                    reduction='mean',
                )
            return loss
        else:
            # inference: just return the logits directly
            return logits

    def _forward_both_mtp(self, idx, targets, attn_mask, loss_extras):
        """
        Forward pass for both_mtp stage: combines Stage 1 (MTP) and Stage 2 (denoising) in one pass.

        Input structure (2L total):
        - First L (idx/xt): group tokens at block positions -> Stage 2 predicts pure tokens
        - Second L (targets/x0): pure tokens -> Stage 1 predicts K future group tokens via MTP

        Args:
            idx: (B, L) input tokens (xt: group tokens at block positions)
            targets: (B, L) target tokens (x0: pure tokens)
            attn_mask: (2L, 2L) block diffusion attention mask
            loss_extras: dict with "loss_mask" (B, L) and "mtp_targets" (B, L, K, overlap_k)

        Returns:
            combined_loss: scalar loss = mtp_loss_weight * stage1_loss + stage2_loss
        """
        from nanochat.mtp_head import compute_mtp_loss

        B, T = idx.size()
        assert attn_mask is not None, "Train should have attn mask"
        assert self.config.sequence_len == T, "use double seq length when train"
        assert targets.size(1) == T, "Targets should match the base sequence length"

        # Concatenate [xt | x0] to form (B, 2L) input
        combined_idx = torch.cat((idx, targets), dim=1)  # (B, 2L)

        # Get rotary embeddings for 2L sequence (same positions for both halves)
        cos = self.cos[:, :T]
        sin = self.sin[:, :T]
        cos_sin = (torch.cat((cos, cos), dim=1), torch.cat((sin, sin), dim=1))

        # Forward through transformer
        x = self.transformer.wte(combined_idx)
        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache=None, attn_mask=attn_mask)
        x = norm(x)

        # Compute logits
        softcap = 15
        logits = self.lm_head(x)  # (B, 2L, pure_vocab_size + num_groups)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        # Split logits for Stage 1 and Stage 2
        pure_vocab_size = self.config.pure_vocab_size

        # Stage 2 Loss: Group → Pure denoising (first L positions, pure vocab logits)
        stage2_logits = logits[:, :T, :pure_vocab_size]  # (B, T, pure_vocab_size)
        loss_mask = loss_extras["loss_mask"]  # (B, T)

        log_probs = F.log_softmax(stage2_logits, dim=-1)
        target_log_probs = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1))
        nll = -target_log_probs.squeeze(-1)  # (B, T)
        stage2_loss = (nll * loss_mask).sum() / loss_mask.sum().clamp(min=1)

        # Stage 1 Loss: Pure → Group MTP (second L positions, group vocab logits)
        # Main model's group logits for first token prediction
        main_group_logits = logits[:, T:, pure_vocab_size:]  # (B, T, num_groups)

        # Get hidden states from x0 portion for MTP head
        x0_hidden = x[:, T:, :]  # (B, T, D)

        # MTP targets: (B, T, K, overlap_k) - all K future group tokens
        mtp_targets = loss_extras["mtp_targets"]  # (B, T, K, overlap_k)
        K = mtp_targets.size(2)

        # First group token for teacher forcing (use ground truth from mtp_targets[:,:,0,:])
        # Take the first valid group (index 0) for teacher forcing
        first_group_tok = mtp_targets[:, :, 0, 0]  # (B, T)

        # MTP head predicts K-1 additional tokens (2nd through Kth)
        # Rotary embeddings: [1, T, 1, head_dim//2] broadcasts with [B, T, n_head, head_dim]
        mtp_cos = cos
        mtp_sin = sin

        # Teacher forcing targets for MTP: use ground truth for k=0..K-2 to predict k=1..K-1
        # mtp_targets[:, :, k, 0] gives the k-th future group token (using first overlap option)
        mtp_teacher_targets = mtp_targets[:, :, :-1, 0]  # (B, T, K-1) - targets for teacher forcing

        mtp_logits = self.mtp_head(
            h=x0_hidden,
            cos=mtp_cos,
            sin=mtp_sin,
            first_group_tok=first_group_tok,
            targets=mtp_teacher_targets,
        )  # (B, T, K-1, num_groups)

        # Compute MTP loss using any-correct loss
        stage1_loss = compute_mtp_loss(
            main_logits=main_group_logits,
            mtp_logits=mtp_logits,
            targets=mtp_targets,
            mtp_loss_beta=self.config.mtp_loss_beta,
        )

        # Combine losses
        combined_loss = self.config.mtp_loss_weight * stage1_loss + stage2_loss

        return combined_loss

    def _forward_stage1_block(self, idx, targets, attn_mask, loss_extras, return_nll=False):
        """
        Forward pass for stage1_block (pure-target mode): L pure tokens in, L×L block-causal mask.
        Position k in block i predicts the pure token at position k in block i+1.
        No 2L concatenation — just direct forward through transformer with block-causal mask.

        Training: Standard CE loss against pure tokens.
        Inference: Collapse lm_head to groups via group_to_pure_mask @ lm_head.weight.

        Args:
            idx: (B, L) pure token inputs
            targets: (B, L) pure tokens (unused directly, kept for API consistency)
            attn_mask: (L, L) block-causal mask
            loss_extras: dict with "pure_targets" (B, T) and "loss_mask" (B, T)
            return_nll: if True, return per-token NLL (B, T) instead of scalar loss

        Returns:
            loss: scalar loss, or (B, T) per-token NLL if return_nll=True
        """
        B, T = idx.size()
        assert attn_mask is not None, "stage1_block requires attention mask"

        # Rotary embeddings for L tokens
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        # Forward through transformer (no 2L concatenation)
        x = self.transformer.wte(idx)
        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache=None, attn_mask=attn_mask)
        x = norm(x)

        # Compute logits: (B, T, pure_vocab_size)
        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        # Branch on target mode
        assert loss_extras is not None
        if "block_targets" in loss_extras:
            # Group-target mode: any_correct_ce_loss against group targets
            block_targets = loss_extras["block_targets"]  # (B, T, overlap_k)
            block_loss_mask = loss_extras["block_loss_mask"]  # (B, T)
            V = logits.size(-1)
            overlap_k = block_targets.size(-1)
            if return_nll:
                nll_flat = any_correct_ce_loss(
                    logits.reshape(-1, V),
                    block_targets.reshape(-1, overlap_k),
                    reduction="none",
                )
                return nll_flat.reshape(B, T)  # (B, T) per-token NLL
            loss = any_correct_ce_loss(
                logits.reshape(-1, V),
                block_targets.reshape(-1, overlap_k),
                block_loss_mask.reshape(-1),
            )
        else:
            # Pure-target mode: standard CE against pure tokens
            assert "pure_targets" in loss_extras
            pure_targets = loss_extras["pure_targets"]  # (B, T)
            loss_mask = loss_extras["loss_mask"]  # (B, T)

            log_probs = F.log_softmax(logits, dim=-1)
            target_log_probs = torch.gather(log_probs, dim=-1, index=pure_targets.unsqueeze(-1))
            nll = -target_log_probs.squeeze(-1)
            if return_nll:
                return nll  # (B, T) per-token NLL, before masking/reducing
            loss = (nll * loss_mask).sum() / loss_mask.sum().clamp(min=1)

        return loss

    def _forward_both_block(self, idx, targets, attn_mask, loss_extras, return_separate_losses=False, return_nll=False):
        """
        Forward pass for both_block stage: combines Stage 1 (block→block) and Stage 2 (denoising).

        Same 2L structure as both_mtp:
        - First L (idx/xt): group tokens at block positions -> Stage 2 predicts pure tokens
        - Second L (targets/x0): pure tokens -> Stage 1 predicts group tokens for next block

        Stage 1 uses block→block prediction via lm_head (no MTP head).

        Args:
            idx: (B, L) input tokens (xt: group tokens at block positions)
            targets: (B, L) target tokens (x0: pure tokens)
            attn_mask: (2L, 2L) block diffusion attention mask
            loss_extras: dict with "loss_mask" (B, L), "block_targets" (B, L, overlap_k),
                         "block_loss_mask" (B, L)
            return_separate_losses: if True, return (combined_loss, stage1_loss, stage2_loss)

        Returns:
            combined_loss: scalar loss = mtp_loss_weight * stage1_loss + stage2_loss
            If return_separate_losses=True: (combined_loss, stage1_loss.detach(), stage2_loss.detach())
        """
        B, T = idx.size()
        assert attn_mask is not None, "Train should have attn mask"
        assert self.config.sequence_len == T, "use double seq length when train"
        assert targets.size(1) == T, "Targets should match the base sequence length"

        # Concatenate [xt | x0] to form (B, 2L) input
        combined_idx = torch.cat((idx, targets), dim=1)  # (B, 2L)

        # Get rotary embeddings for 2L sequence (same positions for both halves)
        cos = self.cos[:, :T]
        sin = self.sin[:, :T]
        cos_sin = (torch.cat((cos, cos), dim=1), torch.cat((sin, sin), dim=1))

        # Forward through transformer
        x = self.transformer.wte(combined_idx)
        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache=None, attn_mask=attn_mask)
        x = norm(x)

        # Compute logits
        softcap = 15
        logits = self.lm_head(x)  # (B, 2L, pure_vocab_size + num_groups)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        pure_vocab_size = self.config.pure_vocab_size

        # Stage 2 Loss: Group -> Pure denoising (first L positions, pure vocab logits)
        stage2_logits = logits[:, :T, :pure_vocab_size]  # (B, T, pure_vocab_size)
        loss_mask = loss_extras["loss_mask"]  # (B, T)

        log_probs = F.log_softmax(stage2_logits, dim=-1)
        target_log_probs = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1))
        nll = -target_log_probs.squeeze(-1)  # (B, T)
        stage2_loss = (nll * loss_mask).sum() / loss_mask.sum().clamp(min=1)

        # Stage 1 Loss: block->block on x0 half (second L positions, group logits)
        group_logits = logits[:, T:, pure_vocab_size:]  # (B, T, num_groups)
        block_targets = loss_extras["block_targets"]  # (B, T, overlap_k)
        block_loss_mask = loss_extras["block_loss_mask"]  # (B, T)
        V_group = group_logits.size(-1)
        overlap_k = block_targets.size(-1)

        stage1_loss = any_correct_ce_loss(
            group_logits.reshape(-1, V_group),
            block_targets.reshape(-1, overlap_k),
            block_loss_mask.reshape(-1),
        )

        if return_nll:
            return stage1_loss, stage2_loss

        # Combine losses
        combined_loss = self.config.mtp_loss_weight * stage1_loss + stage2_loss

        if return_separate_losses:
            return combined_loss, stage1_loss.detach(), stage2_loss.detach()
        return combined_loss

    def _forward_block_pdlm_inference(self, idx, targets, attn_mask, loss_extras, return_separate_losses=False, return_nll=False):
        """
        Forward pass for block_pdlm_inference stage.

        Same 2L structure as both_block:
        - First L (idx/xt): mixed pure/group tokens -> Stage 2 predicts pure tokens
        - Second L (targets/x0): pure tokens -> Stage 1 predicts next block pure tokens

        Stage 2: all positions predict pure tokens. Targets vary by input type:
          - Group-input positions: target = current block's pure token (denoise)
          - Pure-input positions: target = next block's pure token (predict ahead)

        Stage 1: pure tokens predict next block's pure tokens (shifted by block_size).

        Args:
            idx: (B, L) input tokens (xt: mixed pure/group tokens)
            targets: (B, L) target tokens (x0: pure tokens)
            attn_mask: (2L, 2L) block diffusion attention mask
            loss_extras: dict with "stage2_targets", "stage2_loss_mask",
                         "stage1_targets", "stage1_loss_mask"
            return_separate_losses: if True, return (combined_loss, stage1_loss, stage2_loss)

        Returns:
            combined_loss: scalar loss = mtp_loss_weight * stage1_loss + stage2_loss
        """
        B, T = idx.size()
        assert attn_mask is not None, "Train should have attn mask"
        assert self.config.sequence_len == T, "use double seq length when train"
        assert targets.size(1) == T, "Targets should match the base sequence length"

        # Concatenate [xt | x0] to form (B, 2L) input
        combined_idx = torch.cat((idx, targets), dim=1)  # (B, 2L)

        # Get rotary embeddings for 2L sequence (same positions for both halves)
        cos = self.cos[:, :T]
        sin = self.sin[:, :T]
        cos_sin = (torch.cat((cos, cos), dim=1), torch.cat((sin, sin), dim=1))

        # Forward through transformer
        x = self.transformer.wte(combined_idx)
        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache=None, attn_mask=attn_mask)
        x = norm(x)

        # Compute logits (lm_head outputs pure_vocab_size)
        softcap = 15
        logits = self.lm_head(x)  # (B, 2L, pure_vocab_size)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        # Stage 2 Loss: xt half (first L positions) predicts pure tokens
        stage2_logits = logits[:, :T, :]  # (B, T, pure_vocab_size)
        stage2_targets = loss_extras["stage2_targets"]  # (B, T)
        stage2_loss_mask = loss_extras["stage2_loss_mask"]  # (B, T)

        log_probs_s2 = F.log_softmax(stage2_logits, dim=-1)
        target_log_probs_s2 = torch.gather(log_probs_s2, dim=-1, index=stage2_targets.unsqueeze(-1))
        nll_s2 = -target_log_probs_s2.squeeze(-1)  # (B, T)
        stage2_loss = (nll_s2 * stage2_loss_mask).sum() / stage2_loss_mask.sum().clamp(min=1)

        # Stage 1 Loss: x0 half (second L positions) predicts next block pure tokens
        stage1_logits = logits[:, T:, :]  # (B, T, pure_vocab_size)
        stage1_targets = loss_extras["stage1_targets"]  # (B, T)
        stage1_loss_mask = loss_extras["stage1_loss_mask"]  # (B, T)

        log_probs_s1 = F.log_softmax(stage1_logits, dim=-1)
        target_log_probs_s1 = torch.gather(log_probs_s1, dim=-1, index=stage1_targets.unsqueeze(-1))
        nll_s1 = -target_log_probs_s1.squeeze(-1)  # (B, T)
        stage1_loss = (nll_s1 * stage1_loss_mask).sum() / stage1_loss_mask.sum().clamp(min=1)

        if return_nll:
            return stage1_loss, stage2_loss

        # Combine losses
        combined_loss = self.config.mtp_loss_weight * stage1_loss + stage2_loss

        if return_separate_losses:
            return combined_loss, stage1_loss.detach(), stage2_loss.detach()
        return combined_loss

    def _forward_mask_pdlm(self, idx, targets, attn_mask, loss_extras, return_nll=False):
        """
        Forward pass for mask_pdlm stage.

        Same 2L structure as block_pdlm_inference:
        - First L (idx/xt): mixed mask/group/pure tokens -> predicts self's pure tokens
        - Second L (targets/x0): pure tokens (context only, no loss)

        Unified loss on xt half only, at mask+group positions (via loss_mask).
        Target = self's pure token (no shifted targets).

        Args:
            idx: (B, L) input tokens (xt: mixed mask/group/pure tokens)
            targets: (B, L) target tokens (x0: pure tokens)
            attn_mask: (2L, 2L) block diffusion attention mask
            loss_extras: dict with "loss_mask" (B, L)
            return_nll: if True, return single NLL scalar for gradient tracking

        Returns:
            loss: scalar loss
        """
        B, T = idx.size()
        assert attn_mask is not None, "Train should have attn mask"
        assert self.config.sequence_len == T, "use double seq length when train"
        assert targets.size(1) == T, "Targets should match the base sequence length"

        # Concatenate [xt | x0] to form (B, 2L) input
        combined_idx = torch.cat((idx, targets), dim=1)  # (B, 2L)

        # Get rotary embeddings for 2L sequence (same positions for both halves)
        cos = self.cos[:, :T]
        sin = self.sin[:, :T]
        cos_sin = (torch.cat((cos, cos), dim=1), torch.cat((sin, sin), dim=1))

        # Forward through transformer
        x = self.transformer.wte(combined_idx)
        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache=None, attn_mask=attn_mask)
        x = norm(x)

        # Compute logits (lm_head outputs pure_vocab_size)
        softcap = 15
        logits = self.lm_head(x)  # (B, 2L, pure_vocab_size)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        # Unified loss on xt half only
        xt_logits = logits[:, :T, :]  # (B, T, pure_vocab_size)
        loss_mask = loss_extras["loss_mask"]  # (B, T)

        log_probs = F.log_softmax(xt_logits, dim=-1)
        target_log_probs = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1))
        nll = -target_log_probs.squeeze(-1)  # (B, T)
        loss = (nll * loss_mask).sum() / loss_mask.sum().clamp(min=1)

        if return_nll:
            return loss

        return loss

    def forward_for_eval_mask_pdlm(self, idx, targets, attn_mask):
        """
        Forward pass for mask_pdlm evaluation that returns xt-half logits.

        Returns:
            logits: (B, L, pure_vocab_size) logits from xt half
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

        # Compute logits (lm_head outputs pure_vocab_size)
        softcap = 15
        logits = self.lm_head(x)  # (B, 2L, pure_vocab_size)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        # Return only xt half logits
        return logits[:, :T, :]  # (B, L, pure_vocab_size)

    def _forward_pdlm_emb(self, idx, targets, attn_mask, loss_extras, return_nll=False):
        """
        Forward pass for pdlm_emb stage.

        Same 2L structure as mask_pdlm but replaces discrete group tokens with
        averaged noise embeddings at block positions.

        For each block position, construct:
            norm(mean(norm(wte(tok_i)) for tok_i in noise_set))
        where noise_set is randomly sampled pure tokens (including the target).

        Args:
            idx: (B, L) input tokens (xt: placeholder at block positions)
            targets: (B, L) target tokens (x0: pure tokens)
            attn_mask: (2L, 2L) block diffusion attention mask
            loss_extras: dict with "loss_mask" (B, L) and "noise_tokens" (B, block_region_len, noise_count)
            return_nll: if True, return single NLL scalar for gradient tracking

        Returns:
            loss: scalar loss
        """
        B, T = idx.size()
        assert attn_mask is not None, "Train should have attn mask"
        assert self.config.sequence_len == T, "use double seq length when train"
        assert targets.size(1) == T, "Targets should match the base sequence length"

        # Concatenate [xt | x0] to form (B, 2L) input
        combined_idx = torch.cat((idx, targets), dim=1)  # (B, 2L)

        # Get rotary embeddings for 2L sequence (same positions for both halves)
        cos = self.cos[:, :T]
        sin = self.sin[:, :T]
        cos_sin = (torch.cat((cos, cos), dim=1), torch.cat((sin, sin), dim=1))

        # Forward through transformer: embed tokens then override block positions
        x = self.transformer.wte(combined_idx)
        x = norm(x)

        # Override xt block positions with averaged noise embeddings
        noise_tokens = loss_extras["noise_tokens"]  # (B, block_region_len, noise_count)
        B_n, block_region_len, noise_count = noise_tokens.shape
        D = x.size(-1)
        block_start = 0  # pdlm_emb uses prefix_sliding_tokens=0

        norm_of_emb = norm(self.transformer.wte.weight)  # (vocab_size, D)
        noise_embs = norm_of_emb[noise_tokens]  # (B, block_region_len, noise_count, D)
        avg_embs = norm(noise_embs.mean(dim=2)).view(B_n, block_region_len, D)
        x[:, block_start:block_start + block_region_len] = avg_embs

        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache=None, attn_mask=attn_mask)
        x = norm(x)

        # Compute logits (lm_head outputs pure_vocab_size)
        softcap = 15
        logits = self.lm_head(x)  # (B, 2L, pure_vocab_size)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        # Unified loss on xt half only
        xt_logits = logits[:, :T, :]  # (B, T, pure_vocab_size)
        loss_mask = loss_extras["loss_mask"]  # (B, T)

        log_probs = F.log_softmax(xt_logits, dim=-1)
        target_log_probs = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1))
        nll = -target_log_probs.squeeze(-1)  # (B, T)
        loss = (nll * loss_mask).sum() / loss_mask.sum().clamp(min=1)

        if return_nll:
            return loss

        return loss

    def forward_for_eval_pdlm_emb(self, idx, targets, attn_mask, noise_tokens):
        """
        Forward pass for pdlm_emb evaluation that returns xt-half logits.

        Args:
            idx: (B, L) input tokens (xt: placeholder at block positions)
            targets: (B, L) target tokens (x0: pure tokens)
            attn_mask: (2L, 2L) block diffusion attention mask
            noise_tokens: (B, block_region_len, noise_count)

        Returns:
            logits: (B, L, pure_vocab_size) logits from xt half
        """
        B, T = idx.size()
        assert targets.size(1) == T, "Targets should match input length"

        # Concatenate [xt | x0] = [idx | targets]
        combined_idx = torch.cat((idx, targets), dim=1)  # (B, 2L)

        # Get rotary embeddings for 2L sequence
        cos = self.cos[:, :T]
        sin = self.sin[:, :T]
        cos_sin = (torch.cat((cos, cos), dim=1), torch.cat((sin, sin), dim=1))

        # Forward through transformer: embed tokens then override block positions
        x = self.transformer.wte(combined_idx)
        x = norm(x)

        # Override xt block positions with averaged noise embeddings
        B_n, block_region_len, noise_count = noise_tokens.shape
        D = x.size(-1)
        block_start = 0

        norm_of_emb = norm(self.transformer.wte.weight)  # (vocab_size, D)
        noise_embs = norm_of_emb[noise_tokens]  # (B, block_region_len, noise_count, D)
        avg_embs = norm(noise_embs.mean(dim=2)).view(B_n, block_region_len, D)
        x[:, block_start:block_start + block_region_len] = avg_embs

        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache=None, attn_mask=attn_mask)
        x = norm(x)

        # Compute logits
        softcap = 15
        logits = self.lm_head(x)  # (B, 2L, pure_vocab_size)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        # Return only xt half logits
        return logits[:, :T, :]  # (B, L, pure_vocab_size)

    @torch.inference_mode()
    def generate_mask_pdlm(self, tokens, max_new_tokens, block_size=None,
                           temperature=0.0, topk=0, seed=42,
                           collapse_topk=256, num_denoise_steps=4,
                           stop_token=None):
        """
        Generate tokens using mask_pdlm model (iterative denoising).

        Each block is generated in (1 + num_denoise_steps) steps:
        1. Append K mask tokens to xt, zeros to x0 → forward 2L → pure logits
           at mask positions → collapse to groups → initial group tokens
        2-5. Build xt with current pure/group mix, forward 2L, sample pure token,
             collapse remaining positions back to groups

        Args:
            tokens: list of pure token ids (prompt)
            max_new_tokens: number of new pure tokens to generate
            block_size: K, tokens per block (default from config.bucket_size)
            temperature: sampling temperature (0 = greedy)
            topk: top-k sampling (0 = disabled)
            seed: random seed for sampling
            collapse_topk: top-k for pure→group collapse (256 default, -1 for dense)
            num_denoise_steps: number of denoising steps per block (default 4)
            stop_token: if set, stop generation when this token is produced (e.g. bos_token_id)

        Returns:
            generated_tokens: tensor of pure token ids (1, total_len)
            debug_blocks: list of per-block generation details
        """
        assert self.config.stage == "mask_pdlm", \
            "generate_mask_pdlm requires mask_pdlm stage"
        assert isinstance(tokens, list), "tokens must be a list"

        device = self.get_device()
        if block_size is None:
            block_size = self.config.bucket_size
        K = block_size

        pure_vocab_size = self.config.pure_vocab_size
        group_offset = pure_vocab_size
        mask_token_id = pure_vocab_size + self.config.num_groups  # last token in wte

        # Setup RNG for sampling
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)

        # Current sequence (pure tokens)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)  # (1, prompt_len)
        prompt_len = ids.size(1)
        target_len = prompt_len + max_new_tokens

        debug_blocks = []
        block_step = 0

        while ids.size(1) < target_len:
            current_len = ids.size(1)
            T = ids.size(1)

            # === Step 1: Forward with mask tokens → collapse to groups ===
            # Build xt: pure context + K mask tokens
            mask_tokens = torch.full((1, K), mask_token_id, dtype=torch.long, device=device)
            xt = torch.cat([ids, mask_tokens], dim=1)  # (1, T+K)

            # Build x0: pure context + placeholder zeros
            x0 = torch.cat([ids, torch.zeros(1, K, dtype=torch.long, device=device)], dim=1)

            total_len = T + K
            combined_idx = torch.cat([xt, x0], dim=1)  # (1, 2*(T+K))

            cos = self.cos[:, :total_len]
            sin = self.sin[:, :total_len]
            cos_sin_2l = (torch.cat([cos, cos], dim=1), torch.cat([sin, sin], dim=1))

            from nanochat.attn_masks import gen_mask
            attn_mask = gen_mask(total_len, K, attn_backend="sdpa",
                                 is_causal=self._is_causal,
                                 prefix_sliding_tokens=0).to(device)

            softcap = 15
            x = self.transformer.wte(combined_idx)
            x = norm(x)
            for block in self.transformer.h:
                x = block(x, cos_sin_2l, kv_cache=None, attn_mask=attn_mask)
            x = norm(x)

            logits_2l = self.lm_head(x)
            logits_2l = logits_2l.float()
            logits_2l = softcap * torch.tanh(logits_2l / softcap)

            # Get pure logits at mask positions (xt half, positions T:T+K)
            block_logits = logits_2l[:, T:T+K, :]  # (1, K, pure_vocab_size)

            # Initialize block state
            block_pure = torch.full((1, K), -1, dtype=torch.long, device=device)

            if self.config.mask_pdlm_4state:
                # 4-state: step 1 samples p0 directly + collapses rest to groups
                pos_logits = block_logits[:, 0, :]
                if temperature > 0:
                    if topk > 0:
                        v, _ = torch.topk(pos_logits, min(topk, pos_logits.size(-1)), dim=-1)
                        pos_logits[pos_logits < v[:, [-1]]] = float('-inf')
                    probs = F.softmax(pos_logits / temperature, dim=-1)
                    sampled = torch.multinomial(probs, num_samples=1, generator=rng)
                else:
                    sampled = pos_logits.argmax(dim=-1, keepdim=True)
                block_pure[:, 0] = sampled.squeeze(-1)

                # Collapse remaining positions (1..K-1) to groups
                if K > 1:
                    remaining_logits = block_logits[:, 1:, :]
                    remaining_groups = self.collapse_pure_to_group(remaining_logits, collapse_topk=collapse_topk)
                    block_groups = torch.full((1, K), -1, dtype=torch.long, device=device)
                    block_groups[:, 1:] = remaining_groups
                else:
                    block_groups = torch.full((1, K), -1, dtype=torch.long, device=device)

                denoise_start = 1
            else:
                # 5-state: step 1 collapses all to groups, no sampling
                initial_groups = self.collapse_pure_to_group(block_logits, collapse_topk=collapse_topk)
                block_groups = initial_groups
                denoise_start = 0

            debug_info = {
                "block_step": block_step,
                "context_len": current_len,
                "initial_groups": block_groups.cpu().tolist()[0] if denoise_start == 0 else None,
                "denoise_steps": [],
            }

            # === Iterative denoising ===
            for denoise_step in range(denoise_start, num_denoise_steps):
                # Build xt: pure context + current block (mix of pure and group tokens)
                block_tokens = torch.where(
                    block_pure >= 0,
                    block_pure,
                    block_groups + group_offset,
                )  # (1, K)
                xt = torch.cat([ids, block_tokens], dim=1)

                x0 = torch.cat([ids, torch.zeros(1, K, dtype=torch.long, device=device)], dim=1)

                total_len = T + K
                combined_idx = torch.cat([xt, x0], dim=1)

                cos = self.cos[:, :total_len]
                sin = self.sin[:, :total_len]
                cos_sin_2l = (torch.cat([cos, cos], dim=1), torch.cat([sin, sin], dim=1))

                attn_mask = gen_mask(total_len, K, attn_backend="sdpa",
                                     is_causal=self._is_causal,
                                     prefix_sliding_tokens=0).to(device)

                x = self.transformer.wte(combined_idx)
                x = norm(x)
                for block in self.transformer.h:
                    x = block(x, cos_sin_2l, kv_cache=None, attn_mask=attn_mask)
                x = norm(x)

                logits_2l = self.lm_head(x)
                logits_2l = logits_2l.float()
                logits_2l = softcap * torch.tanh(logits_2l / softcap)

                block_logits = logits_2l[:, T:T+K, :]

                # Sample next pure token (left-to-right: reveal position denoise_step)
                pos = denoise_step
                if pos < K:
                    pos_logits = block_logits[:, pos, :]

                    if temperature > 0:
                        if topk > 0:
                            v, _ = torch.topk(pos_logits, min(topk, pos_logits.size(-1)), dim=-1)
                            pos_logits[pos_logits < v[:, [-1]]] = float('-inf')
                        probs = F.softmax(pos_logits / temperature, dim=-1)
                        sampled = torch.multinomial(probs, num_samples=1, generator=rng)
                    else:
                        sampled = pos_logits.argmax(dim=-1, keepdim=True)

                    block_pure[:, pos] = sampled.squeeze(-1)

                    # Update remaining group tokens via collapse
                    for remaining_pos in range(pos + 1, K):
                        remaining_logits = block_logits[:, remaining_pos:remaining_pos+1, :]
                        block_groups[:, remaining_pos] = self.collapse_pure_to_group(
                            remaining_logits, collapse_topk=collapse_topk
                        ).squeeze(0)

                debug_info["denoise_steps"].append({
                    "step": denoise_step,
                    "revealed_pos": pos if pos < K else -1,
                    "pure_token": block_pure[:, pos].item() if pos < K else -1,
                })

            # After all denoise steps, fill remaining positions with argmax
            for pos in range(num_denoise_steps, K):
                if block_pure[:, pos].item() < 0:
                    block_pure[:, pos] = block_logits[:, pos, :].argmax(dim=-1)

            debug_blocks.append(debug_info)

            # Append revealed pure tokens to sequence
            ids = torch.cat([ids, block_pure], dim=1)
            block_step += 1

            # Stop if stop_token found in this block
            if stop_token is not None:
                stop_positions = (block_pure[0] == stop_token).nonzero(as_tuple=True)[0]
                if len(stop_positions) > 0:
                    # Truncate to just before the stop token
                    stop_pos_in_ids = ids.size(1) - K + stop_positions[0].item()
                    ids = ids[:, :stop_pos_in_ids]
                    break

            if block_step > 1000:
                break

        # Truncate to exact target length
        if ids.size(1) > target_len:
            ids = ids[:, :target_len]
        return ids, debug_blocks

    def forward_for_eval_block_pdlm_inference(self, idx, targets, attn_mask):
        """
        Forward pass for block_pdlm_inference evaluation that returns both stage logits.

        Returns:
            stage2_logits: (B, L, pure_vocab_size) logits from xt half
            stage1_logits: (B, L, pure_vocab_size) logits from x0 half
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

        # Compute logits (lm_head outputs pure_vocab_size)
        softcap = 15
        logits = self.lm_head(x)  # (B, 2L, pure_vocab_size)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        # Stage 2: pure vocab logits from xt half (first L positions)
        stage2_logits = logits[:, :T, :]  # (B, L, pure_vocab_size)

        # Stage 1: pure vocab logits from x0 half (second L positions)
        stage1_logits = logits[:, T:, :]  # (B, L, pure_vocab_size)

        return stage2_logits, stage1_logits

    def forward_for_eval(self, idx, targets, attn_mask):
        """
        Forward pass for evaluation that returns logits instead of loss.

        Args:
            idx: (B, L) input tokens (group tokens at block positions for stage2)
            targets: (B, L) clean target tokens (pure tokens)
            attn_mask: attention mask for block diffusion

        Returns:
            logits: (B, L, pure_vocab_size) logits for the xt (input) positions
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
        # For both_block: slice to pure_vocab_size so argmax works correctly
        # (without this, argmax would incorrectly consider group logits)
        if self.config.stage == "both_block":
            return logits[:, :T, :self.config.pure_vocab_size]
        return logits[:, :T, :]

    def forward_for_eval_both_block(self, idx, targets, attn_mask):
        """
        Forward pass for both_block evaluation that returns both stage logits.

        Same transformer body as forward_for_eval, but splits the output into
        stage2 (pure vocab) and stage1 (group vocab) logits.

        Args:
            idx: (B, L) input tokens (xt: group tokens at block positions)
            targets: (B, L) clean target tokens (x0: pure tokens)
            attn_mask: attention mask for block diffusion

        Returns:
            stage2_logits: (B, L, pure_vocab_size) logits for pure token prediction (xt half)
            stage1_logits: (B, L, num_groups) logits for group token prediction (x0 half)
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
        logits = self.lm_head(x)  # (B, 2L, pure_vocab_size + num_groups)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        pure_vocab_size = self.config.pure_vocab_size

        # Stage 2: pure vocab logits from xt half (first L positions)
        stage2_logits = logits[:, :T, :pure_vocab_size]  # (B, L, pure_vocab_size)

        # Stage 1: group logits from x0 half (second L positions)
        stage1_logits = logits[:, T:, pure_vocab_size:]  # (B, L, num_groups)

        return stage2_logits, stage1_logits

    @torch.inference_mode()
    def generate_with_blocks(self, tokens, max_new_tokens,
                             attn_mask=None, bucket_size=None,
                             topk=5, temperature=0, seed=42,
                             transit_topk=10):
        # outdated — use generate_both_block instead
        pass

    @torch.inference_mode()
    def noisy_denoisy_by_model(self, tokens, attn_mask=None, bucket_size=None, noisy_level=1, topk=3, transit_topk=10):
        # outdated — use generate_both_block instead
        pass

    @torch.inference_mode()
    def generate_both_mtp(
        self,
        tokens,
        max_new_tokens,
        block_size=None,
        temperature=0.0,
        topk=0,
        seed=42,
        constrain_to_group=True,
    ):
        """
        Generate tokens using both_mtp model (two-step per block).

        Each block is generated in two steps:
        1. MTP (Pure → Group): predict K group tokens from pure context
        2. Denoise (Group → Pure): predict K pure tokens from group tokens

        Args:
            tokens: list of pure token ids (prompt)
            max_new_tokens: number of new pure tokens to generate
            block_size: K, tokens per block (default from config.n_future_tokens)
            temperature: sampling temperature (0 = greedy)
            topk: top-k sampling (0 = disabled)
            seed: random seed for sampling
            constrain_to_group: if True, constrain pure predictions to tokens within predicted group

        Returns:
            generated_tokens: tensor of pure token ids (1, total_len)
            debug_blocks: list of per-block generation details
        """
        assert self.config.stage == "both_mtp", "generate_both_mtp requires both_mtp stage"
        assert isinstance(tokens, list), "tokens must be a list"

        device = self.get_device()
        if block_size is None:
            block_size = self.config.n_future_tokens
        K = block_size

        # Setup token map (only needed if constraining to group)
        if constrain_to_group:
            if self._token_map is None or self._token_map.device != device:
                self._token_map = get_token_map(device=device)

        # Setup RNG for sampling
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)

        pure_vocab_size = self.config.pure_vocab_size
        group_offset = pure_vocab_size  # group tokens start after pure vocab

        # Current sequence (pure tokens)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)  # (1, prompt_len)
        prompt_len = ids.size(1)
        target_len = prompt_len + max_new_tokens

        debug_blocks = []
        step = 0

        while ids.size(1) < target_len:
            current_len = ids.size(1)

            # === Step 1: MTP (Pure → Group) ===
            # Predict K group tokens from pure context
            group_tokens = self._mtp_predict_block(ids, K)  # (1, K) group indices (0-based)

            # === Step 2: Denoise (Group → Pure) ===
            # Predict K pure tokens from group tokens
            pure_tokens, pure_logits = self._denoise_predict_block(
                ids, group_tokens, K,
                temperature=temperature,
                topk=topk,
                rng=rng,
                constrain_to_group=constrain_to_group,
            )  # (1, K)

            # Debug info
            debug_blocks.append({
                "step": step,
                "context_len": current_len,
                "group_tokens": group_tokens.cpu().tolist()[0],
                "pure_tokens": pure_tokens.cpu().tolist()[0],
            })

            # Append pure tokens to sequence
            ids = torch.cat([ids, pure_tokens], dim=1)
            step += 1

            # Safety break
            if step > 1000:
                break

        # Truncate to exact target length
        ids = ids[:, :target_len]
        return ids, debug_blocks

    def _mtp_predict_block(self, context_ids, K):
        """
        Predict K group tokens using MTP from pure context.

        Uses the x0 path of the model:
        - Main lm_head predicts 1st group token
        - MTP head predicts 2nd..Kth group tokens autoregressively

        Args:
            context_ids: (1, T) pure token context
            K: number of group tokens to predict

        Returns:
            group_tokens: (1, K) group token indices (0-based, not offset)
        """
        device = context_ids.device
        T = context_ids.size(1)
        pure_vocab_size = self.config.pure_vocab_size

        # For MTP, we need to forward through transformer with pure tokens
        # This is similar to training x0 path, but inference only

        # Get rotary embeddings
        cos = self.cos[:, :T]
        sin = self.sin[:, :T]
        cos_sin = (cos, sin)

        # Forward through transformer (x0 path - pure tokens)
        x = self.transformer.wte(context_ids)
        x = norm(x)
        for block in self.transformer.h:
            # Causal attention for pure context
            x = block(x, cos_sin, kv_cache=None, attn_mask=None)
        x = norm(x)

        # Get hidden state at last position
        h = x[:, -1:, :]  # (1, 1, D)

        # Compute logits for group tokens
        softcap = 15
        logits = self.lm_head(h)  # (1, 1, pure_vocab + num_groups)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        # Main model predicts 1st group token
        group_logits = logits[:, :, pure_vocab_size:]  # (1, 1, num_groups)
        g1 = group_logits.argmax(dim=-1).squeeze(1)  # (1,)

        group_tokens = [g1]

        # MTP head predicts 2nd..Kth group tokens autoregressively
        prev_h = h  # (1, 1, D)
        prev_tok = g1  # (1,)

        # Rotary for single position (use position T since we're predicting "future")
        mtp_cos = self.cos[:, T:T+1]
        mtp_sin = self.sin[:, T:T+1]

        for k in range(K - 1):
            # Get token embedding for previous group prediction
            tok_emb = self.group_wte(prev_tok).unsqueeze(1)  # (1, 1, D)

            # Combine: [norm(h), norm(embed)] -> project
            combined = torch.cat([norm(prev_h), norm(tok_emb)], dim=-1)  # (1, 1, 2D)
            x_mtp = self.mtp_head.proj(combined)  # (1, 1, D)

            # Forward through MTP block
            prev_h = self.mtp_head.block(x_mtp, mtp_cos, mtp_sin)
            prev_h = norm(prev_h)

            # Compute group logits
            step_logits = self.mtp_head.lm_head(prev_h)  # (1, 1, num_groups)
            g_next = step_logits.argmax(dim=-1).squeeze(1)  # (1,)

            group_tokens.append(g_next)
            prev_tok = g_next

        return torch.stack(group_tokens, dim=1)  # (1, K)

    def _denoise_predict_block(
        self,
        context_ids,
        group_tokens,
        K,
        temperature=0.0,
        topk=0,
        rng=None,
        constrain_to_group=True,
    ):
        """
        Predict K pure tokens from group tokens using Stage 2 denoising.

        Uses the xt path of the model with block diffusion attention.

        Args:
            context_ids: (1, T) pure token context
            group_tokens: (1, K) group token indices (0-based)
            K: block size
            temperature: sampling temperature
            topk: top-k sampling
            rng: random generator
            constrain_to_group: if True, mask logits to only tokens in each group

        Returns:
            pure_tokens: (1, K) predicted pure tokens
            pure_logits: (1, K, pure_vocab) logits for debugging
        """
        device = context_ids.device
        T = context_ids.size(1)
        pure_vocab_size = self.config.pure_vocab_size
        group_offset = pure_vocab_size

        # Build xt: [pure context..., G1+offset, G2+offset, ...]
        group_token_ids = group_tokens + group_offset  # (1, K) - add offset to get actual token ids
        xt = torch.cat([context_ids, group_token_ids], dim=1)  # (1, T+K)

        # Build x0: [pure context..., placeholder (zeros)]
        # The x0 positions for the block will use placeholder - they provide context via mask
        x0 = torch.cat([context_ids, torch.zeros(1, K, dtype=torch.long, device=device)], dim=1)

        # Total length for attention
        total_len = T + K

        # Create attention mask for inference
        # For Stage 2 inference: xt positions with group tokens attend to pure prefix
        # We need a simplified mask since we're not doing full 2L training
        attn_mask = self._create_inference_denoise_mask(T, K, device)

        # Concatenate [xt | x0] for the forward pass
        combined_idx = torch.cat([xt, x0], dim=1)  # (1, 2*(T+K))

        # Get rotary embeddings
        cos = self.cos[:, :total_len]
        sin = self.sin[:, :total_len]
        cos_sin = (torch.cat([cos, cos], dim=1), torch.cat([sin, sin], dim=1))

        # Forward through transformer
        x = self.transformer.wte(combined_idx)
        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache=None, attn_mask=attn_mask)
        x = norm(x)

        # Compute logits
        softcap = 15
        logits = self.lm_head(x)  # (1, 2*(T+K), pure_vocab + num_groups)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        # Extract pure vocab logits at group positions (xt path, positions T:T+K)
        pure_logits = logits[:, T:T+K, :pure_vocab_size]  # (1, K, pure_vocab)

        # Optionally constrain to tokens within each group
        if constrain_to_group:
            for i in range(K):
                group_idx = group_tokens[0, i].item()
                group_mask = self._token_map.group_to_pure_mask[group_idx]  # (pure_vocab,)
                pure_logits[:, i, ~group_mask] = float('-inf')

        # Sample or argmax
        if temperature > 0:
            if topk > 0:
                # Top-k filtering
                v, _ = torch.topk(pure_logits, min(topk, pure_logits.size(-1)), dim=-1)
                pure_logits[pure_logits < v[:, :, [-1]]] = float('-inf')
            probs = F.softmax(pure_logits / temperature, dim=-1)
            probs_2d = probs.view(-1, probs.size(-1))
            pure_tokens = torch.multinomial(probs_2d, num_samples=1, generator=rng)
            pure_tokens = pure_tokens.view(1, K)
        else:
            pure_tokens = pure_logits.argmax(dim=-1)  # (1, K)

        return pure_tokens, pure_logits

    def _create_inference_denoise_mask(self, prefix_len, block_size, device):
        # outdated — use generate_both_block instead
        pass

    @torch.inference_mode()
    def generate_both_block(self, tokens, max_new_tokens, block_size=None,
                            temperature=0.0, topk=0, seed=42):
        """
        Generate tokens using both_block model (two-step per block).

        Each block is generated in two steps:
        1. Forward pure context through transformer with block-causal attention
           → get group logits at x0 positions → argmax/sample group tokens
        2. Build xt with predicted group tokens, forward through 2L path
           → get pure vocab logits → argmax/sample K pure tokens

        Args:
            tokens: list of pure token ids (prompt)
            max_new_tokens: number of new pure tokens to generate
            block_size: K, tokens per block (default from config.bucket_size)
            temperature: sampling temperature (0 = greedy)
            topk: top-k sampling (0 = disabled)
            seed: random seed for sampling

        Returns:
            generated_tokens: tensor of pure token ids (1, total_len)
            debug_blocks: list of per-block generation details
        """
        # TODO: implement with block-causal mask for step 1 (see design/block_pdlm.md RoPE Position Analysis)
        pass

    def collapse_pure_to_group(self, pure_logits, collapse_topk=256):
        """
        Collapse pure vocab logits to group token indices.

        For each position, compute P(group) by summing P(pure) for all pure tokens
        in each group, then take argmax.

        Args:
            pure_logits: (*, pure_vocab_size) logits over pure vocabulary
            collapse_topk: if > 0, only consider top-k pure tokens for efficiency.
                          if -1, use dense computation (all pure tokens).

        Returns:
            group_indices: (*,) group token indices (0-based, not offset)
        """
        assert self.group_to_pure_mask is not None, \
            "group_to_pure_mask must be registered before calling collapse_pure_to_group"

        original_shape = pure_logits.shape[:-1]
        V = pure_logits.size(-1)
        flat_logits = pure_logits.reshape(-1, V)  # (N, V)

        if collapse_topk > 0 and collapse_topk < V:
            # Sparse: only consider top-k pure tokens
            topk_vals, topk_indices = torch.topk(flat_logits, collapse_topk, dim=-1)  # (N, topk)
            topk_probs = F.softmax(topk_vals, dim=-1)  # (N, topk)

            # Scatter into group probs
            # group_to_pure_mask: (num_groups, pure_vocab_size) bool
            num_groups = self.group_to_pure_mask.size(0)
            group_probs = torch.zeros(flat_logits.size(0), num_groups,
                                      device=flat_logits.device, dtype=flat_logits.dtype)

            # For each top-k token, add its probability to all groups it belongs to
            # topk_indices: (N, topk) -> look up which groups each token belongs to
            token_group_membership = self.group_to_pure_mask[:, topk_indices.view(-1)]  # (num_groups, N*topk)
            token_group_membership = token_group_membership.view(num_groups, flat_logits.size(0), collapse_topk)  # (G, N, topk)
            token_group_membership = token_group_membership.permute(1, 0, 2)  # (N, G, topk)

            # Weight by probabilities and sum
            group_probs = (token_group_membership * topk_probs.unsqueeze(1)).sum(dim=-1)  # (N, G)
        else:
            # Dense: softmax over all pure tokens, then aggregate by group
            probs = F.softmax(flat_logits, dim=-1)  # (N, V)
            # group_to_pure_mask: (num_groups, pure_vocab_size) float
            group_probs = probs @ self.group_to_pure_mask.t()  # (N, num_groups)

        group_indices = group_probs.argmax(dim=-1)  # (N,)
        return group_indices.reshape(original_shape)

    @torch.inference_mode()
    def generate_block_pdlm_inference(self, tokens, max_new_tokens, block_size=None,
                                       temperature=0.0, topk=0, seed=42,
                                       collapse_topk=256, num_denoise_steps=4):
        """
        Generate tokens using block_pdlm_inference model (iterative denoising).

        Each block is generated in (1 + num_denoise_steps) steps:
        1. Forward pure context → stage1 logits → collapse to groups → initial group tokens
        2-5. Build xt with current pure/group mix, forward 2L, sample pure token,
             collapse remaining positions back to groups

        Args:
            tokens: list of pure token ids (prompt)
            max_new_tokens: number of new pure tokens to generate
            block_size: K, tokens per block (default from config.bucket_size)
            temperature: sampling temperature (0 = greedy)
            topk: top-k sampling (0 = disabled)
            seed: random seed for sampling
            collapse_topk: top-k for pure→group collapse (256 default, -1 for dense)
            num_denoise_steps: number of denoising steps per block (default 4)

        Returns:
            generated_tokens: tensor of pure token ids (1, total_len)
            debug_blocks: list of per-block generation details
        """
        assert self.config.stage == "block_pdlm_inference", \
            "generate_block_pdlm_inference requires block_pdlm_inference stage"
        assert isinstance(tokens, list), "tokens must be a list"

        device = self.get_device()
        if block_size is None:
            block_size = self.config.bucket_size
        K = block_size

        pure_vocab_size = self.config.pure_vocab_size
        group_offset = pure_vocab_size

        # Setup RNG for sampling
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)

        # Current sequence (pure tokens)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)  # (1, prompt_len)
        prompt_len = ids.size(1)
        target_len = prompt_len + max_new_tokens

        debug_blocks = []
        block_step = 0

        while ids.size(1) < target_len:
            current_len = ids.size(1)

            # === Step 1: Stage 1 — predict group tokens for next block ===
            # Forward pure context with causal attention
            T = ids.size(1)
            cos_sin = self.cos[:, :T], self.sin[:, :T]

            x = self.transformer.wte(ids)
            x = norm(x)
            for block in self.transformer.h:
                x = block(x, cos_sin, kv_cache=None, attn_mask=None)
            x = norm(x)

            softcap = 15
            logits = self.lm_head(x)  # (1, T, pure_vocab_size)
            logits = logits.float()
            logits = softcap * torch.tanh(logits / softcap)

            # Get logits at last position → collapse to K group tokens
            last_logits = logits[:, -1:, :]  # (1, 1, pure_vocab_size)
            initial_groups = self.collapse_pure_to_group(
                last_logits.expand(-1, K, -1), collapse_topk=collapse_topk
            )  # (1, K)

            # Initialize block state: all positions start as group tokens
            block_pure = torch.full((1, K), -1, dtype=torch.long, device=device)
            block_groups = initial_groups  # (1, K)

            debug_info = {
                "block_step": block_step,
                "context_len": current_len,
                "initial_groups": initial_groups.cpu().tolist()[0],
                "denoise_steps": [],
            }

            # === Steps 2..num_denoise_steps+1: Iterative denoising ===
            for denoise_step in range(num_denoise_steps):
                # Build xt: pure context + current block (mix of pure and group tokens)
                block_tokens = torch.where(
                    block_pure >= 0,
                    block_pure,
                    block_groups + group_offset,
                )  # (1, K)
                xt = torch.cat([ids, block_tokens], dim=1)  # (1, T+K)

                # Build x0: pure context + placeholder (zeros for block positions)
                x0 = torch.cat([ids, torch.zeros(1, K, dtype=torch.long, device=device)], dim=1)

                # 2L forward
                total_len = T + K
                combined_idx = torch.cat([xt, x0], dim=1)  # (1, 2*(T+K))

                cos = self.cos[:, :total_len]
                sin = self.sin[:, :total_len]
                cos_sin_2l = (torch.cat([cos, cos], dim=1), torch.cat([sin, sin], dim=1))

                # Build attention mask for inference
                # Use block diffusion mask structure
                from nanochat.attn_masks import gen_mask
                attn_mask = gen_mask(total_len, K, attn_backend="sdpa",
                                     is_causal=self._is_causal,
                                     prefix_sliding_tokens=0).to(device)

                x = self.transformer.wte(combined_idx)
                x = norm(x)
                for block in self.transformer.h:
                    x = block(x, cos_sin_2l, kv_cache=None, attn_mask=attn_mask)
                x = norm(x)

                logits_2l = self.lm_head(x)  # (1, 2*(T+K), pure_vocab_size)
                logits_2l = logits_2l.float()
                logits_2l = softcap * torch.tanh(logits_2l / softcap)

                # Get logits for block positions in xt half
                block_logits = logits_2l[:, T:T+K, :]  # (1, K, pure_vocab_size)

                # Sample next pure token (left-to-right: reveal position denoise_step)
                pos = denoise_step
                if pos < K:
                    pos_logits = block_logits[:, pos, :]  # (1, pure_vocab_size)

                    if temperature > 0:
                        if topk > 0:
                            v, _ = torch.topk(pos_logits, min(topk, pos_logits.size(-1)), dim=-1)
                            pos_logits[pos_logits < v[:, [-1]]] = float('-inf')
                        probs = F.softmax(pos_logits / temperature, dim=-1)
                        sampled = torch.multinomial(probs, num_samples=1, generator=rng)
                    else:
                        sampled = pos_logits.argmax(dim=-1, keepdim=True)

                    block_pure[:, pos] = sampled.squeeze(-1)

                    # Update remaining group tokens via collapse
                    for remaining_pos in range(pos + 1, K):
                        remaining_logits = block_logits[:, remaining_pos:remaining_pos+1, :]
                        block_groups[:, remaining_pos] = self.collapse_pure_to_group(
                            remaining_logits, collapse_topk=collapse_topk
                        ).squeeze(0)

                debug_info["denoise_steps"].append({
                    "step": denoise_step,
                    "revealed_pos": pos if pos < K else -1,
                    "pure_token": block_pure[:, pos].item() if pos < K else -1,
                })

            # After all denoise steps, fill remaining positions with argmax
            for pos in range(num_denoise_steps, K):
                if block_pure[:, pos].item() < 0:
                    # Need one more forward to get logits for this position
                    # For simplicity, use the last block_logits
                    block_pure[:, pos] = block_logits[:, pos, :].argmax(dim=-1)

            debug_blocks.append(debug_info)

            # Append revealed pure tokens to sequence
            ids = torch.cat([ids, block_pure], dim=1)
            block_step += 1

            if block_step > 1000:
                break

        # Truncate to exact target length
        ids = ids[:, :target_len]
        return ids, debug_blocks

