
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


def any_correct_ce_loss(logits, valid_targets, loss_mask=None):
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

    Returns:
        loss: scalar mean loss
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

    # Apply loss_mask if provided
    if loss_mask is not None:
        nll = nll * loss_mask.float()
        return nll.sum() / loss_mask.sum().clamp(min=1)
    else:
        return nll.mean()

# Stage types for PDLM
PDLMStage = Literal["stage1_mtp", "stage1_mask", "stage2", "both_mtp", "both_mask"]

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

    # MTP (Multi-Token Prediction) config for both_mtp stage
    n_future_tokens: int = 4       # K: number of group tokens to predict for Stage 1
    mtp_loss_beta: float = 0.8     # Exponential decay factor for MTP loss weights
    mtp_loss_weight: float = 1.0   # Stage 1 MTP loss weight relative to Stage 2 (which is 1.0)

    def __post_init__(self):
        valid_stages = {"stage1_mtp", "stage1_mask", "stage2", "both_mtp", "both_mask"}
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
        elif self.stage == "stage2":
            wte_size = self.pure_vocab_size + self.num_groups
            lm_head_size = self.pure_vocab_size
        elif self.stage == "both_mtp":
            wte_size = self.pure_vocab_size + self.num_groups
            lm_head_size = self.pure_vocab_size + self.num_groups
        elif self.stage == "both_mask":
            assert self.mask_token_id != -1, "both_mask requires mask_token_id"
            wte_size = self.pure_vocab_size + self.num_groups + 1
            lm_head_size = self.pure_vocab_size + self.num_groups
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

    def forward(self, idx, targets=None, kv_cache=None, attn_mask=None, loss_extras=None):
        """Training: idx/targets are length L; we concat to 2L inside this and apply block mask."""
        # Dispatch to both_mtp forward if in both_mtp stage with targets
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
                loss = (nll * loss_mask).sum() / loss_mask.sum()
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
        stage2_loss = (nll * loss_mask).sum() / loss_mask.sum()

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
        return logits[:, :T, :]

    @torch.inference_mode()
    def generate_with_blocks(self, tokens, max_new_tokens, 
                             attn_mask=None, 
                             bucket_size=None, 
                             topk=5, 
                             temperature=0, 
                             seed=42,
                             transit_topk=10):
        """
        Like generate(), but also returns per-step noisy/pure blocks.
        If transit_topk > 0, it uses top-k pure token probabilities to transit to the next noisy token.
        max_new_tokens: The number of new tokens to generate (excluding prompt).
        """
        assert isinstance(tokens, list) # B == 1
        assert self.config.mask_token_id != -1, "mask_token_id must be set for generate"
        device = self.get_device()
        
        if bucket_size is None:
            bucket_size = self.bucket_size
        assert bucket_size > 0, "bucket_size must be set in config or passed as arg"
        
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
            
        if self._token_map is None or self._token_map.device != device:
            self._token_map = get_token_map(device=device)

        if not self._is_causal:
            assert attn_mask is not None, "need attn-mask when the model is non-causal"
        
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        prompt_ids = ids.clone()
        prompt_len = ids.size(1)
        target_len = prompt_len + max_new_tokens
        
        mask_id = self.config.mask_token_id

        if self._is_causal:
            ids = F.pad(ids, (0, bucket_size), value=mask_id) # add bucket
            current_mask = None
        else:
            pad_len = bucket_size - (ids.size(1) % bucket_size)
            ids = F.pad(ids, (0, pad_len), value=mask_id)
            T = ids.size(1)
            current_mask = attn_mask[:T, :T]

        block_debug = []
        step = 0
        while True:
            logits = self.forward(ids, attn_mask=current_mask) # (B, T, pure_vocab_size)
            logits = logits[:, -bucket_size:, :] # (B, bucket_size, pure_vocab_size)
            max_topk_logits = logits.size(-1)
            
            # Reporting/Sampling (topk)
            k_report = min(topk, max_topk_logits)
            if k_report < 1: raise ValueError("topk must be >= 1")
            _, topk_ids_report = torch.topk(logits, k=k_report, dim=-1) # (B, bucket, k_report)
            probs = torch.softmax(logits.float(), dim=-1)
            topk_probs_report = torch.gather(probs, -1, topk_ids_report) # (B, bucket, k_report)
            
            noisy_ids = ids[:, -bucket_size:] # (B, bucket)
            entry = {
                "step": step,
                "noisy_ids": noisy_ids.detach().cpu(),
                "pure_ids": topk_ids_report.detach().cpu(),
                "pure_probs": topk_probs_report.detach().cpu(),
            }

            if transit_topk > 0:
                k_transit = min(transit_topk, max_topk_logits)
                _, topk_ids_transit = torch.topk(logits, k=k_transit, dim=-1)
                topk_probs_transit = torch.gather(probs, -1, topk_ids_transit)
                
                next_ids = self._token_map.transit_noisy_tokens(
                    topk_ids_transit, noisy_ids, pure_probs=topk_probs_transit
                )
                entry["sampled_ids"] = topk_ids_report[..., 0].detach().cpu()
            else:
                if topk > 0:
                    v, _ = torch.topk(logits, min(topk, logits.size(-1)), dim=-1)
                    logits[logits < v[:, :, [-1]]] = -float('Inf')
                if temperature > 0:
                    logits_temp = logits / temperature
                    probs_temp = F.softmax(logits_temp, dim=-1)
                    probs_2d = probs_temp.reshape(-1, probs_temp.size(-1))
                    pure_ids = torch.multinomial(probs_2d, num_samples=1, generator=rng)
                    pure_ids = pure_ids.reshape(probs.shape[:-1])
                else:
                    pure_ids = topk_ids_report[..., 0] # (B, bucket)
                
                entry["sampled_ids"] = pure_ids.detach().cpu()
                
                next_ids = self._token_map.transit_noisy_tokens(pure_ids, noisy_ids)
            
            next_is_pure = self._token_map.is_all_pure_tokens(next_ids)
            if next_is_pure:
                entry["next_ids"] = next_ids.detach().cpu()
            block_debug.append(entry)
            ids = torch.cat((ids[:, :-next_ids.size(1)], next_ids), dim=1)
            
            # Restore prompt
            if ids.size(1) >= prompt_ids.size(1):
                ids[:, :prompt_ids.size(1)] = prompt_ids
            
            if next_is_pure:
                if ids.numel() >= target_len:
                    # Truncate to exact target length
                    ids = ids[:, :target_len]
                    break
                else:
                    ids = F.pad(ids, (0, bucket_size), value=mask_id)
                    if not self._is_causal:
                        T = ids.size(1)
                        current_mask = attn_mask[:T, :T]
            step += 1
        return ids, block_debug


    @torch.inference_mode()
    def noisy_denoisy_by_model(self, tokens, attn_mask=None, bucket_size=None, noisy_level=1, topk=3, transit_topk=10):
        """
        This func is not for generate new tokens, but to add noisy to 
        the last bucket_size of the sequence, and then denoise the
        added noisy. So this func would be only several steps.
        this func will also return a block_debug like generate_with_blocks,
        there should be step, the original real pure token, the current noisy token,
        and the predicted pure token by the model, with prob, topk,
        If transit_topk > 0, it uses top-k pure token probabilities to transit to the next noisy token.
        """
        assert isinstance(tokens, list) # B == 1
        device = self.get_device()
        
        if bucket_size is None:
            bucket_size = self.bucket_size
        assert bucket_size > 0, "bucket_size must be set in config or passed as arg"
        
        if self._token_map is None or self._token_map.device != device:
            self._token_map = get_token_map(device=device)

        # Prepare IDs
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # (1, L)
        L = ids.size(1)
        if L < bucket_size:
             bucket_size = L
        
        # Split into prefix and target part
        prefix_ids = ids[:, :-bucket_size]
        target_pure_ids = ids[:, -bucket_size:] # (1, bucket)
        
        # Add noise
        noisy_levels = torch.full_like(target_pure_ids, noisy_level)
        noisy_ids = self._token_map.noise_tokens(target_pure_ids, noisy_levels)
        
        # Combine
        ids = torch.cat([prefix_ids, noisy_ids], dim=1)
        
        current_mask = None
        if not self._is_causal and attn_mask is not None:
            current_mask = attn_mask[:ids.size(1), :ids.size(1)]

        block_debug = []
        step = 0
        
        # Denoising loop
        while True:
            logits = self.forward(ids, attn_mask=current_mask) # (1, L, pure_vocab)
            logits = logits[:, -bucket_size:, :] # (1, bucket, pure_vocab)
            max_topk_logits = logits.size(-1)
            
            # Get topk pure predictions for reporting
            k_report = min(topk, max_topk_logits)
            _, topk_ids_report = torch.topk(logits, k=k_report, dim=-1) # (1, bucket, topk)
            probs = torch.softmax(logits.float(), dim=-1)
            topk_probs_report = torch.gather(probs, -1, topk_ids_report) # (1, bucket, topk)
            
            predicted_pure_ids = topk_ids_report[..., 0] # Top 1 prediction
            
            # Record debug info
            entry = {
                "step": step,
                "original_ids": target_pure_ids.detach().cpu(),
                "noisy_ids": noisy_ids.detach().cpu(),
                "pure_ids": topk_ids_report.detach().cpu(), # Predicted pure (topk)
                "pure_probs": topk_probs_report.detach().cpu(), # Probs of predicted pure
            }
            
            # Transit
            if transit_topk > 0:
                k_transit = min(transit_topk, max_topk_logits)
                _, topk_ids_transit = torch.topk(logits, k=k_transit, dim=-1)
                topk_probs_transit = torch.gather(probs, -1, topk_ids_transit)
                
                next_ids = self._token_map.transit_noisy_tokens(
                    topk_ids_transit, noisy_ids, pure_probs=topk_probs_transit
                )
            else:
                next_ids = self._token_map.transit_noisy_tokens(predicted_pure_ids, noisy_ids)
            
            # Check if converged (all pure)
            next_is_pure = self._token_map.is_all_pure_tokens(next_ids)
            if next_is_pure:
                entry["next_ids"] = next_ids.detach().cpu()
            
            block_debug.append(entry)
            
            # Update loop vars
            noisy_ids = next_ids
            ids = torch.cat([prefix_ids, noisy_ids], dim=1)
            step += 1
            
            if next_is_pure:
                break
                
            # Safety break
            if step > 100: 
                break
                
        return ids, block_debug
        
