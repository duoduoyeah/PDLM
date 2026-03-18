"""
BD3-LM-Prime: Block Discrete Denoising Diffusion Language Model with Prime sub-token encoding.

Combines BD3-LM's block-AR architecture with MDM-Prime's base-b sub-token decomposition.
Instead of masking entire tokens, individual sub-tokens are masked independently,
creating intermediate states between fully masked and fully revealed.
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
from nanochat.bd3lm_utils.prime_encoding import encode, decode, get_prime_params
from nanochat.bd3lm import norm, apply_rotary_emb, CausalSelfAttention, MLP, Block


@dataclass
class BD3LMPrimeConfig:
    sequence_len: int = 512
    pure_vocab_size: int = 50304
    all_vocab_size: int = -1
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768

    bucket_size: int = -1
    is_causal: bool = False

    model_name: str = "bd3lm_prime"
    prefix_pure_tokens: int = 1
    mask_token_id: int = -1
    target_shift: int = -1

    # Prime-specific
    target_length: int = 2        # ℓ: number of sub-tokens per token
    base: int = -1                # auto-computed from pure_vocab_size and target_length
    sub_token_vocab_size: int = -1  # base + 1 (including mask sub-token)
    mask_sub_token: int = -1      # sub-token ID for mask (= base)

    def __post_init__(self):
        if self.base < 0:
            self.base, self.mask_sub_token, self.sub_token_vocab_size = \
                get_prime_params(self.pure_vocab_size, self.target_length)


class BD3LMPrime(nn.Module):
    def __init__(self, config: BD3LMPrimeConfig):
        super().__init__()
        self.config = config

        sub_emb_dim = config.n_embd // config.target_length
        assert config.n_embd % config.target_length == 0, \
            f"n_embd ({config.n_embd}) must be divisible by target_length ({config.target_length})"

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.sub_token_vocab_size, sub_emb_dim),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.pure_vocab_size, bias=False)

        self.rotary_seq_len = max(config.sequence_len, 1024) * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self._is_causal = self.config.is_causal
        self.inference_mask = None
        self.bucket_size = config.bucket_size

    def _embed_sub_tokens(self, sub_tokens):
        """Embed sub-token sequence and reshape to token-level representations.

        Args:
            sub_tokens: (B, T * ℓ) sub-token IDs

        Returns:
            (B, T, D) token-level embeddings
        """
        B = sub_tokens.shape[0]
        l = self.config.target_length
        T = sub_tokens.shape[1] // l

        # Embed each sub-token: (B, T*l, D/l)
        x = self.transformer.wte(sub_tokens)
        # Reshape: concatenate l sub-token embeddings per token -> (B, T, D)
        x = x.view(B, T, -1)
        return x

    def init_weights(self):
        self.apply(self._init_weights)
        torch.nn.init.zeros_(self.lm_head.weight)
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        l, h, q, t = self.config.n_layer, self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return num_flops_per_token

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        matrix_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params)
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
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def forward(self, idx_sub, targets=None, kv_cache=None, attn_mask=None, loss_extras=None):
        """Training forward pass.

        Args:
            idx_sub: (B, T * ℓ) sub-token IDs for noisy input (xt).
            targets: (B, T) clean token IDs in original token space.
            attn_mask: (2T, 2T) block-causal attention mask.
            loss_extras: {"loss_scale": (B, T), "loss_mask": (B, T)}.
        """
        if targets is not None:
            l = self.config.target_length
            B = idx_sub.shape[0]
            T = idx_sub.shape[1] // l
            assert attn_mask is not None, "Train should have attn mask"
            assert self.config.sequence_len == T, f"Expected seq_len={self.config.sequence_len}, got T={T}"
            assert targets.size(1) == T, "Targets should match the base sequence length"

            # Encode targets to sub-tokens for x0 half
            targets_sub = encode(targets, self.config.base, l,
                                 mask_token_id=self.config.mask_token_id,
                                 mask_sub_token=self.config.mask_sub_token)

            # Embed sub-tokens and reshape to token level
            xt_emb = self._embed_sub_tokens(idx_sub)    # (B, T, D)
            x0_emb = self._embed_sub_tokens(targets_sub)  # (B, T, D)

            # Concatenate [xt | x0] at token level
            x = torch.cat((xt_emb, x0_emb), dim=1)  # (B, 2T, D)
        else:
            l = self.config.target_length
            B = idx_sub.shape[0]
            T = idx_sub.shape[1] // l
            x = self._embed_sub_tokens(idx_sub)  # (B, T, D)

        # Rotary embeddings at token level
        assert T <= self.cos.size(1), f"Sequence length {T} > rotary cache {self.cos.size(1)}"
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        if targets is not None:
            cos = self.cos[:, T0:T0+T]
            sin = self.sin[:, T0:T0+T]
            cos_sin = (torch.cat((cos, cos), dim=1), torch.cat((sin, sin), dim=1))
        else:
            cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        # Transformer
        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache, attn_mask=attn_mask)
        x = norm(x)

        # Logits
        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            logits = logits[:, :T, :]

            assert loss_extras is not None and "loss_scale" in loss_extras
            assert "loss_mask" in loss_extras
            loss_scale = loss_extras["loss_scale"]
            loss_mask = loss_extras["loss_mask"]
            attention_mask = loss_mask.float()

            loss, _ = compute_bd3lm_loss(logits, targets, loss_scale, attention_mask=attention_mask)
            return loss
        else:
            return logits

    def forward_for_eval(self, idx, targets, attn_mask):
        """Eval forward pass. Accepts token-level inputs (same interface as BDLM).

        Args:
            idx: (B, L) input token IDs (with mask tokens).
            targets: (B, L) clean target tokens.
            attn_mask: attention mask for block diffusion.

        Returns:
            logits: (B, L, pure_vocab_size)
        """
        B, T = idx.size()
        l = self.config.target_length
        assert targets.size(1) == T

        # Encode both to sub-tokens
        idx_sub = encode(idx, self.config.base, l,
                         mask_token_id=self.config.mask_token_id,
                         mask_sub_token=self.config.mask_sub_token)
        targets_sub = encode(targets, self.config.base, l,
                             mask_token_id=self.config.mask_token_id,
                             mask_sub_token=self.config.mask_sub_token)

        # Embed and reshape to token level
        xt_emb = self._embed_sub_tokens(idx_sub)    # (B, T, D)
        x0_emb = self._embed_sub_tokens(targets_sub)  # (B, T, D)

        # Concatenate [xt | x0]
        x = torch.cat((xt_emb, x0_emb), dim=1)  # (B, 2T, D)

        # Rotary embeddings
        cos = self.cos[:, :T]
        sin = self.sin[:, :T]
        cos_sin = (torch.cat((cos, cos), dim=1), torch.cat((sin, sin), dim=1))

        # Transformer
        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache=None, attn_mask=attn_mask)
        x = norm(x)

        # Logits
        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        return logits[:, :T, :]

    def forward_for_eval_sub(self, idx_sub, targets, attn_mask):
        """Eval forward accepting sub-token inputs directly (for half-decoded states).

        Args:
            idx_sub: (B, T * target_length) sub-token IDs (already in sub-token space).
            targets: (B, T) clean target tokens (in original token space).
            attn_mask: attention mask for block diffusion.

        Returns:
            logits: (B, T, pure_vocab_size)
        """
        B = idx_sub.shape[0]
        l = self.config.target_length
        T = idx_sub.shape[1] // l
        assert targets.size(1) == T

        # idx_sub is already in sub-token space — skip encode
        # Encode targets to sub-tokens for x0 half
        targets_sub = encode(targets, self.config.base, l,
                             mask_token_id=self.config.mask_token_id,
                             mask_sub_token=self.config.mask_sub_token)

        # Embed and reshape to token level
        xt_emb = self._embed_sub_tokens(idx_sub)       # (B, T, D)
        x0_emb = self._embed_sub_tokens(targets_sub)   # (B, T, D)

        # Concatenate [xt | x0]
        x = torch.cat((xt_emb, x0_emb), dim=1)  # (B, 2T, D)

        # Rotary embeddings
        cos = self.cos[:, :T]
        sin = self.sin[:, :T]
        cos_sin = (torch.cat((cos, cos), dim=1), torch.cat((sin, sin), dim=1))

        # Transformer
        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache=None, attn_mask=attn_mask)
        x = norm(x)

        # Logits
        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        return logits[:, :T, :]
