"""
GPT model with Multi-Token Prediction for Stage 1 MTP.

Input: pure_tokens (B, T)
   ↓
[Main GPT] → hidden h → lm_head → 1st group token
                ↓
[MTP Head] (single block reused K-1 times) → 2nd..Kth group tokens

Output: K group tokens per position
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
from nanochat.mtp_head import MTPHead, compute_mtp_loss


@dataclass
class GPTMTPConfig:
    """Configuration for GPT-MTP model."""
    sequence_len: int = 1024
    pure_vocab_size: int = -1  # Input vocab (pure tokens)
    num_groups: int = -1       # Output vocab (group tokens)
    n_future_tokens: int = 4   # K: number of tokens to predict per position
    mtp_loss_beta: float = 0.8 # Exponential decay factor for MTP loss weights
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    stage1_target_mode: str = "pure"  # "pure" (default) or "group" (legacy)


def norm(x):
    """Purely functional rmsnorm with no learnable params"""
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3)
    out = out.to(x.dtype)
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

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2)
        Tk = k.size(2)

        enable_gqa = self.n_head != self.n_kv_head
        if kv_cache is None or Tq == Tk:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
        elif Tq == 1:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
        else:
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device)
            prefix_len = Tk - Tq
            attn_mask[:, :prefix_len] = True
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)

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

    def forward(self, x, cos_sin, kv_cache):
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPTMTP(nn.Module):
    """
    GPT model with Multi-Token Prediction head for Stage 1.

    Architecture:
    - wte: pure token embeddings (input)
    - transformer blocks
    - lm_head: outputs group token logits (1st prediction)
    - group_wte: group token embeddings (for MTP teacher forcing)
    - mtp_head: predicts 2nd..Kth group tokens
    """

    def __init__(self, config: GPTMTPConfig):
        super().__init__()
        self.config = config

        # Main transformer components
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.pure_vocab_size, config.n_embd),  # pure tokens input
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })

        # Output head: size depends on stage1_target_mode
        if config.stage1_target_mode == "pure":
            # Pure mode: predict full vocabulary
            self.lm_head = nn.Linear(config.n_embd, config.pure_vocab_size, bias=False)
        else:
            # Group mode (legacy): predict group tokens
            self.lm_head = nn.Linear(config.n_embd, config.num_groups, bias=False)

        # Group token embeddings for MTP head teacher forcing
        # Always needed - teacher forcing uses group tokens regardless of target mode
        self.group_wte = nn.Embedding(config.num_groups, config.n_embd)

        # MTP head (predicts 2nd..Kth tokens)
        self.mtp_head = MTPHead(
            n_embd=config.n_embd,
            n_head=config.n_head,
            n_future_tokens=config.n_future_tokens,
            group_wte=self.group_wte,
            lm_head=self.lm_head,  # shared with main model
            stage1_target_mode=config.stage1_target_mode,
        )

        # Rotary embeddings
        self.rotary_seq_len = max(config.sequence_len, 1024) * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # Group mask buffer for inference collapse (registered later via register_group_mask)
        # Shape: (num_groups, pure_vocab_size) - bool mask of which pure tokens belong to each group
        self.register_buffer("group_to_pure_mask", None, persistent=False)

    def init_weights(self):
        self.apply(self._init_weights)
        # Zero out classifier weights
        torch.nn.init.zeros_(self.lm_head.weight)
        # Zero out c_proj weights in all blocks
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
        # Init MTP head weights
        self.mtp_head.init_weights()
        # Init rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast embeddings to bf16
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
            self.group_wte.to(dtype=torch.bfloat16)

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

    def register_group_mask(self, group_to_pure_mask):
        """
        Register the group-to-pure mask for inference collapse.

        In pure mode, this allows collapsing pure token logits to group predictions
        at inference time via: group_head = group_to_pure_mask @ lm_head.weight

        Args:
            group_to_pure_mask: (num_groups, pure_vocab_size) bool tensor
        """
        self.register_buffer("group_to_pure_mask", group_to_pure_mask.float(), persistent=False)

    def get_group_head_weight(self):
        """
        Get collapsed group head weight for inference.

        In pure mode: collapses lm_head to group predictions via matrix multiply.
        In group mode: returns lm_head weight directly.

        Returns:
            weight: (num_groups, n_embd) tensor
        """
        if self.config.stage1_target_mode == "pure":
            assert self.group_to_pure_mask is not None, \
                "group_to_pure_mask not registered. Call register_group_mask() first."
            # group_to_pure_mask: (num_groups, pure_vocab_size)
            # lm_head.weight: (pure_vocab_size, n_embd)
            # Result: (num_groups, n_embd)
            return self.group_to_pure_mask @ self.lm_head.weight
        else:
            return self.lm_head.weight

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """Return the estimated FLOPs per token for the model."""
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel() + self.group_wte.weight.numel()
        l, h, q, t = self.config.n_layer, self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Main model FLOPs
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        # MTP head adds some extra FLOPs (K-1 passes through a single block)
        # Rough estimate: (K-1) * single_block_flops
        return num_flops_per_token

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate parameters into groups
        matrix_params = list(self.transformer.h.parameters()) + list(self.mtp_head.block.parameters()) + list(self.mtp_head.proj.parameters())
        embedding_params = list(self.transformer.wte.parameters()) + list(self.group_wte.parameters())
        lm_head_params = list(self.lm_head.parameters())

        # Scale LR by 1/sqrt(dmodel)
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

    def forward(self, idx, targets=None, pure_to_group=None, kv_cache=None, loss_reduction='mean'):
        """
        Forward pass for MTP training.

        Args:
            idx: (B, T) pure token ids
            targets: Target tokens for K future positions. Format depends on stage1_target_mode:
                     - pure mode: (B, T, K) pure token ids
                     - group mode: (B, T, K, overlap_k) group token ids with overlap
                     Or None for inference
            pure_to_group: (pure_vocab_size, overlap_k) mapping for pure->group conversion.
                           Required in pure mode for teacher forcing.
            kv_cache: optional KV cache for inference
            loss_reduction: 'mean' or 'none'

        Returns:
            If targets is None: main_logits
            If targets provided: loss scalar
        """
        B, T = idx.size()
        target_mode = self.config.stage1_target_mode

        # Check rotary cache
        assert T <= self.cos.size(1), f"Sequence length grew beyond rotary cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device
        assert self.cos.dtype == torch.bfloat16

        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        # Forward through main transformer
        x = self.transformer.wte(idx)
        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)
        h = norm(x)  # hidden states for MTP head

        # Compute logits for 1st token prediction
        softcap = 15
        main_logits = self.lm_head(h)  # (B, T, V) where V = pure_vocab_size or num_groups
        main_logits = main_logits.float()
        main_logits = softcap * torch.tanh(main_logits / softcap)

        if targets is None:
            # Inference: return logits only (no MTP during inference)
            return main_logits

        # Training: compute MTP loss
        K = self.config.n_future_tokens

        # Get cos/sin for MTP head (same positions, MTP doesn't extend sequence)
        mtp_cos = self.cos[:, T0:T0+T]
        mtp_sin = self.sin[:, T0:T0+T]

        if target_mode == "pure":
            # Pure mode: targets are (B, T, K) pure token IDs
            assert targets.ndim == 3 and targets.shape == (B, T, K), \
                f"Expected targets shape (B, T, K)=({B}, {T}, {K}), got {targets.shape}"
            assert pure_to_group is not None, "pure_to_group required for pure mode teacher forcing"

            # Convert 1st pure target to group token for MTP head input (teacher forcing)
            # pure_to_group: (pure_vocab_size, overlap_k)
            first_pure = targets[:, :, 0]  # (B, T)
            first_groups = pure_to_group[first_pure]  # (B, T, overlap_k)
            first_group_tok = first_groups[:, :, 0]  # (B, T) use first valid group

            # MTP targets for teacher forcing: convert pure targets to group tokens
            mtp_pure_targets = targets[:, :, 1:]  # (B, T, K-1) pure tokens
            mtp_group_targets = pure_to_group[mtp_pure_targets][:, :, :, 0]  # (B, T, K-1) group indices

            # Forward MTP head
            mtp_logits = self.mtp_head(h, mtp_cos, mtp_sin, first_group_tok,
                                       targets=mtp_group_targets, pure_to_group=pure_to_group)
            mtp_logits = mtp_logits.float()
            mtp_logits = softcap * torch.tanh(mtp_logits / softcap)

            # Compute combined loss (pure mode: standard CE)
            loss = compute_mtp_loss(
                main_logits=main_logits,
                mtp_logits=mtp_logits,
                targets=targets,
                mtp_loss_beta=self.config.mtp_loss_beta,
                stage1_target_mode="pure",
            )
        else:
            # Group mode (legacy): targets are (B, T, K, overlap_k) group token IDs
            assert targets.ndim == 4 and targets.shape[:3] == (B, T, K), \
                f"Expected targets shape (B, T, K, overlap_k)=({B}, {T}, {K}, *), got {targets.shape}"

            # Get 1st group token for MTP head input (teacher forcing)
            first_group_tok = targets[:, :, 0, 0]  # (B, T)

            # MTP targets for teacher forcing: use first valid group (index 0)
            mtp_targets = targets[:, :, 1:, 0]  # (B, T, K-1)

            # Forward MTP head
            mtp_logits = self.mtp_head(h, mtp_cos, mtp_sin, first_group_tok, targets=mtp_targets)
            mtp_logits = mtp_logits.float()
            mtp_logits = softcap * torch.tanh(mtp_logits / softcap)

            # Compute combined loss (group mode: any-correct CE)
            loss = compute_mtp_loss(
                main_logits=main_logits,
                mtp_logits=mtp_logits,
                targets=targets,
                mtp_loss_beta=self.config.mtp_loss_beta,
                stage1_target_mode="group",
            )

        return loss

    def forward_for_eval(self, idx, targets=None, pure_to_group=None, attn_mask=None):
        """
        Forward pass for evaluation that returns logits without computing loss.

        Args:
            idx: (B, T) pure token ids
            targets: Target tokens for K future positions. Format depends on stage1_target_mode:
                     - pure mode: (B, T, K) pure token ids
                     - group mode: (B, T, K, overlap_k) group token ids with overlap
                     Required for teacher forcing in MTP head.
            pure_to_group: (pure_vocab_size, overlap_k) mapping for pure->group conversion.
                           Required in pure mode for teacher forcing.
            attn_mask: ignored (MTP uses standard causal attention)

        Returns:
            main_logits: (B, T, V) logits for k=0 (V = pure_vocab_size or num_groups)
            mtp_logits: (B, T, K-1, V) logits for k=1..K-1
        """
        B, T = idx.size()
        K = self.config.n_future_tokens
        target_mode = self.config.stage1_target_mode

        # Check rotary cache
        assert T <= self.cos.size(1), f"Sequence length exceeds rotary cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device

        cos_sin = self.cos[:, :T], self.sin[:, :T]

        # Forward through main transformer
        x = self.transformer.wte(idx)
        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache=None)
        h = norm(x)  # hidden states for MTP head

        # Compute logits for 1st token prediction
        softcap = 15
        main_logits = self.lm_head(h)  # (B, T, V)
        main_logits = main_logits.float()
        main_logits = softcap * torch.tanh(main_logits / softcap)

        # Get teacher forcing tokens for MTP head
        assert targets is not None, "targets required for forward_for_eval (teacher forcing)"

        # Get cos/sin for MTP head
        mtp_cos = self.cos[:, :T]
        mtp_sin = self.sin[:, :T]

        if target_mode == "pure":
            # Pure mode: targets are (B, T, K) pure token IDs
            assert targets.ndim == 3 and targets.shape == (B, T, K), \
                f"Expected targets shape ({B}, {T}, {K}), got {targets.shape}"
            assert pure_to_group is not None, "pure_to_group required for pure mode"

            # Convert pure targets to group tokens for teacher forcing
            first_pure = targets[:, :, 0]  # (B, T)
            first_groups = pure_to_group[first_pure]  # (B, T, overlap_k)
            first_group_tok = first_groups[:, :, 0]  # (B, T)

            # MTP targets for teacher forcing
            mtp_pure_targets = targets[:, :, 1:]  # (B, T, K-1)
            mtp_group_targets = pure_to_group[mtp_pure_targets][:, :, :, 0]  # (B, T, K-1)

            # Forward MTP head
            mtp_logits = self.mtp_head(h, mtp_cos, mtp_sin, first_group_tok,
                                       targets=mtp_group_targets, pure_to_group=pure_to_group)
        else:
            # Group mode: targets are (B, T, K, overlap_k) group token IDs
            assert targets.ndim == 4 and targets.shape[:3] == (B, T, K), \
                f"Expected targets shape ({B}, {T}, {K}, overlap_k), got {targets.shape}"

            # Use first valid group (index 0) for teacher forcing
            first_group_tok = targets[:, :, 0, 0]  # (B, T)
            mtp_targets = targets[:, :, 1:, 0]  # (B, T, K-1)

            # Forward MTP head
            mtp_logits = self.mtp_head(h, mtp_cos, mtp_sin, first_group_tok, targets=mtp_targets)

        mtp_logits = mtp_logits.float()
        mtp_logits = softcap * torch.tanh(mtp_logits / softcap)

        return main_logits, mtp_logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Autoregressive streaming inference (1st group token only).
        For full MTP inference, use a different method.
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        for _ in range(max_tokens):
            logits = self.forward(ids)
            logits = logits[:, -1, :]
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
