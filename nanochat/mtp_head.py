"""
Multi-Token Prediction Head for Stage 1 MTP.

A single transformer block reused K-1 times to predict K group tokens.
The main model predicts the 1st group token, MTP head predicts 2nd..Kth.

Usage:
    mtp = MTPHead(config, shared_group_wte, shared_lm_head)
    # h: hidden states from main model (B, T, D)
    # targets: ground truth group tokens for teacher forcing (B, T, K-1)
    logits = mtp(h, cos, sin, targets=targets)  # (B, T, K-1, num_groups)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def norm(x):
    """RMSNorm without learnable params"""
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    """Apply rotary embeddings to x"""
    assert x.ndim == 4  # (B, T, H, D)
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=3).to(x.dtype)


class MTPAttention(nn.Module):
    """
    Causal self-attention for MTP block.
    Simplified - no KV cache needed for MTP.
    """
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.c_q = nn.Linear(n_embd, n_embd, bias=False)
        self.c_k = nn.Linear(n_embd, n_embd, bias=False)
        self.c_v = nn.Linear(n_embd, n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)

        # Apply rotary embeddings BEFORE transpose (matches gpt_mtp.py pattern)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)  # QK norm
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # Causal attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MTPMLP(nn.Module):
    """MLP for MTP block"""
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()  # relu^2 activation
        return self.c_proj(x)


class MTPBlock(nn.Module):
    """Single transformer block for MTP, reused K-1 times"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.attn = MTPAttention(n_embd, n_head)
        self.mlp = MTPMLP(n_embd)

    def forward(self, x, cos, sin):
        x = x + self.attn(norm(x), cos, sin)
        x = x + self.mlp(norm(x))
        return x


class MTPHead(nn.Module):
    """
    Multi-token prediction head for Stage 1 MTP.

    Predicts K-1 additional tokens (2nd..Kth) given:
    - Hidden states h from main model
    - 1st token prediction from main model's lm_head

    Args:
        n_embd: embedding dimension
        n_head: number of attention heads
        n_future_tokens: K, total number of tokens to predict (head predicts K-1)
        group_wte: embedding layer for group tokens (used for teacher forcing)
        lm_head: shared output head (outputs V logits, where V depends on target mode)
        stage1_target_mode: "pure" or "group" - determines loss computation
    """
    def __init__(self, n_embd, n_head, n_future_tokens, group_wte, lm_head, stage1_target_mode="pure"):
        super().__init__()
        self.n_future_tokens = n_future_tokens
        self.group_wte = group_wte  # shared, not owned (always used for teacher forcing)
        self.lm_head = lm_head  # shared, not owned
        self.stage1_target_mode = stage1_target_mode

        # Projection to combine hidden state + token embedding
        # DeepSeek style: concat + project
        self.proj = nn.Linear(2 * n_embd, n_embd, bias=False)

        # Single block, reused K-1 times
        self.block = MTPBlock(n_embd, n_head)

    def init_weights(self):
        """Initialize MTP-specific weights"""
        # Zero out projection weights (like c_proj in gpt.py)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.block.attn.c_proj.weight)
        nn.init.zeros_(self.block.mlp.c_proj.weight)

    def forward(self, h, cos, sin, first_group_tok, targets=None, pure_to_group=None):
        """
        Forward pass for MTP head.

        Teacher forcing always uses group embeddings (group_wte). In pure mode,
        if using autoregressive (targets=None), predictions are converted to
        group tokens via pure_to_group for the next iteration.

        Args:
            h: hidden states from main model, shape (B, T, D)
            cos, sin: rotary embeddings from main model
            first_group_tok: (B, T) first group token index (already converted from pure if needed)
            targets: ground truth GROUP tokens for teacher forcing, shape (B, T, K-1) or None.
                     In pure mode, caller should convert pure targets to group tokens first.
            pure_to_group: (pure_vocab_size, overlap_k) mapping for autoregressive in pure mode.
                           Only used when targets=None and stage1_target_mode="pure".

        Returns:
            all_logits: (B, T, K-1, V) logits for 2nd..Kth tokens
                        V = pure_vocab_size in pure mode, num_groups in group mode
        """
        B, T, D = h.shape
        K = self.n_future_tokens
        all_logits = []

        prev_h = h  # start from main model's final hidden state
        prev_group_tok = first_group_tok  # group token index for embedding lookup

        # Predict K-1 additional tokens (2nd through Kth)
        for k in range(K - 1):
            # Get token embedding for previous prediction (always group embedding)
            tok_emb = self.group_wte(prev_group_tok)  # (B, T, D)

            # Combine: [norm(h), norm(embed)] -> project
            combined = torch.cat([norm(prev_h), norm(tok_emb)], dim=-1)  # (B, T, 2D)
            x = self.proj(combined)  # (B, T, D)

            # Forward through shared block (use same rotary for all positions)
            prev_h = self.block(x, cos, sin)
            prev_h = norm(prev_h)

            # Compute logits
            logits = self.lm_head(prev_h)  # (B, T, V)
            all_logits.append(logits)

            # Get next group token for next iteration
            if targets is not None:
                # Teacher forcing: use ground truth (already group tokens)
                prev_group_tok = targets[:, :, k]  # (B, T)
            else:
                # Autoregressive: use prediction
                if self.stage1_target_mode == "pure":
                    # In pure mode, logits are over pure vocab
                    # Convert predicted pure token to group token
                    pred_pure = logits.argmax(dim=-1)  # (B, T)
                    assert pure_to_group is not None, "pure_to_group required for autoregressive pure mode"
                    pred_groups = pure_to_group[pred_pure]  # (B, T, overlap_k)
                    prev_group_tok = pred_groups[:, :, 0]  # (B, T) use first group
                else:
                    # In group mode, logits are over group vocab
                    prev_group_tok = logits.argmax(dim=-1)  # (B, T)

        return torch.stack(all_logits, dim=2)  # (B, T, K-1, V)


def compute_mtp_loss(main_logits, mtp_logits, targets, loss_weights=None,
                     mtp_loss_beta=0.8, stage1_target_mode="pure"):
    """
    Compute combined MTP loss with exponential decay weights.

    In pure mode: standard cross-entropy loss against pure token targets.
    In group mode: any-correct loss where predicting ANY valid group is correct.

    Args:
        main_logits: (B, T, V) logits from main model for 1st token
        mtp_logits: (B, T, K-1, V) logits from MTP head for 2nd..Kth tokens
        targets: Target format depends on stage1_target_mode:
                 - pure mode: (B, T, K) pure token ids
                 - group mode: (B, T, K, overlap_k) group token ids with all valid groups
        loss_weights: optional (K,) weights, default exponential decay
        mtp_loss_beta: decay factor for exponential weights
        stage1_target_mode: "pure" or "group"

    Returns:
        loss: scalar combined loss
    """
    V = main_logits.shape[-1]

    if stage1_target_mode == "pure":
        # Pure mode: targets are (B, T, K) pure token ids
        B, T, K = targets.shape
    else:
        # Group mode: targets are (B, T, K, overlap_k) group token ids
        B, T, K, overlap_k = targets.shape

    if loss_weights is None:
        loss_weights = exponential_decay_weights(K, mtp_loss_beta, device=main_logits.device)

    total_loss = 0.0

    if stage1_target_mode == "pure":
        # Pure mode: standard cross-entropy against pure tokens
        # Loss for 1st token (from main model)
        step_targets = targets[:, :, 0]  # (B, T)
        step_loss = F.cross_entropy(
            main_logits.reshape(-1, V),
            step_targets.reshape(-1)
        )
        total_loss = total_loss + loss_weights[0] * step_loss

        # Loss for 2nd..Kth tokens (from MTP head)
        for k in range(K - 1):
            step_logits = mtp_logits[:, :, k, :]  # (B, T, V)
            step_targets = targets[:, :, k + 1]  # (B, T)
            step_loss = F.cross_entropy(
                step_logits.reshape(-1, V),
                step_targets.reshape(-1)
            )
            total_loss = total_loss + loss_weights[k + 1] * step_loss
    else:
        # Group mode: any-correct loss for overlap tokenizer
        # Loss for 1st token (from main model)
        step_targets = targets[:, :, 0, :]  # (B, T, overlap_k)
        step_loss = any_correct_ce_loss(
            main_logits.reshape(-1, V),
            step_targets.reshape(-1, overlap_k)
        )
        total_loss = total_loss + loss_weights[0] * step_loss

        # Loss for 2nd..Kth tokens (from MTP head)
        for k in range(K - 1):
            step_logits = mtp_logits[:, :, k, :]  # (B, T, V)
            step_targets = targets[:, :, k + 1, :]  # (B, T, overlap_k)
            step_loss = any_correct_ce_loss(
                step_logits.reshape(-1, V),
                step_targets.reshape(-1, overlap_k)
            )
            total_loss = total_loss + loss_weights[k + 1] * step_loss

    return total_loss


def exponential_decay_weights(K, beta=0.8, device=None):
    """
    Exponential decay weights for MTP loss.
    Earlier tokens weighted more heavily.

    Args:
        K: number of tokens
        beta: decay factor (0 < beta < 1 means faster decay)

    Returns:
        weights: (K,) tensor, normalized to sum to 1
    """
    weights = torch.tensor([beta ** k for k in range(K)], device=device)
    return weights / weights.sum()


def any_correct_ce_loss(logits, valid_targets):
    """
    Cross-entropy loss where ANY of the valid targets is considered correct.

    Loss = -log(Σ P(g) for g in valid_groups)

    This allows the model to predict any of the valid group assignments
    when overlap_k > 1.

    Args:
        logits: (*, num_groups) - logits over group vocabulary
        valid_targets: (*, overlap_k) - ALL valid group indices for each position
                       Use -1 for padding (will be masked out)

    Returns:
        loss: scalar mean loss
    """
    # Compute log probabilities
    log_probs = F.log_softmax(logits, dim=-1)

    # Mask invalid targets (where valid_targets == -1)
    valid_mask = valid_targets >= 0  # (*, overlap_k)

    # Clamp to valid indices for gather (masked positions will be ignored)
    safe_targets = valid_targets.clamp(min=0)

    # Gather log probs for valid targets
    valid_log_probs = torch.gather(log_probs, dim=-1, index=safe_targets)  # (*, overlap_k)

    # Mask out invalid positions with -inf before logsumexp
    valid_log_probs = valid_log_probs.masked_fill(~valid_mask, float('-inf'))

    # Log-sum-exp over valid groups: log(Σ P(valid_g))
    log_valid_prob = torch.logsumexp(valid_log_probs, dim=-1)  # (*)

    # Loss = -log(P(any valid))
    return -log_valid_prob.mean()
