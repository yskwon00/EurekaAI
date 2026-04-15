"""
EurekaAI — Core Transformer Architecture (TinyLearnAI-30M)
Decoder-only GPT-style model with:
  - RoPE positional encoding (expandable context)
  - Pre-LayerNorm (more stable training)
  - Causal self-attention with causal masking
  - Weight tying (embedding ↔ lm_head)
  - HuggingFace-compatible save/load
"""

import math
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import EurekaConfig


# ── RoPE (Rotary Position Embedding) ───────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    Encodes position info into Q/K directly — no extra parameters,
    and generalizes to longer sequences than seen in training.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)        # (seq_len, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)       # (seq_len, dim)
        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)

    def forward(self, seq_len: int):
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        return self.cos_cache[:seq_len], self.sin_cache[:seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors."""
    # cos/sin: (seq_len, head_dim) → (1, 1, seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


# ── Attention ───────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE."""

    def __init__(self, config: EurekaConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)

        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len, config.rope_base)

        # Register causal mask buffer (will be expanded on-the-fly)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool)),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape

        # Project Q, K, V
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rope(T)
        q, k = apply_rope(q, k, cos, sin)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, T, T)

        # Apply causal mask
        causal = self.causal_mask[:T, :T]
        attn_weights = attn_weights.masked_fill(~causal, float("-inf"))

        # Apply optional padding mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Aggregate values
        out = torch.matmul(attn_weights, v)              # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


# ── Feed-Forward Network ────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    """Standard FFN with GELU activation."""

    def __init__(self, config: EurekaConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))


# ── Transformer Block ───────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """Pre-LayerNorm transformer block (more stable than post-LN)."""

    def __init__(self, config: EurekaConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.ffn = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attention_mask)
        x = x + self.ffn(self.ln2(x))
        return x


# ── EurekaModel ─────────────────────────────────────────────────────────────────

class EurekaModel(nn.Module):
    """
    TinyLearnAI-30M: Decoder-only Transformer for curriculum self-learning.

    Architecture summary:
      - Token embedding: vocab_size × hidden_size
      - 6 × TransformerBlock (Pre-LN, RoPE, Causal attention, GELU FFN)
      - Final LayerNorm + LM Head (weights tied to embedding)
      - ~29M parameters total
    """

    def __init__(self, config: EurekaConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.emb_dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )

        self.ln_final = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Weight tying — drastically reduces parameter count
        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self):
        """GPT-2 style weight initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Scale residual stream output projections (GPT-2 trick)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.num_layers))

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            input_ids:      (B, T) token ids
            labels:         (B, T) shifted targets; -100 = ignore
            attention_mask: (B, T) 1=real token, 0=pad (optional)

        Returns:
            dict with 'logits' (B, T, vocab) and optionally 'loss'
        """
        B, T = input_ids.shape
        x = self.emb_dropout(self.token_embedding(input_ids))

        # Convert padding mask to additive bias
        bias = None
        if attention_mask is not None:
            # (B, 1, 1, T) → broadcast to (B, H, T, T)
            bias = (1.0 - attention_mask[:, None, None, :].float()) * -1e9

        for layer in self.layers:
            x = layer(x, bias)

        x = self.ln_final(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if labels is not None:
            # Flatten for cross-entropy
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive generation with temperature + top-k/p sampling."""
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to max context
            ctx = input_ids[:, -self.config.max_seq_len:]
            logits = self(ctx)["logits"][:, -1, :] / max(temperature, 1e-8)

            # Top-k filtering
            if top_k > 0:
                topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < topk_vals[:, [-1]]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumprobs = sorted_logits.softmax(-1).cumsum(-1)
                remove = cumprobs - sorted_logits.softmax(-1) > top_p
                sorted_logits[remove] = float("-inf")
                logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, next_tok], dim=1)

            if eos_token_id is not None and (next_tok == eos_token_id).all():
                break

        return input_ids

    def num_parameters(self, trainable_only: bool = False) -> int:
        params = self.parameters() if not trainable_only else (
            p for p in self.parameters() if p.requires_grad
        )
        return sum(p.numel() for p in params)

    def save_pretrained(self, save_dir: str):
        """Save model + config in HuggingFace-compatible format."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_path / "pytorch_model.bin")
        self.config.save_json(str(save_path / "config.json"))
        print(f"✅ Model saved to {save_path}")

    @classmethod
    def from_pretrained(cls, load_dir: str, map_location: str = "cpu") -> "EurekaModel":
        """Load model from saved checkpoint."""
        load_path = Path(load_dir)
        config = EurekaConfig.from_json(str(load_path / "config.json"))
        model = cls(config)
        state = torch.load(load_path / "pytorch_model.bin", map_location=map_location)
        model.load_state_dict(state)
        print(f"✅ Model loaded from {load_path} ({model.num_parameters()/1e6:.1f}M params)")
        return model

    def __repr__(self):
        return (
            f"EurekaModel(\n"
            f"  config={self.config}\n"
            f"  parameters={self.num_parameters()/1e6:.1f}M\n"
            f")"
        )
