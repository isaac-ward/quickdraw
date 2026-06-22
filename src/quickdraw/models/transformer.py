"""Causal, pre-norm transformer with RoPE and a sliding-window mask.

Adapted from seamstress (blocks.py / attention.py / positional.py): made causal, pre-norm,
RoPE-only, with a causal + sliding-window attention mask. FlexAttention ONLY — block-sparse, skips
out-of-window blocks. There is no SDPA fallback by design: if FlexAttention is unavailable we fail
hard (it is required for the speed this project targets).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch.nn.attention.flex_attention import create_block_mask, flex_attention


# ----------------------------- RoPE -----------------------------
def build_rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype) -> tuple[Tensor, Tensor]:
    half = head_dim // 2
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    freqs = torch.outer(pos, inv_freq)  # (T, half)
    emb = torch.cat((freqs, freqs), dim=-1)  # (T, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """x: (B, H, T, Dh); cos/sin: (T, Dh)."""
    cos = cos[None, None]
    sin = sin[None, None]
    return x * cos + _rotate_half(x) * sin


# ----------------------------- masks -----------------------------
def _sliding_causal_mask_mod(window: int):
    def mask_mod(b, h, q_idx, kv_idx):
        return (kv_idx <= q_idx) & (q_idx - kv_idx < window)

    return mask_mod


# ----------------------------- attention -----------------------------
class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, window: int, rope_theta: float):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.window = window
        self.rope_theta = rope_theta
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, T, Dh)
        cos, sin = build_rope_cache(T, self.head_dim, self.rope_theta, x.device, x.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        block_mask = create_block_mask(_sliding_causal_mask_mod(self.window), B=None, H=None, Q_LEN=T, KV_LEN=T, device=x.device)
        o = flex_attention(q, k, v, block_mask=block_mask)
        o = o.transpose(1, 2).reshape(B, T, self.heads * self.head_dim)
        return self.out(o)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float):
        super().__init__()
        hidden = int(round(dim * mlp_ratio))
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    """Pre-norm residual block."""

    def __init__(self, dim: int, heads: int, window: int, mlp_ratio: float, rope_theta: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, heads, window, rope_theta)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, mlp_ratio)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, window: int, mlp_ratio: float, rope_theta: float = 10000.0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [Block(dim, heads, window, mlp_ratio, rope_theta) for _ in range(depth)]
        )
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.norm_out(x)
