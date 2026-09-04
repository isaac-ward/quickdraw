"""Factorized space-time transformer (design/models/vision.md).

Operates on a per-step TOKEN BAG: x is (B, T, N, d) — T timesteps, N tokens per step (proprio + action +
image tokens). Each block does, in order:
  1. SPATIAL attention  — within a step, the N tokens attend each other, BIDIRECTIONAL (no mask).
  2. TEMPORAL attention — per token-slot, across the T steps, CAUSAL + sliding-window + RoPE.
  3. MLP.
This is the ViViT / TimeSformer / Genie "divided" attention: cost O(T·N² + N·T²) vs joint O((T·N)²).
The temporal pass REUSES the existing causal `SelfAttention` (RoPE + FlexAttention + the pad/eager mask
paths) verbatim — only the spatial pass is new (plain bidirectional SDPA). A learned per-slot embedding
marks which token is which (proprio / action / image_i) so the shared spatial weights can tell them apart.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

from .transformer import FeedForward, SelfAttention


class _SpatialAttention(nn.Module):
    """Bidirectional multi-head attention over the N tokens of a single step (no mask, no RoPE)."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    # SDPA's CUDA kernel launch overflows when the batch dim M=B*T is very large (MPPI eval batches thousands
    # of candidate rollouts) -> cudaErrorInvalidConfiguration. Spatial attention is independent per row, so we
    # slice M into chunks under this cap and concat — exact, and a no-op for the small batches seen in training.
    _M_CHUNK = 8192

    def forward(self, x: Tensor, attn_eager: bool = False) -> Tensor:   # x: (M, N, d)
        M, N, _ = x.shape
        qkv = self.qkv(x).view(M, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                         # (M, H, N, Dh)

        def _attend(qq, kk, vv):
            if attn_eager:   # MATH backend supports double-backward (contraction penalty); default SDPA does not
                with sdpa_kernel(SDPBackend.MATH):
                    return F.scaled_dot_product_attention(qq, kk, vv)
            return F.scaled_dot_product_attention(qq, kk, vv)   # full (bidirectional)

        if M <= self._M_CHUNK:
            o = _attend(q, k, v)
        else:
            o = torch.cat([_attend(q[i:i + self._M_CHUNK], k[i:i + self._M_CHUNK], v[i:i + self._M_CHUNK])
                           for i in range(0, M, self._M_CHUNK)], dim=0)
        return self.out(o.transpose(1, 2).reshape(M, N, self.heads * self.head_dim))


class SpaceTimeBlock(nn.Module):
    """Pre-norm: spatial (within-step, bidirectional) -> temporal (across-step, causal) -> MLP."""

    def __init__(self, dim: int, heads: int, window: int, mlp_ratio: float, rope_theta: float):
        super().__init__()
        self.s_norm = nn.LayerNorm(dim)
        self.s_attn = _SpatialAttention(dim, heads)
        self.t_norm = nn.LayerNorm(dim)
        self.t_attn = SelfAttention(dim, heads, window, rope_theta)   # reuse causal+RoPE+FlexAttention
        self.m_norm = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, mlp_ratio)

    def forward(self, x: Tensor, temporal_block_mask=None, attn_eager: bool = False) -> Tensor:
        B, T, N, d = x.shape
        # spatial: (B,T,N,d) -> (B*T, N, d), attend over N, bidirectional
        x = x + self.s_attn(self.s_norm(x).reshape(B * T, N, d), attn_eager).reshape(B, T, N, d)
        # temporal: (B,T,N,d) -> (B*N, T, d), causal over T per slot, then back
        y = self.t_norm(x).permute(0, 2, 1, 3).reshape(B * N, T, d)
        if getattr(self, "t_fp32", False):
            # §8.64: run the temporal attention in fp32 even under bf16 autocast. MEASURED at r10B-ctrl's
            # dead ep87 checkpoint: identical loss (0.1817), total grad norm 1.76e10 (bf16 Flex backward)
            # vs 4.25 (fp32). The blow-ups that killed r10B (and r9's ep-159 collapse signature) are a
            # bf16 FlexAttention BACKWARD overflow under unbounded temporal-logit growth (block-3 max
            # |logit| reaches ~17k), not an optimization pathology. Default off = bit-identical.
            with torch.autocast("cuda", enabled=False):
                t = self.t_attn(y.float(), block_mask=temporal_block_mask, attn_eager=attn_eager).to(x.dtype)
        else:
            t = self.t_attn(y, block_mask=temporal_block_mask, attn_eager=attn_eager)
        x = x + t.reshape(B, N, T, d).permute(0, 2, 1, 3)
        # mlp (token-wise)
        return x + self.mlp(self.m_norm(x))

    def forward_cached(self, x: Tensor, ring, pos: int) -> Tensor:
        """Single-step counterpart of forward with a temporal KV-cache. x: (B, N, d) — the NEW step's bag;
        `ring`: this block's KVRing; `pos`: the step's GLOBAL time index (RoPE). Spatial (within-step,
        bidirectional over N) and the MLP are per-step-local so they are recomputed; only temporal attention
        reads/writes the cache across time. Returns (B, N, d)."""
        B, N, d = x.shape
        x = x + self.s_attn(self.s_norm(x))                     # spatial, single step (M=B)
        y = self.t_norm(x).reshape(B * N, 1, d)                 # temporal per-slot: (B*N, 1, d)
        positions = torch.full((1,), pos, device=x.device, dtype=torch.long)
        if getattr(self, "t_fp32", False):
            with torch.autocast("cuda", enabled=False):
                t = self.t_attn.forward_cached(y.float(), ring, positions).to(x.dtype)
        else:
            t = self.t_attn.forward_cached(y, ring, positions)  # (B*N, 1, d)
        x = x + t.reshape(B, N, d)
        return x + self.mlp(self.m_norm(x))


class SpaceTimeTransformer(nn.Module):
    """Stack of factorized blocks over a token bag (B, T, N, d). `n_slots` = N (fixed per model from the
    enabled modalities); a learned per-slot embedding is added once at the input."""

    def __init__(self, dim: int, depth: int, heads: int, window: int, mlp_ratio: float,
                 n_slots: int, rope_theta: float = 10000.0):
        super().__init__()
        self.window = window
        self.slot_emb = nn.Parameter(torch.zeros(1, 1, n_slots, dim))
        nn.init.trunc_normal_(self.slot_emb, std=0.02)
        self.blocks = nn.ModuleList(
            [SpaceTimeBlock(dim, heads, window, mlp_ratio, rope_theta) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x: Tensor, temporal_block_mask=None, attn_eager: bool = False) -> Tensor:
        x = x + self.slot_emb[:, :, : x.shape[2]]                # per-slot (token-type) embedding
        for blk in self.blocks:
            x = blk(x, temporal_block_mask, attn_eager)
        return self.norm_out(x)

    def make_cache(self):
        """Fresh per-block temporal K/V ring buffers for one cached rollout (one KVRing per block)."""
        from .transformer import KVRing
        return [KVRing() for _ in self.blocks]

    def forward_cached(self, x: Tensor, cache, pos: int) -> Tensor:
        """One rollout step through the stack with a temporal KV-cache. x: (B, N, d) — the NEW step's token
        bag; `cache`: list[KVRing] from make_cache(); `pos`: the step's GLOBAL time index. Returns (B, N, d) —
        the backbone output at this step, equal (within fp tolerance) to the parallel path's h[:, -1]."""
        x = x + self.slot_emb[:, 0, : x.shape[1]]                # (1,N,d) per-slot emb, broadcast over B
        for blk, ring in zip(self.blocks, cache):
            x = blk.forward_cached(x, ring, pos)
        return self.norm_out(x)
