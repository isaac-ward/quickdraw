"""Vision modality — a 100% ViT image autoencoder (design/models/vision.md).

No CNNs: patchify / unpatchify are plain Linear maps over flattened patches (a non-overlapping patch
projection is mathematically a linear map, so there is no convolutional inductive bias). No pretraining,
no checkpoints — trained end-to-end with the world model under plain MSE recon. The latent is a flat LIST
of `num_tokens` tokens (never a spatial grid), produced by a Perceiver-style bottleneck (learned queries
cross-attend the patch tokens) and consumed the same way on decode.

  Linear-patchify -> ViT encoder -> num_tokens learned queries cross-attend -> ViT decoder -> Linear-unpatchify

Bidirectional attention everywhere (no causal mask) via SDPA. The temporal world-model backbone is a
SEPARATE, causal/RoPE transformer (models/transformer.py); P3 unifies the block where it reduces
duplication. Image tensors are channels-last [0,1] floats: (B, H, W, 3)."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VisionAEConfig:
    img_size: int = 128
    patch: int = 16
    d: int = 256
    enc_depth: int = 4
    dec_depth: int = 4
    heads: int = 8
    num_tokens: int = 8     # latent token-list length (NOT the diffusion step count K)
    channels: int = 3
    mlp_ratio: float = 4.0


def _heads(x, heads):                                    # (B,N,d) -> (B,heads,N,hd)
    B, N, d = x.shape
    return x.view(B, N, heads, d // heads).transpose(1, 2)


class ViTBlock(nn.Module):
    """Pre-norm bidirectional transformer block (full attention, no mask)."""

    def __init__(self, d: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.heads = heads
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        h = int(d * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, d))

    def forward(self, x):
        y = self.n1(x)
        q, k, v = self.qkv(y).chunk(3, dim=-1)
        o = F.scaled_dot_product_attention(_heads(q, self.heads), _heads(k, self.heads), _heads(v, self.heads))
        o = o.transpose(1, 2).reshape(x.shape)
        x = x + self.proj(o)
        return x + self.mlp(self.n2(x))


class CrossAttn(nn.Module):
    """Pre-norm cross-attention: `q_tokens` attend to `ctx` (Perceiver-style bottleneck / expansion)."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads = heads
        self.nq = nn.LayerNorm(d)
        self.nkv = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, q_tokens, ctx):
        q = self.q(self.nq(q_tokens))
        k, v = self.kv(self.nkv(ctx)).chunk(2, dim=-1)
        o = F.scaled_dot_product_attention(_heads(q, self.heads), _heads(k, self.heads), _heads(v, self.heads))
        return q_tokens + self.proj(o.transpose(1, 2).reshape(q_tokens.shape))


class ImageAutoencoder(nn.Module):
    """ViT AE. encode: (B,H,W,3)[0,1] -> latent (B,num_tokens,d). decode: latent -> (B,H,W,3)."""

    def __init__(self, cfg: VisionAEConfig):
        super().__init__()
        self.cfg = cfg
        gp = cfg.img_size // cfg.patch
        assert gp * cfg.patch == cfg.img_size, "img_size must be divisible by patch"
        self.gp, self.np = gp, gp * gp
        pdim = cfg.patch * cfg.patch * cfg.channels
        d, h = cfg.d, cfg.heads
        # encoder
        self.patch_embed = nn.Linear(pdim, d)
        self.enc_pos = nn.Parameter(torch.zeros(1, self.np, d))
        self.enc_blocks = nn.ModuleList([ViTBlock(d, h, cfg.mlp_ratio) for _ in range(cfg.enc_depth)])
        self.enc_norm = nn.LayerNorm(d)
        # perceiver bottleneck -> token list
        self.latent_q = nn.Parameter(torch.zeros(1, cfg.num_tokens, d))
        self.to_latent = CrossAttn(d, h)
        self.latent_norm = nn.LayerNorm(d)
        # decoder
        self.dec_pos = nn.Parameter(torch.zeros(1, self.np, d))   # output-patch query tokens
        self.from_latent = CrossAttn(d, h)
        self.dec_blocks = nn.ModuleList([ViTBlock(d, h, cfg.mlp_ratio) for _ in range(cfg.dec_depth)])
        self.dec_norm = nn.LayerNorm(d)
        self.unpatch = nn.Linear(d, pdim)
        for p in (self.enc_pos, self.dec_pos, self.latent_q):
            nn.init.trunc_normal_(p, std=0.02)

    # ---- linear (de)patchify, no conv ----
    def patchify(self, img):                                  # (B,H,W,C) -> (B, np, patch*patch*C)
        B, H, W, C = img.shape
        p, gp = self.cfg.patch, self.gp
        return img.reshape(B, gp, p, gp, p, C).permute(0, 1, 3, 2, 4, 5).reshape(B, gp * gp, p * p * C)

    def unpatchify(self, x):                                  # (B, np, patch*patch*C) -> (B,H,W,C)
        p, gp, C = self.cfg.patch, self.gp, self.cfg.channels
        x = x.reshape(x.shape[0], gp, gp, p, p, C).permute(0, 1, 3, 2, 4, 5)
        return x.reshape(x.shape[0], gp * p, gp * p, C)

    def encode(self, img):                                    # -> (B, num_tokens, d)
        x = self.patch_embed(self.patchify(img)) + self.enc_pos
        for blk in self.enc_blocks:
            x = blk(x)
        x = self.enc_norm(x)
        z = self.to_latent(self.latent_q.expand(x.shape[0], -1, -1), x)
        return self.latent_norm(z)

    def decode(self, z):                                      # (B, num_tokens, d) -> (B,H,W,C)
        x = self.from_latent(self.dec_pos.expand(z.shape[0], -1, -1), z)
        for blk in self.dec_blocks:
            x = blk(x)
        x = self.dec_norm(x)
        return self.unpatchify(self.unpatch(x))

    def forward(self, img):
        z = self.encode(img)
        return self.decode(z), z
