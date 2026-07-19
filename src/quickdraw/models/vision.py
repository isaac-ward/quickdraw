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
    build_decoder: bool = True  # False when a generative flow decode head replaces the mse decoder (no dead weight)


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
        # decoder (mse). Skipped entirely when a generative flow decode head replaces it (build_decoder=False)
        # — otherwise these would be dead, never-called, never-trained params.
        if cfg.build_decoder:
            self.dec_pos = nn.Parameter(torch.zeros(1, self.np, d))   # output-patch query tokens
            self.from_latent = CrossAttn(d, h)
            self.dec_blocks = nn.ModuleList([ViTBlock(d, h, cfg.mlp_ratio) for _ in range(cfg.dec_depth)])
            self.dec_norm = nn.LayerNorm(d)
            self.unpatch = nn.Linear(d, pdim)
            nn.init.trunc_normal_(self.dec_pos, std=0.02)
        for p in (self.enc_pos, self.latent_q):
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


class _FiLMResBlock(nn.Module):
    """Conv residual block with FiLM (per-channel scale+shift) from a global conditioning vector `g`."""

    def __init__(self, cin: int, cout: int, gdim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, cin), cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, cout), cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.film = nn.Linear(gdim, 2 * cout)
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)   # zero-init -> s=0,b=0 at init, so the
        #   block starts as an exact identity modulation (ADM/DiT standard). Prevents FiLM's multiplicative
        #   (1+s) term from amplifying activations early, which is how the flow-decode U-Net overflowed to inf.
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, g):
        h = self.conv1(F.silu(self.norm1(x)))
        s, b = self.film(g).chunk(2, dim=-1)
        h = self.norm2(h) * (1 + s[..., None, None]) + b[..., None, None]
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class _ConvResBlock(nn.Module):
    """Plain (unconditioned) conv residual block — the encoder counterpart of _FiLMResBlock (no FiLM: the encoder
    has nothing to condition on)."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, cin), cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, cout), cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class ConvImageEncoder(nn.Module):
    """Convolutional image encoder that MIRRORS ConditionalUNet's down-path, so a conv encoder can pair with the
    conv (unet) decoder — symmetric inductive bias, no patch grid. Same output contract as ImageAutoencoder.encode:
    (M,H,W,3)[0,1] -> (M, num_tokens, d). Conv pyramid -> a small bottleneck feature map -> num_tokens learned
    queries cross-attend it (Perceiver bottleneck, same as the ViT encoder) so the token count/interface is
    identical and the projection stays cheap (no dense flatten)."""

    def __init__(self, cfg: VisionAEConfig, *, base: int = 32):
        super().__init__()
        import math
        self.cfg = cfg
        C, d, T = cfg.channels, cfg.d, cfg.num_tokens
        n_levels = max(1, int(math.log2(max(8, cfg.img_size) // 8)))   # keep the bottleneck ~8px (mirrors the U-Net)
        chs = [base * min(4, 2 ** i) for i in range(n_levels)]
        self.in_conv = nn.Conv2d(C, chs[0], 3, padding=1)
        prev, self.downs = chs[0], nn.ModuleList()
        for ch in chs:
            self.downs.append(_ConvResBlock(prev, ch)); prev = ch
        self.bott_hw = cfg.img_size // (2 ** len(chs))                 # 8 at 128px
        self.to_d = nn.Conv2d(chs[-1], d, 1)                          # channels -> model dim
        self.pos = nn.Parameter(torch.zeros(1, self.bott_hw * self.bott_hw, d))
        self.latent_q = nn.Parameter(torch.zeros(1, T, d))
        self.to_latent = CrossAttn(d, cfg.heads)
        self.latent_norm = nn.LayerNorm(d)
        for p in (self.pos, self.latent_q):
            nn.init.trunc_normal_(p, std=0.02)

    def encode(self, img):                                            # (M,H,W,C)[0,1] -> (M,T,d)
        h = self.in_conv(img.permute(0, 3, 1, 2))
        for down in self.downs:
            h = down(h); h = F.avg_pool2d(h, 2)
        x = self.to_d(h).flatten(2).transpose(1, 2) + self.pos        # (M, bott_hw^2, d)
        z = self.to_latent(self.latent_q.expand(x.shape[0], -1, -1), x)
        return self.latent_norm(z)


class ConditionalUNet(nn.Module):
    """Conv U-Net over an image, conditioned on the latent tokens (spatial injection at the bottleneck) + an
    optional time/step embedding. The convolutional alternative to the all-ViT image head — no patch grid, so
    smooth fields don't block. ONE module serves both decode kinds via `velocity(x, temb, cond, demb)`:
      - flow:  x = the noised image, temb = tau embedding  -> denoiser (velocity or x0).
      - mse:   x = zeros, temb = None                      -> pure tokens->image decoder.
    Decodes from `cond` regardless of `x` (bottleneck injection), so the eps=0 deterministic sample works."""

    def __init__(self, ae_cfg, *, base: int = 32, time_dim: int = 32):
        super().__init__()
        import math
        self.cfg = ae_cfg
        C, d, T = ae_cfg.channels, ae_cfg.d, ae_cfg.num_tokens
        n_levels = max(1, int(math.log2(max(8, ae_cfg.img_size) // 8)))   # keep the bottleneck ~8px (128->4, 64->3, 32->2)
        chs = [base * min(4, 2 ** i) for i in range(n_levels)]            # e.g. 128px -> [base,2b,4b,4b]
        self.gdim = d
        self.t_proj = nn.Linear(time_dim, d)                  # time (flow); unused for mse (temb=None)
        self.d_proj = nn.Linear(time_dim, d)                  # step-size (shortcut); unused unless demb given
        self.in_conv = nn.Conv2d(C, chs[0], 3, padding=1)
        prev, self.downs = chs[0], nn.ModuleList()
        for ch in chs:
            self.downs.append(_FiLMResBlock(prev, ch, d)); prev = ch
        self.bott_hw = ae_cfg.img_size // (2 ** len(chs))     # 128/16 = 8
        self.seed_hw = 2                                       # tokens -> a small 2x2 seed, upsampled to the bottleneck
        self.cond_to_spatial = nn.Linear(T * d, chs[-1] * self.seed_hw * self.seed_hw)   # (was a dense 8x8 map = the 8M term)
        self.mid = _FiLMResBlock(chs[-1], chs[-1], d)
        self.ups, prev = nn.ModuleList(), chs[-1]
        for ch in reversed(chs):
            self.ups.append(_FiLMResBlock(prev + ch, ch, d)); prev = ch  # concat skip
        self.out_norm = nn.GroupNorm(min(8, chs[0]), chs[0])
        self.out_conv = nn.Conv2d(chs[0], C, 3, padding=1)

    def velocity(self, x, temb=None, cond=None, demb=None):   # x:(M,H,W,C) cond:(M,T,d) -> (M,H,W,C)
        M = cond.shape[0]
        g = cond.mean(1)                                      # (M,d) global conditioning
        if temb is not None:
            g = g + self.t_proj(temb.reshape(M, -1))
        if demb is not None:
            g = g + self.d_proj(demb.reshape(M, -1))
        h = self.in_conv(x.permute(0, 3, 1, 2))               # (M,C,H,W)
        skips = []
        for down in self.downs:
            h = down(h, g); skips.append(h); h = F.avg_pool2d(h, 2)
        seed = self.cond_to_spatial(cond.reshape(M, -1)).reshape(M, -1, self.seed_hw, self.seed_hw)
        h = h + F.interpolate(seed, size=(self.bott_hw, self.bott_hw), mode="nearest")
        h = self.mid(h, g)
        for up, skip in zip(self.ups, reversed(skips)):
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            h = up(torch.cat([h, skip], dim=1), g)
        return self.out_conv(F.silu(self.out_norm(h))).permute(0, 2, 3, 1)   # (M,H,W,C)
