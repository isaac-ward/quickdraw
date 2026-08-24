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


def img_hw(img_size) -> tuple[int, int]:
    """img_size int (square) or (H, W) -> (H, W). The one spot that normalizes the two forms."""
    return (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)


@dataclass
class VisionAEConfig:
    img_size: int | tuple[int, int] = 128   # int -> square (torus); (H, W) -> non-square (recorded cams)
    patch: int = 16
    d: int = 256
    enc_depth: int = 4
    dec_depth: int = 4
    heads: int = 8
    num_tokens: int = 8     # latent token-list length (NOT the diffusion step count K)
    channels: int = 3
    mlp_ratio: float = 4.0
    build_decoder: bool = True  # False when a generative flow decode head replaces the mse decoder (no dead weight)
    bottleneck: int = 8     # TARGET spatial size of the conv pyramid's bottleneck, for BOTH ConvImageEncoder and
    #                         ConditionalUNet. 8 = the previous hardcoded value = bit-identical.
    #                         WHY IT IS A KNOB (2026-08-21): both classes computed
    #                         `n_levels = log2(short_side // 8)`, i.e. they pooled to 8x8 BY CONSTRUCTION at every
    #                         resolution -- so the encoder discarded all spatial detail below 8x8 BEFORE the
    #                         num_tokens queries ever saw it. That is why sweeping num_tokens 8->64 (an 8x range
    #                         of latent floats), decode_base 32->64, ae_depth 4->6 and latent_loss_weight 10->30
    #                         ALL left the bespoke reconstruction floor flat at 18.7-20.4 dB: none of them touch
    #                         the binding constraint. Raising this to 16 makes the bottleneck 16x16 (a 64x spatial
    #                         reduction at 128px instead of 256x) and gives the token budget something to carry.
    #                         NOT exposed as `n_levels` directly: the encoder pools AFTER a stride-2 stem and the
    #                         decoder pools from full resolution, so at 128px they use 3 and 4 levels respectively
    #                         -- one shared n_levels would desynchronise them. A shared TARGET cannot.


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
        H, W = img_hw(cfg.img_size)
        gh, gw = H // cfg.patch, W // cfg.patch
        assert gh * cfg.patch == H and gw * cfg.patch == W, "img_size must be divisible by patch"
        self.gh, self.gw, self.np = gh, gw, gh * gw
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
        p, gh, gw = self.cfg.patch, self.gh, self.gw
        return img.reshape(B, gh, p, gw, p, C).permute(0, 1, 3, 2, 4, 5).reshape(B, gh * gw, p * p * C)

    def unpatchify(self, x):                                  # (B, np, patch*patch*C) -> (B,H,W,C)
        p, gh, gw, C = self.cfg.patch, self.gh, self.gw, self.cfg.channels
        x = x.reshape(x.shape[0], gh, gw, p, p, C).permute(0, 1, 3, 2, 4, 5)
        return x.reshape(x.shape[0], gh * p, gw * p, C)

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


# ---- pretrained-AE (TAESD) grid<->token adapters (issue #12 §2): bridge a spatial latent GRID and the token
# BAG so the dynamics still sees num_tokens tokens (never a 16x16=256-token grid). Same Perceiver pattern as
# ImageAutoencoder (learned queries cross-attend), reused here on the frozen pretrained-AE latent. ----
def adapter_mode(lat_ch: int, grid_hw: tuple[int, int], num_tokens: int, d: int, dense: bool = False) -> dict:
    """Decide (and DESCRIBE) how a pretrained-AE latent grid maps onto the num_tokens x d token bag.

    L = lat_ch*gh*gw floats in the grid; M = num_tokens*d floats in the bag. The base path is a fixed
    index rearrangement (NO parameters, exactly invertible); a learned residual refines on top.

      EXACT     M == L            reshape only, no padding, no wasted decode width  -> identity at init
      PADDED    M >  L, !dense    reshape + zero-pad each token to d, strip on the inverse -> identity at init
                                  BUT: `identity_at_init` is an ADAPTER-ONLY property. The model runs
                                  encode -> LayerNorm -> decode, and LN is PER TOKEN, so the pad floats join
                                  the mean/std the real floats are normalized by. PADDED is therefore NOT
                                  identity end-to-end and is measurably WORSE than any EXACT split: 16.03 dB
                                  vs 20.41 dB on robocasa 128px + frozen TAESD (2026-08-09). Prefer EXACT.
      PROJECTED M >  L,  dense    learned per->d projection (dense tokens, no idle width) -> NOT identity
      LOSSY     M <  L            impossible without discarding latent floats -> caller must raise

    `per` = ceil(L/num_tokens) real floats per token; per <= d is guaranteed whenever M >= L (if per > d then
    L > num_tokens*d = M, contradicting M >= L)."""
    gh, gw = grid_hw
    L, M = lat_ch * gh * gw, num_tokens * d
    per = -(-L // num_tokens)                                    # ceil
    if M < L:
        mode = "LOSSY"
    elif M == L:
        mode = "EXACT"
    else:
        mode = "PROJECTED" if dense else "PADDED"
    return {"mode": mode, "L": L, "M": M, "per": per, "pad_per_token": max(0, d - per),
            "identity_at_init": mode in ("EXACT", "PADDED"),
            "grid": (lat_ch, gh, gw), "bag": (num_tokens, d)}


def _zero_init_last(module: nn.Module) -> nn.Module:
    """Zero the LAST Linear's weight+bias so the module outputs exactly 0 -> a residual branch that starts as
    a no-op (ControlNet / DiT adaLN-zero trick). This is what makes the round-trip an EXACT identity at init
    rather than a well-initialised approximation."""
    last = [m for m in module.modules() if isinstance(m, nn.Linear)][-1]
    nn.init.zeros_(last.weight)
    if last.bias is not None:
        nn.init.zeros_(last.bias)
    return module


class GridToTokens(nn.Module):
    """down-adapter: latent GRID (B, C, gh, gw) -> token LIST (B, num_tokens, d).

    BASE PATH = a fixed index rearrangement, no parameters, exactly invertible by TokensToGrid. The grid is
    permuted to (H, W, C) BEFORE flattening so each token is a contiguous SPATIAL PATCH carrying all channels
    (ViT-patch-like), rather than a channel-major slab. Nothing is destroyed — the bag holds the same floats,
    and the backbone's spatial attention re-mixes across tokens anyway.
    REFINE = a zero-initialised residual, so at step 0 the adapter is EXACTLY the rearrangement."""

    def __init__(self, lat_ch: int, grid_hw: tuple[int, int], num_tokens: int, d: int, heads: int, depth: int,
                 dense: bool = False):
        super().__init__()
        self.info = adapter_mode(lat_ch, grid_hw, num_tokens, d, dense)
        if self.info["mode"] == "LOSSY":
            raise ValueError(
                f"pretrained-AE adapter is LOSSY: latent {self.info['grid']} = {self.info['L']} floats does not "
                f"fit the token bag {self.info['bag']} = {self.info['M']} floats. Raise model.modalities."
                f"<image>.num_tokens or model.d so that num_tokens*d >= {self.info['L']}.")
        self.lat_ch, (self.gh, self.gw) = lat_ch, grid_hw
        self.T, self.d, self.per = num_tokens, d, self.info["per"]
        self.dense = self.info["mode"] == "PROJECTED"
        if self.dense:                                            # learned per->d (no identity guarantee)
            self.proj = nn.Linear(self.per, d)
        self.refine = _zero_init_last(nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(),
                                                    nn.Linear(4 * d, d)))

    def base(self, grid):                                        # (B,C,gh,gw) -> (B,T,d), parameter-free
        B = grid.shape[0]
        flat = grid.permute(0, 2, 3, 1).reshape(B, -1)            # (B, gh*gw*C) — spatial-major, channels inner
        need = self.T * self.per
        if flat.shape[1] < need:                                  # pad the FLAT vector so the split is even
            flat = F.pad(flat, (0, need - flat.shape[1]))
        x = flat.reshape(B, self.T, self.per)                     # (B,T,per) — each token a contiguous patch
        if self.dense:
            return self.proj(x)
        # Widen to d with zeros. THIS IS NOT FREE, AND NOT MERELY IDLE WIDTH. Zero-pad + strip is a bijection,
        # so the ADAPTER round-trip stays an exact identity -- but the model runs encode -> LayerNorm -> decode
        # and _ln is PER TOKEN, not per element: it takes the mean and std over all d entries of the token,
        # INCLUDING these zeros, and normalizes the `per` real floats by them. The pad therefore sits inside the
        # statistic the signal is divided by, and stripping it afterwards cannot undo a scale that was already
        # applied. Measured (robocasa 128px + frozen TAESD, 2026-08-09): bag 32x128 (75% pad) floors at 16.03 dB
        # vs 20.41 dB for the EXACT 8x128 bag -- -4.4 dB, a bigger loss than ANY num_tokens choice. Keep
        # num_tokens*d == the latent size (EXACT); see design/capacity.md and conf/model/mm_flow.yaml.
        return F.pad(x, (0, self.d - self.per))

    def forward(self, grid):
        b = self.base(grid)
        return b + self.refine(b)


class TokensToGrid(nn.Module):
    """up-adapter: token LIST (B, num_tokens, d) -> latent GRID (B, C, gh, gw). EXACT inverse of
    GridToTokens.base (strip the pad, unflatten, un-permute) + a zero-initialised residual, so
    up(down(g)) == g exactly at init whenever the mode is EXACT or PADDED."""

    def __init__(self, lat_ch: int, grid_hw: tuple[int, int], num_tokens: int, d: int, heads: int, depth: int,
                 dense: bool = False):
        super().__init__()
        self.info = adapter_mode(lat_ch, grid_hw, num_tokens, d, dense)
        self.lat_ch, (self.gh, self.gw) = lat_ch, grid_hw
        self.T, self.d, self.per = num_tokens, d, self.info["per"]
        self.dense = self.info["mode"] == "PROJECTED"
        if self.dense:
            self.unproj = nn.Linear(d, self.per)
        self.refine = _zero_init_last(nn.Sequential(nn.LayerNorm(lat_ch), nn.Linear(lat_ch, 4 * lat_ch),
                                                    nn.GELU(), nn.Linear(4 * lat_ch, lat_ch)))

    def base(self, tokens):                                      # (B,T,d) -> (B,C,gh,gw), parameter-free
        B = tokens.shape[0]
        x = self.unproj(tokens) if self.dense else tokens[..., :self.per]      # (B,T,per)
        flat = x.reshape(B, -1)[:, :self.lat_ch * self.gh * self.gw]           # strip the flat-vector pad
        return flat.reshape(B, self.gh, self.gw, self.lat_ch).permute(0, 3, 1, 2)   # un-permute to (B,C,gh,gw)

    def forward(self, tokens):
        g = self.base(tokens)                                    # (B,C,gh,gw)
        r = self.refine(g.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)   # refine over the CHANNEL dim
        return g + r


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
        stem = 2                                                       # stride-2 stem: the full-res activation is
        h0, w0 = (s // stem for s in img_hw(cfg.img_size))             #   1/4 the memory (standard conv-encoder stem)
        bott = max(1, int(getattr(cfg, "bottleneck", 8)))              # target bottleneck (see VisionAEConfig)
        n_levels = max(1, int(math.log2(max(bott, min(h0, w0)) // bott)))   # pool to ~bott px on the short side
        chs = [base * min(4, 2 ** i) for i in range(n_levels)]
        self.in_conv = nn.Conv2d(C, chs[0], 3, stride=stem, padding=1)
        prev, self.downs = chs[0], nn.ModuleList()
        for ch in chs:
            self.downs.append(_ConvResBlock(prev, ch)); prev = ch
        self.bott_hw = (h0 // (2 ** len(chs)), w0 // (2 ** len(chs)))  # (8, 8) at 128px (stem/2 then n_levels pools)
        self.to_d = nn.Conv2d(chs[-1], d, 1)                          # channels -> model dim
        self.pos = nn.Parameter(torch.zeros(1, self.bott_hw[0] * self.bott_hw[1], d))
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
        H, W = img_hw(ae_cfg.img_size)
        bott = max(1, int(getattr(ae_cfg, "bottleneck", 8)))              # SAME target as ConvImageEncoder
        n_levels = max(1, int(math.log2(max(bott, min(H, W)) // bott)))    # bottleneck ~bott px (at bott=8: 128->4)
        chs = [base * min(4, 2 ** i) for i in range(n_levels)]            # e.g. 128px -> [base,2b,4b,4b]
        self.gdim = d
        self.t_proj = nn.Linear(time_dim, d)                  # time (flow); unused for mse (temb=None)
        self.d_proj = nn.Linear(time_dim, d)                  # step-size (shortcut); unused unless demb given
        self.in_conv = nn.Conv2d(C, chs[0], 3, padding=1)
        prev, self.downs = chs[0], nn.ModuleList()
        for ch in chs:
            self.downs.append(_FiLMResBlock(prev, ch, d)); prev = ch
        self.bott_hw = (H // (2 ** len(chs)), W // (2 ** len(chs)))   # (8, 8) at 128px
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
        h = h + F.interpolate(seed, size=self.bott_hw, mode="nearest")
        h = self.mid(h, g)
        for up, skip in zip(self.ups, reversed(skips)):
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            h = up(torch.cat([h, skip], dim=1), g)
        return self.out_conv(F.silu(self.out_norm(h))).permute(0, 2, 3, 1)   # (M,H,W,C)
