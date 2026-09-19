"""Shared conditioning primitives for the up-backed decoders.

Token SET -> spatial GRID (`TokenGridReadout`), token SET -> global vector (`TokenPool`), and per-level
token re-selection (`LevelCrossAttn`). Written as standalone modules so BOTH the deterministic decoder
(`up-mse`, models/decoders/up.py) and the flow denoiser (`up-flow`, models/decoders/up_flow.py) use the
identical readout — see the package `__init__` for the up-vs-unet history."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..vision import CrossAttn, ViTBlock


class TokenGridReadout(nn.Module):
    """Token SET -> spatial GRID, by cross-attention from a learned query grid.

    Replaces `ConditionalUNet.cond_to_spatial` (a Linear(T*d -> chs[-1]*4) that flattened the bag in fixed
    order into 4 spatial cells). A fixed-order flatten is wrong for a SET three ways: it caps the readout at
    its output width regardless of token count, it cannot scale when num_tokens changes (the weight shape is
    baked to T*d), and it denies the decoder per-token addressing. Cross-attention fixes all three -- and
    `grid_q` doubles as the positional encoding the conv path otherwise lacks entirely.

    THE MID SELF-ATTENTION IS NOT OPTIONAL (adversarial audit, 2026-08-26). `grid_q` is content-INDEPENDENT,
    so before its keys learn to discriminate slots, attention is near-uniform and EVERY cell receives
    approximately the token mean -- i.e. at init this module momentarily reproduces the very mean-pooling
    pathology it exists to remove. One self-attention block over the grid cells breaks that symmetry."""

    def __init__(self, d: int, heads: int, grid_hw: tuple[int, int], *, mlp_ratio: float = 4.0):
        super().__init__()
        self.grid_hw = grid_hw
        gh, gw = grid_hw
        self.grid_q = nn.Parameter(torch.zeros(1, gh * gw, d))
        nn.init.trunc_normal_(self.grid_q, std=0.02)          # house style (vision.py pos/latent_q)
        self.readout = CrossAttn(d, heads)                    # cells attend the token bag
        self.mix = ViTBlock(d, heads, mlp_ratio)              # see the init-degeneracy note above

    def forward(self, cond: Tensor) -> Tensor:                # (M,T,d) -> (M,d,gh,gw)
        M = cond.shape[0]
        h = self.readout(self.grid_q.expand(M, -1, -1), cond)  # (M, gh*gw, d)
        h = self.mix(h)
        gh, gw = self.grid_hw
        return h.transpose(1, 2).reshape(M, -1, gh, gw)        # (M,d,gh,gw)


class TokenPool(nn.Module):
    """Token SET -> one global vector, by attention pooling (a single learned query).

    Replaces `g = cond.mean(1)`. The mean's crime was not being a mean -- it was being the ONLY per-block
    conditioning, so token identity was annihilated on that path. Attention pooling is content-adaptive."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.q = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.q, std=0.02)
        self.pool = CrossAttn(d, heads)

    def forward(self, cond: Tensor) -> Tensor:                # (M,T,d) -> (M,d)
        return self.pool(self.q.expand(cond.shape[0], -1, -1), cond)[:, 0]


class LevelCrossAttn(nn.Module):
    """Let one level RE-SELECT from the token bag, instead of reusing the bottleneck readout.

    The trunk runs at `ch` channels while attention lives at `d`, so this projects in, cross-attends the bag
    with every spatial position as its own query, and projects back through a ZERO-INIT conv -- so the level
    is an exact identity at step 0 and this can only be learned into. This is the pattern Stable Diffusion's
    U-Net uses (cross-attention to the conditioning at several resolutions). Queries are H*W, so it is only
    affordable at low resolution; callers gate it by resolution."""

    def __init__(self, ch: int, d: int, heads: int):
        super().__init__()
        self.to_d = nn.Conv2d(ch, d, 1)
        self.xa = CrossAttn(d, heads)
        self.to_ch = nn.Conv2d(d, ch, 1)
        nn.init.zeros_(self.to_ch.weight); nn.init.zeros_(self.to_ch.bias)

    def forward(self, h: Tensor, cond: Tensor) -> Tensor:      # (M,ch,H,W), (M,T,d) -> (M,ch,H,W)
        M, _, H, W = h.shape
        q = self.to_d(h).flatten(2).transpose(1, 2)            # (M, H*W, d) -- one query per spatial position
        a = self.xa(q, cond).transpose(1, 2).reshape(M, -1, H, W)
        return h + self.to_ch(a)
