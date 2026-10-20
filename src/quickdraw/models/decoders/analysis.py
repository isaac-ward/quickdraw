"""`_UpAnalysis` -- the flow denoiser's down/analysis path, built on `up`'s principles.

Reads the noised iterate and produces a bottleneck feature + per-level skips for `UpBackend.synthesize`.
Design decisions carried from `up`: FiLM on the attention-pooled `g` (not a mean), optional per-level token
re-injection (`inject`/`xattn`, the same modules as the up path, gated + zero-init), and an EXPLICIT learned
positional grid so position never comes from a padding-halo artifact. The bottleneck feature (`to_bott`) and
every skip (`skip_proj`) are ZERO-INIT, so at start the analysis contributes nothing and the flow head equals
`up-mse` exactly (superset). Geometry (`chs`, resolutions) is taken from the backend so the two stacks match."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..vision import _FiLMResBlock, img_hw
from .components import LevelCrossAttn


class _UpAnalysis(nn.Module):
    def __init__(self, ae_cfg, *, chs: list[int], down_inject: bool = False, down_xattn_max_res: int = 0):
        super().__init__()
        c = ae_cfg
        H, W = img_hw(c.img_size)
        d, C = c.d, c.channels
        self.in_conv = nn.Conv2d(C, chs[0], 3, padding=1)
        self.pos = nn.Parameter(torch.zeros(1, chs[0], H, W))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.downs, prev = nn.ModuleList(), chs[0]
        self.inject_d = nn.ModuleList() if down_inject else None
        self.xattn_d = nn.ModuleDict()
        for i, ch in enumerate(chs):
            self.downs.append(_FiLMResBlock(prev, ch, d))
            if down_inject:
                cv = nn.Conv2d(d, ch, 1)
                nn.init.zeros_(cv.weight); nn.init.zeros_(cv.bias)
                self.inject_d.append(cv)
            res = H // (2 ** i)
            if 0 < down_xattn_max_res and res <= down_xattn_max_res:
                self.xattn_d[str(i)] = LevelCrossAttn(ch, d, c.heads)
            prev = ch
        # up path insertion channels: h BEFORE up[i] has up_prev[i] channels
        up_prev = [chs[-1]] + list(reversed(chs))[:-1]
        # skip_proj[i] maps down level i (chs[i]) -> the up level (len-1-i) it feeds; zero-init -> no-op at start
        self.skip_proj = nn.ModuleList()
        for i in range(len(chs)):
            up_i = len(chs) - 1 - i
            sp = nn.Conv2d(chs[i], up_prev[up_i], 1)
            nn.init.zeros_(sp.weight); nn.init.zeros_(sp.bias)
            self.skip_proj.append(sp)
        self.to_bott = nn.Conv2d(chs[-1], chs[-1], 1)
        nn.init.zeros_(self.to_bott.weight); nn.init.zeros_(self.to_bott.bias)
        self._n = len(chs)

    def forward(self, x: Tensor, g: Tensor, cond: Tensor, r: Tensor):
        """x:(M,H,W,C) noised iterate -> (seed_extra at bottleneck, skips list aligned to the up levels)."""
        h = self.in_conv(x.permute(0, 3, 1, 2)) + self.pos
        raw = []
        for i, down in enumerate(self.downs):
            h = down(h, g)                                          # FiLM on pooled g (every block)
            if self.inject_d is not None:                          # per-level readout map (zero-init)
                h = h + self.inject_d[i](F.interpolate(r, size=h.shape[-2:], mode="bilinear", align_corners=False))
            if str(i) in self.xattn_d:                             # per-level re-attend (zero-init)
                h = self.xattn_d[str(i)](h, cond)
            raw.append(h)
            h = F.avg_pool2d(h, 2)
        seed_extra = self.to_bott(h)                                # zero-init -> 0 at start
        skips = [None] * self._n
        for i in range(self._n):                                    # down level i feeds up level (n-1-i)
            skips[self._n - 1 - i] = self.skip_proj[i](raw[i])      # zero-init -> 0 at start
        return seed_extra, skips
