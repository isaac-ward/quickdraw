"""`UpBackend` -- the `up` decoder's readout + synthesis trunk, shared by the deterministic head
(`up-mse`, TokenGridDecoder) and the flow denoiser (`up-flow`, UpFlowDecoder).

CONSTRUCTION ORDER IS LOAD-BEARING. The submodules are created in the SAME order the pre-package
`TokenGridDecoder.__init__` used (readout, to_ch, gpool, mid, per-level [inject, xattn, up], out_norm,
out_conv), so a seeded fresh-init is bit-identical to the old decoder -- asserted by smoke/up_flow_parity.py
(T1). Do not reorder.

The synthesis body is byte-for-byte the old `TokenGridDecoder.velocity` when `seed_extra`/`skips` are None
(the up-mse path); both are additive and default None, so the flow head's analysis contributions are a strict
superset that starts as a no-op."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..vision import _FiLMResBlock, img_hw
from .components import LevelCrossAttn, TokenGridReadout, TokenPool


class UpBackend(nn.Module):
    def __init__(self, ae_cfg, *, base: int = 32, inject: bool = False, xattn_max_res: int = 0,
                 out_act: str = "none"):
        super().__init__()
        c = ae_cfg
        H, W = img_hw(c.img_size)
        d, T = c.d, c.num_tokens
        bott = max(1, int(getattr(c, "bottleneck", 8)))
        n_levels = max(1, int(math.log2(max(bott, min(H, W)) // bott)))
        chs = [base * min(4, 2 ** i) for i in range(n_levels)]      # e.g. [64,128,256,256]
        self.chs = chs                                              # exposed for the analysis path geometry
        self.bott_hw = (H // (2 ** len(chs)), W // (2 ** len(chs)))
        self.readout = TokenGridReadout(d, c.heads, self.bott_hw, mlp_ratio=c.mlp_ratio)
        self.to_ch = nn.Conv2d(d, chs[-1], 1)
        self.gpool = TokenPool(d, c.heads)
        self.mid = _FiLMResBlock(chs[-1], chs[-1], d)
        self.ups, prev = nn.ModuleList(), chs[-1]
        self.inject = nn.ModuleList() if inject else None          # Feature 2 (per-level readout map)
        self.xattn = nn.ModuleDict()                               # Feature 3 (per-level re-attend)
        for i, ch in enumerate(reversed(chs)):
            if inject:
                cv = nn.Conv2d(d, prev, 1)
                nn.init.zeros_(cv.weight); nn.init.zeros_(cv.bias)
                self.inject.append(cv)
            res = self.bott_hw[0] * (2 ** (i + 1))
            if 0 < xattn_max_res and res <= xattn_max_res:
                self.xattn[str(i)] = LevelCrossAttn(prev, d, c.heads)
            self.ups.append(_FiLMResBlock(prev, ch, d))            # NO concat: skips (flow) are additive
            prev = ch
        self.out_norm = nn.GroupNorm(min(8, chs[0]), chs[0])
        self.out_conv = nn.Conv2d(chs[0], c.channels, 3, padding=1)
        if out_act not in ("none", "sigmoid"):
            raise ValueError(f"decode_out_act={out_act!r}; expected 'none' or 'sigmoid'")
        self.out_act = out_act

    def pool(self, cond: Tensor) -> Tensor:                        # tokens -> global g (attention pool)
        return self.gpool(cond)

    def readout_map(self, cond: Tensor) -> Tensor:                 # tokens -> (M,d,bh,bw) full-rank seed
        return self.readout(cond)

    def synthesize(self, cond: Tensor, g: Tensor, r: Tensor,
                   seed_extra: Tensor | None = None, skips=None) -> Tensor:
        """(cond,g,r) -> (M,H,W,C). up-mse passes seed_extra=None, skips=None -> byte-identical to the old
        TokenGridDecoder.velocity body. Flow passes the analysis bottleneck + per-level additive skips."""
        h = self.to_ch(r)
        if seed_extra is not None:
            h = h + seed_extra                                     # flow: analysis bottleneck (zero-init at start)
        h = self.mid(h, g)
        for i, up in enumerate(self.ups):
            h = F.interpolate(h, scale_factor=2, mode="nearest")   # resize-conv (Odena et al.)
            if skips is not None and skips[i] is not None:         # flow: additive zero-init skip
                s = skips[i]
                if s.shape[-2:] != h.shape[-2:]:
                    s = F.interpolate(s, size=h.shape[-2:], mode="nearest")
                h = h + s
            if self.inject is not None:                            # Feature 2
                h = h + self.inject[i](F.interpolate(r, size=h.shape[-2:], mode="bilinear", align_corners=False))
            if str(i) in self.xattn:                               # Feature 3
                h = self.xattn[str(i)](h, cond)
            h = up(h, g)
        out = self.out_conv(F.silu(self.out_norm(h)))
        if self.out_act == "sigmoid":
            out = torch.sigmoid(out)
        return out.permute(0, 2, 3, 1)
