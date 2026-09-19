"""`decode_arch: "up"`, `decode_kind: "flow"` -- the up backend as a generative denoiser.

Flow needs to read its noised iterate, which needs an analysis path, which makes it a U-Net -- so `up-flow`
IS "unet-flow with `up`'s readout." It reuses the SAME `UpBackend` as `up-mse` plus `_UpAnalysis`. Every
analysis contribution is zero-init (skips, bottleneck, time projections), so at init it computes EXACTLY
`up-mse` (superset), and it can warm-start from an `up-mse` checkpoint's backend. See design/up_flow_decoder.md."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from ..flow import TransportHead
from ..vision import img_hw
from .analysis import _UpAnalysis
from .backend import UpBackend


class UpFlowDecoder(TransportHead):
    def __init__(self, ae_cfg, *, base: int = 32, chunk: int = 0, param: str = "x0", shortcut: bool = False,
                 inject: bool = False, xattn_max_res: int = 0, down_inject: bool = False,
                 down_xattn_max_res: int = 0, out_act: str = "none", n_freq: int = 16, time_dim: int = 32):
        super().__init__(param=param, shortcut=shortcut, event_dims=3, n_freq=n_freq, time_dim=time_dim,
                         no_noise=False, chunk=chunk)
        self.cfg = ae_cfg
        self.back = UpBackend(ae_cfg, base=base, inject=inject, xattn_max_res=xattn_max_res, out_act=out_act)
        self.ana = _UpAnalysis(ae_cfg, chs=self.back.chs, down_inject=down_inject,
                               down_xattn_max_res=down_xattn_max_res)
        # time / step-size projections onto g. ZERO-INIT so at start the flow head == up-mse for ANY temb
        # (the time signal is learned in from zero), which is what makes the superset + warm-start exact.
        self.t_proj = nn.Linear(time_dim, ae_cfg.d)
        self.d_proj = nn.Linear(time_dim, ae_cfg.d)
        for m in (self.t_proj, self.d_proj):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def velocity(self, x: Tensor, temb: Tensor | None, cond: Tensor, demb: Tensor | None = None) -> Tensor:
        M = cond.shape[0]
        g = self.back.pool(cond)
        if temb is not None:
            g = g + self.t_proj(temb.reshape(M, -1))
        if demb is not None:
            g = g + self.d_proj(demb.reshape(M, -1))
        r = self.back.readout_map(cond)
        seed_extra, skips = self.ana(x, g, cond, r)
        return self.back.synthesize(cond, g, r, seed_extra=seed_extra, skips=skips)

    def sample(self, cond: Tensor, *, steps: int, deterministic: bool, eps: Tensor | None = None,
               record_path: bool = False):
        c = self.cfg
        return self._sample(cond, event_shape=(*img_hw(c.img_size), c.channels), lead=cond.shape[:-2],
                            steps=steps, deterministic=deterministic, eps=eps, record_path=record_path)
