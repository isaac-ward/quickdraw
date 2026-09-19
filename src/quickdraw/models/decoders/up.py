"""`decode_arch: "up"`, `decode_kind: "mse"` -- the deterministic tokens->image decoder.

Behavior is byte-identical to the pre-package `TokenGridDecoder`: it holds an `UpBackend` (created in the same
order, so a seeded init matches) and its `velocity` calls the same synthesis with no analysis contributions.
`no_noise=True` -- x/temb/demb are ignored (a pure decoder decodes from `cond`)."""

from __future__ import annotations

from torch import Tensor

from ..flow import TransportHead
from ..vision import img_hw
from .backend import UpBackend


class TokenGridDecoder(TransportHead):
    """UP-ONLY decoder (no analysis path, query-grid readout). Deterministic; never a denoiser.

    Keeps the `velocity(x, temb, cond, demb)` signature even though it ignores x/temb/demb, because the
    frozen-decoder probe (`functional_call`), `TransportHead._chunked_velocity` (decode_chunk_train) and
    smoke/decode_recon.py depend on that contract."""

    def __init__(self, ae_cfg, *, base: int = 32, chunk: int = 0,
                 inject: bool = False, xattn_max_res: int = 0, out_act: str = "none"):
        # x0 + no_noise: predicts the clean image directly and never sees noise (matches the old head).
        super().__init__(param="x0", shortcut=False, event_dims=3, no_noise=True, chunk=chunk)
        self.cfg = ae_cfg
        self.back = UpBackend(ae_cfg, base=base, inject=inject, xattn_max_res=xattn_max_res, out_act=out_act)

    def velocity(self, x=None, temb=None, cond=None, demb=None) -> Tensor:
        """cond (M,T,d) -> (M,H,W,C). x/temb/demb ignored."""
        assert cond is not None, "TokenGridDecoder decodes from `cond`; x carries no information."
        r = self.back.readout_map(cond)
        g = self.back.pool(cond)
        return self.back.synthesize(cond, g, r)

    def sample(self, cond: Tensor, *, steps: int, deterministic: bool, eps: Tensor | None = None,
               record_path: bool = False):
        c = self.cfg
        return self._sample(cond, event_shape=(*img_hw(c.img_size), c.channels), lead=cond.shape[:-2],
                            steps=steps, deterministic=deterministic, eps=eps, record_path=record_path)
