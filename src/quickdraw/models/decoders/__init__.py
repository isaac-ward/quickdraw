"""Up-backed image decoders: one readout+synthesis backend, two heads.

WHY `up` EXISTS (2026-08-26). `vision.ConditionalUNet` served both a DECODER (`decode_kind=mse`, x=zeros)
and a DENOISER (`decode_kind=flow`, x=noised) through one `velocity`, with two measured pathologies:
  1. THE DOWN PATH CONVOLVES ZEROS in mse mode -- its only spatial signal was a zero-padding halo artifact.
  2. THE LATENT REACHED PIXELS THROUGH A RANK-<=640 CHOKE (`cond_to_spatial` Linear + `g=cond.mean(1)`),
     84% of the latent in the null space, and the token MEAN annihilated per-block identity.
`up` (TokenGridDecoder) fixed both: no down path, and a learned query-grid cross-attention readout that
preserves token identity at full bandwidth. See models/decoders/components.py.

THE PACKAGE (2026-09-19). `up` is mse-only (no analysis path). Flow needs to read its noised iterate, which
needs an analysis path, which makes it a U-Net -- so "up-flow" IS "unet-flow with `up`'s readout." Rather
than route flow back onto the rank-640 `ConditionalUNet`, both heads now share `UpBackend`:

  - `TokenGridDecoder` (up.py)   -- up-mse: UpBackend alone. Byte-identical to the pre-package decoder.
  - `UpFlowDecoder`    (up_flow.py) -- up-flow: UpBackend + `_UpAnalysis`, all analysis contributions
                                        zero-init so at start it == up-mse (superset; can warm-start from it).

`ConditionalUNet`/`ImageUNetFlowHead` remain as legacy (mm_flow/bsp32mse defaults, the Dreamer configs, and
the `_oneoff_decoder_ab.py` harness). `decode_arch=up` never routes to them. See design/up_flow_decoder.md."""

from .analysis import _UpAnalysis
from .backend import UpBackend
from .components import LevelCrossAttn, TokenGridReadout, TokenPool
from .up import TokenGridDecoder
from .up_flow import UpFlowDecoder

__all__ = [
    "TokenGridDecoder", "UpFlowDecoder", "UpBackend", "_UpAnalysis",
    "TokenGridReadout", "TokenPool", "LevelCrossAttn",
]
