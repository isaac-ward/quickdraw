"""∫-noise: temporally-correlated noise for stochastic decode (Chang et al., "How I Warped Your Noise",
ICLR 2024; infinite-resolution follow-up ICLR 2025 -- https://warpyournoise.github.io/,
official code github.com/yitongdeng-projects/infinite_resolution_integral_noise_warping_code, taichi).

WHY (record §8.29): stochastic image decode gives sharp frames but flickers (independent noise per frame);
`decode_shared_noise` reuses one field but it is glued to the CAMERA, so moving content swims through a static
noise texture ("texture-sticking"). ∫-noise advects the noise along the scene's optical flow so it sticks to
the CONTENT.

THIS IS A TORCH TRANSLATION OF THE PAPER'S EQUATIONS, not a port of their taichi code (which I could not read).
It implements the two pieces the paper's Algorithm 1/2 specify:

  * REPRESENTATION (their Eq 3 / Alg 1): a pixel is the integral of a white field over its area. We realise the
    field on an `up`x-finer grid. For a FRESH field, drawing iid N(0,1) on the fine grid IS the correct marginal
    (Eq 3's conditional-upsample only matters when refining an EXISTING coarse field, which we don't do).

  * TRANSPORT (their Eq 5 / Alg 2): G(p) = (1/sqrt|Ω_p|) * Σ_{sub ∈ Ω_p} W(sub), where Ω_p is the set of fine
    sub-pixels covered by output pixel p's back-warped area. We realise this as a forward SCATTER: each fine
    cell is advected by the (upsampled) flow and added into the output pixel it lands in; each output pixel then
    divides its accumulated sum by sqrt(count of cells it received). This is the discrete area-integral, and the
    variable count is exactly the discrete Jacobian/area factor (Eq 27's |∇T|^-1/2): a compressed region
    collects more cells and sqrt(count) divides it back to unit variance. Disjoint cell sets -> outputs stay
    spatially WHITE (unlike bilinear grid_sample, which averages shared neighbours and DISSIPATES variance --
    the artifact the paper was written to remove, and the ~10% loss the previous bilinear version here showed).

APPROXIMATIONS vs. the exact paper (honest): (a) forward-splat can leave a pixel with zero cells (a "hole")
where the flow diverges; the paper's backward polygon-rasterization cannot. We fill holes with a fresh draw.
(b) cells are assigned to one pixel by floor(), not by exact polygon-overlap area weighting. Both shrink as
`up` grows. The exact version is the official taichi code.

SCOPE: DECODE-SIDE / IMAGE-only render tool (Ceiling A). It does NOT touch the latent walk (Ceiling B). Needs
an optical-flow field from the caller (classical / pretrained / known motion). No learned parameters here.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def fine_noise(b: int, c: int, h: int, w: int, up: int, *, device=None, dtype=None,
               generator: torch.Generator | None = None) -> Tensor:
    """White noise on an `up`x-finer grid: (B, C, H*up, W*up). Its area-integral over each pixel (`pixelize`)
    is standard-normal pixel noise. For a fresh field iid is the correct marginal (see module docstring)."""
    return torch.randn(b, c, h * up, w * up, device=device, dtype=dtype, generator=generator)


def pixelize(fine: Tensor, up: int) -> Tensor:
    """Area-integrate a fine field (B,C,Hf,Wf) to pixels (B,C,H,W): sum the up*up cells of each pixel and divide
    by sqrt(up*up)=up to keep unit variance. This is Eq 5 with the identity flow (Ω_p = the pixel's own block)."""
    return F.avg_pool2d(fine, kernel_size=up, stride=up) * up          # avg*up == sum/up == sum/sqrt(up^2)


def warp_integral(fine: Tensor, flow_px: Tensor, up: int) -> Tensor:
    """Transport a fine field by `flow_px` (B,H,W,2) forward optical flow in PIXEL units, per the paper's Eq 5:
    forward-scatter each fine cell into the output pixel it advects to, then divide each output pixel by
    sqrt(count) of cells it received. Returns pixel noise (B,C,H,W), ~N(0,1) and ~white.

    This replaces bilinear grid_sample (the dissipative baseline the paper improves on). flow_px[...,0]=dx (px)."""
    b, c, hf, wf = fine.shape
    h, w = hf // up, wf // up
    dev, dt = fine.device, fine.dtype
    # per-fine-cell forward flow, in PIXEL units, on the fine grid
    flow = F.interpolate(flow_px.permute(0, 3, 1, 2), size=(hf, wf), mode="bilinear", align_corners=False)  # (B,2,Hf,Wf)
    fj = torch.arange(wf, device=dev, dtype=dt)[None, :].expand(hf, wf)          # fine col
    fi = torch.arange(hf, device=dev, dtype=dt)[:, None].expand(hf, wf)          # fine row
    # fine-cell centre in OUTPUT-pixel coords, advected forward by the flow -> destination pixel via floor()
    dst_x = (fj[None] + 0.5) / up + flow[:, 0]                                   # (B,Hf,Wf)
    dst_y = (fi[None] + 0.5) / up + flow[:, 1]
    px = dst_x.floor().long().clamp_(0, w - 1)
    py = dst_y.floor().long().clamp_(0, h - 1)
    idx = (py * w + px).reshape(b, 1, hf * wf)                                   # flat output-pixel index
    src = fine.reshape(b, c, hf * wf)
    acc = torch.zeros(b, c, h * w, device=dev, dtype=dt).scatter_add_(2, idx.expand(b, c, hf * wf), src)
    cnt = torch.zeros(b, 1, h * w, device=dev, dtype=dt).scatter_add_(2, idx, torch.ones_like(src[:, :1]))
    out = acc / cnt.clamp_min(1.0).sqrt()                                        # Eq 5: sum / sqrt|Ω_p|
    holes = cnt == 0                                                             # forward-splat gaps: fresh draw
    if holes.any():
        out = torch.where(holes.expand_as(out), torch.randn_like(out), out)
    return out.reshape(b, c, h, w)


def advect_and_pixelize(fine: Tensor, flow_px: Tensor, up: int) -> tuple[Tensor, Tensor]:
    """Convenience: (warped_fine_for_next_step, this_frame_pixel_noise). We keep the SAME fine field and return
    it unchanged for carry-forward; the transport is applied at pixelize time. flow_px=None -> constant case
    (recovers `decode_shared_noise` exactly: pixelize with the identity flow)."""
    if flow_px is None:
        return fine, pixelize(fine, up)
    return fine, warp_integral(fine, flow_px, up)
