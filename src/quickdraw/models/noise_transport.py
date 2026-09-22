"""∫-noise: temporally-correlated noise for stochastic decode (Chang et al., "How I Warped Your Noise",
ICLR 2024 -- https://warpyournoise.github.io/).

WHY (record §8.29): stochastic image decode gives sharp, valid frames but flickers, because each frame draws
independent noise. `decode_shared_noise` fixes that by reusing ONE noise field for every frame -- but a fixed
field is glued to the CAMERA, so when a cube slides across the image it swims through a stationary noise
texture ("texture-sticking"). ∫-noise instead ADVECTS the noise along the scene's optical flow, so the noise
sticks to the CONTENT: a moving cube carries its own consistent noise.

THE TRICK. You cannot bilinearly warp a noise image and reuse it -- resampling averages neighbours, which
collapses variance and injects spatial correlation, so the decoder then sees out-of-distribution "noise". The
paper's fix is to treat each pixel's noise as the INTEGRAL of an infinite-resolution white field over the
pixel's area; warping the fine field and re-integrating preserves the white-noise statistics. This module
implements the practical finite approximation the paper's own code uses: keep the field on an `up`x-finer
grid, warp THAT, and area-integrate (avg-pool, rescale by `up`) back to pixel resolution. It is an
APPROXIMATION of the exact continuous integral -- bilinear sampling on the fine grid still smooths a little,
which is why the smoke checks the output stays ~unit-variance and ~white; raise `up` to tighten it.

SCOPE. This is a DECODE-SIDE / IMAGE-only tool (a Ceiling-A render lever). It does NOT touch the latent
random-walk (Ceiling B): it changes how a latent is DRAWN to pixels, not the latent. It is meaningless for
the proprio head (a 17-vector has no spatial grid to advect) and for the dynamics flow (a sequential latent
transition, not a per-frame field). Training-free: no learned parameters here; the only external input is an
optical-flow field, which the caller supplies (classical, a pretrained net, or derived from known motion).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def fine_noise(b: int, c: int, h: int, w: int, up: int, *, device=None, dtype=None,
               generator: torch.Generator | None = None) -> Tensor:
    """White noise on an `up`x-finer grid: (B, C, H*up, W*up). Its area-integral over each pixel (see
    `pixelize`) is standard-normal pixel noise; this fine field is what gets advected between frames."""
    return torch.randn(b, c, h * up, w * up, device=device, dtype=dtype, generator=generator)


def pixelize(fine: Tensor, up: int) -> Tensor:
    """Area-integrate a fine field (B,C,Hf,Wf) to pixel resolution (B,C,H,W). Averaging up*up iid cells scales
    variance by 1/up^2, so multiply by `up` to restore unit variance -- this is the discrete ∫ over the pixel."""
    return F.avg_pool2d(fine, kernel_size=up, stride=up) * up


def warp_fine(fine: Tensor, flow_px: Tensor, up: int) -> Tensor:
    """Advect the fine field along `flow_px` (B,H,W,2), a PIXEL-resolution forward optical flow in pixel units
    (flow[...,0]=dx, dy). Backward-warp: output location p samples the field at p - flow, so the texture moves
    WITH the content. Bilinear grid_sample on the fine grid; out-of-frame samples are reflected (keeps energy).

    Returns the warped fine field (same shape as `fine`), ready to `pixelize` for this frame and to warp again
    for the next -- carry the fine field across the rollout, advecting by each step's flow."""
    b, c, hf, wf = fine.shape
    h, w = hf // up, wf // up
    # upsample the pixel-res flow to the fine grid; scale displacements to fine-grid units (x up).
    flow = flow_px.permute(0, 3, 1, 2)                                   # (B,2,H,W)
    flow = F.interpolate(flow, size=(hf, wf), mode="bilinear", align_corners=False) * up   # (B,2,Hf,Wf)
    ys, xs = torch.meshgrid(torch.arange(hf, device=fine.device, dtype=fine.dtype),
                            torch.arange(wf, device=fine.device, dtype=fine.dtype), indexing="ij")
    src_x = xs[None] - flow[:, 0]                                        # backward warp: sample from p - flow
    src_y = ys[None] - flow[:, 1]
    gx = 2.0 * src_x / max(wf - 1, 1) - 1.0                              # -> normalized [-1,1] grid
    gy = 2.0 * src_y / max(hf - 1, 1) - 1.0                              #    (align_corners=True convention:
    grid = torch.stack([gx, gy], dim=-1)                                 #     -1/+1 hit pixel CENTERS, so an
    #                                                                     integer displacement samples exactly)
    return F.grid_sample(fine, grid, mode="bilinear", padding_mode="reflection", align_corners=True)


def advect_and_pixelize(fine: Tensor, flow_px: Tensor, up: int) -> tuple[Tensor, Tensor]:
    """Convenience: warp the fine field by `flow_px`, return (warped_fine, pixel_noise). Feed warped_fine back
    in for the next frame; pixel_noise (B,C,H,W) is this frame's decode eps. flow_px=None -> no motion (the
    `decode_shared_noise` constant case, recovered exactly)."""
    warped = fine if flow_px is None else warp_fine(fine, flow_px, up)
    return warped, pixelize(warped, up)
