"""`VisualLoss` — the ONE pixel-space reconstruction mix, shared by both image loss sites.

WHY THIS FILE EXISTS (2026-08-29). Two different call sites train the same image decoder, and until now both
were plain `F.mse_loss` with no way to change one without silently diverging from the other:

    site (a)  AR decode loss     flow.TransportHead.loss, no_noise branch     weight 1.0
    site (b)  roundtrip anchor   multimodal.roundtrip_losses                  weight 10.0

L2's minimiser under uncertainty is the conditional MEAN, i.e. blur, and design/collapse.md already recorded
the consequence: "MSE loves blur... LPIPS was 0.18 the whole time -- the blur was there from the start; MSE
never saw it." We optimise a loss structurally blind to sharpness and then rank runs on LPIPS.

ONE INSTANCE, BOTH SITES. `ImageModality` owns a single `VisualLoss` and hands the same object to (a) via
`TransportHead.loss(recon_loss=...)` and to (b) directly. The site weights (1.0 / 10.0) are applied OUTSIDE.
That is deliberate and it dissolves a real hazard: if site (b) kept extra PURE pixel loss outside the module,
its 10x weight would dominate a small perceptual term and re-blur the decoder. Because the site weight scales
the whole mix, the pixel:perceptual RATIO is identical at both sites and only the magnitude differs.

LITERATURE (all code-verified 2026-08-29; see record section 22). Two families:
  * 1:1 with LPIPS-VGG16 -- VQGAN (L1 1.0 + LPIPS 1.0), LDM/SD-VAE (same), IRIS (same, NO GAN),
    ViTok stage 1 (L2 1.0 + LPIPS 1.0, swept 0/0.5/1.0), SoftVQ-VAE (L2 1.0 + LPIPS 1.0).
  * 0.1:1 -- MAGVIT-v2 and TiTok. DO NOT COPY THIS ONE. Both have a GAN carrying sharpness, and neither uses
    LPIPS at all (MAGVIT-v2 scores ResNet50 logits, TiTok a ConvNeXt-S). We have no GAN.
IRIS is the closest published relative: bespoke encoder, 64px frames, no GAN, L1 + LPIPS-VGG16 at 1:1. Every
one of them enables the perceptual term from STEP 0; warmup is reserved for the GAN.

VGG FOR TRAINING, SQUEEZE FOR EVAL -- NOT A STYLE CHOICE. `evaluation.openloop.image_curves` computes the
REPORTED metric with SqueezeNet. Training against that same network optimises the metric's own features and
produces a number not comparable to any of the 25 historical runs on this dataset. The LPIPS README also says
plainly that alex is the better forward METRIC while vgg "is closer to the traditional perceptual loss", and
VQGAN/LDM/IRIS/SoftVQ all hardcode VGG16. So `lpips_net` defaults to "vgg" here and eval keeps its squeeze.
(E-LPIPS/R-LPIPS show optimising against an LPIPS net finds metric-specific minima; different backbones
attenuate that, they do not eliminate it, so keep reading PSNR/SSIM alongside.)

THE ONE NUMBER THE LITERATURE CANNOT GIVE US is the site-(b) weight. On [0,1] images MSE ~ 0.005 while
L1 ~ 0.05 and LPIPS ~ 0.2, so swapping 10*MSE (~0.05) for 10*(L1+LPIPS) (~2.5) is a ~50x increase in the
anchor's contribution against an unchanged dynamics loss -- testing loss SHAPE and a 50x reweighting at once.
Use `terms()` to MEASURE the three magnitudes at init and solve w_anchor * (mix) = 10 * MSE, rather than
trusting those ballpark figures. No published system has our two-site structure to copy from.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class VisualLoss(nn.Module):
    """w_l2*L2 + w_l1*L1 + w_lpips*LPIPS(net). Inputs (M,H,W,C) in [0,1].

    Defaults are `w_l2=1.0` and nothing else, which is EXACTLY `F.mse_loss` -- so constructing one and wiring
    it into both sites is bit-identical to the previous behaviour until a weight is set. That is the point:
    the plumbing lands as a no-op and the experiment is a config change.
    """

    def __init__(self, *, w_l2: float = 1.0, w_l1: float = 0.0, w_lpips: float = 0.0,
                 lpips_net: str = "vgg", frames: int = 128):
        super().__init__()
        self.w_l2, self.w_l1, self.w_lpips = float(w_l2), float(w_l1), float(w_lpips)
        self.lpips_net, self.frames = str(lpips_net), int(frames)
        self._net = None                      # built lazily on first use: it needs a device, and constructing
        #                                       it in __init__ would download weights during a --help.

    # ---- internals ----
    def _lpips(self, device):
        if self._net is None:
            from ..evaluation.openloop import _lpips_net
            self._net = _lpips_net(device, net_type=self.lpips_net)
        return self._net

    def _subsample(self, pred: Tensor, target: Tensor):
        """Random subset of frames. LPIPS on all B*F frames (2048 at batch 32 / F 64) would dominate the step;
        a random subset is an unbiased estimate of the same expectation. Applies at BOTH sites -- the anchor
        decodes the full batch too, so leaving it off there was the more expensive omission."""
        n = self.frames
        if not (0 < n < pred.shape[0]):
            return pred, target
        idx = torch.randperm(pred.shape[0], device=pred.device)[:n]
        return pred[idx], target[idx]

    def _lpips_term(self, pred: Tensor, target: Tensor) -> Tensor:
        net = self._lpips(pred.device)
        if net is None:                        # weights unavailable -> fail soft, exactly as eval does
            return pred.new_zeros(())
        p, t = self._subsample(pred, target)
        # torchmetrics is built with normalize=True, i.e. it expects [0,1] and rescales to [-1,1] itself -- so
        # do NOT pre-scale here. It also VALIDATES the range and raises on anything outside it, and a plain
        # clamp() would zero the gradient exactly where the decoder overshoots, which is where we most want it
        # pulled back. Hence a STRAIGHT-THROUGH clamp: forward value clipped, backward pass the identity.
        pc = p + (p.clamp(0.0, 1.0) - p).detach()
        out = net(pc.permute(0, 3, 1, 2).float(), t.permute(0, 3, 1, 2).clamp(0, 1).float())
        # RESET, every call. LearnedPerceptualImagePatchSimilarity is a stateful torchmetrics Metric: every
        # __call__ appends the batch score to `all_scores`, and the net is cached for the whole process, so
        # without this the list grows WITHOUT BOUND -- measured 22 -> 42 -> 62 entries over 60 calls. This
        # site runs twice per training step (decode loss + roundtrip anchor), i.e. ~2,700 times an epoch.
        # `evaluation.openloop.image_curves` already does this and says why; VisualLoss did not copy it.
        # reset() clears the accumulated STATE only -- `out` keeps its autograd graph, so the gradient is
        # unaffected (asserted in smoke/visual_loss.py).
        net.reset()
        return out

    # ---- public ----
    @staticmethod
    def _flatten(pred: Tensor, target: Tensor):
        """Both sites hand this DIFFERENT RANKS and only one of them is safe for LPIPS.

        The AR decode loss flattens to (M,H,W,C) before calling the head, but the roundtrip anchor scores
        `to_obs(...)` output, which keeps its (B,F) lead -- so it arrives as (B,F,H,W,C). F.mse_loss and
        F.l1_loss are rank-agnostic (they reduce over everything), which is exactly why a pure-MSE anchor
        never cared and why this was invisible until LPIPS was wired in. It matters twice: `permute(0,3,1,2)`
        needs rank 4, and `_subsample` must draw over FRAMES, not whole trajectories.

        Images are event_dims=3, so the last three axes are always (H,W,C)."""
        if pred.dim() > 4:
            return pred.reshape(-1, *pred.shape[-3:]), target.reshape(-1, *target.shape[-3:])
        return pred, target

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pred, target = self._flatten(pred, target)     # (B,F,H,W,C) from the anchor -> (B*F,H,W,C)
        loss = pred.new_zeros(())
        if self.w_l2:
            loss = loss + self.w_l2 * F.mse_loss(pred, target)
        if self.w_l1:
            loss = loss + self.w_l1 * F.l1_loss(pred, target)
        if self.w_lpips:
            loss = loss + self.w_lpips * self._lpips_term(pred, target)
        return loss

    @torch.no_grad()
    def terms(self, pred: Tensor, target: Tensor) -> dict[str, float]:
        """RAW (unweighted) magnitude of each term -- for choosing the site weights by measurement.

        The site-(b) weight of 10 was calibrated for MSE's ~0.005 scale. Anything perceptual is ~40x larger, so
        reusing 10 silently rescales the anchor by more than an order of magnitude. Solve
        `w = 10 * mse / (w_l1*l1 + w_lpips*lpips + w_l2*mse)` from THESE numbers, on a real batch."""
        pred, target = self._flatten(pred, target)
        out = {"l2": float(F.mse_loss(pred, target)), "l1": float(F.l1_loss(pred, target))}
        out["lpips"] = float(self._lpips_term(pred, target)) if self.w_lpips else float("nan")
        return out

    def extra_repr(self) -> str:
        return (f"w_l2={self.w_l2}, w_l1={self.w_l1}, w_lpips={self.w_lpips}, "
                f"lpips_net={self.lpips_net!r}, frames={self.frames}")
