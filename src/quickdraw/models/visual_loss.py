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


# ---- TERMS ------------------------------------------------------------------------------------
# ONE declaration per term, used by BOTH reductions. Before this, `forward()` and `temporal()` each
# carried their own `if self.w_*:` chain, so a term added to one and not the other was silently
# absent from half the objective and nothing failed.
#
# A term never sees HOW frames were chosen. `_subsample` (randperm over flattened rows) and `_pairs`
# (contiguous pairs) are genuinely different and both load-bearing -- `_subsample` would hand back
# (b=3,t=17), (b=0,t=52)..., useless for differencing -- so that choice stays in VisualLoss.
#
# `difference` returning None means "no temporal form", which is how w_l2's absence from the
# derivative term became greppable instead of an undocumented divergence a reader had to notice.


class _Term:
    """One loss term. `pointwise` compares frames; `difference` compares temporal differences."""

    name: str = ""

    def weight(self, vl: "VisualLoss") -> float:
        raise NotImplementedError

    def pointwise(self, vl: "VisualLoss", p: Tensor, t: Tensor) -> Tensor | None:
        raise NotImplementedError

    def difference(self, vl: "VisualLoss", p0: Tensor, p1: Tensor, g0: Tensor, g1: Tensor) -> Tensor | None:
        """Terms difference THEMSELVES, so a feature term can difference EMBEDDINGS rather than embed
        a difference -- see LpipsTerm.difference. None = this term has no temporal form.

        MAY RETURN A LIST, and LpipsTerm does. The caller then folds each element into the running
        total separately, as `total = total + w * part`. That is not cosmetic: the pre-refactor code
        multiplied w_lpips into the total ONCE PER VGG LAYER, and summing the five layers first and
        multiplying once is mathematically identical but differs in fp32 by ~1e-7 -- which the parity
        golden catches, correctly, because a 1e-7 drift in `visual_lpips` silently makes 25+ historical
        runs incomparable."""
        raise NotImplementedError


class L2Term(_Term):
    name = "l2"
    full_sequence = True

    def weight(self, vl): return vl.w_l2

    def pointwise(self, vl, p, t): return F.mse_loss(p, t)

    def difference(self, vl, p0, p1, g0, g1):
        """NO temporal form, deliberately. A temporal difference image is SPARSE -- almost all zero
        with a blob where something moved -- and L2 squares, so the single biggest change swamps the
        rest. L1 does not let a few large values dominate and its constant gradient keeps small
        motions visible. `temporal()` has always skipped L2; now it says so."""
        return None


class L1Term(_Term):
    name = "l1"
    full_sequence = True

    def weight(self, vl): return vl.w_l1

    def pointwise(self, vl, p, t): return F.l1_loss(p, t)

    def difference(self, vl, p0, p1, g0, g1): return F.l1_loss(p1 - p0, g1 - g0)


class LpipsTerm(_Term):
    """The two reductions compute DIFFERENT functions of the same VGG features, deliberately.

    `pointwise` calls the torchmetrics metric as a BLACK BOX, learned per-layer weights included --
    that is the number the historical runs are ranked on, and reproducing its internals to "share
    code" would change it. `difference` uses raw unit-normalised features and does NOT apply those
    learned weights, which were fitted to human judgements of IMAGE similarity with nothing
    calibrating them for the similarity of temporal DIFFERENCES.

    They share the NETWORK (one process-level cache) and the `_sanitise` guard -- not the distance.
    """

    name = "lpips"
    full_sequence = False            # sampled contiguous pairs -- the net is the cost

    def weight(self, vl): return vl.w_lpips

    def pointwise(self, vl, p, t): return vl._lpips_term(p, t)

    def difference(self, vl, p0, p1, g0, g1):
        fp0, fp1 = vl._layer_features(vl._sanitise(p0)), vl._layer_features(vl._sanitise(p1))
        fg0, fg1 = vl._layer_features(g0.clamp(0, 1)), vl._layer_features(g1.clamp(0, 1))
        if not fp0:                       # weights unavailable -> fail soft, as _lpips_term does
            return None
        # ONE ELEMENT PER LAYER, not a pre-summed scalar -- see _Term.difference on fp32 ordering.
        return [((a1 - a0) - (b1 - b0)).pow(2).mean()
                for a0, a1, b0, b1 in zip(fp0, fp1, fg0, fg1)]


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
        self._diag: dict = {}                 # per-site RAW output range, drained per epoch -- see range_stats
        # ONE list, both reductions. Adding a term here makes it apply to the decode loss, the
        # roundtrip anchor AND the derivative term, for every modality that owns a VisualLoss.
        self._terms: list[_Term] = [L2Term(), L1Term(), LpipsTerm()]

    # ---- diagnostics ----
    # WHY THIS EXISTS (2026-09-03). The decoder head is an unbounded nn.Conv2d and NOTHING here reads its raw
    # magnitude: every consumer clamps first (LPIPS validates [0,1], and so do the eval metrics and the image
    # writers). `torus_vl128b` therefore trained 9 hours with a strip of decoder output reaching +77 across the
    # top rows of every frame while every clamped metric read healthy -- the only trace was `roundtrip_*_mse`,
    # which is logged at weight 0 and read by nobody. These four numbers are the missing alarm.
    #
    # WHY MAX/MIN AND *TWO* FRACTIONS, and not percentiles or a histogram. Benign overshoot and a runaway are
    # different by ORDERS of magnitude, not by shape: measured, robocasa's decoder tops out at 1.034 while
    # torus reached 20.0 / 77.2 / 52.5 over three epochs. `max`/`min` separate those on sight. The fractions
    # then separate benign-but-widespread (32% of torus pixels sit just over 1.0, median 1.052, because 67% of
    # its TARGETS are exactly 1.0) from spreading (the >5.0 population grew 0.51% -> 1.04%). Percentiles added
    # nothing to that diagnosis when it was done by hand.
    #
    # BOTH BOUNDS, not just the top. The mechanism is symmetric -- a target of exactly 0.0 makes an output of
    # -5 clamp to a PIXEL-PERFECT 0.0, so LPIPS is equally blind below -- and the low side is real (min reached
    # -1.592, with 8.6%/9.6%/1.3% of pixels below 0). It is also unexplained why the low excursion stayed 48x
    # smaller than the high one, which is precisely why it is measured rather than assumed to mirror.
    #
    # AND BOTH SITES, keyed by `site`. One VisualLoss serves the AR decode loss (rolled latents -- the output
    # that `@+128` actually scores) and the roundtrip anchor (encoded real frames). The DIFFERENCE between them
    # is diagnostic: comparable => the loss cannot hold the range; much worse on rolled latents => the dynamics
    # are pushing the decoder off-manifold and the range violation is downstream of that.
    @torch.no_grad()
    def _record(self, site: str, pred: Tensor) -> None:
        p = pred.detach().float()
        d = self._diag.setdefault(site, {"max": -float("inf"), "min": float("inf"),
                                         "hi": 0.0, "lo": 0.0, "n": 0})
        d["max"] = max(d["max"], float(p.max()))
        d["min"] = min(d["min"], float(p.min()))
        d["hi"] += float((p >= 1.0 - 1e-3).float().mean())
        d["lo"] += float((p <= 1e-3).float().mean())
        d["n"] += 1

    def pop_diagnostics(self) -> dict:
        """Drain the accumulated range stats -> {site: {max,min,frac_hi,frac_lo}, "nonfinite": n}, and reset.

        Called once per epoch from `lit.on_train_epoch_end`. Draining (rather than reading) is what makes the
        non-finite warning per-EPOCH: it was `_warned_nonfinite`, a one-shot per PROCESS, so a run could take
        thousands of non-finite steps and print a single line.

        `max`/`min` are extrema over the epoch's steps; the fractions are step means. At the AR decode site the
        tensor is the `recon_frac` frame subset for that step, so the fractions stay unbiased but `max` is a
        max over a subset and reads slightly below the true per-step maximum -- the two sites' `max` are
        therefore not exactly like-for-like. It does not weaken the alarm (77 vs 1.03 survives any subset).
        """
        out = {s: {"max": d["max"], "min": d["min"],
                   "frac_hi": d["hi"] / max(1, d["n"]), "frac_lo": d["lo"] / max(1, d["n"])}
               for s, d in self._diag.items() if d["n"]}
        out["nonfinite"] = int(getattr(self, "_nonfinite_calls", 0))
        self._diag, self._nonfinite_calls = {}, 0
        return out

    # ---- internals ----
    def _lpips(self, device):
        if self._net is None:
            from ..evaluation.openloop import _lpips_net
            self._net = _lpips_net(device, net_type=self.lpips_net)
        return self._net

    def _sanitise(self, p: Tensor) -> Tensor:
        """Decoder output -> a tensor a frozen feature net can be fed. SHARED by every feature-space term.

        torchmetrics is built with normalize=True, i.e. it expects [0,1] and VALIDATES it, raising on
        anything outside. A plain clamp() would zero the gradient exactly where the decoder overshoots,
        which is where we most want it pulled back -- hence a STRAIGHT-THROUGH clamp: forward value
        clipped, backward pass the identity.

        BUT THE NAIVE STRAIGHT-THROUGH IS NUMERICALLY UNSAFE, and it killed a run (torus_vl128, 2026-09-02,
        dead 7.5 h before anyone noticed). `p + (p.clamp(0,1) - p)` is exact only for moderate magnitudes.
        Measured: 1e8 -> 0.0 (catastrophic cancellation) and +-inf or nan -> NaN. A FRESH decoder at step 0
        is unbounded and can emit exactly those, and then torchmetrics raises mid-training-step. So:
        sanitise the non-finites FIRST, then straight-through, then a final clamp as a belt.

        THIS LIVES IN ONE PLACE ON PURPOSE. It is a hard-won fix whose correctness is not obvious from
        reading it, so a second copy would drift. Every term that feeds a frozen net calls this one."""
        finite = torch.isfinite(p)
        if not bool(finite.all()):
            # COUNTED, and the count is drained per EPOCH by pop_diagnostics -- this used to be a one-shot
            # `_warned_nonfinite` per PROCESS, so a run taking thousands of these printed one line.
            n = self._nonfinite_calls = int(getattr(self, "_nonfinite_calls", 0)) + 1
            if n <= 3:                     # first few in detail; the per-epoch total comes from the report
                print(f"[visual_loss] NON-FINITE decoder output into a feature net "
                      f"({int((~finite).sum())}/{p.numel()} elements) -- sanitised so the step survives. "
                      f"This is a symptom, not the disease: check grad/norm_preclip and the decode loss.",
                      flush=True)
            p = torch.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0)
        return (p + (p.clamp(0.0, 1.0) - p).detach()).clamp(0.0, 1.0)

    def _layer_features(self, x: Tensor) -> list[Tensor]:
        """(M,H,W,C) in [0,1] -> the LPIPS backbone's 5 per-layer UNIT-NORMALISED feature maps.

        WHY THIS IS NOT USED BY `_lpips_term`. That method calls the torchmetrics metric as a BLACK BOX,
        which is correct and tested; reproducing its internals here to "share code" would risk changing a
        number that 25+ historical runs are ranked on. These two share the NETWORK (one process-level cache
        keyed by (device, net_type) in evaluation/openloop) and the sanitise guard above -- not the
        distance. They compute different functions of the same features.

        RAW unit-normalised features, NOT LPIPS's learned per-layer weights. Those weights were fitted to
        match HUMAN JUDGEMENTS OF IMAGE SIMILARITY; nothing calibrates them for the similarity of temporal
        DIFFERENCES, and borrowing a calibration across tasks is the kind of thing that looks rigorous and
        is not. `_normalize_tensor` is torchmetrics' own, so the normalisation matches LPIPS exactly."""
        from torchmetrics.functional.image.lpips import _normalize_tensor
        net = self._lpips(x.device)
        if net is None:
            return []
        inner = net.net                                        # _NoTrainLpips: .net = Vgg16, .L = 5
        feats = inner.net.forward(inner.scaling_layer(x.permute(0, 3, 1, 2).float()))
        return [_normalize_tensor(f) for f in feats]

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
        # torchmetrics is built with normalize=True, i.e. it expects [0,1] and VALIDATES it, raising on
        # anything outside. A plain clamp() would zero the gradient exactly where the decoder overshoots,
        # which is where we most want it pulled back -- hence a STRAIGHT-THROUGH clamp: forward value
        # clipped, backward pass the identity.
        #
        # BUT THE NAIVE STRAIGHT-THROUGH IS NUMERICALLY UNSAFE, and it killed a run (torus_vl128, 2026-09-02,
        # dead 7.5 h before anyone noticed). `p + (p.clamp(0,1) - p)` is exact only for moderate magnitudes.
        # Measured: 1e8 -> 0.0 (catastrophic cancellation) and +-inf or nan -> NaN. A FRESH decoder at step 0
        # is unbounded and can emit exactly those, and then torchmetrics raises mid-training-step:
        #   "Expected both input arguments to be normalized tensors ... values in range [0., 2.]"
        # So: sanitise the non-finites FIRST, then straight-through, then a final clamp as a belt. That last
        # clamp is a NO-OP for any value already inside [0,1], so it does not touch the gradient in the region
        # that matters -- it only catches precision artifacts, which have no meaningful gradient anyway.
        pc = self._sanitise(p)
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

    def _assemble(self, zero: Tensor, mode: str, *args) -> Tensor:
        """Sum w * term over the registry. `mode` picks which reduction each term contributes."""
        total = zero
        for term in self._terms:
            w = float(term.weight(self) or 0.0)
            if not w:
                continue
            part = term.pointwise(self, *args) if mode == "pointwise" else term.difference(self, *args)
            if part is None:
                continue
            for sub in (part if isinstance(part, list) else [part]):
                total = total + w * sub
        return total

    def forward(self, pred: Tensor, target: Tensor, site: str = "decode") -> Tensor:
        pred, target = self._flatten(pred, target)     # (B,F,H,W,C) from the anchor -> (B*F,H,W,C)
        self._record(site, pred)                       # RAW range, before any term clamps -- see pop_diagnostics
        return self._assemble(pred.new_zeros(()), "pointwise", pred, target)

    # ---- the FIRST-ORDER term (design/derivative_loss.md) ------------------------------------------
    def _pairs(self, pred: Tensor, target: Tensor, stride: int):
        """(B,F,...) -> four (n,...) tensors: pred_t, pred_{t+k}, true_t, true_{t+k}. CONTIGUOUS pairs.

        WHY PAIRS AND NOT `_subsample`. `_subsample` is `randperm` over the FLATTENED B*F rows. Pick 128 of
        1344 and you get (b=3,t=17), (b=0,t=52), (b=14,t=6)...; differencing consecutive entries of that
        would compute frame(b=3,t=17) - frame(b=0,t=52) -- two frames from DIFFERENT EPISODES at unrelated
        times. Not a derivative, noise. So the sampler picks (b, t) and takes (t, t+k) together.

        `self.frames` is the budget in FRAMES, so n_pairs = frames // 2 keeps the feature-net cost equal to
        the ordinary LPIPS term's. Indexing along dim 1 also means episode seams are impossible by
        construction -- differencing adjacent rows of the flat tensor would fabricate a huge spurious delta
        at every b -> b+1 boundary."""
        B, F = pred.shape[:2]
        nt = F - stride
        if nt <= 0:
            return None
        n = max(1, int(self.frames) // 2) if self.frames else B * nt
        bi = torch.randint(0, B, (n,), device=pred.device)
        ti = torch.randint(0, nt, (n,), device=pred.device)
        return pred[bi, ti], pred[bi, ti + stride], target[bi, ti], target[bi, ti + stride]

    def temporal(self, pred: Tensor, target: Tensor, strides=(1,)) -> Tensor:
        """Distance between TEMPORAL DIFFERENCES of two sequences. Both (B, F, H, W, C) in [0,1].

            w_l1    * L1(dp, dg)
          + w_lpips * sum_l || (phi_l(p_t+k) - phi_l(p_t)) - (phi_l(g_t+k) - phi_l(g_t)) ||^2

        DIFFERENCE OF EMBEDDINGS, never embedding of differences. `self(dp, dg)` would be the latter: it
        would run VGG on a signed, sparse, near-zero tensor it was never trained on, and the features would
        be meaningless. Here every input to the net is a real frame. For the PIXEL term the two readings are
        identical (f = identity, differencing commutes), which is why the distinction is easy to miss.

        `strides` is LOCKED at (1,) in config (design/derivative_loss.md §2.1): arXiv 2102.05822 §4.5
        tested K > 1 and found results "almost identical", and our per-frame term already catches the drift
        a larger K would add. The plural signature exists so a reader need not re-derive the generalisation
        to learn it was considered.

        Why this is not `w_l2`: a temporal difference image is SPARSE -- almost all zero, with a blob where
        something moved. L1 does not let a few large values dominate and its constant gradient keeps small
        motions visible; L2 squares, so the biggest change swamps the rest."""
        # RANK 5 exactly. (B,F,H,W,C) is 5-D; the flattened (B*F,H,W,C) the other loss sites use is 4-D,
        # and a rank>=3 check could not tell them apart -- which is the whole failure this guards against,
        # since a flattened input would silently difference frames from different episodes.
        assert pred.dim() == 5 and pred.shape[:2] == target.shape[:2], (
            f"temporal() needs (B, F, H, W, C) sequences, got {tuple(pred.shape)} vs "
            f"{tuple(target.shape)}. A 4-D input means the time axis was already flattened away "
            f"(see design/derivative_loss.md §6)")
        total = pred.new_zeros(())
        for k in (strides if isinstance(strides, (list, tuple)) else (strides,)):
            k = int(k)
            if k < 1 or k >= pred.shape[1]:
                continue
            # TWO PAIRINGS, and the split is not arbitrary. The PIXEL terms are cheap, so they see
            # ALL frames via a plain slice. The FEATURE terms are not, so they see `_pairs`' sampled
            # contiguous pairs. Terms declare which they want with `full_sequence`.
            for term in self._terms:
                w = float(term.weight(self) or 0.0)
                if not w:
                    continue
                if getattr(term, "full_sequence", False):
                    args = (pred[:, :-k], pred[:, k:], target[:, :-k], target[:, k:])
                else:
                    got = self._pairs(pred, target, k)
                    if got is None:
                        continue
                    p0, p1, g0, g1 = got
                    args = (p0, p1, g0, g1)
                part = term.difference(self, *args)
                if part is None:
                    continue
                for sub in (part if isinstance(part, list) else [part]):
                    total = total + w * sub
        return total

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
