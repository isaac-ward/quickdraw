"""DINOv3 patch-similarity: a perceptual term that is LOCAL by construction.

Wired into `VisualLoss` as one more `_Term`, so it reaches the decode loss, the roundtrip anchor
and the derivative term alike. This file owns ONLY the frozen backbone cache and the two distances;
`_sanitise`, `_subsample`, `_pairs`, `_record` and the rank handling all stay in VisualLoss and
serve every term.

WHY, IN ONE MEASUREMENT. On `bs_stride10` open-loop, `cos(dpred, dtrue)` on the flattened frame
difference is 0.07. For scale: displacing the TRUE change by 1 px scores 0.85, by 4 px 0.26, by 8 px
0.03, and shuffling its pixels to random positions scores 0.002. The model repaints roughly the right
QUANTITY (||dpred||/||dtrue|| reaches 0.87) in close to the WRONG PLACES. LPIPS reduces with a
spatial mean over frames that are 84.6% static, diluting the moving region 5x -- it is structurally
poorly placed to see this.

THE DIFFERENCE FROM LPIPS IS THE REDUCTION ORDER. LPIPS takes a squared difference per (position,
channel) and averages all of it at once, so magnitude leaks across positions and bright, large-area
regions dominate. Here the cosine is taken over the FEATURE axis INDEPENDENTLY PER PATCH, and only
then averaged over patches -- so a dim 4-pixel block and a bright table texture contribute EQUALLY.
That is local emphasis by construction, not a reweighting bolted on top.

    f_p(x) in R^d                      the token for patch p          (d=384, P=48 at 96x128)

    pointwise   mean_p [ 1 - cos( f_p(pred), f_p(true) ) ]
    temporal    mean_p [ 1 - cos( df_p(pred), df_p(true) ) ],  df_p(x) = f_p(x_t+k) - f_p(x_t)

The temporal form is the DIFFERENCE OF EMBEDDINGS, never the embedding of a difference -- the same
distinction visual_loss.temporal already makes for LPIPS. Feeding a ViT a signed, near-zero tensor
it never saw in training would produce meaningless features.

PATCH 16 DIVIDES OUR FRAMES EXACTLY. 96/16 = 6, 128/16 = 8, so 48 patch tokens with no resize and no
pad -- and the same 6x8 grid as the decoder's query grid at ae_bottleneck=6. DINOv2's patch 14
divides neither side. DINOv3 also uses RoPE rather than a learned position table, so a non-square
input needs no interpolation.

WHY DINOv3 AND NOT DINOv2, when P-DINO (arXiv 2602.02493) was published on v2: v3's headline
contribution is the mechanism this term relies on. During long training, dense features degrade --
"an increasing number of irrelevant patches with high similarity to the reference patch" -- which is
a FALSE POSITIVE in exactly the patch cosine computed here. Gram anchoring regularises the Gram
matrix of patch-to-patch similarities against an earlier checkpoint. Result vs DINOv2: +6 mIoU on
ADE20K and +6.7 J&F on video tracking, the latter being the closest published proxy for "does a
patch stay identifiable over time".

GATED. `facebook/dinov3-*` needs licence acceptance on the HF account; HF_TOKEN then picks it up.
This module fails SOFT if the weights are unavailable, exactly as the LPIPS path does.
"""

from __future__ import annotations

import torch
from torch import Tensor

# One frozen backbone per (name, device) for the whole process. The net is ~21M params for ViT-S and
# is reused by every modality's VisualLoss and by both loss sites, which is ~2700 calls an epoch.
_CACHE: dict[tuple[str, str], object] = {}

_REPOS = {
    "vits16":     "facebook/dinov3-vits16-pretrain-lvd1689m",
    "vitsplus16": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "vitb16":     "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "vitl16":     "facebook/dinov3-vitl16-pretrain-lvd1689m",
}

# ImageNet statistics, which is what DINOv3 was trained with. Our frames arrive in [0,1].
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)

PATCH = 16
N_SPECIAL = 5          # 1 cls + 4 register tokens, dropped before the patch cosine


def get_net(name: str, device):
    """The frozen DINOv3 backbone, cached. Returns None if the weights cannot be had (gated repo,
    no network, transformers too old) -- the caller then contributes nothing, which is how the LPIPS
    path already behaves when its weights are missing."""
    key = (str(name), str(device))
    if key in _CACHE:
        return _CACHE[key]
    net = None
    try:
        from transformers import AutoModel
        net = AutoModel.from_pretrained(_REPOS[str(name)])
        net.eval().to(device)
        for p in net.parameters():
            p.requires_grad_(False)       # FROZEN: a critic that moves is not a critic
    except Exception as exc:              # noqa: BLE001 -- fail soft, and say why once
        print(f"[dino_loss] DINOv3 '{name}' unavailable, term contributes nothing "
              f"({type(exc).__name__}: {str(exc)[:120]})", flush=True)
        net = None
    _CACHE[key] = net
    return net


def patch_tokens(net, x: Tensor, layer: int) -> Tensor | None:
    """(M,H,W,C) in [0,1] -> (M,P,d) patch tokens from `layer`. -1 = the final block.

    `layer` indexes `hidden_states`, which is [embeddings, block_1, ..., block_L], so -1 is the last
    block's output. PixelGen found the FINAL block best for this kind of loss -- "benefits from
    high-level semantic features instead of low-level features" -- and that stacking several layers
    actively hurts, "conflicting supervision and performs poorly". That is the OPPOSITE of LPIPS,
    which sums five layers, so the instinct to stack must be resisted.
    """
    if net is None:
        return None
    m, h, w, _ = x.shape
    if h % PATCH or w % PATCH:
        raise ValueError(f"dino_loss needs H and W divisible by {PATCH}, got {h}x{w}. Our frames are "
                         f"96x128 (6x8 patches); a different img_size needs a resize decision made "
                         f"deliberately rather than silently.")
    z = x.permute(0, 3, 1, 2).float()
    mean = torch.as_tensor(_MEAN, device=z.device).view(1, 3, 1, 1)
    std = torch.as_tensor(_STD, device=z.device).view(1, 3, 1, 1)
    out = net(pixel_values=(z - mean) / std, output_hidden_states=True)
    return out.hidden_states[layer][:, N_SPECIAL:, :]      # drop cls + registers


def patch_cosine(fp: Tensor, fg: Tensor, eps: float = 1e-6) -> Tensor:
    """mean over patches and frames of (1 - cos), the cosine taken PER PATCH over the feature axis.

    The per-patchness is the point: `cos` normalises WITHIN a patch, so every patch contributes on
    equal footing regardless of its contrast or area. Contrast LPIPS, whose squared difference lets
    high-magnitude regions dominate -- the measured 5x dilution on our 84.6% static frames.

    KNOWN LIMITATION OF THE TEMPORAL FORM, measured and left visible rather than papered over: most
    patches are static, so their `df_p` is ~0, and the cosine of two near-zero vectors is noise. On
    real pred/gt pairs chopped into the exact 6x8 grid of 16x16 patches, 72.2% of patches have
    ||dtrue|| below 10% of the p90, and on those the cosine is mean +0.041 / std 0.155 -- centred on
    zero. On the moving patches it is +0.161. So a plain mean is ~72% noise for the TEMPORAL form.
    The POINTWISE form has no such problem: f_p is never near zero.

    Starting with the plain mean anyway (user, 2026-09-12) because it is the published P-DINO form
    and the simplest thing that can work; `visual_dino_v3_patch_weight` exists to weight by motion
    instead, and which wins is a measurement, not an assertion. See design/dino_loss.md section 3.
    """
    num = (fp * fg).sum(-1)
    den = fp.norm(dim=-1) * fg.norm(dim=-1)
    return (1.0 - num / (den + eps)).mean()


def weighted_patch_cosine(fp: Tensor, fg: Tensor, mode: str, eps: float = 1e-6) -> Tensor:
    """`patch_cosine` with a per-patch weight. mode: 'none' (plain mean) | 'true' | 'max'.

    'true' weights patch p by ||fg_p||, 'max' by max(||fp_p||, ||fg_p||). The weight is DETACHED:
    without that, the model can lower the loss by shrinking ||fp_p|| -- i.e. by freezing -- rather
    than by pointing the change in the right direction. Focal Frequency Loss locks the gradient
    through its spectrum weight matrix for exactly this reason.

    'max' rather than 'true' alone so that a patch where the model HALLUCINATES change onto a static
    region still carries weight; weighting by truth only would leave that unpenalised.
    """
    if mode == "none":
        return patch_cosine(fp, fg, eps)
    num = (fp * fg).sum(-1)
    nfp, nfg = fp.norm(dim=-1), fg.norm(dim=-1)
    per = 1.0 - num / (nfp * nfg + eps)
    if mode == "true":
        w = nfg.detach()
    elif mode == "max":
        w = torch.maximum(nfp, nfg).detach()
    else:
        raise ValueError(f"patch_weight must be none|true|max, got {mode!r}")
    return (per * w).sum() / (w.sum() + eps)
