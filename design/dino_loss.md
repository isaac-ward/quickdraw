# A DINOv3 patch-similarity term, alongside LPIPS

**Status: PLAN.** Depends on `design/visual_loss_terms.md` (the term registry) landing first, and on
DINOv3 access — `facebook/dinov3-vits16-pretrain-lvd1689m` is a GATED HF repo; access requested
2026-09-12. `transformers 5.15.1` already supports DINOv3, so nothing else to install.

## 1. Why, in one measurement

`bs_stride10`, open-loop, `cos(dpred, dtrue)` on the flattened frame difference = **0.07**. Scale:
a 1 px displacement of the TRUE change scores 0.85, 4 px scores 0.26, 8 px scores 0.03, and the
true change with its pixels randomly shuffled scores 0.002. So the model repaints about the right
QUANTITY (`‖dpred‖/‖dtrue‖ → 0.87` by ep9) in close to the WRONG PLACES.

LPIPS reduces with a spatial `.mean()` over frames that are **84.6% static**, diluting the moving
region **5x** (measured, scene_right at stride 10). It is structurally poorly placed to see this.

## 2. The exact reduction — this is the whole point

`f_p(x) ∈ R^d` is the DINOv3 token for patch `p` of frame `x`. For ViT-S/16 at 96x128: `d = 384`,
`P = 48` patches (6x8, exact — no resize, no pad), plus a cls token and 4 registers which are
DROPPED.

**Pointwise (the LPIPS-slot form):**

```
                 1    M    1    P
  L_point   =   ---  SUM ---  SUM   [ 1 - cos( f_p(x̂_m) , f_p(x_m) ) ]
                 M   m=1   P   p=1
```

**Temporal (the derivative-term form), difference of EMBEDDINGS:**

```
  Δf_p(x)   =   f_p(x_{t+k}) - f_p(x_t)                    a d-vector, per patch

                 1    M    1    P
  L_temporal =  ---  SUM ---  SUM   [ 1 - cos( Δf_p(pred) , Δf_p(true) ) ]
                 M   m=1   P   p=1
```

**Where the per-patchness lives:** the cosine is taken over the FEATURE axis `d`, INDEPENDENTLY for
each patch `p`, and only then averaged over patches and frames. Two consequences:

* **cos normalises WITHIN a patch**, so a dim 4-pixel block and a bright table texture contribute
  EQUALLY. That is precisely the 5x dilution LPIPS suffers, removed by construction rather than by
  a reweighting bolted on top.
* Contrast LPIPS, which takes a squared difference per (position, channel) and averages over ALL of
  them at once — magnitude leaks across positions, so big bright regions dominate the score.

NOT `cos` of the flattened frame (that is the diagnostic in §1, one number per frame). NOT a mean
of features then a cosine. Per patch, then mean.

## 3. THE PROBLEM THAT WILL BITE, and the decided fix

Most patches are static, so `Δf_p ≈ 0`, and **the cosine of two near-zero vectors is numerically
meaningless** — it swings over [-1, 1] on floating-point noise.

MEASURED, on real (pred, gt) pairs from `bs_stride10` ep9, chopped into exactly the 6x8 grid of
16x16 patches DINOv3 would see (79,200 patch-differences):

| | |
|---|---|
| patches with `‖Δtrue‖` < 10% of the p90 | **72.2%** |
| cosine on those STATIC patches | mean **+0.041**, std 0.155 -> centred on zero, i.e. noise |
| cosine on the MOVING patches | mean **+0.161**, std 0.322 -> real signal |

So a plain mean would be **72% composed of patches whose cosine is meaningless**.

### The design space, stated algebraically

`‖a − b‖² = ‖a‖² + ‖b‖² − 2‖a‖‖b‖·cos(a,b)`, so an L2 ALREADY contains the cosine, weighted by the
product of the magnitudes. The question is therefore only HOW MUCH magnitude weighting:

| reduction | magnitude weighting | outcome |
|---|---|---|
| cosine, plain mean | none | 72% noise (measured above) |
| plain L2 | quadratic | back to LPIPS-style dilution |
| **weighted cosine** | **linear** | the middle ground — TAKE THIS |

Measured on the same data: plain mean `0.9258`, weighted `0.8823`, moving-patches-only `0.8393`.

### The rule

```
  w_p       =  stopgrad( max( ‖Δf_p(true)‖ , ‖Δf_p(pred)‖ ) )

                 1    M     SUM_p  w_p · [ 1 - cos( Δf_p(pred), Δf_p(true) ) ]
  L_temporal =  ---  SUM   ------------------------------------------------------
                 M   m=1                    SUM_p  w_p
```

**`max`, not `‖Δf_true‖` alone.** Weighting by truth only means a patch where the model HALLUCINATES
motion onto a static background gets ~0 weight, so the hallucination goes unpenalised — the
background-neglect risk. `max` closes it: missed motion is weighted by the true magnitude, invented
motion by the predicted one.

**stopgrad on the weight.** Otherwise the model lowers the loss by shrinking `‖Δf_p(pred)‖` — i.e.
FREEZING — to reduce its own weight. Detached, freezing gains nothing: a genuinely moving patch
keeps weight `‖Δf_true‖`, and `cos(0, Δf_true) = 0` gives the full penalty. Direct precedent:
Focal Frequency Loss locks the gradient through its spectrum weight matrix for the same reason.

**NOT max or top-k OVER PATCHES.** With ~13 moving patches of 48, a max lets one patch drive the
whole gradient — high variance, and it is the OHEM-ranked-by-current-loss failure that the
segmentation literature reports degrading as training proceeds.

`visual_dino_v3_weight` selects `max` (default) / `true` / `none`, so the choice stays measurable
in section 6 rather than asserted.

**The POINTWISE form has none of this problem** — `f_p` is never near zero, so it takes a plain
mean over patches. Only the temporal form needs the weight.

## 4. Layer choice

**Default: the final block**, configurable, and settle it empirically in §6.

PixelGen (2602.02493, DINOv2-B) found layer 12 of 12 best — *"P-DINO benefits from high-level
semantic features instead of low-level features"* — and that **multiple layers actively hurt**:
*"conflicting supervision and performs poorly"*. That is the OPPOSITE of LPIPS, which sums 5 layers,
so the instinct to stack layers must be resisted.

DINOv3 specifically earns this: its Gram anchoring exists to stop dense features degrading during
long training, where *"an increasing number of irrelevant patches [get] high similarity to the
reference patch"* — a false-positive failure in exactly the patch cosine we compute. It regularises
the Gram matrix of patch-to-patch similarities `G = XX^T` against an earlier checkpoint. Result vs
DINOv2: **+6 mIoU** ADE20K, **+6.7 J&F** video tracking.

## 5. Design: a Term, not a switch

`models/dino_loss.py` holds ONLY the backbone cache and the two distances. It is wired in as one
more `Term` in the registry (`design/visual_loss_terms.md`), exactly as LPIPS is.

```
visual_lpips: 1.0           # unchanged, still VGG
visual_dino_v3: 0.0         # new, default OFF -> byte-identical to not having it
visual_dino_v3_net:   vits16    # vits16 | vitsplus16 | vitb16
visual_dino_v3_layer: -1        # -1 = FINAL block (see section 4); configurable, settle it in section 6
visual_dino_v3_weight: max      # patch reduction: max | true | none  (see section 3)
```

**Two weights, not a mode switch.** Setting `visual_lpips=0 visual_dino_v3=1` swaps them; leaving both
nonzero runs both, which is what PixelGen found best (LPIPS for local texture, DINO for semantics:
FID 23.67 -> 10.00 with LPIPS -> 7.46 adding DINO). A `perceptual: lpips|dino` enum could not
express that, and would need a new value for every future combination.

Reuse comes from the registry, not from sharing code with LPIPS: `_sanitise`, `_subsample`,
`_pairs`, `_record` and the rank handling all stay in `VisualLoss` and serve every term. The DINO
file owns the frozen net (one process-level cache keyed by `(device, name)`, as `_lpips` already
does) and nothing else.

## 6. Settle it on data we already have, before any training run

978 `raw_filmstrip_*.npz` files across four runs hold real (pred, gt) pairs at KNOWN quality levels.
Score the same pairs with LPIPS, with this term at several layers, and with the three §3 variants,
then ask which best separates rollouts we already have opinions about and which tracks the visible
popping. A candidate that cannot out-rank LPIPS on footage we can already judge will not help as a
loss. No GPU-hours, no training run.

## 7. Cost — not an objection

FLOPs per image at 96x128, against the VGG16 LPIPS already pays for:

| | params | GFLOPs/img | vs VGG16 |
|---|---|---|---|
| VGG16 (current) | — | 3.75 | 1.00x |
| **DINOv3 ViT-S/16** | 21M | **2.30** | **0.61x** |
| ViT-B/16 | 86M | 9.11 | 2.43x |
| ViT-L/16 | 300M | 32.3 | 8.62x |

ViT-S is CHEAPER than what we run today: a conv backbone pays per pixel, a ViT pays per patch, and
48 patches is very few. Analytical FLOPs, not wall clock — at 53 tokens it will be latency-bound,
so the real margin is smaller. It rules out "too expensive" at ViT-S.
