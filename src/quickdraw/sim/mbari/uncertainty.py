"""OOD detection from the world model's own predictions, and pixel-level uncertainty maps.

THE METHOD IS NOT NEW HERE -- it is the one `wizard/records/starling.md` §4 already validated, applied
to three OOD conditions instead of two, and the choices it already settled are reused rather than
re-litigated:

  SCORE. Violation of Expectation: the ONE-STEP re-grounded prediction error, per head. Chosen first
  because it needs no calibrated uncertainty -- just the model's own error against what happened.
  Episode score is the 90th percentile over steps, not the mean: an intruder is in frame for part of
  an episode and a mean dilutes it with the nominal remainder.

  PIXEL MAP. Decode K samples of the next latent, take |obs - mean| / (std + eps) per pixel, pool over
  patches. Measured on starling at AUC 0.961 (8x8 pool) and 0.944 (4x4). The alternatives were measured
  and REJECTED there: raw |absdiff| 0.909, bare `std` 0.530, background-normalised 0.917. Do not
  re-try those.

  CALIBRATION. Leave-one-episode-out WITHIN the nominal set, not a threshold fitted on val. Fitting on
  val gave recall 0.016 at AUC 0.94 on starling, because val is a different regime. The threshold for
  episode i is the 90% quantile of the OTHER nominal episodes, which makes ~10% FPR true by
  construction and leaves TPR as the thing being measured.

  PER-HEAD, because the records show the channels separate: a visual intruder shows up in image LPIPS
  and latent surprise, a dynamics fault in the proprio channels. Reporting one blended score would
  hide exactly the structure that distinguishes the three conditions.

`latent_cos` is included as a channel but never as the headline -- it is a diagnostic, it has been
wrong before, and the records say so three times.
"""
from __future__ import annotations

import numpy as np

CHANNELS = ("image_lpips", "image_l1", "latent_surprise", "proprio_pos", "proprio_vel", "proprio_angvel")
PCTL = 90.0


def episode_score(per_step: dict[str, np.ndarray], pctl=PCTL) -> dict[str, float]:
    """Per-step channel errors -> one number per channel for the episode."""
    return {k: float(np.percentile(np.asarray(v, np.float64), pctl)) for k, v in per_step.items()}


def loo_threshold(nominal_scores: np.ndarray, i: int, q=PCTL) -> float:
    """The q-th percentile of every nominal episode EXCEPT i."""
    other = np.delete(np.asarray(nominal_scores, np.float64), i)
    return float(np.percentile(other, q))


def detect(nominal: np.ndarray, ood: np.ndarray, q=PCTL):
    """Leave-one-out FPR on nominals, TPR on OOD at the same threshold, plus threshold-free AUC."""
    nominal = np.asarray(nominal, np.float64)
    ood = np.asarray(ood, np.float64)
    fp = sum(nominal[i] > loo_threshold(nominal, i, q) for i in range(len(nominal)))
    thr = float(np.percentile(nominal, q))
    tp = float((ood > thr).mean())
    return {"fpr": fp / max(1, len(nominal)), "tpr": tp, "threshold": thr,
            "auc": auc(ood, nominal), "n_nom": len(nominal), "n_ood": len(ood)}


def auc(pos, neg):
    """P(a positive outranks a negative), ties counted as half. 0.5 = chance under any imbalance."""
    pos, neg = np.asarray(pos, np.float64), np.asarray(neg, np.float64)
    if not len(pos) or not len(neg):
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def confusion_3way(scores_by_class: dict[str, dict[str, np.ndarray]], channels=CHANNELS):
    """Nearest-centroid classification of the three OOD types, on z-scored channel vectors.

    The question the user asked is not just "is this OOD" but "WHICH of the three". A linear
    nearest-centroid on the channel vector is deliberately the simplest thing that can answer it: if
    the conditions are separable at all, they are separable here, and a stronger classifier on 50
    episodes per class would mostly be fitting the split."""
    classes = list(scores_by_class)
    X, y = [], []
    for ci, c in enumerate(classes):
        v = np.stack([np.asarray(scores_by_class[c][ch], np.float64) for ch in channels], 1)
        X.append(v)
        y.append(np.full(len(v), ci))
    X, y = np.concatenate(X), np.concatenate(y)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    # leave-one-out so the centroid a point is scored against never contains that point
    pred = np.zeros(len(Z), int)
    for i in range(len(Z)):
        keep = np.ones(len(Z), bool)
        keep[i] = False
        cent = np.stack([Z[keep & (y == k)].mean(0) for k in range(len(classes))])
        pred[i] = int(np.argmin(((cent - Z[i]) ** 2).sum(1)))
    cm = np.zeros((len(classes), len(classes)), int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    per_class = (cm.diagonal() / np.maximum(1, cm.sum(1)))
    return {"classes": classes, "confusion": cm, "per_class_acc": per_class,
            "balanced_acc": float(per_class.mean())}


def surprise_map(samples, obs, eps=1e-3, pool=8, smooth=False):
    """|obs - mean| / (std + eps) per pixel, mean-pooled over `pool` x `pool` patches.

    `samples` (K,H,W,3) decoded draws of the SAME next latent; `obs` (H,W,3) what actually happened.
    Both in [0,1]. Returns (H,W) float -- the pooling is why 8x8 scored 0.961 against raw absdiff's
    0.909 on starling: a single-pixel disagreement is noise, a patch of them is an object.

    `smooth` separates the DENOISING from the BLOCKINESS (2026-10-22). The 0.961 came from averaging
    over a patch; nothing in that result requires the output to be a staircase of 8x8 tiles, which is
    purely an artefact of expanding the pooled grid back with np.kron. With smooth=True the same patch
    average is applied as a moving window at full resolution, so the map denoises identically but reads
    as a continuous heat map instead of a chunky grid. Scoring keeps the blocky default so the measured
    AUC still describes the thing that produced it; the VIDEO asks for smooth.
    """
    s = np.asarray(samples, np.float32)
    mu, sd = s.mean(0), s.std(0)
    z = np.abs(np.asarray(obs, np.float32) - mu) / (sd + eps)
    z = z.mean(-1)
    if pool > 1:
        H, W = z.shape
        if smooth:                                   # box filter at full resolution, same window
            # float64 SAT: the running sum reaches ~H*W*max(z), and in float32 a window difference of
            # ~z against a total of ~1e4 loses most of its significant digits -- a constant field came
            # back non-constant. The table is one array, so the cost of f64 here is noise.
            c = np.cumsum(np.cumsum(np.pad(z.astype(np.float64), ((1, 0), (1, 0))), 0), 1)
            r = pool // 2
            y0 = np.clip(np.arange(H) - r, 0, H); y1 = np.clip(np.arange(H) - r + pool, 0, H)
            x0 = np.clip(np.arange(W) - r, 0, W); x1 = np.clip(np.arange(W) - r + pool, 0, W)
            area = np.maximum(1, (y1 - y0)[:, None] * (x1 - x0)[None, :])
            z = (c[np.ix_(y1, x1)] - c[np.ix_(y0, x1)]
                 - c[np.ix_(y1, x0)] + c[np.ix_(y0, x0)]) / area
            z = z.astype(np.float32)
        else:
            ph, pw = H // pool * pool, W // pool * pool
            blocks = z[:ph, :pw].reshape(ph // pool, pool, pw // pool, pool).mean((1, 3))
            z = np.kron(blocks, np.ones((pool, pool), np.float32))
            z = np.pad(z, ((0, H - z.shape[0]), (0, W - z.shape[1])), mode="edge")
    return z
