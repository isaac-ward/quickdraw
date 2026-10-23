"""Split-conformal OOD detection on the controlled onset set, plus the onset trace.

WHAT THIS REPLACES. `uncertainty.confusion_3way` is leave-one-out nearest-centroid with NO held-out
calibration: the centroids are fitted on the same episodes being classified, minus the point itself.
LOO keeps that from being circular but it still has no calibration/test separation, so the detection
threshold has no coverage guarantee and the reported rates are optimistic by an unknown amount.

The controlled set was generated to fix exactly this. 60 nominal episodes split 30 CALIBRATION /
30 TEST, so:

  * the threshold is fitted on calibration nominals ONLY, with the finite-sample correction
    k = ceil((n+1)(1-alpha)) -- at n=30, alpha=0.1 that is the 28th smallest of 30. Split conformal
    then guarantees P(false positive) <= alpha on exchangeable data, rather than hoping a 90th
    percentile of 30 points lands in the right place.
  * the false-positive rate is measured on the 30 TEST nominals, which the threshold never saw.
    A split-conformal FPR should come out near alpha; if it does not, the exchangeability
    assumption is broken and that is worth knowing.

THE ONSET IS THE OTHER HALF. Every episode is 10 s clean then 30 s anomalous, so each one carries
its own before/after contrast and a per-episode score need not be compared across episodes to mean
something. `onset_trace` reports the channel scores on both windows: a detector that works should
show PRE indistinguishable from nominal and POST separated, on the SAME episode. A split that only
separates on the post-window average could be riding an episode-level confound; this cannot.

    python -m quickdraw.sim.mbari.conformal <ckpt_or_run_dir> [--alpha 0.1] [--limit N]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import pathlib

import numpy as np

ROOT = "/app/logs/mbari/data/ood_controlled"
CALIB, TEST = "nominal_calib", "nominal_test"
OOD = ("ood_thruster", "ood_fish", "ood_buoy")
ALPHA = 0.1


def conformal_threshold(calib, alpha=ALPHA):
    """One-sided split-conformal threshold: the k-th smallest calibration score.

    k = ceil((n+1)(1-alpha)) is the finite-sample correction. Taking a plain (1-alpha) empirical
    quantile of n points under-covers for small n -- at n=30 it is the 27th value where the
    guarantee needs the 28th, which sounds like nothing and is the difference between a bound that
    holds and a number that usually looks fine."""
    # DROP NON-FINITE FIRST. A dynamics-only model reports its image channels as NaN (it has no
    # opinion, see one_step), and np.sort puts NaN LAST -- so the k-th smallest silently becomes NaN,
    # every `score > NaN` is False, and the channel reports FPR 0 / TPR 0, which reads as a
    # beautifully specific detector rather than as no measurement at all.
    c = np.asarray(calib, np.float64)
    c = np.sort(c[np.isfinite(c)])
    n = len(c)
    if n == 0:
        return float("nan"), 0
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:                      # too few calibration points to certify this alpha at all
        return float("inf"), k
    return float(c[k - 1]), k


def episode_windows(core, norm, path, key, P, dev, onset):
    """Per-channel scores on the PRE-onset and POST-onset windows of one episode.

    `one_step` returns errors indexed from absolute step P, so the post window begins at
    `onset - P` in its own indexing -- off-by-P here would quietly mix 8 anomalous steps into the
    clean baseline."""
    from .run_uncertainty import CHANNELS, one_step

    z = np.load(path)
    e = one_step(core, norm, z["obs"], z["act"], z[f"frames_{key}"], key, P, dev)
    cut = max(0, int(onset) - int(P))
    pre, post = {}, {}
    for c in CHANNELS:
        v = np.asarray(e[c], np.float64)
        pre[c] = float(np.percentile(v[:cut], 90)) if cut > 2 else float("nan")
        post[c] = float(np.percentile(v[cut:], 90)) if len(v) - cut > 2 else float("nan")
    return pre, post


def score_split(core, norm, root, split, key, P, dev, onset, limit=None, log=print):
    files = sorted(glob.glob(str(pathlib.Path(root) / split / "ep_*.npz")))[:limit]
    pres, posts = [], []
    for n, f in enumerate(files, 1):
        pre, post = episode_windows(core, norm, f, key, P, dev, onset)
        pres.append(pre)
        posts.append(post)
        if n % 10 == 0 or n == len(files):
            log(f"    {split} {n}/{len(files)}")
    keys = pres[0].keys() if pres else []
    return ({c: np.array([p[c] for p in pres]) for c in keys},
            {c: np.array([p[c] for p in posts]) for c in keys})


def run(ckpt, root=ROOT, out="/app/logs/mbari/uncertainty/conformal", key="fpv",
        alpha=ALPHA, limit=None, onset=300, dev="cuda", log=print):
    from ...evaluation.standalone import load_checkpoint
    from ...training.setup import build_model, normalizer
    from .run_uncertainty import CHANNELS, _cfg_for

    cfg = _cfg_for(ckpt)
    core = build_model(cfg).to(dev)
    load_checkpoint(core, ckpt)
    core.eval()
    norm = normalizer(cfg)
    P = int(cfg.data.P)

    scores = {}
    for sp in (CALIB, TEST, *OOD):
        if not (pathlib.Path(root) / sp).exists():
            log(f"  !! missing split {sp}")
            continue
        scores[sp] = score_split(core, norm, root, sp, key, P, dev, onset, limit=limit, log=log)

    rep = {"alpha": alpha, "onset": onset, "channels": list(CHANNELS), "detection": {}, "onset_trace": {}}
    log(f"\n  split-conformal, alpha={alpha}, calibration on {CALIB} ({len(scores[CALIB][1][CHANNELS[0]])} eps)")
    log(f"  {'channel':16s} {'thr':>9} {'k':>3}  {'FPR':>6} | " + "  ".join(f"{s.replace('ood_',''):>9}" for s in OOD))
    for c in CHANNELS:
        thr, k = conformal_threshold(scores[CALIB][1][c], alpha)
        if not np.isfinite(thr):                      # no calibration signal on this channel
            rep["detection"][c] = {"threshold": None, "k": k, "fpr_test": None, "tpr": {},
                                   "note": "channel absent for this model (no image head)" if k == 0
                                           else "too few calibration points to certify alpha"}
            log(f"  {c:16s} {'--':>9} {k:3d}  {'--':>6} |      (no measurement)")
            continue
        fpr = float(np.mean(scores[TEST][1][c][np.isfinite(scores[TEST][1][c])] > thr))
        tpr = {s: float(np.mean(scores[s][1][c][np.isfinite(scores[s][1][c])] > thr))
               for s in OOD if s in scores}
        rep["detection"][c] = {"threshold": thr, "k": k, "fpr_test": fpr, "tpr": tpr}
        log(f"  {c:16s} {thr:9.4f} {k:3d}  {fpr:6.2f} | " + "  ".join(f"{tpr.get(s, float('nan')):9.2f}" for s in OOD))

    # THE IN-EPISODE CONTRAST. pre should sit at the nominal level and post above it, on the same
    # episode -- that is what rules out an episode-level confound explaining the separation.
    log(f"\n  onset trace (90th pct, PRE [0,{onset}) vs POST [{onset},end))")
    log(f"  {'split':16s} " + "  ".join(f"{c.replace('image_','im_').replace('proprio_','pro_'):>13}" for c in CHANNELS))
    for sp in (CALIB, TEST, *OOD):
        if sp not in scores:
            continue
        pre, post = scores[sp]
        rep["onset_trace"][sp] = {c: {"pre": float(np.nanmean(pre[c])), "post": float(np.nanmean(post[c]))}
                                  for c in CHANNELS}
        log(f"  {sp:16s} " + "  ".join(f"{np.nanmean(pre[c]):5.3f}->{np.nanmean(post[c]):6.3f}" for c in CHANNELS))

    o = pathlib.Path(out)
    o.mkdir(parents=True, exist_ok=True)
    (o / "conformal.json").write_text(json.dumps(rep, indent=1, default=str))
    (o / "scores.json").write_text(json.dumps(
        {sp: {"pre": {c: v[0][c].tolist() for c in CHANNELS},
              "post": {c: v[1][c].tolist() for c in CHANNELS}} for sp, v in scores.items()}, indent=1))
    log(f"\n  wrote {o}")
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt")
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default="/app/logs/mbari/uncertainty/conformal")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--onset", type=int, default=300)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    run(a.ckpt, root=a.root, out=a.out, alpha=a.alpha, limit=a.limit, onset=a.onset)


if __name__ == "__main__":
    main()
