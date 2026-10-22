"""World-model OOD detection: one-step SURPRISE + split-CONFORMAL InD/OOD classification.

Detector: at each step predict obs[t+1] from the P-window of obs+actions (proprio WM) and score surprise =
||pred - actual|| in NORMALIZED obs space. The first WARM steps (startup transient: drone accelerating from rest
+ WM warm-start context) are DISCARDED -- evaluation runs only in the settled nominal regime, and the onset t* is
placed after WARM so every trajectory has a clean pre-onset window.

Conformal: a held-out NOMINAL calibration set gives per-trajectory scores; the split-conformal threshold at level
alpha is the finite-sample (1-alpha) quantile, which guarantees P(nominal flagged) <= alpha. Classify a trajectory
InD/OOD by score vs threshold; report accuracy, TPR per OOD type, FPR, AUC, and detection latency after t*.
"""
from __future__ import annotations

import numpy as np

WARM = 300                      # steps (10 s @30Hz) of startup transient to discard before evaluating
GROUPS = {"pos": slice(0, 3), "quat": slice(3, 7), "vel": slice(7, 10), "rate": slice(10, 13)}


def proprio_surprise(core, norm, obs, act, P, dev="cuda", batch=1024):
    """obs (T,13), act (T,4) -> dict of per-group + 'total' one-step surprise (T,), NaN for the first P steps."""
    import torch
    T = len(obs)
    on = norm.norm_obs(torch.from_numpy(np.asarray(obs)).float())      # (T,13) normalized
    an = norm.norm_act(torch.from_numpy(np.asarray(act)).float())      # (T,4)
    B = T - P
    ctx = torch.stack([on[i:i + P] for i in range(B)])                 # (B,P,13)
    seq = torch.stack([an[i:i + P] for i in range(B)])                 # (B,P,4) = P-1 past + 1 future action
    preds = []
    for s in range(0, B, batch):
        c = {"proprio": ctx[s:s + batch].to(dev)}
        with torch.no_grad():
            out = core.imagine_shared(c, seq[s:s + batch].to(dev), 1, 1, heads=["proprio"], norm=norm)
        preds.append(out["proprio"][:, 0].cpu())
    err = (torch.cat(preds) - on[P:P + B]).numpy()                     # (B,13) normalized-space error
    res = {}
    for g, sl in GROUPS.items():
        v = np.full(T, np.nan); v[P:P + B] = np.linalg.norm(err[:, sl], axis=1); res[g] = v
    v = np.full(T, np.nan); v[P:P + B] = np.linalg.norm(err, axis=1); res["total"] = v
    return res


def joint_surprise(core, norm, obs, act, frames, P, dev="cuda", stride=1, batch=12, patch=16, hipx_thr=0.05):
    """One-step surprise from the JOINT cave_fpv WM, BOTH heads from a single imagine_shared pass:
      - 'proprio': ||pred - actual|| in NORMALIZED obs space (same quantity as proprio_surprise['total']),
      - 'fpv': pixel-space MSE of the clamp-to-[0,1] decoded next frame vs the real next frame -- EXACTLY the
        trainer's val/metric/fpv/mse (lit.py), so it's comparable to the published 0.0068 (GLOBAL mean).
      - 'fpv_patchmax': worst PxP-patch mean sq-error -- a LOCALIZED statistic: a small/distant novel object is a
        local error hot-spot that a whole-frame mean buries, so patch-max rescues the far-person misses.
      - 'fpv_hipx': count of pixels whose sq-error exceeds hipx_thr -- another localized (area) statistic.
    The vision model needs BOTH modalities in context (encode_state encodes the full layout). frames (T,144,192,3)
    uint8 -> float/255, HWC, NO resize, NO normalizer (images bypass it). `stride` subsamples the window starts to
    bound the expensive image-decode cost; returns per-step (T,) arrays, NaN except at predicted positions."""
    import torch
    T = len(obs)
    on = norm.norm_obs(torch.from_numpy(np.asarray(obs)).float())                  # (T,13) normalized
    an = norm.norm_act(torch.from_numpy(np.asarray(act)).float())                  # (T,4)  normalized
    fn = torch.from_numpy(np.asarray(frames)).float().div_(255.0)                  # (T,144,192,3) in [0,1], HWC
    B = T - P
    idxs = list(range(0, B, stride))
    pro = np.full(T, np.nan); fpv = np.full(T, np.nan)
    pmax = np.full(T, np.nan); hipx = np.full(T, np.nan)
    for s in range(0, len(idxs), batch):
        chunk = idxs[s:s + batch]
        ctx = {"proprio": torch.stack([on[i:i + P] for i in chunk]).to(dev),
               "fpv": torch.stack([fn[i:i + P] for i in chunk]).to(dev)}
        seq = torch.stack([an[i:i + P] for i in chunk]).to(dev)                    # (b,P,4) = P-1 past + 1 future
        with torch.no_grad():
            o = core.imagine_shared(ctx, seq, 1, 1, heads=["fpv", "proprio"], norm=norm)
        pe = (o["proprio"][:, 0].cpu() - torch.stack([on[i + P] for i in chunk])).numpy()   # (b,13)
        pf = o["fpv"][:, 0].clamp(0, 1).cpu()                                      # (b,144,192,3)
        emap = ((pf - torch.stack([fn[i + P] for i in chunk])) ** 2).mean(-1)      # (b,144,192) per-pixel sq-err
        b, H, W = emap.shape
        glob = emap.reshape(b, -1).mean(1).numpy()
        pm = emap.reshape(b, H // patch, patch, W // patch, patch).mean((2, 4)).amax((1, 2)).numpy()
        hp = (emap > hipx_thr).reshape(b, -1).sum(1).float().numpy()
        for j, i in enumerate(chunk):
            pro[i + P] = np.linalg.norm(pe[j]); fpv[i + P] = glob[j]; pmax[i + P] = pm[j]; hipx[i + P] = hp[j]
    return {"proprio": pro, "fpv": fpv, "fpv_patchmax": pmax, "fpv_hipx": hipx}


def traj_score(surp, tail_n=300):
    """Per-trajectory detector score = MEAN surprise over the last tail_n steps (10 s). Both proprio anomalies
    PERSIST to the end and the onset t* is always <= 18 s, so the last 10 s (20-30 s) is fully post-onset for OOD
    and smooth cruise for nominal -- the tail mean separates a sustained dynamics step-up (0.13 -> ~0.25) and
    glitch's repeated spikes from the flat nominal tail, without diluting by the pre-onset half (global mean) or
    tripping on rare nominal corner-spikes (max)."""
    v = surp[-tail_n:]
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else float("nan")


def perstep_pool(surps, warm=WARM):
    """Pool all settled per-step surprise values across a set of trajectories (for the per-step threshold)."""
    return np.concatenate([s[warm:][np.isfinite(s[warm:])] for s in surps])


def chroma_perframe(frames, thresh=100.0):
    """Per-frame count of STRONGLY-red pixels: (R - max(G,B)) > thresh; (T,). Cheap, model-free. The bright-red
    person (R-max ~190) produces hundreds of such pixels; brown/grey rock never clears ~40 even as the drone flies
    erratically (dynamics/glitch), so this is truly BLIND to motion-only OOD and fires only on the novel red object
    (cf [[ood-surprise-tools]]: chroma separates a foreign colour far better than ensemble sigma)."""
    f = np.asarray(frames, np.float32)
    red = f[..., 0] - np.maximum(f[..., 1], f[..., 2])
    return (red > thresh).reshape(len(f), -1).sum(1).astype(np.float32)


def traj_max(score, warm=WARM):
    """Per-trajectory score = MAX settled per-frame value. For VISION/person: the person is unambiguous when in
    view but only TRANSIENTLY visible (the drone flies past), so max (not tail-mean) is the right statistic."""
    v = score[warm:]
    v = v[np.isfinite(v)]
    return float(v.max()) if len(v) else float("nan")


def conformal_threshold(cal_scores, alpha):
    """Split-conformal threshold at level alpha from nominal calibration scores: the finite-sample (1-alpha)
    quantile ceil((n+1)(1-alpha))/n, which bounds the nominal flag rate by alpha."""
    s = np.sort(np.asarray(cal_scores, float))
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))                           # rank (1-indexed) of the threshold
    return float(s[-1] if k > n else s[k - 1])


def detection_latency(surp, tau, t_star, warm=WARM, fps=30.0):
    """Seconds from onset t* to the first settled step whose surprise exceeds tau (inf = missed)."""
    start = max(warm, int(t_star))
    idx = np.where(np.nan_to_num(surp[start:], nan=-np.inf) > tau)[0]
    return float(idx[0] / fps) if len(idx) else float("inf")


def roc_auc(neg, pos):
    """AUC that the score separates OOD (pos) from InD (neg); rank-based (Mann-Whitney) with AVERAGE ranks for
    ties, so all-equal scores give 0.5 (a blind detector), not 1.0."""
    neg = np.asarray(neg, float); pos = np.asarray(pos, float)
    if not len(neg) or not len(pos):
        return float("nan")
    allv = np.concatenate([neg, pos])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    sv = allv[order]                                              # average tied ranks
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    r_pos = ranks[len(neg):].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
