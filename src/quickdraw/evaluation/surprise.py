"""General OOD surprise / anomaly maps via ensemble one-step VoE.

The world model as an anomaly detector: at step t, draw N stochastic one-step predictions of obs[t] from the
TRUE context obs[t-P:t] (+ actions) -- so the N draws are samples of p(obs_t | true context) -- and score how
surprising the real obs[t] is against that ensemble. OOD content (an object/colour/region the model never
trained on) has HIGH surprise; in-distribution content is low.

  surprise = |obs - mu| / (sigma + eps)        # z-score of the truth under the ensemble, per pixel / per dim

Ported from the starling paper scripts (localise_ood_pixels / viz_ood_surprise on `main`) and GENERALIZED:
env-agnostic, works on ANY image head, adds a PROPRIO branch (per-dimension z-score -- did not exist before),
takes the ground-truth region as an OPTIONAL mask argument (no baked-in colour rule), and has no starling I/O.

REQUIRES model.stochastic_eval (else the N draws are identical -> sigma=0). image maps patch-pooled 8x8 (the
paper's AUC-0.961 winner: frames don't line up perfectly even one step ahead, and pooling tolerates a pixel or
two of misalignment that a per-pixel map punishes).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def _ensemble(model, norm, o, a, fr, head, P, t, n, device, return_bag=False):
    """N one-step predictions of step t from the TRUE context [t-P:t]. Image head -> (n,H,W,3) in [0,1];
    proprio -> (n,obs_dim) NORMALIZED (the z-score is scale-free, and pred/obs must share a space).
    return_bag -> also return the predicted LATENT bag (n, n_tok, d) for a latent-space surprise metric."""
    # the model is multimodal -> the CONTEXT needs EVERY head (all image heads + proprio), even when we only
    # SCORE one. Build ctx from all heads present in `fr` plus proprio.
    ctx = {"proprio": norm.norm_obs(torch.from_numpy(o[t - P:t])).float()[None].expand(n, -1, -1).to(device)}
    for h in (fr or {}):
        ctx[h] = torch.from_numpy(fr[h][t - P:t]).float().div(255.0)[None].expand(n, -1, -1, -1, -1).to(device)
    acts = norm.norm_act(torch.from_numpy(a[t - P:t])).float()[None].expand(n, -1, -1).to(device)
    out = model.imagine_eval(ctx, acts, 1, heads=[head], norm=norm, return_bag=return_bag)
    s = out[head][:, 0]
    s = s.clamp(0, 1) if head != "proprio" else s
    return (s, out["_bag"][:, 0]) if return_bag else s


@torch.no_grad()
def encode_true_latent(model, norm, o, fr, t, device):
    """The model's TRUE latent bag for the real obs[t] -> (n_tok, d). Pairs with _ensemble(..., return_bag=True):
    latent surprise = ||z_true - mean(pred_latents)|| (per token). anchor matches imagine_eval (None when rel off,
    e.g. block-stack); if a model uses egocentric relativization, the single-step anchor is an approximation."""
    obs = {"proprio": norm.norm_obs(torch.from_numpy(o[t:t + 1])).float()[None].to(device)}
    for h in (fr or {}):
        obs[h] = torch.from_numpy(fr[h][t:t + 1]).float().div(255.0)[None].to(device)
    anchor = model.rel_anchor(obs) if model._rel_on() else None
    return model.encode_state(obs, anchor)[0, 0]


def image_maps(samples, obs, patch=8):
    """samples (n,H,W,3), obs (H,W,3) in [0,1] -> {absdiff, std, surprise, surprise_patch}, each (H,W)."""
    mu, sd = samples.mean(0), samples.std(0)
    diff = (obs - mu).abs().mean(-1)
    std = sd.mean(-1)
    sur = ((obs - mu).abs() / (sd + 1e-3)).mean(-1)
    p = F.avg_pool2d(sur[None, None], patch, stride=1, padding=patch // 2)[0, 0][:sur.shape[0], :sur.shape[1]]
    return {"absdiff": diff, "std": std, "surprise": sur, "surprise_patch": p}


def proprio_zscore(samples, obs, eps=0.1):
    """samples (n,obs_dim), obs (obs_dim,) NORMALIZED -> per-dimension z-score |obs-mu|/(sigma+eps), (obs_dim,).
    The proprio analogue of the pixel surprise map: which observation dimensions are anomalous.

    eps is a SIGMA FLOOR (default 0.1, in normalized units ~ 10% of a std): proprio dims the model predicts
    near-deterministically have sigma->0, and a tiny eps (1e-3) made their z-score explode to ~1e8 and swamp
    every normal dim. 0.1 means "surprise relative to at least 10% of a standard deviation", keeping the scale
    readable while still flagging genuinely-off dims."""
    mu, sd = samples.mean(0), samples.std(0)
    return ((obs - mu).abs() / (sd + eps))


@torch.no_grad()
def surprise_image(model, norm, o, a, fr, head, P, t, n=64, patch=8, device="cuda"):
    """The per-frame image surprise maps at step t (+ the ensemble mean mu for viz). Returns (maps, mu)."""
    samples = _ensemble(model, norm, o, a, fr, head, P, t, n, device)
    obs = torch.from_numpy(fr[head][t]).float().div(255.0).to(device)
    return image_maps(samples, obs, patch=patch), samples.mean(0)


@torch.no_grad()
def surprise_proprio(model, norm, o, a, fr, P, t, n=64, device="cuda"):
    """The per-frame proprio surprise vector at step t (per observation dimension, (obs_dim,)). `fr` (image
    frames) is still needed -- the model is multimodal, so the proprio prediction conditions on the images too."""
    samples = _ensemble(model, norm, o, a, fr, "proprio", P, t, n, device)
    obs = norm.norm_obs(torch.from_numpy(o[t])).float().to(device)
    return proprio_zscore(samples, obs)


def chroma_err(obs_u8, pred_u8):
    """Prediction error in COLOUR only: Euclidean distance between obs and pred in CIELAB (a,b), luminance (L)
    DISCARDED. obs_u8, pred_u8: (H,W,3) uint8 RGB -> (H,W) float. Isolates an OOD colour (model paints it wrong ->
    big a,b shift) from motion (arm shifts position/brightness but keeps its grey colour -> small a,b shift):
    measured AUC(ood-region > moving-arm) 0.998, where |obs-pred| and ensemble sigma both fail. No colour baked in."""
    import cv2

    lo = cv2.cvtColor(obs_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    lp = cv2.cvtColor(pred_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    return np.linalg.norm(lo[..., 1:] - lp[..., 1:], axis=-1)


def auc_vs_mask(sur, mask):
    """Rank-AUC of surprise values INSIDE the GT mask vs OUTSIDE it. sur, mask: same (H,W). mask bool/{0,1}.
    1.0 = surprise perfectly separates the OOD region from the rest; 0.5 = no localization. NaN if a side empty."""
    sur = np.asarray(sur).ravel()
    m = np.asarray(mask).ravel().astype(bool)
    pos, neg = sur[m], sur[~m]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    lo = np.sort(neg)
    r = (np.searchsorted(lo, pos, "left") + np.searchsorted(lo, pos, "right")) / 2.0
    return float(r.mean() / neg.size)
