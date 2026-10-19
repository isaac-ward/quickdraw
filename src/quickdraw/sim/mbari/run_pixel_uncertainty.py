"""Pixel-level uncertainty videos: where in the frame the model is surprised.

Per frame: encode the real context, draw K samples of the NEXT latent from the flow, decode each,
and take |observed - mean| / (std + eps) per pixel, pooled over patches. The map is rendered beside
the observation and the predicted mean, so a reviewer can see both that the model was surprised and
WHERE.

K SAMPLES OF THE SAME NEXT LATENT, not K independent rollouts. The question is "how much does the
model's own predictive distribution disagree with what happened, locally", which is a one-step
quantity -- a multi-step rollout would conflate surprise with accumulated drift.

The 8x8 pooling and the std normalisation are not arbitrary: starling measured this exact map at
pixel AUC 0.961, against raw |absdiff| 0.909 and bare `std` 0.530. The rejected variants are listed
in `uncertainty.surprise_map` so they do not get retried.
"""
from __future__ import annotations

import pathlib

import numpy as np
import torch

from .uncertainty import surprise_map


def _heat(x, lo=None, hi=None):
    """Scalar field -> inferno RGB uint8, with a FIXED scale so frames are comparable in a video."""
    import matplotlib.cm as cm

    lo = float(np.nanpercentile(x, 2)) if lo is None else lo
    hi = float(np.nanpercentile(x, 99)) if hi is None else hi
    z = np.clip((x - lo) / max(1e-9, hi - lo), 0, 1)
    return (cm.inferno(z)[..., :3] * 255).astype(np.uint8)


@torch.no_grad()
def episode_maps(core, norm, o, a, fr, key="fpv", P=8, dev="cuda", K=32, stride=2, pool=8,
                 max_frames=150):
    """-> (obs, pred_mean, surprise) each (N,H,W,3)/(N,H,W), subsampled by `stride`."""
    T = len(o)
    ts = list(range(P, T, stride))[:max_frames]
    obs_out, mu_out, sur_out = [], [], []
    for t in ts:
        ctx = {"proprio": norm.norm_obs(torch.from_numpy(o[t - P:t]))[None].float().to(dev),
               key: torch.from_numpy(fr[t - P:t])[None].float().div(255.0).to(dev)}
        acts = norm.norm_act(torch.from_numpy(a[t - P:t]))[None].float().to(dev)
        draws = []
        for _ in range(K):
            out = core.imagine_eval({k: v.clone() for k, v in ctx.items()}, acts, 1,
                                    heads=[key], norm=norm)
            draws.append(out[key][0, 0].clamp(0, 1).cpu().numpy())
        s = np.stack(draws)
        truth = fr[t].astype(np.float32) / 255.0
        obs_out.append((truth * 255).astype(np.uint8))
        mu_out.append((s.mean(0) * 255).astype(np.uint8))
        sur_out.append(surprise_map(s, truth, pool=pool))
    return np.stack(obs_out), np.stack(mu_out), np.stack(sur_out)


def render(core, norm, root, split, ep_idx, out_dir, key="fpv", P=8, dev="cuda", K=32, log=print):
    """Write one <split>_ep<i>.mp4/.gif: observed | predicted mean | surprise."""
    from ...logging import viz

    f = sorted(pathlib.Path(root, split).glob("ep_*.npz"))[ep_idx]
    z = np.load(f)
    obs, mu, sur = episode_maps(core, norm, z["obs"], z["act"], z[f"frames_{key}"],
                                key=key, P=P, dev=dev, K=K)
    lo, hi = float(np.percentile(sur, 2)), float(np.percentile(sur, 99))
    heat = np.stack([_heat(s, lo, hi) for s in sur])
    canvas = viz.label_panels([obs, mu, heat], ["Observed", f"Predicted mean of {K}", "Surprise"],
                              panel_wh=(288, 216), header_px=49, font_px=30)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    viz.save_mp4(str(out_dir / f"{split}_ep{ep_idx}.mp4"), canvas, 15)
    mb = viz.save_gif(str(out_dir / f"{split}_ep{ep_idx}.gif"), canvas, fps=12, max_mb=45, log=log)
    log(f"  {split} ep{ep_idx}: {len(obs)} frames, surprise {sur.min():.2f}-{sur.max():.2f}, gif {mb:.1f} MB")
    return sur


if __name__ == "__main__":
    import argparse
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    ap = argparse.ArgumentParser(description="Pixel-level uncertainty videos, one per split.")
    ap.add_argument("--ckpt", required=True, help="training run dir or .ckpt")
    ap.add_argument("--root", default="/app/logs/mbari/data/v6")
    ap.add_argument("--out", default="/app/logs/mbari/uncertainty/pixel")
    ap.add_argument("--splits", default="train,ood_thruster,ood_fish,ood_buoy")
    ap.add_argument("--ep", type=int, default=0)
    ap.add_argument("--K", type=int, default=32)
    a = ap.parse_args()

    from ...evaluation.standalone import load_checkpoint
    from ...training.setup import build_model, normalizer
    from .run_uncertainty import _cfg_for

    cfg = _cfg_for(a.ckpt)
    core = build_model(cfg).to("cuda")
    load_checkpoint(core, a.ckpt)
    core.eval()
    norm = normalizer(cfg)
    for sp in a.splits.split(","):
        render(core, norm, a.root, sp, a.ep, a.out, key=str(cfg.data.cam), P=int(cfg.data.P),
               K=a.K, log=lambda m: print(m, flush=True))
