"""WHICH PIXELS are OOD? Localise the pool noodle, and measure whether the localisation is real.

Four candidate maps, computed from an ENSEMBLE of one-step predictions (the model is sampled N times from
the same context, so stochastic_eval gives N draws of "what should come next"):

  absdiff        |observed - ensemble mean|. The naive pixel diff. Included as the baseline to beat.
  std            the ensemble's per-pixel standard deviation -- "where the model is unsure". PREDICTED TO
                 FAIL, and worth measuring for that reason: a model that has never seen a noodle does not
                 know it might be there, so it predicts the wall CONFIDENTLY. Epistemic uncertainty only
                 covers hypotheses inside the model's own distribution, and the anomaly is outside it.
  surprise       |observed - mean| / (std + eps). The fix: judge the difference in units of the model's own
                 spread, so a big difference where the model was uncertain (edges, moving structure) counts
                 for little and a modest one on a flat confident wall counts for a lot.
  surprise_bg    MEASURED AND REJECTED (0.917 against surprise_patch's 0.961): surprise_patch divided
                 by a PER-PIXEL BACKGROUND surprise, estimated from the same
                 episode's out-of-window frames. The surprise map still fires on room edges -- thin, high
                 frequency structure the model always gets slightly wrong -- and those pixels are hard in
                 EVERY frame, anomaly or not. Dividing them out asks "is this pixel harder than it usually
                 is HERE", which is the question. Valid because these clips are near-static, so a pixel
                 means the same thing across frames; it would not transfer to a moving camera. It is
                 kept in the table because the reasoning was sound and the result says otherwise -- four
                 out-of-window frames at 16 samples is too noisy a denominator to divide by, and the noise
                 it adds costs more than the edge response it removes.
  surprise_patch surprise averaged over 8x8 patches. Frames do not line up perfectly even one step ahead,
                 and patch pooling tolerates a pixel or two of misalignment that punishes a per-pixel map.

ONE-STEP, not open-loop, on purpose: the prediction has to be spatially aligned with the observation for
any per-pixel comparison to mean anything, and an open-loop rollout has drifted.

GROUND TRUTH FOR LOCALISATION comes from the colour mask (saturated warm pixels), which is independent of
the world model -- so pixel-level AUC answers "does this map put the noodle above everything else",
not merely "does the frame look anomalous".

    CUDA_VISIBLE_DEVICES=0 python scratch/localise_ood_pixels.py <wm_ckpt> [out_dir]
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from quickdraw.data.dataset import load_split_episodes_mm, set_action_aggregate, set_subsample
from quickdraw.data.ood_windows import kept, window_steps
from quickdraw.training.setup import (build_model, image_head_cams, image_head_sizes, load_checkpoint,
                                      normalizer, resolve_data_root)

SPLIT, SUB, N_SAMP, PATCH = "eval_ood_noodle", 4, 64, 8
MAPS = ["absdiff", "std", "surprise", "surprise_patch", "surprise_bg"]


def pink_mask(frame_u8):
    """Ground-truth noodle pixels, from colour alone -- independent of the world model."""
    hsv = cv2.cvtColor(frame_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = hsv[..., 0], hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    return (((h < 25) | (h > 160)) & (s > 0.35) & (v > 0.25))


@torch.no_grad()
def ensemble(core, norm, o, a, fr, key, P, t, dev, n=N_SAMP):
    """n one-step predictions of frame t from the true context. -> (n,H,W,3) in [0,1]."""
    ctx = {"proprio": norm.norm_obs(torch.from_numpy(o[t - P:t])).float()[None].expand(n, -1, -1).to(dev),
           key: torch.from_numpy(fr[t - P:t]).float().div(255.0)[None].expand(n, -1, -1, -1, -1).to(dev)}
    acts = norm.norm_act(torch.from_numpy(a[t - P:t])).float()[None].expand(n, -1, -1).to(dev)
    out = core.imagine_eval(ctx, acts, 1, heads=[key], norm=norm)
    return out[key][:, 0].clamp(0, 1)


def maps_from(samples, obs, bg=None):
    """samples (n,H,W,3), obs (H,W,3) -> the four candidate maps, each (H,W)."""
    mu, sd = samples.mean(0), samples.std(0)
    diff = (obs - mu).abs().mean(-1)
    std = sd.mean(-1)
    sur = ((obs - mu).abs() / (sd + 1e-3)).mean(-1)
    p = F.avg_pool2d(sur[None, None], PATCH, stride=1, padding=PATCH // 2)[0, 0][:sur.shape[0], :sur.shape[1]]
    out = {"absdiff": diff, "std": std, "surprise": sur, "surprise_patch": p}
    out["surprise_bg"] = p / (bg + 1e-3) if bg is not None else p
    return out, mu, sd


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    lo = np.sort(neg)
    r = np.searchsorted(lo, pos, side="left") + 0.5 * (
        np.searchsorted(lo, pos, side="right") - np.searchsorted(lo, pos, side="left"))
    return float(r.mean() / len(neg))


def main(ckpt: str, out_root: str = "logs/paper_icra_2027") -> int:
    run = os.path.dirname(os.path.dirname(ckpt)) if ckpt.endswith(".ckpt") else ckpt
    cfg = OmegaConf.create(json.load(open(os.path.join(run, "logs", "config.json"))))
    dev = "cuda"
    set_subsample(SUB); set_action_aggregate("concat")
    m = build_model(cfg).to(dev); load_checkpoint(m, ckpt); m.eval()
    core = getattr(m, "_orig_mod", m); norm = normalizer(cfg)
    P = int(cfg.data.P)
    key = next((n for n, _ in core.layout if n != "proprio"))
    eps = load_split_episodes_mm(resolve_data_root(cfg), SPLIT, img_size=image_head_sizes(cfg),
                                 cam=image_head_cams(cfg), repo_id="starling-2")
    d = os.path.join(out_root, "eval_ood", SPLIT); os.makedirs(d, exist_ok=True)
    pooled = {k: ([], []) for k in MAPS}
    shown, rows = 0, []
    print(f"  {SPLIT}: {N_SAMP} samples/step, one-step predictions, patch {PATCH}")
    print(f"  {'ep':>3s} {'step':>5s} {'noodle px':>10s} " + " ".join(f"{k:>15s}" for k in MAPS))
    for i in kept(SPLIT):
        o, a, fr = eps[i]
        w0, w1 = window_steps(SPLIT, i, SUB)
        cand = [t for t in range(max(P, w0), min(w1, len(o))) ]
        if not cand:
            continue
        # per-pixel background surprise: the same map on OUT-OF-WINDOW steps of this episode
        outw = [t for t in range(P, len(o)) if not (w0 <= t < w1)][:4]
        bg = None
        if outw:
            acc = []
            for t in outw:
                sb = ensemble(core, norm, o, a, fr[key], key, P, t, dev, n=16)
                mb, _, _ = maps_from(sb, torch.from_numpy(fr[key][t]).float().div(255.0).to(dev))
                acc.append(mb["surprise_patch"])
            bg = torch.stack(acc).mean(0)
        for t in cand[:3]:
            obs_u8 = fr[key][t]
            gt = pink_mask(obs_u8)
            if gt.sum() < 50:
                continue
            obs = torch.from_numpy(obs_u8).float().div(255.0).to(dev)
            s = ensemble(core, norm, o, a, fr[key], key, P, t, dev)
            mp, mu, sd = maps_from(s, obs, bg)
            line = ""
            for k in MAPS:
                v = mp[k].cpu().numpy()
                pooled[k][0].extend(v[gt].tolist()); pooled[k][1].extend(v[~gt].tolist())
                line += f" {auc(v[gt], v[~gt]):>15.3f}"
            rows.append({"episode": i, "step": t, "noodle_px": int(gt.sum())})
            print(f"  {i:>3d} {t:>5d} {int(gt.sum()):>10d}{line}")
            if shown < 4:                            # the inspectable overlay
                best = mp["surprise_patch"].cpu().numpy()
                hm = (best - np.percentile(best, 50)) / (np.percentile(best, 99.5) - np.percentile(best, 50) + 1e-9)
                hm = np.clip(hm, 0, 1)
                ov = obs_u8.astype(np.float32).copy()
                ov[..., 0] = np.clip(ov[..., 0] + 255 * hm * 0.85, 0, 255)      # red where surprising
                ov[..., 1] *= (1 - 0.5 * hm); ov[..., 2] *= (1 - 0.5 * hm)
                fig, ax = plt.subplots(1, 5, figsize=(17, 3.1))
                for A, img, ttl in ((ax[0], obs_u8, f"observed (ep{i:02d} step {t})"),
                                    (ax[1], (mu.cpu().numpy() * 255).astype(np.uint8),
                                     f"model mean of {N_SAMP} samples"),
                                    (ax[2], None, "ensemble std"),
                                    (ax[3], None, "surprise, patch-pooled (the winner)"),
                                    (ax[4], ov.astype(np.uint8), "OVERLAY — red = OOD")):
                    if img is None:
                        arr = (sd.mean(-1) if "std" in ttl else mp["surprise_patch"]).cpu().numpy()
                        # percentile-clipped: without it the permanent edge response dominates the colour
                        # scale and the anomaly -- which is the EXTREME of the map -- looks unremarkable
                        A.imshow(arr, cmap="inferno", vmin=np.percentile(arr, 50),
                                 vmax=np.percentile(arr, 99.0))
                    else:
                        A.imshow(img)
                    A.set_title(ttl, fontsize=8); A.axis("off")
                ax[4].contour(hm, levels=[0.55], colors="w", linewidths=1.2)
                ax[4].text(4, 14, "THIS OBJECT IS OOD", color="w", fontsize=9, weight="bold",
                           bbox=dict(facecolor="black", alpha=0.55, pad=2))
                fig.tight_layout()
                fig.savefig(os.path.join(d, f"ep{i:02d}_step{t:03d}_ood_localise.png"), dpi=110)
                plt.close(fig); shown += 1
    print(f"\n  {'POOLED over ' + str(len(rows)) + ' frames':>21s} " + " ".join(
        f"{auc(np.array(pooled[k][0]), np.array(pooled[k][1])):>15.3f}" for k in MAPS))
    print("\n  pixel-level AUC: P(a noodle pixel scores higher than a non-noodle pixel). 0.5 = the map")
    print("  carries no spatial information about WHERE the anomaly is.")
    json.dump({"maps": {k: auc(np.array(pooled[k][0]), np.array(pooled[k][1])) for k in MAPS},
               "frames": rows, "n_samples": N_SAMP, "patch": PATCH},
              open(os.path.join(out_root, "ood_pixel_localisation.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
