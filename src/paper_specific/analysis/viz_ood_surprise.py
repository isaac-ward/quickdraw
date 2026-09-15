"""ONE video per noodle episode: every anomaly-localisation method side by side, including the observation.

Panels, left to right -- observation first so the eye has the reference, then the model's own picture of
what should have happened, then the four maps in the order they were measured:

  observed         the true frame
  decode(z_pred)   what the model predicted, decoded. The noodle is absent from it, which is the anomaly.
  surprise_patch   |observed - ensemble mean| / ensemble std, per pixel, pooled over 4x4   (AUC 0.944)

THE OTHER THREE MAPS WERE MEASURED AND DROPPED, in case they are reached for again: unpooled `surprise`
(0.895) is noisier for no gain; `recon_diff` (0.853), which decodes BOTH bags so the codec's blur cancels,
loses because a tokenizer that cannot represent the noodle does not put it in decode(z_true) either -- the
anomaly is attenuated on both sides of the subtraction, so comparing against the RAW frame wins; and
`token_attrib` (0.777) is too diffuse, since each of the 32 image tokens influences a broad region.

COLOUR SCALE IS FIXED PER EPISODE PER METHOD, from that episode's OUT-OF-WINDOW frames (50th-99th
percentile). Normalising each frame to itself would make every frame look equally anomalous; the scale has
to come from frames known not to contain the noodle for the anomalous ones to stand out. Brightness is therefore comparable across TIME within the panel.

A white border and an [IN WINDOW] caption mark the reviewed anomaly window, so the highlight can be checked
against the ground truth while watching.

    CUDA_VISIBLE_DEVICES=0 python scratch/viz_ood_surprise.py <wm_ckpt> [out_dir] [n_samples]
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(__file__))   # sibling analyses in this package
from latent_ood_maps import recon_diff_map                                                  # noqa: E402
from localise_ood_pixels import SPLIT, SUB, ensemble                           # noqa: E402

from quickdraw.data.dataset import load_split_episodes_mm, set_action_aggregate, set_subsample
from quickdraw.data.ood_windows import kept, window_steps
from quickdraw.logging import viz
from quickdraw.training.setup import (build_model, image_head_cams, image_head_sizes, load_checkpoint,
                                      normalizer, resolve_data_root)

FPS_OUT = 4.0                 # model steps are 0.27 s apart, so 4 fps is close to real time
PATCH_VIZ = 4          # 4x4, at the user's request. The AUC sweep prefers larger (4x4 0.944,
#                        8x8 0.960, 12x12 0.965) because bigger pools suppress the thin
#                        permanent edge response, but 4x4 keeps the blob tight to the object,
#                        which is what a figure wants. The trade is sharpness vs edge rejection.
MAPS = ["surprise_patch"]
BAR = 16


def label(img, text):
    out = np.zeros((img.shape[0] + BAR, img.shape[1], 3), np.uint8)
    out[BAR:] = img
    cv2.putText(out, text, (3, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def heat(arr, lo, hi):
    x = np.clip((arr - lo) / max(1e-9, hi - lo), 0, 1)
    return cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]


def main(ckpt: str, out_root: str = "logs/paper_icra_2027", n_samp: str = "32") -> int:
    N = int(n_samp)
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
    d = os.path.join(out_root, "eval_ood", SPLIT, "viz"); os.makedirs(d, exist_ok=True)
    rep = {}
    print(f"  {SPLIT} -> {d}   one video per episode, {N} ensemble samples/step, all methods side by side")
    for i in kept(SPLIT):
        o, a, fr = eps[i]
        w0, w1 = window_steps(SPLIT, i, SUB)
        steps = list(range(P, len(o)))
        acc = {k: [] for k in MAPS}
        pred = []
        for t in steps:
            obs = torch.from_numpy(fr[key][t]).float().div(255.0).to(dev)
            s_ = ensemble(core, norm, o, a, fr[key], key, P, t, dev, n=N)
            mu, sd = s_.mean(0), s_.std(0)
            sur = ((obs - mu).abs() / (sd + 1e-3)).mean(-1)
            sp = F.avg_pool2d(sur[None, None], PATCH_VIZ, 1, PATCH_VIZ // 2)[0, 0][:sur.shape[0], :sur.shape[1]]
            _, dp = recon_diff_map(core, norm, o, a, fr[key], key, P, t, dev, patch=PATCH_VIZ)
            acc["surprise_patch"].append(sp.cpu().numpy())
            pred.append((dp.cpu().numpy() * 255).astype(np.uint8))
        A = {k: np.stack(v) for k, v in acc.items()}
        outw = np.array([not (w0 <= t < w1) for t in steps])
        sc = {k: (float(np.percentile(A[k][outw] if outw.any() else A[k], 50)),
                  float(np.percentile(A[k][outw] if outw.any() else A[k], 99))) for k in MAPS}
        frames = []
        for j, t in enumerate(steps):
            inw = w0 <= t < w1
            tag = f" [IN WINDOW]" if inw else ""
            panes = [label(fr[key][t], f"observed  t={t}{tag}"), label(pred[j], "decode(z_pred)")]
            panes += [label(heat(A[k][j], *sc[k]), k) for k in MAPS]
            row = np.concatenate(panes, axis=1)
            if inw:
                row[:2] = 255; row[-2:] = 255; row[:, :2] = 255; row[:, -2:] = 255
            frames.append(row)
        viz.save_mp4(os.path.join(d, f"ep{i:02d}.mp4"), np.stack(frames), FPS_OUT)
        rep[f"ep{i:02d}"] = {"steps": [steps[0], steps[-1]], "window": [w0, w1],
                             "scales": {k: sc[k] for k in MAPS},
                             "mean_in": {k: (float(A[k][~outw].mean()) if (~outw).any() else None) for k in MAPS},
                             "mean_out": {k: (float(A[k][outw].mean()) if outw.any() else None) for k in MAPS}}
        f = lambda x: "  --  " if x is None else f"{x:6.3f}"
        print(f"    ep{i:02d}  {len(steps):2d} steps, window {w0}-{w1} | in/out  " + "  ".join(
            f"{k} {f(rep[f'ep{i:02d}']['mean_in'][k])}/{f(rep[f'ep{i:02d}']['mean_out'][k])}" for k in MAPS))
    json.dump(rep, open(os.path.join(d, "_summary.json"), "w"), indent=1)
    print(f"  -> {len(rep)} videos, 6 panels each (observed, decode(z_pred), {', '.join(MAPS)})")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
