"""First 30 s of each Cosmos posneg prediction, from the RAW frames.

Source is raw_filmstrip_frames_<ep>.npz, which holds the whole rollout at native resolution in float32
-- not the rollout mp4, so there is no h264 pre-pass. Two outputs per clip:

  .mp4  512x384 panels (the house canvas, 1027x471) -- no 256-colour limit, use this when gradients matter
  .gif  288x216 panels, built at final size in ONE resample -- 1.7x sharper per pixel than upscaling to
        512 and shrinking, and small enough to leave headroom under the 40 MB cap
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/app/src")
from quickdraw.logging import viz

SRC = ("/app/logs/eval_ood_horizon_2026_09_22_19_16_38_cosmos_val_posneg/logs/epoch_0000/"
       "eval_ood_horizon")
OUT = "/app/scratch/gifs_spec"
HZ, SEC = 15.0, 30
os.makedirs(OUT, exist_ok=True)

for mode in ("open_loop", "closed_loop_16_steps"):
    for ep in (0, 1):
        z = np.load(f"{SRC}/{mode}/image/raw_filmstrip_frames_{ep}.npz")
        n = min(int(SEC * HZ), len(z["pred"]))
        pred, gt = z["pred"][:n], z["gt"][:n]
        tag = f"{mode}_ep{ep}_first{int(n / HZ)}s"
        big = viz.label_panels([gt, pred], ["Ground truth", "Prediction"])
        viz.save_mp4(f"{OUT}/{tag}.mp4", big, HZ)
        small = viz.label_panels([gt, pred], ["Ground truth", "Prediction"],
                                 panel_wh=(288, 216), header_px=49, font_px=30)
        mb = viz.save_gif(f"{OUT}/{tag}.gif", small, fps=15, log=print)
        print(f"  {tag}: {n} frames | mp4 {big.shape[2]}x{big.shape[1]} | "
              f"gif {small.shape[2]}x{small.shape[1]} {mb:.1f} MB", flush=True)
