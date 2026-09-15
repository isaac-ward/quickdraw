"""figures/plans.png -- three language-steered plans, imagined.

One request per row: the frames the planner imagined while pursuing it, left to right, labelled by their
open-loop step, ending on the step the request named. The prompt sits above its row.

THE THREE PLANS ARE NOT CHOSEN BY EYE. They are plans the VLM labeller independently confirmed reached
the place they were asked for (paper_specific/analysis/steer_vlm_objects.py writes those labels, and
scratch/_pick_plans.py maps them back to their folders) -- so the figure shows plans that a judge agreed
with, not the ones that happened to look good.

The plan mp4 writes a caption bar under each frame, so it is cropped to the image height: the caption
carries the request text, and a figure that showed it would be telling the reader the answer.

    python -m paper_specific.figures.make_plans_fig
"""
from __future__ import annotations

import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np

RUN = "/app/logs/eval_steer_2026_09_14_21_57_53_best_prior_guided"
OUT = "/app/logs/paper_icra_2027/plans.png"
IMG_H = 112                # the starling camera; rows below this in the mp4 are the caption bar
N_SHOW = 6
FS = 9.5
# (request, plan folder) -- each one VLM-confirmed. Two objects and one region, so the figure shows both
# kinds of request the reward head was trained on.
PLANS = [("black panel", "plans/black_panel/ep003_t0367"),
         ("table", "plans/table/ep004_t0055"),
         ("floor to ceiling glass wall", "plans/floor_to_ceiling_glass_wall/ep004_t0055")]


def frames(path):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f[:IMG_H, :, ::-1])          # crop the caption bar: it names the request
    cap.release()
    return np.stack(out)


def main() -> int:
    rows = []
    for req, rel in PLANS:
        d = os.path.join(RUN, "logs", "epoch_0000", "eval_steer", rel)
        fr = frames(os.path.join(d, "image.mp4"))
        H = len(fr)
        ks = np.unique(np.linspace(0, H - 1, N_SHOW).round().astype(int))
        rows.append((req, fr[ks], ks, H))
        print(f"  {req:32s} {H} steps -> " + ", ".join(f"+{k + 1}" for k in ks))

    ih, iw = rows[0][1][0].shape[:2]
    NC = len(rows[0][2])
    fig = plt.figure(figsize=(7.1, 7.1 * (len(rows) * ih * 1.68) / (NC * iw)))
    # THREE rows per plan: the prompt, the frames, and a spacer -- with hspace=0 (which the frames want)
    # the step numbers under one row otherwise land on the next row's prompt.
    gs = fig.add_gridspec(3 * len(rows), NC, hspace=0.0, wspace=0.02,
                          height_ratios=[0.34, 1.0, 0.34] * len(rows))
    for r, (req, imgs, ks, H) in enumerate(rows):
        lab = fig.add_subplot(gs[3 * r, :]); lab.axis("off")
        lab.text(0.0, 0.10, "\u201c" + req + "\u201d", ha="left", va="bottom", fontsize=FS + 1.0,
                 style="italic")
        for c in range(NC):
            A = fig.add_subplot(gs[3 * r + 1, c])
            A.imshow(imgs[c], interpolation="bilinear", aspect="auto")
            A.set_xticks([]); A.set_yticks([])
            for sp in A.spines.values():
                sp.set_linewidth(1.4); sp.set_color("black")
            A.set_xlabel(f"$+${ks[c] + 1}", fontsize=FS - 1.0, labelpad=1.5)
    fig.savefig(OUT, dpi=450, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {OUT}  ({os.path.getsize(OUT) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
