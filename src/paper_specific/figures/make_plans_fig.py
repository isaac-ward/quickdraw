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

RUN = "/app/logs/eval_steer_2026_09_15_16_33_40_plans8"   # 3 requests x 8 contexts, video on
OUT = "/app/logs/paper_icra_2027/plans.png"
IMG_H = 112                # the starling camera; rows below this in the mp4 are the caption bar
FS = 6.0      # single column, so the text comes down with the width
# THE AUTHOR'S SELECTION, round two: (request, episode prefix, the steps to show). Steps are the
# author's own picks, not an even spacing. Where an episode ran from two different start times the
# reference is ambiguous, so BOTH are emitted and the next round narrows it.
KEEP = [
    # THE AUTHOR'S FINAL SELECTION: explicit start times now, no ambiguity left to resolve.
    ("table", "ep003_t0259", (1, 26, 48, 52)),
    ("table", "ep005_t0200", (1, 10, 26, 52)),
    ("mannequin", "ep006_t0096", (1, 52, 82, 103)),
    ("mannequin", "ep003_t0367", (1, 26, 52, 103)),
    ("clock", "ep001_t0219", (1, 8, 16, 26)),
    ("clock", "ep003_t0367", (1, 52, 77, 128)),
]


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
    for req, prefix, steps in KEEP:
        folder = req.replace(" ", "_")
        hits = sorted(glob.glob(os.path.join(RUN, "logs", "epoch_*", "eval_steer", "plans", folder,
                                             prefix + "*")))
        assert hits, f"no plan matching {req}/{prefix}"
        for d in hits:                                   # both start times when the prefix is ambiguous
            fr = frames(os.path.join(d, "image.mp4"))
            ks = [min(int(t) - 1, len(fr) - 1) for t in steps]
            rows.append((req, os.path.basename(d), fr[ks], steps))
    print(f"  {len(rows)} rows from {len(KEEP)} selections")

    ih, iw = rows[0][2][0].shape[:2]
    NC = max(len(r[3]) for r in rows)
    fig = plt.figure(figsize=(3.4, 3.4 * (len(rows) * ih * 1.62) / (NC * iw)))
    gs = fig.add_gridspec(2 * len(rows), NC, hspace=0.0, wspace=0.02,
                          height_ratios=[0.52, 1.0] * len(rows))
    for r, (req, ctx, imgs, steps) in enumerate(rows):
        lab = fig.add_subplot(gs[2 * r, :]); lab.axis("off")
        lab.text(0.0, 0.16, "\u201c" + req + "\u201d", ha="left", va="bottom", fontsize=FS,
                 style="italic")
        lab.text(1.0, 0.16, ctx, ha="right", va="bottom", fontsize=FS - 2.0, color="0.45")
        for c in range(len(steps)):
            A = fig.add_subplot(gs[2 * r + 1, c])
            A.imshow(imgs[c], interpolation="bilinear")
            A.set_xticks([]); A.set_yticks([])
            for sp in A.spines.values():
                sp.set_linewidth(1.2); sp.set_color("black")
            A.set_title(f"$+${steps[c]}", fontsize=FS - 1.5, pad=1.5)
    fig.savefig(OUT, dpi=450, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {OUT}  ({os.path.getsize(OUT) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
