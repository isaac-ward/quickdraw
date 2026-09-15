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
from matplotlib.patches import PathPatch
from matplotlib.path import Path

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


def rounded_path(pts, r):
    """Open polyline with interior corners rounded -- the paper's brace primitive, copied from
    paper_specific/figures/make_sequence_figs.py so the braces here match the ones there."""
    pts = [np.asarray(q, float) for q in pts]
    verts, codes = [pts[0]], [Path.MOVETO]
    for i in range(1, len(pts) - 1):
        prev, cur, nxt = pts[i - 1], pts[i], pts[i + 1]
        u_in, u_out = cur - prev, nxt - cur
        l_in, l_out = np.linalg.norm(u_in) or 1.0, np.linalg.norm(u_out) or 1.0
        d = min(r, 0.5 * l_in, 0.5 * l_out)
        verts += [cur - d * u_in / l_in, cur, cur + d * u_out / l_out]
        codes += [Path.LINETO, Path.CURVE3, Path.CURVE3]
    verts.append(pts[-1]); codes.append(Path.LINETO)
    return Path(verts, codes)


def brace_left(fig, x, y0, y1, x_stem, r=0.006, **kw):
    """A vertical brace spanning y0..y1 at x, nibs turning toward the images, stem out to the label."""
    mid = 0.5 * (y0 + y1)
    for ye in (y0, y1):
        pts = [(x + 0.012, ye), (x, ye), (x, mid), (x - 0.010, mid)]
        fig.add_artist(PathPatch(rounded_path(pts, r), fill=False, transform=fig.transFigure, **kw))
    fig.add_artist(PathPatch(rounded_path([(x - 0.010, mid), (x_stem, mid)], r), fill=False,
                             transform=fig.transFigure, **kw))


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
    for req, name, steps in KEEP:
        folder = req.replace(" ", "_")
        d = os.path.join(RUN, "logs", "epoch_0000", "eval_steer", "plans", folder, name)
        assert os.path.isdir(d), d
        fr = frames(os.path.join(d, "image.mp4"))
        ks = [min(int(t) - 1, len(fr) - 1) for t in steps]
        rows.append((req, fr[ks], steps))
    print(f"  {len(rows)} rows")

    ih, iw = rows[0][1][0].shape[:2]
    NC = max(len(r[2]) for r in rows)
    # ONE LABEL PER REQUEST, not per row: consecutive rows of the same request share it, and the start
    # time is dropped -- it identified a context for the author to choose from and means nothing now.
    groups = []
    for k, (req, _, _) in enumerate(rows):
        if groups and groups[-1][0] == req:
            groups[-1][1].append(k)
        else:
            groups.append((req, [k]))
    fig = plt.figure(figsize=(3.4, 3.4 * (len(rows) * ih * 1.20) / (NC * iw)))
    # NO GAP INSIDE A SEQUENCE: the frames of one plan are one strip. Rows stay apart just enough for
    # the step numbers, and each request is braced across its own pair.
    gs = fig.add_gridspec(len(rows), NC, hspace=0.26, wspace=0.0,
                          left=0.135, right=0.999, top=0.97, bottom=0.004)
    axes = []
    for r, (req, imgs, steps) in enumerate(rows):
        row = []
        for c in range(len(steps)):
            A = fig.add_subplot(gs[r, c])
            A.imshow(imgs[c], interpolation="bilinear")
            A.set_xticks([]); A.set_yticks([])
            for sp in A.spines.values():
                sp.set_linewidth(0.6); sp.set_color("black")
            A.set_title(f"$+${steps[c]}", fontsize=FS - 1.0, pad=1.2)
            row.append(A)
        axes.append(row)
    fig.canvas.draw()
    for req, idxs in groups:
        pos = [axes[i][0].get_position() for i in idxs]
        y0, y1 = pos[-1].y0, pos[0].y1
        brace_left(fig, 0.115, y0, y1, 0.085, color="0.35", lw=0.9)
        fig.text(0.030, 0.5 * (y0 + y1), "\u201c" + req + "\u201d", ha="center", va="center",
                 fontsize=FS * 1.5, style="italic", rotation=90)
    fig.savefig(OUT, dpi=450, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {OUT}  ({os.path.getsize(OUT) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
