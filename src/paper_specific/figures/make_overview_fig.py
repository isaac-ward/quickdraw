"""figures/overview.png -- Figure 1. What is learned from play, and what is added at plan time.

Three panels, left to right. PLAY: real recorded frames with the commands that were flown, no task.
LEARNED: the three parts, each named by the question it answers. PLAN TIME: a sentence arrives, candidate
chunks are drawn from the prior, rolled through the world model and scored by the reward head, and the
best is committed -- frames and path here are a real imagined plan, not a sketch.

Modality hues match the architecture figure (vision green, proprioception orange, action blue).

    python -m paper_specific.figures.make_overview_fig
"""
from __future__ import annotations

import glob
import json

import matplotlib
matplotlib.use("Agg")
import imageio.v3 as iio3
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = glob.glob("/app/scratch/recording_*_starling-2")[0]
PLAN = "/app/logs/eval_steer_2026_09_14_21_57_53_best_prior_guided/logs/epoch_0000/eval_steer/plans"
REQ = "rotate_right"
OUT = "/app/logs/paper_icra_2027/overview.png"
GRN, ORG, BLU = [plt.get_cmap(c) for c in ("Greens", "Oranges", "Blues")]
GREY = "#4d4d4d"
FS = 8.0


def frames_of(mp4, want):
    fr = list(iio3.imiter(mp4, plugin="pyav"))
    return [fr[min(i, len(fr) - 1)] for i in want]


def heading(p):
    """Net heading change in degrees, from the proprio quaternion (qx, qy, qz, qw at columns 6:10)."""
    x, y, z, w = p[:, 6], p[:, 7], p[:, 8], p[:, 9]
    yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1.0 - 2 * (y * y + z * z)))
    return np.degrees(yaw - yaw[0])


def box(ax, x, y, w, h, title, sub, ec, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=3.0",
                                lw=1.4, ec=ec, fc=fc, zorder=5))
    ax.text(x + w / 2, y + h * 0.40, title, ha="center", va="center", fontsize=FS, zorder=6)
    ax.text(x + w / 2, y + h * 0.74, sub, ha="center", va="center", fontsize=FS - 1.2,
            style="italic", color=GREY, zorder=6)


def arrow(ax, a, b, lab="", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, lw=1.1, color=GREY,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=4))
    if lab:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 - 2.0, lab, ha="center", va="bottom",
                fontsize=FS - 1.5, color=GREY, zorder=7)


def main() -> int:
    # ---- real data for the two end panels
    vid = sorted(glob.glob(f"{ROOT}/train/videos/**/*.mp4", recursive=True))[0]
    play = frames_of(vid, [180, 240, 300])
    plans = sorted(glob.glob(f"{PLAN}/{REQ}/*/plan.json"))
    pros = [np.load(q.replace("plan.json", "proprio.npy")) for q in plans]
    # FEATURE THE PLAN THE READOUT PICKS, not the first filename: `rotate right` wants a NEGATIVE net
    # heading change and individual contexts disagree -- the paper's claim is about the mean over sixteen
    # of them, so the figure must not imply that every one obeys.
    best = int(np.argmin([heading(q)[-1] for q in pros]))
    d = json.load(open(plans[best]))
    pro, others = pros[best], [q for k, q in enumerate(pros) if k != best][:3]
    imag = frames_of(plans[best].replace("plan.json", "image.mp4"), [0, 60, 127])
    print(f"  play frames {len(play)} | plan '{d['request']}' ep{d['episode']}@{d['start']} "
          f"| {pro.shape[0]} steps | {len(others)} context paths")

    fig = plt.figure(figsize=(7.0, 2.35))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_xlim(0, 100); ax.set_ylim(34, 0)                       # y inverted: down the page is +y

    IW, IH = 17.0, 17.0 * play[0].shape[0] / play[0].shape[1]

    # ---- A: PLAY ------------------------------------------------------------------------------------
    ax.text(9.5, 2.0, "play", ha="center", va="center", fontsize=FS + 1.5)
    ax.text(9.5, 4.8, "no task, no reward", ha="center", va="center", fontsize=FS - 1.2,
            style="italic", color=GREY)
    for k, f in enumerate(play):
        x, y = 1.0 + 3.6 * k, 7.0 + 1.8 * k
        ax.imshow(f, extent=(x, x + IW, y + IH, y), alpha=0.55 + 0.225 * k, zorder=k,
                  interpolation="bilinear")
        ax.add_patch(Rectangle((x, y), IW, IH, fill=False, ec=GRN(0.78), lw=1.0, zorder=k + 0.5))
    for k in range(4):                                            # the commands that were flown
        ax.add_patch(Rectangle((8.2 + 2.6 * k, 22.5), 2.6, 2.6, fc=BLU(0.25 + 0.16 * k), ec="k", lw=0.7))
    ax.text(7.4, 23.8, "commands", ha="right", va="center", fontsize=FS - 1.0, color=GREY)
    ax.text(9.5, 27.6, "$2$ hours", ha="center", va="center", fontsize=FS - 1.0, color=GREY)

    # ---- B: LEARNED ---------------------------------------------------------------------------------
    ax.text(48.0, 2.0, "learned from play", ha="center", va="center", fontsize=FS + 1.5)
    box(ax, 34.0, 6.0, 28.0, 6.4, "world model", "what will happen", GRN(0.78), GRN(0.05))
    box(ax, 34.0, 14.0, 28.0, 6.4, "action prior", "what I might do", BLU(0.78), BLU(0.05))
    box(ax, 34.0, 22.0, 28.0, 6.4, "reward head", "what is good", ORG(0.78), ORG(0.05))
    arrow(ax, (20.0, 16.0), (33.0, 12.0))
    arrow(ax, (48.0, 12.4), (48.0, 14.0))
    arrow(ax, (48.0, 20.4), (48.0, 22.0))

    # ---- C: PLAN TIME -------------------------------------------------------------------------------
    ax.text(83.0, 2.0, "plan time", ha="center", va="center", fontsize=FS + 1.5)
    ax.add_patch(FancyBboxPatch((66.0, 4.2), 34.0, 4.6, boxstyle="round,pad=0,rounding_size=2.2",
                                lw=1.0, ec=GREY, fc="white", zorder=6))
    ax.text(83.0, 6.5, "\u201c" + d["request"] + "\u201d", ha="center", va="center", fontsize=FS,
            zorder=7)
    arrow(ax, (63.0, 14.0), (66.5, 14.0))
    for k, f in enumerate(imag):                                  # what the plan imagined (HUD cropped)
        g = f[: int(f.shape[0] * 0.76)]
        h = 10.6 * g.shape[0] / g.shape[1]
        x = 66.5 + 11.4 * k
        ax.imshow(g, extent=(x, x + 10.6, 10.6 + h, 10.6), zorder=2, interpolation="bilinear")
        ax.add_patch(Rectangle((x, 10.6), 10.6, h, fill=False, ec=GRN(0.78), lw=1.0,
                               ls=(0, (2, 1.4)), zorder=3))
    ax.text(66.5, 19.6, "imagined", ha="left", va="top", fontsize=FS - 1.5, color=GREY)

    hs = [heading(pro)] + [heading(o) for o in others]
    span = max(np.abs(np.concatenate(hs)).max(), 1.0)
    X0, X1, YM, AMP = 70.0, 95.0, 27.0, 5.2                       # the little axes, in panel units
    ax.plot([X0, X0], [YM - AMP, YM + AMP], color=GREY, lw=0.8, zorder=4)
    ax.plot([X0, X1], [YM, YM], color=GREY, lw=0.8, zorder=4)     # zero: heading unchanged
    for k, hh in enumerate(hs):                                   # the same request, four contexts
        t = X0 + (X1 - X0) * np.arange(len(hh)) / (len(hh) - 1)
        ax.plot(t, YM + AMP * hh / span, lw=1.6 if k == 0 else 0.7,
                color=BLU(0.85) if k == 0 else GREY, alpha=1.0 if k == 0 else 0.5, zorder=5)
    ax.text(X0 - 1.0, YM, "heading\nchange", ha="right", va="center", fontsize=FS - 1.8,
            color=GREY, linespacing=1.1)
    ax.text(X1 + 0.6, YM, "$34$ s", ha="left", va="center", fontsize=FS - 1.8, color=GREY)
    ax.text(X1 + 0.6, YM + AMP * hs[0][-1] / span, f"${hs[0][-1]:+.0f}^\\circ$", ha="left",
            va="center", fontsize=FS - 1.8, color=BLU(0.9))

    fig.savefig(OUT, dpi=450, bbox_inches="tight")
    print("  wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
