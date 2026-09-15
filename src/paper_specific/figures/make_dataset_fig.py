"""figures/dataset.png -- what a training sample IS, in the same visual language as the architecture.

Two sequences, one above the other, six consecutive model steps each. Every step carries the camera
frame, the 16-dimensional proprioceptive vector and the 4-dimensional commanded action, in the three
modality hues the architecture figure uses (vision green, proprioception orange, action blue) and with
the same colour-map window, so a reader can carry the colours between the two figures.

Cells are normalised PER DIMENSION across the sequence, so colour shows how that dimension CHANGED over
the span rather than which dimensions are large -- the same rule as make_sequence_figs. An action cell's
arrow is REAL: up for a positive stick, down for negative, length proportional to how far it is pushed
within that axis's own range over the sequence.

Resolution is matched to the architecture figure in pixels per column-inch, since the two sit at the
same size on the page.

    python -m paper_specific.figures.make_dataset_fig
"""
from __future__ import annotations

import glob

import matplotlib
matplotlib.use("Agg")
import imageio.v3 as iio3
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from matplotlib.patches import FancyArrow, Rectangle

ROOT = glob.glob("/app/scratch/recording_*_starling-2")[0]
OUT = "/app/logs/paper_icra_2027/dataset.png"
FPS, STRIDE = 15.0, 4                 # capture rate and the frame stride the model is trained at
N, ENDS = 6, (304, 1064)              # columns per sequence, and the frame each sequence ends on
IMG_W = 300.0                         # image width in figure units; one proprio cell is IMG_W / 16
LW, EDGE = 0.9, "#000000"
CMAP = {"vision": "Greens", "proprio": "Oranges", "action": "Blues"}
CMAP_LO, CMAP_HI = 0.20, 0.95         # the window make_sequence_figs uses; a white cell reads as empty
# make_sequence_figs draws the architecture on a 7663 px canvas for one \textwidth (7.16 in) = 1070
# px/in. This figure is one \columnwidth (3.4 in) wide on the page, so it needs ~3640 px to match.
DPI = 520

tab = pq.read_table(sorted(glob.glob(f"{ROOT}/train/data/**/*.parquet", recursive=True))[0]).to_pydict()
ep = np.asarray(tab["episode_index"]); keep = np.flatnonzero(ep == ep[0])
obs = np.stack([np.asarray(x, np.float32) for x in tab["observation_vector"]])[keep]
act = np.stack([np.asarray(x, np.float32) for x in tab["action"]])[keep]
vid = sorted(glob.glob(f"{ROOT}/train/videos/**/*.mp4", recursive=True))[0]
IDX = [[e - (N - 1 - i) * STRIDE for i in range(N)] for e in ENDS]
allf = [f for k, f in enumerate(iio3.imiter(vid, plugin="pyav")) if k <= max(max(x) for x in IDX)]
SEQ = [([allf[i] for i in ix], obs[ix], act[ix]) for ix in IDX]
print(f"{len(SEQ)} sequences x {N} steps | frames {IDX} | {1000 * STRIDE / FPS:.0f} ms apart")


def strip(ax, mat, x0, y, cmap, cell, arrows=False):
    """One row of D cells per step, left-aligned at x0, normalised per dimension across the sequence."""
    lo, hi = mat.min(0, keepdims=True), mat.max(0, keepdims=True)
    flat = (hi - lo) < 1e-9
    u = np.where(flat, 0.5, (mat - lo) / np.where(flat, 1.0, hi - lo))
    cm = plt.get_cmap(cmap)
    rng = np.abs(mat).max(axis=0, keepdims=True)                    # per-axis, so every axis is visible
    for i in range(mat.shape[0]):
        for k in range(mat.shape[1]):
            cx = x0[i] + k * cell
            ax.add_patch(Rectangle((cx, y), cell, cell, lw=LW, ec=EDGE,
                                   fc=cm(CMAP_LO + (CMAP_HI - CMAP_LO) * u[i, k])))
            if arrows:
                # EVERY cell gets an arrow, so a centred stick still reads as a direction rather than as
                # a missing glyph: sign sets the direction, |value| sets the length above a floor.
                v = mat[i, k] / max(float(rng[0, k]), 1e-9)
                L = cell * (0.30 + 0.42 * min(abs(v), 1.0))
                dy = -np.sign(v if v != 0 else 1.0) * L
                ax.add_patch(FancyArrow(cx + cell / 2, y + cell / 2 - dy / 2, 0.0, dy,
                                        width=LW * 0.8, head_width=cell * 0.30,
                                        head_length=cell * 0.26, length_includes_head=True,
                                        fc=EDGE, ec=EDGE))
        ax.add_patch(Rectangle((x0[i], y), mat.shape[1] * cell, cell, fill=False, ec=EDGE, lw=LW))


def main() -> int:
    img_h = IMG_W * SEQ[0][0][0].shape[0] / SEQ[0][0][0].shape[1]
    cell = IMG_W / 16.0                       # proprio AND action cells: one size, set by the 16-vector
    gap, colw = 16.0, IMG_W + 26.0
    row_sep = cell + 24.0          # proprio centre to action centre: the labels must not collide
    seq_h = img_h + gap + cell + row_sep + 78.0
    xs = [i * colw for i in range(N)]
    vis = plt.get_cmap(CMAP["vision"])(0.78)
    fig = plt.figure(figsize=(7.0, 7.0 * (len(SEQ) * seq_h) / (N * colw + 150)))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()

    for si, (frames, o, a) in enumerate(SEQ):
        y0 = si * seq_h
        for i, x in enumerate(xs):
            ax.imshow(frames[i], extent=(x, x + IMG_W, y0 + img_h, y0), interpolation="bilinear")
            ax.add_patch(Rectangle((x, y0), IMG_W, img_h, fill=False, ec=vis, lw=LW * 2.2))
        y_s = y0 + img_h + gap
        y_a = y_s + row_sep
        strip(ax, o, xs, y_s, CMAP["proprio"], cell)
        strip(ax, a, xs, y_a, CMAP["action"], cell, arrows=True)
        for y, lab, key in ((y0 + img_h / 2, "image", "vision"), (y_s + cell / 2, "proprio", "proprio"),
                            (y_a + cell / 2, "action", "action")):
            ax.text(-12, y, lab, ha="right", va="center", fontsize=11,
                    color=plt.get_cmap(CMAP[key])(0.9 if key != "vision" else 0.82))
        ax.annotate("", xy=(xs[-1] + IMG_W, y_a + cell + 40), xytext=(0, y_a + cell + 40),
                    arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#4d4d4d"))
        ax.text(xs[-1] + IMG_W, y_a + cell + 36, "Time", ha="right", va="bottom", fontsize=10,
                color="#4d4d4d")

    ax.set_xlim(-130, N * colw + 20); ax.set_ylim(len(SEQ) * seq_h - 14, -14)
    fig.savefig(OUT, dpi=DPI, bbox_inches="tight")
    import os
    print(f"  wrote {OUT}  ({os.path.getsize(OUT) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
