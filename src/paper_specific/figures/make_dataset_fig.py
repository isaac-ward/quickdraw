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
N, K = 6, 8                           # columns per sequence, and MODEL STEPS between columns: 8 steps is
#                                       2.1 s, so six columns cover 10.7 s and the scene actually moves
ENDS = (304, 1664)                    # the frame each sequence ends on. The second was chosen by
#                                       scratch/_pick_seq.py as the most visually diverse window in the
#                                       episode (mean pairwise L1 between the six shown frames).
IMG_W = 300.0                         # image width in figure units
CELL = IMG_W / 8.0                    # TWICE the old cell: the 16-vector is drawn as 8 x 2 rather than
#                                       16 x 1, which doubles the cell without widening the column
LW, EDGE = 1.4, "#000000"   # 0.0375 x CELL, which is figure 4's LW/CELL ratio exactly
IMG_LW = 0.9                          # matched to the architecture figure RELATIVE TO THE IMAGE: its
#                                       outline is 2.4 pt on a 3 in image, and here the image is 1.1 in
CMAP = {"vision": "Greens", "proprio": "Oranges", "action": "Blues"}
CMAP_LO, CMAP_HI = 0.20, 0.95         # the window make_sequence_figs uses; a white cell reads as empty
# NO SEPARATE ACTION WINDOW: figure 4 draws all three streams through CMAP_LO..CMAP_HI, and a different
# window here made the same data look like a different figure.
# make_sequence_figs draws the architecture on a 7663 px canvas for one \textwidth (7.16 in) = 1070
# px/in. This figure is one \columnwidth (3.4 in) wide on the page, so it needs ~3640 px to match.
DPI = 520

tab = pq.read_table(sorted(glob.glob(f"{ROOT}/train/data/**/*.parquet", recursive=True))[0]).to_pydict()
ep = np.asarray(tab["episode_index"]); keep = np.flatnonzero(ep == ep[0])
obs = np.stack([np.asarray(x, np.float32) for x in tab["observation_vector"]])[keep]
act = np.stack([np.asarray(x, np.float32) for x in tab["action"]])[keep]
vid = sorted(glob.glob(f"{ROOT}/train/videos/**/*.mp4", recursive=True))[0]
IDX = [[e - (N - 1 - i) * K * STRIDE for i in range(N)] for e in ENDS]
allf = [f for k, f in enumerate(iio3.imiter(vid, plugin="pyav")) if k <= max(max(x) for x in IDX)]
SEQ = [([allf[i] for i in ix], obs[ix], act[ix]) for ix in IDX]
print(f"{len(SEQ)} sequences x {N} steps, {K * STRIDE / FPS:.1f} s apart, span "
      f"{(N - 1) * K * STRIDE / FPS:.1f} s | frames {IDX}")


def cells(ax, mat, x0, y0, cmap, rows, lo=CMAP_LO, needles=False, lw=LW):
    """mat (N, D) -> one block of `rows` x D/rows cells per step, left-aligned at x0[i].

    Normalised PER DIMENSION across the sequence, so colour shows how that dimension changed rather than
    which dimensions are large -- the rule make_sequence_figs uses."""
    n, D = mat.shape
    per = D // rows
    m_lo, m_hi = mat.min(0, keepdims=True), mat.max(0, keepdims=True)
    flat = (m_hi - m_lo) < 1e-9
    u = np.where(flat, 0.5, (mat - m_lo) / np.where(flat, 1.0, m_hi - m_lo))
    cm = plt.get_cmap(cmap)
    rng = np.abs(mat).max(axis=0, keepdims=True)
    for i in range(n):
        for k in range(D):
            r, c = divmod(k, per)
            cx, cy = x0[i] + c * CELL, y0 + r * CELL
            ax.add_patch(Rectangle((cx, cy), CELL, CELL, lw=lw, ec=EDGE,
                                   fc=cm(lo + (CMAP_HI - lo) * u[i, k])))
            if needles:
                # A NEEDLE, NOT AN UP/DOWN ARROW: the angle sweeps from straight up (stick fully
                # positive) through horizontal (centred) to straight down (fully negative), so a
                # part-pushed stick reads as a diagonal and every cell carries a direction. SNAPPED to
                # 45 degrees -- eight directions read at a glance where a continuous angle does not.
                v = float(mat[i, k] / max(float(rng[0, k]), 1e-9))
                q = np.pi / 4.0
                ang = round((1.0 - v) * np.pi / 2.0 / q) * q
                L = CELL * 0.62
                dx, dy = L * np.sin(ang), -L * np.cos(ang)
                ax.add_patch(FancyArrow(cx + CELL / 2 - dx / 2, cy + CELL / 2 - dy / 2, dx, dy,
                                        width=LW * 1.1, head_width=CELL * 0.32,
                                        head_length=CELL * 0.32, length_includes_head=True,
                                        fc=EDGE, ec=EDGE, lw=0.0))
        ax.add_patch(Rectangle((x0[i], y0), per * CELL, rows * CELL, fill=False, ec=EDGE, lw=lw))


def main() -> int:
    img_h = IMG_W * SEQ[0][0][0].shape[0] / SEQ[0][0][0].shape[1]
    gap, colw = 12.0, IMG_W + 10.0            # columns nearly touch: the room buys the bigger cells
    seq_h = img_h + 2 * gap + 3 * CELL + 64.0  # image, gap, 2 proprio rows, THE SAME gap, action row
    xs = [i * colw for i in range(N)]
    vis = plt.get_cmap(CMAP["vision"])(0.78)
    fig = plt.figure(figsize=(7.0, 7.0 * (len(SEQ) * seq_h) / (N * colw + 30)))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()

    for si, (frames, o, a) in enumerate(SEQ):
        y0 = si * seq_h
        for i, x in enumerate(xs):
            ax.imshow(frames[i], extent=(x, x + IMG_W, y0 + img_h, y0), interpolation="bilinear")
            ax.add_patch(Rectangle((x, y0), IMG_W, img_h, fill=False, ec=vis, lw=IMG_LW / 2))
        y_s = y0 + img_h + gap
        cells(ax, o, xs, y_s, CMAP["proprio"], rows=2, lw=LW / 2)
        cells(ax, a, xs, y_s + 2 * CELL + gap, CMAP["action"], rows=1, needles=True, lw=LW / 2)
        y_t = y_s + 3 * CELL + gap + 30
        ax.annotate("", xy=(xs[-1] + IMG_W, y_t), xytext=(0, y_t),
                    arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#4d4d4d"))
        ax.text(xs[-1] + IMG_W + 14, y_t, "$t$", ha="left", va="center", fontsize=12, color="#4d4d4d")

    ax.set_xlim(-8, N * colw + 34); ax.set_ylim(len(SEQ) * seq_h - 22, -10)
    fig.savefig(OUT, dpi=DPI, bbox_inches="tight")
    import os
    print(f"  wrote {OUT}  ({os.path.getsize(OUT) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
