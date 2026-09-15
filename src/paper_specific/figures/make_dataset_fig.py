"""figures/dataset.png -- what one training sample IS, drawn with the same primitives as the architecture.

Four consecutive model steps left to right. Each column carries the camera frame, the 16-dimensional
proprioceptive vector and the 4-dimensional commanded action, in the three modality hues the architecture
figure uses (vision green, proprioception orange, action blue) so a reader can carry the colours between
the two figures. Cells are normalised PER DIMENSION across the four columns, so colour shows how that
dimension changed over the span rather than which dimensions are large -- the same rule as
make_sequence_figs.

Real data: the same starling-2 recording and the same `concat` reconstruction of the action.

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
N, END = 4, 304                       # columns, and the frame the last column sits on
CELL, LW, EDGE = 17.0, 0.9, "#000000"
IMG_W = 300.0
CMAP = {"vision": "Greens", "proprio": "Oranges", "action": "Blues"}
CMAP_LO, CMAP_HI = 0.20, 0.95

tab = pq.read_table(sorted(glob.glob(f"{ROOT}/train/data/**/*.parquet", recursive=True))[0]).to_pydict()
ep = np.asarray(tab["episode_index"]); keep = np.flatnonzero(ep == ep[0])
obs = np.stack([np.asarray(x, np.float32) for x in tab["observation_vector"]])[keep]
act = np.stack([np.asarray(x, np.float32) for x in tab["action"]])[keep]
idx = [END - (N - 1 - i) * STRIDE for i in range(N)]
vid = sorted(glob.glob(f"{ROOT}/train/videos/**/*.mp4", recursive=True))[0]
allf = [f for k, f in enumerate(iio3.imiter(vid, plugin="pyav")) if k <= idx[-1]]
FRAMES = [allf[i] for i in idx]
OBS, ACT = obs[idx], act[idx]
print(f"{N} columns | frames {idx} | {STRIDE / FPS * 1000:.0f} ms apart | obs {OBS.shape} act {ACT.shape}")


def strip(ax, mat, xs, y, cmap, cell, arrows=False):
    """One ROW of D cells per column, normalised per dimension across the N columns."""
    lo, hi = mat.min(0, keepdims=True), mat.max(0, keepdims=True)
    flat = (hi - lo) < 1e-9
    u = np.where(flat, 0.5, (mat - lo) / np.where(flat, 1.0, hi - lo))
    cm = plt.get_cmap(cmap)
    for i in range(mat.shape[0]):
        for k in range(mat.shape[1]):
            cx = xs[i] + k * cell
            ax.add_patch(Rectangle((cx, y), cell, cell, lw=LW, ec=EDGE,
                                   fc=cm(CMAP_LO + (CMAP_HI - CMAP_LO) * u[i, k])))
            if arrows:
                # the RAW command, not its normalisation: direction is the stick's sign and length is
                # how far it is pushed, so a centred stick draws nothing
                v = mat[i, k] / max(np.abs(mat).max(), 1e-9)
                dy = -v * cell * 0.72
                if abs(dy) > cell * 0.08:
                    ax.add_patch(FancyArrow(cx + cell / 2, y + cell / 2 - dy / 2, 0.0, dy,
                                            width=LW * 0.8, head_width=cell * 0.26,
                                            head_length=cell * 0.22, length_includes_head=True,
                                            fc=EDGE, ec=EDGE))
        ax.add_patch(Rectangle((xs[i], y), mat.shape[1] * cell, cell, fill=False, ec=EDGE, lw=LW))


def main() -> int:
    img_h = IMG_W * FRAMES[0].shape[0] / FRAMES[0].shape[1]
    s_cell, a_cell = IMG_W / 16.0, IMG_W / 16.0 * 2          # 16 state cells span the frame; 4 action
    gap, colw = 16.0, IMG_W + 34.0                           #   cells are drawn twice as wide
    xs = [i * colw for i in range(N)]
    y_state = img_h + gap
    y_act = y_state + s_cell + gap * 0.6
    y_bot = y_act + a_cell
    fig = plt.figure(figsize=(7.0, 7.0 * (y_bot + 60) / (N * colw + 170)))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    vis = plt.get_cmap(CMAP["vision"])(0.78)

    for i, x in enumerate(xs):                                   # the camera frame
        ax.imshow(FRAMES[i], extent=(x, x + IMG_W, img_h, 0.0), interpolation="bilinear")
        ax.add_patch(Rectangle((x, 0), IMG_W, img_h, fill=False, ec=vis, lw=LW * 2.2))
        ax.text(x + IMG_W / 2, -8, "$t$" if i == N - 1 else f"$t{i - N + 1}$",
                ha="center", va="bottom", fontsize=13)

    strip(ax, OBS, xs, y_state, CMAP["proprio"], s_cell)
    strip(ax, ACT, [x + (IMG_W - 4 * a_cell) / 2 for x in xs], y_act, CMAP["action"], a_cell, arrows=True)
    ax.text(-12, y_state + s_cell / 2, "state", ha="right", va="center", fontsize=12,
            color=plt.get_cmap(CMAP["proprio"])(0.9))
    ax.text(-12, y_act + a_cell / 2, "action", ha="right", va="center", fontsize=12,
            color=plt.get_cmap(CMAP["action"])(0.9))

    ax.set_xlim(-130, N * colw + 20); ax.set_ylim(y_bot + 16, -34)
    fig.savefig(OUT, dpi=450, bbox_inches="tight")
    print("  wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
