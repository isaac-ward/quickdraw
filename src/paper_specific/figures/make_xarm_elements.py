"""The four data streams of the xArm 7 block-stack corpus, drawn in the paper figure's own language.

    xarm7/elements/<name>.png      one stream, transparent, on the SHARED canvas -- they overlay
    xarm7/sync_gifs/<name>.png     the same stream cropped to its own ink at CLOSEUP_X the pixels
    xarm7/sync_gifs/_streams.png   all four on one labelled sheet, for choosing

THE CLOSE-UPS LIVE BESIDE THE GIFS, at the author's ask: one folder holds everything this data stream
has to show -- the four animated views and the four still cascades they were cut from. `elements/` keeps
the shared-canvas copies, which are the ones that overlay each other and are useless on their own.

    items_cam_scene    the fixed third-person camera (scene_right)      144 x 192 x 3
    items_cam_wrist    the gripper-mounted camera (gripper_right_top)   144 x 192 x 3
    items_proprio      the observation vector                           17
    items_action       the Xbox pad                                     5

THE CELL SIZE IS DERIVED, NOT CHOSEN. A cell is (image height / 17), so the 17-D observation column is
exactly as tall as an image item and the 5-D action column is 5/17 of it. That ratio IS the data format --
sizing the two vectors independently would have them lie about their own lengths, which is the one thing
this figure is for. (The starling version drew 7 of 16 observation dims as a schematic; this one draws
every dim it has.)

TWO IMAGE STREAMS, same hue. Green means VISION in this figure's colour code, and a second green would
say the gripper camera is a different modality. What tells them apart is what is in them.

    python src/paper_specific/figures/make_xarm_elements.py
"""
from __future__ import annotations

import io
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                    # noqa: E402
import numpy as np                                                                 # noqa: E402
from matplotlib.patches import FancyArrow, Rectangle                               # noqa: E402
from matplotlib.transforms import Bbox                                             # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xarm_stream as X                                                            # noqa: E402

ROOT = "/app/logs/paper_icra_2027/xarm7"
DPI = 300
CLOSEUP_X = 8

# THE SAME WINDOW make_xarm_gifs calls `a` (episode 29, the one with real grasping in it), so the stills
# and the animations are the same few seconds of the same run.
ROW0 = 84723
N = 8                       # the model's own window, P = 8
ITEM_STEP = 30              # rows between items = 1.0 s at 30 Hz: far enough apart that the arm moves

OFFSET_X, OFFSET_Y = 48.0, 24.0      # the cascade step, and its half-step drop -- the figure's angle
IMG_W = 300.0
IMG_H = IMG_W * X.H / X.W            # 225: the frame is 4:3, and squashing it to square is the one
#                                      mistake every config in this repo warns about
CELL = IMG_H / len(X.STATE_KEYS)     # see the header
MARGIN = 16.0
GAP = 70.0                           # between one stream's band and the next
LW = 1.2
EDGE = "#000000"
ALPHA_LO = 0.32                      # oldest item's opacity; newest is 1.0
CMAP_LO, CMAP_HI = 0.20, 0.95        # a sequential map's bottom end is white, and white reads as empty
MOD_CMAP = {"vision": "Greens", "proprio": "Oranges", "action": "Blues"}
VIS_EC = plt.get_cmap(MOD_CMAP["vision"])(0.78)


def cascade_h(ph):
    return ph + (N - 1) * OFFSET_Y


def vector_drawer(mat, cmap, arrows=False):
    """PER-DIM normalisation across the N drawn items, so a cell's colour says how that dim CHANGED over
    the window rather than which dims happen to be large -- `ee_x_mm` is in the hundreds and `gripper` is
    0..1, and one shared scale would paint the gripper column a single flat colour."""
    lo, hi = mat.min(0, keepdims=True), mat.max(0, keepdims=True)
    flat = (hi - lo) < 1e-9
    norm = np.where(flat, 0.5, (mat - lo) / np.where(flat, 1.0, hi - lo))
    cm = plt.get_cmap(cmap)
    D = mat.shape[1]

    def draw(ax, x, y, i, a, z):
        for k in range(D):
            cy0 = y + k * CELL
            ax.add_patch(Rectangle((x, cy0), CELL, CELL, alpha=a, zorder=z, lw=LW, ec=EDGE,
                                   fc=cm(CMAP_LO + (CMAP_HI - CMAP_LO) * float(norm[i, k]))))
            if arrows:
                # up for a positive command, down for a negative one -- the same convention the starling
                # figure uses. The trigger is unipolar, so its arrow only ever points up, which is true.
                L = CELL * 0.54
                dy = -L if mat[i, k] >= 0.0 else L
                ax.add_patch(FancyArrow(x + CELL / 2, cy0 + CELL / 2 - dy / 2, 0.0, dy,
                                        width=LW * 0.8 * CELL / 32.0, head_width=CELL * 0.30,
                                        head_length=CELL * 0.26, length_includes_head=True,
                                        fc=EDGE, ec=EDGE, alpha=a, zorder=z + 0.4))
        ax.add_patch(Rectangle((x, y), CELL, D * CELL, fill=False, ec=EDGE, lw=LW, alpha=a, zorder=z + 0.5))
    return draw, CELL, D * CELL


def image_drawer(frames):
    def draw(ax, x, y, i, a, z):
        ax.imshow(frames[i], extent=(x, x + IMG_W, y + IMG_H, y), alpha=a, zorder=z,
                  interpolation="bilinear")
        ax.add_patch(Rectangle((x, y), IMG_W, IMG_H, fill=False, ec=VIS_EC, lw=LW * 2, alpha=a,
                               zorder=z + 0.5))
    return draw, IMG_W, IMG_H


def save_closeup(fig, path):
    """Crop to the ink at CLOSEUP_X the pixels WITHOUT rasterising the whole canvas at 8x: measure the
    alpha bbox from a cheap DPI-resolution render and hand it back as an INCHES bbox, which is dpi-free."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, dpi=DPI, pad_inches=0)
    im = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_UNCHANGED)
    rows = np.where(im[:, :, 3].max(axis=1) > 0)[0]
    cols = np.where(im[:, :, 3].max(axis=0) > 0)[0]
    assert len(rows) and len(cols), f"{path}: nothing was drawn"
    h, pad = im.shape[0], 2.0
    fig.savefig(path, transparent=True, dpi=DPI * CLOSEUP_X, pad_inches=0,
                bbox_inches=Bbox.from_extents((cols[0] - pad) / DPI, (h - rows[-1] - 1 - pad) / DPI,
                                              (cols[-1] + 1 + pad) / DPI, (h - rows[0] + pad) / DPI))
    print(f"  {os.path.basename(path):22s} {cols[-1] - cols[0] + 1:5d} x {rows[-1] - rows[0] + 1:5d} px "
          f"of ink -> {CLOSEUP_X}x")


def main() -> int:
    rows = ROW0 + np.arange(N) * ITEM_STEP
    obs, act, ep = X.table()
    eps = X.episodes()
    assert ep[rows[0]] == ep[rows[-1]], "the drawn items cross an episode boundary"
    span = int(rows[-1] - rows[0]) + 1
    cams = {c: X.frames(c, ROW0, span, eps)[rows - ROW0] for c in X.CAMS}
    print(f"  ep {int(ep[ROW0])}, rows {rows[0]}..{rows[-1]}, {N} items {ITEM_STEP / X.FPS:.1f} s apart "
          f"| obs {obs[rows].shape} act {act[rows].shape} | cell {CELL:.1f} u")

    LAYERS = [("items_cam_scene", *image_drawer(cams[X.CAMS[0]])),
              ("items_cam_wrist", *image_drawer(cams[X.CAMS[1]])),
              ("items_proprio", *vector_drawer(obs[rows], MOD_CMAP["proprio"])),
              ("items_action", *vector_drawer(act[rows], MOD_CMAP["action"], arrows=True))]

    # ONE CANVAS, one band per stream, so the four PNGs drop on top of each other exactly the way the
    # paper figure's elements do.
    ys, y = [], MARGIN
    for _, _, _, ph in LAYERS:
        ys.append(y)
        y += cascade_h(ph) + GAP
    W = 2 * MARGIN + (N - 1) * OFFSET_X + max(pw for _, _, pw, _ in LAYERS)
    H = y - GAP + MARGIN

    for d in ("elements", "sync_gifs"):
        os.makedirs(f"{ROOT}/{d}", exist_ok=True)
    for (name, draw, pw, ph), y0 in zip(LAYERS, ys):
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=DPI)
        ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H)
        ax.axis("off"); ax.invert_yaxis()
        for i in range(N):                               # oldest first -> newest painted on top
            draw(ax, MARGIN + i * OFFSET_X, y0 + i * OFFSET_Y, i,
                 ALPHA_LO + (1.0 - ALPHA_LO) * i / (N - 1), 10 + i)
        fig.savefig(f"{ROOT}/elements/{name}.png", transparent=True, dpi=DPI, pad_inches=0)
        save_closeup(fig, f"{ROOT}/sync_gifs/{name}.png")
        plt.close(fig)

    # ...and all four at once, labelled, so the set can be judged at a glance
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.axis("off"); ax.invert_yaxis(); fig.patch.set_facecolor("white")
    for (name, draw, pw, ph), y0 in zip(LAYERS, ys):
        for i in range(N):
            draw(ax, MARGIN + i * OFFSET_X, y0 + i * OFFSET_Y, i,
                 ALPHA_LO + (1.0 - ALPHA_LO) * i / (N - 1), 10 + i)
        ax.text(W - MARGIN, y0 + ph / 2, name.replace("items_", ""), ha="right", va="center",
                fontsize=13, color="#444444")
    fig.savefig(f"{ROOT}/sync_gifs/_streams.png", dpi=150, pad_inches=0.02,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  canvas {int(W * DPI / 100)} x {int(H * DPI / 100)} px, identical for every stream")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
