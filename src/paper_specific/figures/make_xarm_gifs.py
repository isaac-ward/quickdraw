"""Time-aligned GIF sets of the xArm 7 block-stack data -- the starling sync_gifs, re-cut for this corpus.

    xarm7/sync_gifs/cam_scene_<x>.gif    the fixed third-person camera, framed in the figure's VISION green
    xarm7/sync_gifs/cam_wrist_<x>.gif    the gripper-mounted camera, same frame, same clock
    xarm7/sync_gifs/controller_<x>.gif   both analog sticks AND the right trigger, labelled, in ACTION blue
    xarm7/sync_gifs/proprio_<x>.gif      7 of the 17 observation dims as labelled traces, in PROPRIO orange

ONE TIME BASE, SHARED. The four files of a set sample the same rows at the same output rate, so frame k of
one is frame k of the others. Different letters are different episodes and are NOT interchangeable.

WHAT IS DIFFERENT FROM THE STARLING SET, and all of it is forced by the data: 30 Hz rather than 15, two
cameras rather than one, a 17-D observation rather than 16, a 5-D action rather than 4, and the fifth
action is a TRIGGER -- unipolar, 0..1 -- so it is drawn as a bar rather than as a third stick.

20 fps, NOT the source's 30. A GIF frame delay is an integer number of centiseconds: 30 fps is 3.33 cs, so
every player rounds it and a set drifts apart at four different rates. 20 fps is exactly 5 cs.

    python src/paper_specific/figures/make_xarm_gifs.py            # everything
    python src/paper_specific/figures/make_xarm_gifs.py a proprio  # one window, one component
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

import cv2
import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors                                                           # noqa: E402
import matplotlib.pyplot as plt                                                    # noqa: E402
import numpy as np                                                                 # noqa: E402
from matplotlib.patches import Circle, Rectangle                                   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xarm_stream as X                                                            # noqa: E402

OUT = "/app/logs/paper_icra_2027/xarm7/sync_gifs"
TMP = "/app/logs/paper_icra_2027/xarm7/_gif_frames"

GIF_FPS, SECONDS = 20, 15.0
BORDER = 10                         # the green frame around a camera feed
UPSCALE = 2                         # the only size knob that matters: a GIF's weight IS its pixel count
TILE_W, TILE_H = X.W * UPSCALE + 2 * BORDER, X.H * UPSCALE + 2 * BORDER     # 404 x 308

# CHOSEN BY A SCAN of all 172,835 rows for the 15 s window with the most end-effector travel plus the most
# gripper movement (weighted 300 mm per unit of gripper travel, so a window that picks something up beats
# one that only waves), capped at one per episode. `a` is the one with real grasping in it.
WINDOWS = {"a":  84723,     # ep 29   ee path 1900 mm   gripper travel 12.6
           "b": 112681,     # ep 32   ee path 1510 mm   gripper travel  5.7
           "c":  48698,     # ep 13   ee path 2950 mm   gripper travel  0.1  -- pure reaching, no grasp
           "d":   1783}     # ep  0   ee path 1504 mm   gripper travel  4.6

FS_LABEL, FS_TICK = 12.0, 10.0
VIS, PRO, ACT = (matplotlib.colors.to_hex(plt.get_cmap(c)(0.78)) for c in ("Greens", "Oranges", "Blues"))
GREY_EDGE, GREY_KNOB, GREY_TEXT = "#8a8a8a", "#d9d9d9", "#555555"

# THE PAD, from the collection repo's own controls table and the dataset's PROCESSING.md section 1:
#   left stick            move the end effector in the plane parallel to the ground
#   right stick up/down   raise / lower that plane
#   right stick L/R       rotate the end effector about the vertical axis
#   right trigger         gripper, proportional
# Which planar axis is the stick's X and which its Y is in neither document, so it was MEASURED over all
# 172,835 steps against the end-effector's own velocity:
#   move_x vs d ee_x  r = +0.89      move_y vs d ee_y  r = +0.89
#   height vs d ee_z  r = +0.76      yaw    vs yaw rate r = +0.92
#   trigger vs gripper state r = -0.51  -- squeezing CLOSES it (1 = open), hence the label
STICKS = [((0, 1.0), (1, 1.0), "left stick", "world +X / -X", "world +Y / -Y"),
          ((3, 1.0), (2, 1.0), "right stick", "rotate about vertical", "raise / lower")]
TRIGGER = (4, "right trigger", "squeeze to\nclose gripper")
# The pad is already post-deadzone and post-expo, so its full scale is 1.0 -- but a teleoperator spends
# almost all of a run near centre, and a knob that never leaves the middle reads as a still image. The
# gate is scaled to the 90th percentile of each stick instead: ordinary driving reaches the edge and the
# strongest ~10% clips there.
STICK_PCT = 90

# SEVEN PANELS, matching the starling set's seven. The three end-effector positions and the gripper say
# what the arm is DOING; the three joints are the three with the most motion across the corpus
# (joint5 0.372, joint6 0.345, joint4 0.335 rad). The 6-D rotation block is left out on purpose: those
# are six components of a rotation matrix, not six angles, and on a strip chart they read as noise.
# All 17 are in X.STATE_KEYS -- this list is the only thing that chooses.
PANELS = ("ee_x_mm", "ee_y_mm", "ee_z_mm", "gripper", "joint4_rad", "joint5_rad", "joint6_rad")


def ffmpeg_gif(frames_dir, path, *, photographic):
    """palettegen + paletteuse in one pass: a GIF is 256 colours, and letting the encoder pick them per
    clip is the difference between a usable camera feed and mud. Flat graphics get no dithering."""
    pal = "palettegen=stats_mode=diff" + ("" if photographic else ":max_colors=64")
    use = "paletteuse=diff_mode=rectangle:dither=" + ("bayer:bayer_scale=5" if photographic else "none")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
                    "-framerate", str(GIF_FPS), "-i", f"{frames_dir}/%05d.png",
                    "-filter_complex", f"[0:v]split[a][b];[a]{pal}[p];[b][p]{use}",
                    "-loop", "0", path], check=True)
    print(f"  {os.path.basename(path):22s} {os.path.getsize(path) / 1e6:5.1f} MB")


def dump(frames, name):
    import imageio.v2 as imageio
    d = f"{TMP}/{name}"
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    for k, f in enumerate(frames):
        imageio.imwrite(f"{d}/{k:05d}.png", f)
    ffmpeg_gif(d, f"{OUT}/{name}.gif", photographic=name.startswith("cam_"))
    shutil.rmtree(d, ignore_errors=True)


def fig_rgb(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def camera_frames(raw):
    green = tuple(int(255 * c) for c in matplotlib.colors.to_rgb(VIS))
    out = []
    for im in raw:
        f = np.full((TILE_H, TILE_W, 3), green, np.uint8)
        f[BORDER:-BORDER, BORDER:-BORDER] = cv2.resize(im, (TILE_W - 2 * BORDER, TILE_H - 2 * BORDER),
                                                       interpolation=cv2.INTER_LANCZOS4)
        out.append(f)
    return out


def controller_frames(act, rows, scales):
    """Two gates and a trigger bar, no controller body. The trigger is NOT a third stick: it only ever
    goes one way, so a knob in a gate would imply a centre it never returns to."""
    # THREE EQUAL SLOTS, one per control, labels centred in their own slot. They were packed tight and
    # the right stick's phrase ran straight into the trigger's -- at slide size the two read as one line.
    w = int(TILE_W * 1.9)
    fig = plt.figure(figsize=(w / 100, TILE_H / 100), dpi=100)
    SLOT = 1.0
    fig_w = 3 * SLOT
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, fig_w); ax.set_ylim(0, fig_w * TILE_H / w)
    ax.axis("off"); fig.patch.set_facecolor("white")
    # cy/R are set so the BOLD TITLES fit: the canvas is three slots wide and only 3*TILE_H/w tall, so
    # a gate centred any higher pushes its own name off the top edge.
    cy, R, KR = 0.70, 0.26, 0.092
    cx = [0.5 * SLOT, 1.5 * SLOT]
    knobs = []
    for s_i, (_a, _b, name, xl, yl) in enumerate(STICKS):
        ax.add_patch(Circle((cx[s_i], cy), R, fc="white", ec=ACT, lw=3.0))
        ax.plot([cx[s_i] - R, cx[s_i] + R], [cy, cy], color=GREY_EDGE, lw=0.9, zorder=2)
        ax.plot([cx[s_i], cx[s_i]], [cy - R, cy + R], color=GREY_EDGE, lw=0.9, zorder=2)
        k = Circle((cx[s_i], cy), KR, fc=GREY_KNOB, ec=ACT, lw=2.8, zorder=3)
        ax.add_patch(k); knobs.append(k)
        ax.text(cx[s_i], cy + R + 0.07, name, ha="center", va="bottom", fontsize=FS_LABEL,
                color=GREY_TEXT, fontweight="bold")
        ax.text(cx[s_i], cy - R - 0.07, f"↔  {xl}", ha="center", va="top",
                fontsize=FS_LABEL, color=GREY_TEXT)
        ax.text(cx[s_i], cy - R - 0.22, f"↕  {yl}", ha="center", va="top",
                fontsize=FS_LABEL, color=GREY_TEXT)
    # the trigger: a bar that fills from the bottom, on the same baseline as the gates
    bx, bw = 2.5 * SLOT - 0.10, 0.20
    ax.add_patch(Rectangle((bx, cy - R), bw, 2 * R, fc="white", ec=ACT, lw=3.0, zorder=2))
    fill = Rectangle((bx, cy - R), bw, 0.0, fc=ACT, ec="none", alpha=0.65, zorder=3)
    ax.add_patch(fill)
    ax.text(bx + bw / 2, cy + R + 0.07, TRIGGER[1], ha="center", va="bottom", fontsize=FS_LABEL,
            color=GREY_TEXT, fontweight="bold")
    ax.text(bx + bw / 2, cy - R - 0.07, TRIGGER[2], ha="center", va="top", fontsize=FS_LABEL,
            color=GREY_TEXT, linespacing=1.35)
    out = []
    for i in rows:
        for s_i, ((xa, xs), (ya, ys), _, _, _) in enumerate(STICKS):
            u = float(np.clip(act[i, xa] * xs / scales[s_i], -1, 1))
            v = float(np.clip(act[i, ya] * ys / scales[s_i], -1, 1))
            knobs[s_i].center = (cx[s_i] + u * (R - KR), cy + v * (R - KR))
        fill.set_height(2 * R * float(np.clip(act[i, TRIGGER[0]], 0, 1)))
        out.append(fig_rgb(fig))
    plt.close(fig)
    return out


def proprio_frames(series, labels, n_out):
    """One panel per drawn dim, each with its OWN y axis and its OWN name on the left. The dims are
    millimetres, a 0-1 gripper opening and radians, so a shared scale would mean nothing."""
    from matplotlib.ticker import MaxNLocator
    t = np.arange(n_out) / GIF_FPS
    n = len(labels)
    fig, axes = plt.subplots(n, 1, figsize=(2 * TILE_W / 100, 7.6), dpi=100, sharex=True)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.27, right=0.985, top=0.99, bottom=0.075, hspace=0.12)
    lines, dots = [], []
    for c, a in enumerate(axes):
        lo, hi = (float(v) for v in np.percentile(series[:, c], [1, 99]))
        pad = max(1e-3, 0.14 * (hi - lo))
        a.set_ylim(lo - pad, hi + pad); a.set_xlim(0, SECONDS)
        a.yaxis.set_major_locator(MaxNLocator(3, prune="both"))
        a.set_ylabel(labels[c].replace("_", " "), fontsize=FS_LABEL, color=GREY_TEXT,
                     rotation=0, ha="right", va="center", labelpad=10)
        last = c == n - 1
        a.tick_params(which="both", labelsize=FS_TICK, colors=GREY_TEXT, left=True, right=False,
                      labelleft=True, bottom=last, labelbottom=last, length=3)
        a.grid(axis="x", alpha=0.35, lw=0.7)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.spines["left"].set_color(GREY_EDGE)
        a.spines["bottom"].set_color(GREY_TEXT if last else GREY_EDGE)
        lines.append(a.plot([], [], color=PRO, lw=2.0)[0])
        dots.append(a.plot([], [], "o", color=PRO, ms=5.0)[0])
    axes[-1].set_xlabel("seconds", fontsize=FS_LABEL, color=GREY_TEXT)
    out = []
    for k in range(n_out):
        for c in range(n):
            lines[c].set_data(t[:k + 1], series[:k + 1, c])
            dots[c].set_data(t[k:k + 1], series[k:k + 1, c])
        out.append(fig_rgb(fig))
    plt.close(fig)
    return out


def main() -> int:
    only = set(sys.argv[1:])
    comps = {c for c in ("cam_scene", "cam_wrist", "controller", "proprio") if not only or c in only} \
        or {"cam_scene", "cam_wrist", "controller", "proprio"}
    wins = {k for k in WINDOWS if not only & set(WINDOWS) or k in only}
    os.makedirs(OUT, exist_ok=True)

    obs, act, ep = X.table()
    eps = X.episodes()
    n_out = int(round(SECONDS * GIF_FPS))
    span = int(round(SECONDS * X.FPS))
    cols = [X.STATE_KEYS.index(p) for p in PANELS]
    scales = [max(np.percentile(np.abs(act[:, a]), STICK_PCT) for a, _ in (xa, ya))
              for xa, ya, _, _, _ in STICKS]
    print(f"  stick full-scale at p{STICK_PCT}: " + ", ".join(f"{v:.3f}" for v in scales))

    for k in sorted(wins):
        r0 = WINDOWS[k]
        assert ep[r0] == ep[r0 + span - 1], f"window {k} crosses an episode boundary"
        rows = r0 + np.round(np.arange(n_out) / GIF_FPS * X.FPS).astype(int)
        print(f"  [{k}] ep{int(ep[r0]):3d}  rows {rows[0]}..{rows[-1]}  ({SECONDS:.0f}s at 30 Hz)")
        for cam, name in zip(X.CAMS, ("cam_scene", "cam_wrist")):
            if name in comps:
                raw = X.frames(cam, r0, span, eps)
                dump(camera_frames(raw[rows - r0]), f"{name}_{k}")
        if "controller" in comps:
            dump(controller_frames(act, rows, scales), f"controller_{k}")
        if "proprio" in comps:
            dump(proprio_frames(obs[rows][:, cols], PANELS, n_out), f"proprio_{k}")

    shutil.rmtree(TMP, ignore_errors=True)
    print(f"  every set is {n_out} frames at {GIF_FPS} fps = {n_out / GIF_FPS:.1f}s, frame-for-frame aligned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
