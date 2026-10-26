"""Time-aligned GIF triples of the drone actually flying, one set per candidate window.

    sync_gifs/fpv_<x>.gif          the onboard camera, framed in the figure's VISION green
    sync_gifs/controller_<x>.gif   the two analog sticks, gates and knobs in the ACTION blue
    sync_gifs/proprio_<x>.gif      the 7 drawn observation dims as stacked traces, in the PROPRIO orange

ONE TIME BASE, SHARED. The three files of a set sample the same frame indices at the same output rate, so
dropping them in a timeline beside each other needs no alignment work: frame k of one is frame k of the
others. Different sets are different flights and are NOT interchangeable -- use a matching suffix.

THE WINDOWS ARE NOT THE FIGURE'S. make_sequence_figs anchors its cascade at frame 128 of episode 0, where
the drone is 20-39 cm off the floor and barely moving -- true to the figure, useless as "here is a drone
flying around a room". These windows come from a scan of all 106,708 recorded steps for 15 s stretches
that stay above 0.8 m and cover the most ground, capped at one per episode so the options are different
flights rather than one flight sliced six ways.

20 fps, NOT the source's 15. A GIF frame delay is an integer number of centiseconds: 15 fps is 6.67 cs, so
every player rounds it to 6 or 7 and the three drift apart at different rates. 20 fps is exactly 5 cs. The
extra frames are duplicates in time and cost almost nothing -- an identical frame is a cheap GIF delta.

    python src/paper_specific/figures/make_sync_gifs.py
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess

import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                    # noqa: E402
import numpy as np                                                                 # noqa: E402
import pyarrow.parquet as pq                                                       # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch                              # noqa: E402

from quickdraw.data.rosbag import STATE_COLUMNS                                    # noqa: E402

ROOT = glob.glob("/app/scratch/recording_*_starling-2")[0]
RUN = sorted(glob.glob("/app/logs/train_world_model_*_s2_sub4_concat"))[-1]
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_sequence_figs.py")
OUT = "/app/logs/paper_icra_2027/sync_gifs"
TMP = "/app/logs/paper_icra_2027/_gif_frames"

SRC_FPS, GIF_FPS, SECONDS = 15.0, 20, 15.0
SRC_W, SRC_H = 192, 112             # the onboard camera, asserted against the file below
BORDER = 10                         # the green frame around the camera feed
# UPSCALE IS THE ONLY SIZE KNOB THAT MATTERS. A GIF stores whole frames, so its weight is the pixel count:
# measured on this clip, 2x is 10.5 MB and 3x is 20.9 MB for identical content, and the source is 192x112,
# so 3x invents nothing. (Frame rate is NOT a lever -- 20 fps costs 0.01 MB over 15 fps, because the extra
# frames are exact duplicates and a duplicate is a near-empty GIF delta.)
UPSCALE = 2
TILE_W, TILE_H = SRC_W * UPSCALE + 2 * BORDER, SRC_H * UPSCALE + 2 * BORDER

# Chosen off the scan described above; `path` is the L1 ground track over the 15 s, `z` the mean altitude.
WINDOWS = {"a": 93600,      # ep 52   path 47 m   z 1.4 m
           "b": 82845,      # ep 46   path 46 m   z 1.5 m
           "c": 96480,      # ep 54   path 46 m   z 1.3 m
           "d": 10125,      # ep  5   path 45 m   z 1.6 m
           "e": 95175,      # ep 53   path 44 m   z 2.6 m -- near the ceiling, a different look
           "f": 68220}      # ep 38   path 43 m   z 1.4 m

# TEXT IS SIZED FOR A SLIDE, and these are small canvases, so what matters is size RELATIVE to the frame:
# a 12 pt label on a 244 px tall tile is ~7% of its height, which survives any projector once the GIF is
# scaled up. Absolute point size here is meaningless on its own.
FS_LABEL, FS_TICK = 12.0, 10.0

# The figure's three modality hues, taken the same way palette() takes them: cmap at 0.78.
VIS, PRO, ACT = (matplotlib.colors.to_hex(plt.get_cmap(c)(0.78)) for c in ("Greens", "Oranges", "Blues"))
GREY_BODY, GREY_EDGE, GREY_KNOB, GREY_TEXT = "#ececec", "#8a8a8a", "#d9d9d9", "#555555"

# STICK MAPPING, MEASURED NOT ASSUMED. /joy ships four unnamed axes (rosbag.ACTION_COLUMNS is joy_axis_0..3)
# so the pairing was recovered by correlating each axis against the flight state over all 106,708 recorded
# steps. It comes out as textbook Mode 2:
#     axis 0 vs yaw rate            r = +0.85     left stick X   yaw
#     axis 1 vs climb velocity      r = -0.73     left stick Y   throttle   (negated: up = climb)
#     axis 2 vs lateral velocity    r = +0.69     right stick X  roll
#     axis 3 vs forward velocity    r = -0.70     right stick Y  pitch      (negated: up = forward)
# Nothing is re-centred. axis 3 sits near -0.43 for the whole dataset because the pilot held forward pitch,
# not because of a trim offset, and a stick that rests off centre is the truth about how this was flown.
# The axis->stick assignment below is exactly the one the user described (right stick translates, left
# stick rotates and climbs); only the LABELS changed, from control jargon to what the pilot is doing.
STICKS = [((0, 1.0), (1, -1.0), "rotate left / right", "ascend / descend"),
          ((2, 1.0), (3, -1.0), "move left / right", "forward / back")]

# How far the knob travels. The 99.5th percentile is the honest full-scale of the axis, but the pilot
# almost never goes there, so the knobs barely left the centre and the GIF read as a still. 90 puts
# ordinary flying against the edge of the gate; the strongest ~10% of inputs clip there instead.
STICK_PCT = 90


def _const(name):
    return int(re.search(rf"^{name}\s*=\s*(\d+)", open(FIG).read(), re.M).group(1))


def ffmpeg_gif(frames_dir, path, *, photographic):
    """palettegen + paletteuse in one pass. A GIF is 256 colours: letting the encoder pick them per clip
    (stats_mode=diff weights what CHANGES) is the difference between a usable camera feed and mud.
    Flat graphics get no dithering -- it only adds noise, and noise is what makes a GIF big."""
    pal = "palettegen=stats_mode=diff" + ("" if photographic else ":max_colors=64")
    use = "paletteuse=diff_mode=rectangle:dither=" + ("bayer:bayer_scale=5" if photographic else "none")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
                    "-framerate", str(GIF_FPS), "-i", f"{frames_dir}/%05d.png",
                    "-filter_complex", f"[0:v]split[a][b];[a]{pal}[p];[b][p]{use}",
                    "-loop", "0", path], check=True)
    print(f"  {os.path.basename(path):16s} {os.path.getsize(path) / 1e6:5.1f} MB")


def dump(frames, name, *, photographic, out_dir=None):
    import imageio.v2 as imageio
    d = f"{TMP}/{name}"
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    for k, f in enumerate(frames):
        imageio.imwrite(f"{d}/{k:05d}.png", f)
    ffmpeg_gif(d, f"{out_dir or OUT}/{name}.gif", photographic=photographic)
    shutil.rmtree(d, ignore_errors=True)


def fig_rgb(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def controller_frames(act, idx, scales):
    """Two gates, two knobs, no controller body. The body was a grey rounded box around them; it added no
    information and ate the room the labels need at slide size."""
    fig = plt.figure(figsize=(TILE_W / 100, TILE_H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 2); ax.set_ylim(0, 2 * TILE_H / TILE_W)
    ax.axis("off"); fig.patch.set_facecolor("white")
    cx, cy, R, KR = [0.56, 1.44], 0.78, 0.30, 0.105
    knobs = []
    for s_i, (_, _, xl, yl) in enumerate(STICKS):
        ax.add_patch(Circle((cx[s_i], cy), R, fc="white", ec=ACT, lw=3.0))
        ax.plot([cx[s_i] - R, cx[s_i] + R], [cy, cy], color=GREY_EDGE, lw=0.9, zorder=2)
        ax.plot([cx[s_i], cx[s_i]], [cy - R, cy + R], color=GREY_EDGE, lw=0.9, zorder=2)
        k = Circle((cx[s_i], cy), KR, fc=GREY_KNOB, ec=ACT, lw=2.8, zorder=3)
        ax.add_patch(k); knobs.append(k)
        # The arrow says WHICH axis of the stick each phrase belongs to, so the pair is unambiguous
        # without labelling the gate's four compass points.
        ax.text(cx[s_i], cy - R - 0.09, f"\u2194  {xl}", ha="center", va="top",
                fontsize=FS_LABEL, color=GREY_TEXT)
        ax.text(cx[s_i], cy - R - 0.26, f"\u2195  {yl}", ha="center", va="top",
                fontsize=FS_LABEL, color=GREY_TEXT)
    out = []
    for i in idx:
        for s_i, ((xa, xs), (ya, ys), _, _) in enumerate(STICKS):
            u = np.clip(act[i, xa] * xs / scales[s_i], -1, 1)
            v = np.clip(act[i, ya] * ys / scales[s_i], -1, 1)
            knobs[s_i].center = (cx[s_i] + u * (R - KR), cy + v * (R - KR))
        out.append(fig_rgb(fig))
    plt.close(fig)
    return out


def proprio_frames(series, cells, n_out):
    """One panel per drawn observation dim, each with ITS OWN y axis. The dims are metres, metres per
    second and a bare quaternion component, so a shared scale would be meaningless; ticks per panel say
    what the trace is worth without the reader having to take it on trust."""
    from matplotlib.ticker import MaxNLocator
    t = np.arange(n_out) / GIF_FPS
    fig, axes = plt.subplots(cells, 1, figsize=(2 * TILE_W / 100, 7.6), dpi=100, sharex=True)
    fig.patch.set_facecolor("white")
    # A little air between panels. Flush-stacked, each panel's top tick label lands on its neighbour's
    # bottom one ("0.32" sitting on "-2"), which is exactly the kind of collision per-panel y axes invite.
    fig.subplots_adjust(left=0.30, right=0.985, top=0.99, bottom=0.075, hspace=0.12)
    lines, dots = [], []
    for c, a in enumerate(axes):
        # Limits off the 1st/99th percentile, not min/max. The velocities are a gradient of mocap
        # position, so a single dropped sample becomes a two-frame +-5 m/s spike; scaling to it flattens
        # the fifteen seconds of real motion into a straight line. The spike still draws, clipped by the
        # panel edge -- it is not hidden, just stopped from owning the axis.
        lo, hi = (float(v) for v in np.percentile(series[:, c], [1, 99]))
        pad = max(1e-3, 0.14 * (hi - lo))
        a.set_ylim(lo - pad, hi + pad); a.set_xlim(0, SECONDS)
        a.yaxis.set_major_locator(MaxNLocator(3, prune="both"))
        a.set_ylabel(STATE_COLUMNS[c].replace("_", " "), fontsize=FS_LABEL, color=GREY_TEXT,
                     rotation=0, ha="right", va="center", labelpad=10)
        last = c == cells - 1
        a.tick_params(which="both", labelsize=FS_TICK, colors=GREY_TEXT, left=True, right=False,
                      labelleft=True, bottom=last, labelbottom=last, length=3)
        a.grid(axis="x", alpha=0.35, lw=0.7)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.spines["left"].set_color(GREY_EDGE)
        a.spines["bottom"].set_color(GREY_EDGE if not last else GREY_TEXT)
        lines.append(a.plot([], [], color=PRO, lw=2.0)[0])
        dots.append(a.plot([], [], "o", color=PRO, ms=5.0)[0])
    axes[-1].set_xlabel("seconds", fontsize=FS_LABEL, color=GREY_TEXT)
    out = []
    for k in range(n_out):
        for c in range(cells):
            lines[c].set_data(t[:k + 1], series[:k + 1, c])
            dots[c].set_data(t[k:k + 1], series[k:k + 1, c])
        out.append(fig_rgb(fig))
    plt.close(fig)
    return out


def main() -> int:
    """Argv narrows the rebuild: component names (fpv / controller / proprio) and/or window letters."""
    import sys
    import cv2
    import imageio.v3 as iio3
    only = set(sys.argv[1:])
    parts = {c for c in ("fpv", "controller", "proprio") if not only or c in only} or \
            {"fpv", "controller", "proprio"}
    wins = {k for k in WINDOWS if not only & set(WINDOWS) or k in only}
    os.makedirs(OUT, exist_ok=True)
    CELLS = _const("PROPRIO_CELLS")                      # still the figure's: these are ITS seven dims
    n_out = int(round(SECONDS * GIF_FPS))
    span = int(round(SECONDS * SRC_FPS))

    tab = pq.read_table(sorted(glob.glob(f"{ROOT}/train/data/**/*.parquet", recursive=True))[0]).to_pydict()
    ep = np.asarray(tab["episode_index"])
    obs = np.stack([np.asarray(x, np.float32) for x in tab["observation_vector"]])
    act = np.stack([np.asarray(x, np.float32) for x in tab["action"]])
    scales = [max(np.percentile(np.abs(act[:, a]), STICK_PCT) for a, _ in (xa, ya))
              for xa, ya, _, _ in STICKS]
    print(f"  stick full-scale at p{STICK_PCT}: " + ", ".join(f"{v:.3f}" for v in scales))

    # One shared time base per window: row index == video frame index, the file is in dataset row order.
    bases = {}
    for k, s0 in sorted((k, v) for k, v in WINDOWS.items() if k in wins):
        idx = s0 + np.round(np.arange(n_out) / GIF_FPS * SRC_FPS).astype(int)
        assert ep[s0] == ep[s0 + span - 1], f"window {k} crosses an episode boundary"
        bases[k] = idx

    # ONE sequential pass over the 106,708-frame file for every window at once. Seeking in this file is
    # not free and there are six windows scattered through two hours of flight.
    raw, green = {}, tuple(int(255 * c) for c in matplotlib.colors.to_rgb(VIS))
    if "fpv" in parts:
        need = {int(i) for idx in bases.values() for i in idx}
        stop = max(need)
        vid = sorted(glob.glob(f"{ROOT}/train/videos/**/*.mp4", recursive=True))[0]
        for j, f in enumerate(iio3.imiter(vid, plugin="pyav")):
            if j in need:
                raw[j] = f
            if j >= stop:
                break
        assert raw[stop].shape[:2] == (SRC_H, SRC_W), f"camera is {raw[stop].shape[:2]}, not {(SRC_H, SRC_W)}"

    for k, idx in bases.items():
        print(f"  [{k}] ep{int(ep[idx[0]]):3d} frames {idx[0]}..{idx[-1]}")
        if "fpv" in parts:
            fpv = []
            for i in idx:
                im = cv2.resize(raw[int(i)], (TILE_W - 2 * BORDER, TILE_H - 2 * BORDER),
                                interpolation=cv2.INTER_LANCZOS4)
                f = np.full((TILE_H, TILE_W, 3), green, np.uint8)
                f[BORDER:-BORDER, BORDER:-BORDER] = im
                fpv.append(f)
            dump(fpv, f"fpv_{k}", photographic=True)
        if "controller" in parts:
            dump(controller_frames(act, idx, scales), f"controller_{k}", photographic=False)
        if "proprio" in parts:
            dump(proprio_frames(obs[idx][:, :CELLS], CELLS, n_out), f"proprio_{k}", photographic=False)

    shutil.rmtree(TMP, ignore_errors=True)
    print(f"  every set is {n_out} frames at {GIF_FPS} fps = {n_out / GIF_FPS:.1f}s, frame-for-frame aligned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
