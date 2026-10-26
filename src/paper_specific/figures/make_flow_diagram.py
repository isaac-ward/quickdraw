"""What the flow head is doing, in 3-D: a data manifold, the field that points onto it, and the
sampler's own trajectory walking in.

    losses/flow_manifold.png

THE GEOMETRY IS SCHEMATIC, THE FRAMES ARE REAL. There is a surface where the data lives, a velocity
field pointing back toward it from BOTH sides, and a path that starts off the surface and lands on it
-- that much is drawn, because a 4,224-dimensional latent has no honest 3-D picture. The images are
not drawn: each is the model's own sampling trajectory decoded at that step (make_flow_path.py).

THE ARROWS ARE BUILT BY HAND, not with ax.quiver, which draws its heads as two line segments. A solid
head is a triangle in the plane spanned by the arrow and the camera's right vector, as a
Poly3DCollection, so it is still depth-ordered against everything else.

EVERY Z ORDER IS EXPLICIT (computed_zorder=False). Matplotlib's 3-D depth sort works per ARTIST, so
the whole field would be ordered against the whole surface -- arrows under the sheet and arrows over it
have to be two artists with two zorders, and the trajectory's markers vanish under the sheet otherwise.

    python src/paper_specific/figures/make_flow_path.py      # the decoded trajectory
    python src/paper_specific/figures/make_flow_diagram.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors                                                           # noqa: E402
import matplotlib.pyplot as plt                                                    # noqa: E402
import numpy as np                                                                 # noqa: E402
from matplotlib.lines import Line2D                                                # noqa: E402
from mpl_toolkits.mplot3d import proj3d                                            # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection          # noqa: E402

OUT = "/app/logs/paper_icra_2027/losses"
DPI = 300
W, H = 22.0, 9.0
# Real frames pulled straight from the recording, hung off points ON the surface: the figure's way of
# saying what the manifold IS. Spread through the flight so they are not four views of one corner.
MANIFOLD = ((1.0, 3.9, 1250), (2.0, 1.2, 2600), (4.8, 0.5, 4100))
SRC_FPS = 15.0
SHEET_FC = plt.get_cmap("Blues")(0.16)
SHEET_EC = plt.get_cmap("Blues")(0.52)
ARROW = plt.get_cmap("Oranges")(0.72)
TARGET = (8.8, 1.9)                   # where the next state lands: the field and the path agree on it
GREY = "#555555"
FS, FS_SM = 20.0, 15.0
TAIL_A = 0.16                         # a shaft fades out toward its tail; the head stays solid
LEADER = 2.4                          # callout line width
DOT = dict(marker="o", ms=13, mfc="white", mec="black", mew=2.6, ls="none")
SHOW = (0, 3, 6)                      # which trajectory points get a decoded frame beside them
ELEV, AZIM = 24.0, -58.0
Z_UNDER, Z_SHEET, Z_OVER, Z_PATH = 1, 3, 6, 8


def surf(x, y):
    return 0.62 * np.sin(0.58 * x) + 0.48 * np.cos(0.46 * y) - 0.22 * np.cos(0.3 * x + 0.4 * y)


def view_dir():
    e, a = np.deg2rad(ELEV), np.deg2rad(AZIM)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def arrow3d(ax, pts, color, z, lw=2.4, head=0.42, half=0.17, a_tail=None):
    """An arrow along an arbitrary polyline, with a SOLID head. The head is a flat triangle turned to
    face the camera -- what a line-pair head cannot be, and what reads at slide size. The shaft is
    trimmed back by `head` so the triangle sits ON the end rather than past it."""
    pts = np.asarray(pts, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    tip = pts[-1]
    i = int(np.clip(np.searchsorted(cum, cum[-1] - head), 1, len(pts) - 1))
    f = (cum[-1] - head - cum[i - 1]) / max(seg[i - 1], 1e-9)
    neck = pts[i - 1] + (pts[i] - pts[i - 1]) * float(np.clip(f, 0.0, 1.0))
    d = tip - neck
    d = d / (np.linalg.norm(d) or 1.0)
    r = np.cross(view_dir(), d)
    nr = np.linalg.norm(r)
    r = np.array([1.0, 0.0, 0.0]) if nr < 1e-6 else r / nr
    shaft = np.vstack([pts[:i], neck])
    # ONE SEGMENT PER ALPHA. A single Line3D takes one alpha, so the fade is a collection whose
    # segments ramp from almost nothing at the tail to solid where the head begins.
    rgb = matplotlib.colors.to_rgb(color)
    segs = [[tuple(shaft[k]), tuple(shaft[k + 1])] for k in range(len(shaft) - 1)]
    # the fade is for the FIELD. The trajectories are single objects and are drawn solid: a history
    # that fades out reads as uncertainty about the past, which is not what is being said.
    a = np.linspace(TAIL_A if a_tail is None else a_tail, 1.0, max(len(segs), 1)) ** 0.55
    lc = Line3DCollection(segs, colors=[rgb + (float(v),) for v in a], linewidths=lw,
                          capstyle="round", zorder=z)
    lc.set_sort_zpos(None)
    ax.add_collection3d(lc)
    tri = Poly3DCollection([[tuple(tip), tuple(neck + r * half), tuple(neck - r * half)]],
                           facecolors=color, edgecolors=color, linewidths=0.4, zorder=z)
    tri.set_sort_zpos(None)
    ax.add_collection3d(tri)


def field_path(a, b, h, bow, n=26, stop=0.78):
    """One streamline of the CONDITIONED field. Every arrow aims at the same place -- the next state on
    the trajectory -- because that is what conditioning on h_t means: the field is not a generic pull
    back onto the data, it carries you to one particular point. They stop short of it by `stop` so the
    heads form a ring rather than a blob, and they bow rather than running straight, so the set reads as
    a flow. It is an illustration; this is not the model's own field."""
    p0 = np.array([a, b, surf(a, b) + h])
    T = np.array([TARGET[0], TARGET[1], surf(*TARGET)])
    d = T - p0
    L = float(np.linalg.norm(d)) or 1.0
    k = np.linspace(0.0, max(0.0, 1.0 - stop / L), n)
    perp = np.cross(d, np.array([0.0, 0.0, 1.0]))
    nn = float(np.linalg.norm(perp))
    perp = perp / nn if nn > 1e-6 else np.array([1.0, 0.0, 0.0])
    off = bow * np.sin(np.pi * k / max(k[-1], 1e-6))
    return p0[None, :] + d[None, :] * k[:, None] + perp[None, :] * off[:, None]


def grab(indices):
    """Single frames from the starling recording, by index. `-ss` rather than a sequential read: these
    are thousands of frames apart and decoding up to each one costs minutes."""
    import glob
    import subprocess
    import imageio_ffmpeg
    vid = sorted(glob.glob(glob.glob("/app/scratch/recording_*_starling-2")[0]
                           + "/train/videos/**/*.mp4", recursive=True))[0]
    out = []
    for i in indices:
        raw = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
                              "-ss", f"{i / SRC_FPS:.6f}", "-i", vid, "-frames:v", "1",
                              "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                             capture_output=True, check=True).stdout
        a = np.frombuffer(raw, np.uint8)
        out.append(a.reshape(112, 192, 3))
    return out


def inset(fig, img, x, y, w, h, anchor, side, label=None):
    """One borderless frame, with a hairline back to the point it belongs to, and the step's own tau
    set HALFWAY ALONG that line -- beside the dot it crowded the convergence; on the leader it names
    the pairing it belongs to and nothing else."""
    a = fig.add_axes([x, y, w, h])
    a.imshow(img); a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():          # no frame: a border read as a modality colour it is not
        sp.set_visible(False)
    fig.add_artist(Line2D([anchor[0], x + (w if side == "left" else 0.0)],
                          [anchor[1], y + h / 2], transform=fig.transFigure,
                          color=GREY, lw=LEADER, zorder=5))
    # the point goes ON TOP of its own leader. The line is a FIGURE artist and so paints over the whole
    # 3-D axes; redrawing the marker at the same place in figure coordinates puts it back on top.
    fig.add_artist(Line2D([anchor[0]], [anchor[1]], transform=fig.transFigure, zorder=6, **DOT))
    if label:
        ex = x + (w if side == "left" else 0.0)
        fig.text(0.5 * (anchor[0] + ex), 0.5 * (anchor[1] + y + h / 2), label, ha="center",
                 va="center", fontsize=FS, zorder=7,
                 bbox=dict(fc="white", ec="black", lw=1.4, boxstyle="round,pad=0.28"))


def place(anchors, bh, lo=0.012):
    """Stack slots lined up with their own anchors, or evenly if that would overlap."""
    ys = [float(np.clip(a[1] - bh / 2, lo, 1.0 - lo - bh)) for a in anchors]
    if any(ys[i] - ys[i + 1] < bh + 0.010 for i in range(len(ys) - 1)):
        k = len(anchors)
        ys = [lo + (k - 1 - i) * (1.0 - 2 * lo - bh) / max(k - 1, 1) for i in range(k)]
    return ys


def main() -> int:
    d = np.load(f"{OUT}/flow_path.npz")
    img, tau = d["image"], d["tau"]
    n = len(tau)

    fig = plt.figure(figsize=(W, H), dpi=DPI)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.235, -0.09, 0.545, 1.18], projection="3d", computed_zorder=False)
    ax.set_axis_off()
    ax.patch.set_visible(False)       # or the callout lines are painted over by the axes background
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_box_aspect((1.5, 1.05, 0.9))
    ax.set_xlim(0.2, 10.8); ax.set_ylim(0.2, 7.8); ax.set_zlim(-3.0, 5.4)

    # ---- the field, from both sides ---------------------------------------------------------------
    # THE TWO UPPER LAYERS ARE STAGGERED. On the same grid they are collinear, so the shorter arrow
    # hides inside the longer one's shaft and the pair reads as one arrow with a head in its middle.
    # THEY END ON THE TARGET, not merely near it: the heads stop just short so they read as pointing
    # AT the next state rather than overlapping it. Two layers below the surface and one above, because
    # a field that only ever arrives from overhead looks like gravity rather than like a field.
    for h, z, off, bow, nx, ny in ((3.9, Z_OVER, 0.0, 0.95, 3, 3),
                                   (-2.4, Z_UNDER, 0.8, 0.75, 4, 3),
                                   (-4.0, Z_UNDER, 1.9, -0.60, 3, 2)):
        fx, fy = np.meshgrid(np.linspace(1.2 + off, 9.6, nx), np.linspace(1.0 + off * 0.5, 7.0, ny))
        for a, b in zip(fx.ravel(), fy.ravel()):
            if np.hypot(a - TARGET[0], b - TARGET[1]) < 2.3:
                continue
            arrow3d(ax, field_path(a, b, h, bow), ARROW, z)

    # ---- the manifold -----------------------------------------------------------------------------
    gx, gy = np.meshgrid(np.linspace(0, 11, 23), np.linspace(0, 8, 17))
    ax.plot_surface(gx, gy, surf(gx, gy), color=SHEET_FC, edgecolor=SHEET_EC, linewidth=0.55,
                    rstride=1, cstride=1, shade=False, alpha=0.88, zorder=Z_SHEET)

    # ---- the sampler's own path --------------------------------------------------------------------
    t = tau
    xe, ye = TARGET                   # it lands on the NEAR-RIGHT corner: every callout runs right
    def path_of(u):
        return (xe - 5.2 * (1 - u) + 0.55 * np.sin(5.2 * u),
                ye + 4.9 * (1 - u) + 0.45 * np.cos(4.4 * u + 1.1),
                4.3 * (1 - u))
    px, py, dz = path_of(t)
    pz = surf(px, py) + dz
    fine = np.linspace(0, 0.955, 150)
    fx_, fy_, fz_ = path_of(fine)
    ax.plot(fx_, fy_, surf(fx_, fy_) + fz_, color="black", lw=4.2, solid_capstyle="round",
            zorder=Z_PATH + 0.4)
    _tail = np.stack(path_of(np.linspace(0.90, 1.0, 6)), 1)
    _tail[:, 2] += surf(_tail[:, 0], _tail[:, 1])
    arrow3d(ax, _tail, "black", Z_PATH + 0.5, lw=4.2, head=0.55, half=0.22, a_tail=1.0)
    # A DOT ONLY WHERE THERE IS A FRAME. Seven dots invite the reader to look for seven images.
    ax.scatter(px[list(SHOW)], py[list(SHOW)], pz[list(SHOW)], s=110, c="white",
               edgecolors="black", linewidths=2.6, depthshade=False, zorder=Z_PATH + 1)
    # clear of the tau = 0.50 callout, which it was sitting on at the previous height


    # ---- real frames hung off the surface itself ----------------------------------------------------
    real = grab([m[2] for m in MANIFOLD])
    ax.scatter([m[0] for m in MANIFOLD], [m[1] for m in MANIFOLD],
               [surf(m[0], m[1]) + 0.04 for m in MANIFOLD], s=110, c="white",
               edgecolors="black", linewidths=2.6, depthshade=False, zorder=Z_OVER + 1)

    def to_fig(x, y, z):
        a, b, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
        return fig.transFigure.inverted().transform(ax.transData.transform((a, b)))

    # ---- the decoded frame beside the step it came from ---------------------------------------------
    fig.canvas.draw()
    bh = 0.300
    bw = bh * (H / W) * img.shape[2] / img.shape[1]
    bx = 1.0 - 0.015 - bw
    anchor = [to_fig(px[k], py[k], pz[k]) for k in SHOW]
    # LINED UP WITH THEIR OWN DOTS, so every callout is near-horizontal and none of them cross.
    for j, (k, y) in enumerate(zip(SHOW, place(anchor, bh))):
        inset(fig, img[k], bx, y, bw, bh, anchor[j], "right", label=rf"$\tau={t[k]:.2f}$")

    # the recorded frames match the decoded ones in size: they are the same kind of object
    m_anchor = [to_fig(m[0], m[1], surf(m[0], m[1])) for m in MANIFOLD]
    order = sorted(range(len(MANIFOLD)), key=lambda i: -m_anchor[i][1])
    slots = place([m_anchor[j] for j in order], bh)
    for slot, i in enumerate(order):
        inset(fig, real[i], 0.012, slots[slot], bw, bh, m_anchor[i], "left")

    # IN FIGURE COORDINATES, not on the surface: it is a label for the whole field, and the one band
    # of the page no leader line crosses is the top middle, between the left callouts and the right.
    fig.text(0.455, 0.90, r"$v_\theta$", color=ARROW, fontsize=FS * 1.7, ha="center", va="center",
             zorder=7)

    p = f"{OUT}/flow_manifold.png"
    fig.savefig(p, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"  wrote {p}  {int(W * DPI)} x {int(H * DPI)} px, {n} steps, frames at {SHOW}, "
          f"image box {bw:.3f} x {bh:.3f} of the figure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
