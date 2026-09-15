"""Candidate glyphs for "the rectified-flow head emits a DISTRIBUTION, and x_0 is a SAMPLE from it".

Nothing here touches the main figure. Each candidate is written on its own transparent canvas so it can
be dropped beside the flow head's output arrow, and all of them are also rendered onto one labelled
contact sheet so the set can be compared at a glance:

    elements/dist_<name>.png        one glyph, transparent, nothing else
    elements/dist_candidates.png    all of them side by side, labelled

    python logs/paper_icra_2027/make_dist_glyphs.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                    # noqa: E402
import numpy as np                                                                 # noqa: E402
from matplotlib.patches import Ellipse                                             # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "elements")
EDGE = "#000000"
FC, EC = "#f3d9d9", "#ab5b5b"      # the figure's predicted-block pink: these glyphs describe the RED side
LW, DPI = 2.2, 300
X = np.linspace(-3.2, 3.2, 400)


def _norm(y):
    return y / y.max()


def _mix(x):                                                    # the point of the head is MULTIMODALITY
    return _norm(0.62 * np.exp(-((x + 1.25) ** 2) / (2 * 0.55 ** 2))
                 + 0.38 * np.exp(-((x - 1.30) ** 2) / (2 * 0.72 ** 2)))


def _axis(ax, y=0.0):
    ax.plot([X[0], X[-1]], [y, y], color=EDGE, lw=LW * 0.7, solid_capstyle="round", zorder=1)


def bell(ax):
    """One Gaussian. Reads as 'a distribution' and nothing more -- no claim about shape."""
    y = _norm(np.exp(-(X ** 2) / 2))
    ax.fill_between(X, 0, y, fc=FC, ec="none", zorder=2)
    ax.plot(X, y, color=EC, lw=LW, zorder=3)
    _axis(ax)


def bimodal(ax):
    """Two modes. Says what the action/latent prior actually looks like, which a bell quietly denies."""
    y = _mix(X)
    ax.fill_between(X, 0, y, fc=FC, ec="none", zorder=2)
    ax.plot(X, y, color=EC, lw=LW, zorder=3)
    _axis(ax)


def bimodal_sample(ax):
    """Bimodal PLUS the draw: a dot on the density, dropped to the axis. This is the one that says
    'we SAMPLE from it' rather than merely 'it is a distribution'."""
    bimodal(ax)
    xs = -1.25
    ys = float(_mix(np.array([xs]))[0])
    ax.plot([xs, xs], [0, ys], color=EDGE, lw=LW * 0.8, ls=(0, (3, 3)), zorder=4)
    ax.plot([xs], [0], marker="o", ms=7, mfc=EDGE, mec=EDGE, zorder=5)


def samples(ax):
    """A particle set: no density at all, just the draws. Cheapest to read, weakest about the shape."""
    rng = np.random.default_rng(0)
    xs = np.concatenate([rng.normal(-1.25, 0.42, 14), rng.normal(1.30, 0.55, 9)])
    ys = rng.uniform(0.06, 0.92, xs.size)
    ax.plot(xs, ys, ls="none", marker="o", ms=6, mfc=FC, mec=EC, mew=LW * 0.6, zorder=3)
    _axis(ax)


def contours(ax):
    """The 2-D version: nested level sets with one draw marked. Right when the thing sampled is a
    VECTOR (it is -- a whole token bag), wrong if the reader expects a 1-D density."""
    for k, s in enumerate((1.0, 0.66, 0.34)):
        ax.add_patch(Ellipse((0.0, 0.5), 4.6 * s, 1.5 * s, angle=-18, fc=FC if k == 2 else "none",
                             ec=EC, lw=LW, zorder=2 + k))
    ax.plot([0.95], [0.28], marker="o", ms=7, mfc=EDGE, mec=EDGE, zorder=6)


def violin(ax):
    """The density turned on its side, so it can sit ON the output arrow rather than beside it."""
    y = np.linspace(-3.2, 3.2, 400)
    w = _mix(y) * 0.9
    ax.fill_betweenx(y, -w, w, fc=FC, ec="none", zorder=2)
    ax.plot(np.concatenate([-w, w[::-1]]), np.concatenate([y, y[::-1]]), color=EC, lw=LW, zorder=3)
    ax.plot([0.0], [-1.25], marker="o", ms=7, mfc=EDGE, mec=EDGE, zorder=4)


def notation(ax):
    """No glyph at all -- just the tilde, which is the symbol that actually MEANS 'sampled from'."""
    ax.text(0.0, 0.5, r"$x_0 \sim p(x_0 \mid h)$", ha="center", va="center", fontsize=17, color=EDGE)


GLYPHS = [("bell", bell), ("bimodal", bimodal), ("bimodal_sample", bimodal_sample),
          ("samples", samples), ("contours", contours), ("violin", violin), ("notation", notation)]


def _frame(ax, name):
    ax.set_xlim(-3.6, 3.6)
    ax.set_ylim({"violin": (-3.6, 3.6), "contours": (-0.62, 1.62)}.get(name, (-0.22, 1.25)))
    ax.set_aspect("auto")
    ax.axis("off")


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    for name, fn in GLYPHS:
        fig = plt.figure(figsize=(1.6, 1.1), dpi=DPI)
        ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
        fn(ax); _frame(ax, name)
        fig.savefig(f"{OUT}/dist_{name}.png", transparent=True, dpi=DPI, pad_inches=0)
        plt.close(fig)
        print(f"  wrote dist_{name}.png")

    cols = 4
    rows = int(np.ceil(len(GLYPHS) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.1 * cols, 1.7 * rows), dpi=150)
    for ax, (name, fn) in zip(axes.ravel(), GLYPHS):
        fn(ax); _frame(ax, name)
        ax.set_title(name, fontsize=9, color=EDGE)
    for ax in axes.ravel()[len(GLYPHS):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"{OUT}/dist_candidates.png", dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  wrote dist_candidates.png  ({len(GLYPHS)} candidates)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
