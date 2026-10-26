"""The four training losses as standalone equation PNGs for slides.

    losses/representation.png      the autoencoder anchor
    losses/flow.png                the rectified-flow dynamics loss, three lines
    losses/dynamics_dataspace.png  the rollout loss, in pixel space
    losses/dynamics_latent.png     the rollout loss, in latent space
    losses/explanations.md         three levels of explanation per loss, plus the symbol key
    losses/flow.md                 the flow loss line by line, symbol by symbol

SIMPLIFIED, NOT CHANGED. Each equation is the one in the code with the indices that never vary
folded away: per-modality sums collapse to one l(.,.), the stop-grads are stated in the key rather
than decorating every target, and the shortcut term is left out because it is off in the run these
are written against. Nothing here is a different objective from what multimodal.py optimises.

WHITE, NOT TRANSPARENT, and tight-cropped: these go on light slides, and a transparent PNG dropped
on one shows whatever is behind it through the antialiasing. The symbol key is NOT rendered -- it is
spoken, and it lives in explanations.md.

    python src/paper_specific/figures/make_loss_math.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                    # noqa: E402
import numpy as np                                                                 # noqa: E402

OUT = "/app/logs/paper_icra_2027/losses"
DPI = 600                 # a 10-inch equation lands at 6000 px: enough to fill a projector unscaled
FS = 34
# THE FIGURES' OWN FONT. Every label in architecture.png is DejaVu Sans, so the maths is set in the
# matching math face rather than in Computer Modern -- one slide, one typeface. `cm` or `stix` here is a
# one-word change if the deck is serif.
plt.rcParams["mathtext.fontset"] = "dejavusans"

# (name, equation, figure width in inches, horizontal alignment)
#
# THE TWO ROLLOUT LOSSES ARE ONE FORM. They differ only in the space the distance is taken in, so they
# are written with the same shape and a named DISTANCE -- d_data over observations, d_latent
# over token bags -- rather than one spelling out a decoder and the other a norm. `d`, not `l`: these
# are distances between two things of the same kind, and the only choice being made is which space.
#
# TIME RUNS THE LITERATURE'S WAY HERE, NOT THE CODE'S. models/flow.py puts data at tau=0 and noise at
# tau=1; nearly every paper does the reverse. These plots use the paper convention (tau=0 noise, tau=1
# data) because they are read beside papers, not beside flow.py. It is the SAME objective under
# tau -> 1-tau: the path reverses, so the velocity target flips sign from (eps - Delta) to
# (Delta - eps). losses/flow.md states the mapping so the two can never be confused.
EQS = [
    ("representation",
     r"$\mathcal{L}_{\mathrm{representation}} \;=\; d_{\mathrm{data}}"
     r"\left(\,\mathrm{Dec}(\mathrm{Enc}(x_t)),\ x_t\,\right)$", 11.0, "center"),
    ("dynamics_dataspace",
     r"$\mathcal{L}_{\mathrm{dyn.}}^{\mathrm{data}} \;=\; \frac{1}{F}\sum_{k=1}^{F}\ "
     r"d_{\mathrm{data}}\left(\,\mathrm{Dec}(\hat{z}_{t+k}),\ x_{t+k}\,\right)$", 11.0, "center"),
    ("dynamics_latent",
     r"$\mathcal{L}_{\mathrm{dyn.}}^{\mathrm{latent}} \;=\; \frac{1}{F}\sum_{k=1}^{F}\ "
     r"d_{\mathrm{latent}}\left(\,\hat{z}_{t+k},\ \mathrm{sg}(z_{t+k})\,\right)$", 11.0, "center"),
]


def render(name, eq, width, ha="center", fs=FS, color="black", path=None):
    fig = plt.figure(figsize=(width, 2.4), dpi=DPI)
    # ha anchors the BLOCK, ma aligns the lines inside it -- a left-justified multi-line equation needs
    # both, and setting only ha leaves the lines centred under a left-anchored block.
    fig.text(0.02 if ha == "left" else 0.5, 0.5, eq, ha=ha, ma=ha, va="center", fontsize=fs,
             color=color, linespacing=1.9)
    p = path or f"{OUT}/{name}.png"
    fig.patch.set_facecolor("white")
    fig.savefig(p, dpi=DPI, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    import cv2
    im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    print(f"  {os.path.basename(p):20s} {im.shape[1]:5d} x {im.shape[0]:4d} px")


def brace(ax, x0, x1, y, label, *, tick=0.055, stem=0.085, fs=FS * 0.56, drop=0.10):
    """An UNDERBRACE in the figures' own grammar: a line with a nib turning up at each end toward the
    span it marks, and a stem down from the middle to the label. Not a curly brace -- every brace in
    architecture.png is this shape, and a talk should not carry two kinds."""
    ax.plot([x0, x0, x1, x1], [y + tick, y, y, y + tick], color="black", lw=2.0,
            solid_joinstyle="round", solid_capstyle="round", zorder=3)
    xm = 0.5 * (x0 + x1)
    ax.plot([xm, xm], [y, y - stem], color="black", lw=2.0, solid_capstyle="round", zorder=3)
    ax.text(xm, y - stem - drop, label, ha="center", va="top", fontsize=fs, color="black")
    return y - stem - drop


def flow_figure(path):
    """The flow loss with its third line taken apart: which half is the model's guess, which half is the
    answer, and what the expectation is averaging over.

    The line is laid out piece by piece rather than as one mathtext string, because a brace has to sit
    under a SUB-EXPRESSION and mathtext gives no way to ask where inside a string a sub-expression
    landed. Each piece is measured after it is drawn and the next one starts where it ended, so the
    spacing is the renderer's, not a guess."""
    W, H = 15.0, 7.6
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.text(0.45, H - 0.9, r"$\Delta_t \;=\; \mathrm{sg}\left(\, z_{t+1} - z_t \,\right)$",
            ha="left", va="center", fontsize=FS)
    ax.text(0.45, H - 2.0, r"$x_\tau \;=\; (1-\tau)\,\epsilon \;+\; \tau\,\Delta_t$"
                           r"$\qquad \tau\sim U(0,1),\ \ \epsilon\sim\mathcal{N}(0,I)$",
            ha="left", va="center", fontsize=FS)
    # ...line three, in pieces. `\|` and not `\left\|`: a \left needs its \right in the SAME string.
    y3 = H - 3.5
    pieces = [r"$\mathcal{L}_{\mathrm{flow}} \;=\;$",
              r"$\mathrm{E}_{\,\tau,\epsilon}\ \|\,$",
              r"$v_\theta(x_\tau,\ \tau,\ h_t)$",
              r"$\;-\;$",
              r"$(\Delta_t - \epsilon)$",
              r"$\,\|^2$"]
    # Each piece's bbox is TIGHT AROUND ITS INK, so the thin spaces a single mathtext string would
    # have put between these terms are trimmed away at every seam. The gaps are therefore explicit.
    gaps = [0.17, 0.07, 0.13, 0.13, 0.05, 0.0]
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    x, span = 0.45, []
    for q, g in zip(pieces, gaps):
        t = ax.text(x, y3, q, ha="left", va="baseline", fontsize=FS)
        bb = inv.transform(t.get_window_extent(rend))
        span.append((bb[0][0], bb[1][0]))
        x = bb[1][0] + g
    y_b = y3 - 0.62
    brace(ax, span[2][0], span[2][1], y_b, "Predicted direction")
    brace(ax, span[4][0], span[4][1], y_b, "True direction")
    brace(ax, span[1][0], span[5][1], y_b - 1.15,
          "Averaged over random noise draws and noise levels")
    # CROP ON THE INK, not with bbox_inches="tight": the axes spans the whole figure, so its tight bbox
    # IS the whole figure and the result keeps every inch of blank page below the last brace.
    fig.savefig(path, dpi=DPI, facecolor="white", pad_inches=0)
    plt.close(fig)
    import cv2
    im = cv2.imread(path)
    ink = (im < 250).any(axis=2)
    r = np.where(ink.any(axis=1))[0]; c = np.where(ink.any(axis=0))[0]
    pad = int(0.14 * DPI)
    im = im[max(0, r[0] - pad): r[-1] + 1 + pad, max(0, c[0] - pad): c[-1] + 1 + pad]
    cv2.imwrite(path, im)
    print(f"  {os.path.basename(path):22s} {im.shape[1]:5d} x {im.shape[0]:4d} px")


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    for name, eq, w, ha in EQS:
        render(name, eq, w, ha)
    flow_figure(f"{OUT}/flow.png")
    print(f"  {len(EQS)} equation PNGs at {DPI} dpi, white, cropped to the equation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
