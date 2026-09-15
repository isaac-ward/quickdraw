"""figures/planning.png -- how a sentence becomes a plan, and where the overlap comes from.

(a) the loop: the summariser's context conditions the action prior, which draws K candidate chunks; each
    is rolled through the FROZEN world model; the reward head scores the imagined latents against the
    sentence; the best chunk's first `commit` steps are executed and the loop repeats.
(b) the chunk schedule: chunks are 32 steps, only 16 are committed, and the 16 that overlap are what
    prefix guidance conditions the next draw on -- which is what takes the seam between chunks from
    8.7x the recorded step-to-step change down to 2.6x.

    python -m paper_specific.figures.make_planning_fig
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = "/app/logs/paper_icra_2027/planning.png"
GRN, ORG, BLU = [plt.get_cmap(c) for c in ("Greens", "Oranges", "Blues")]
GREY, FS = "#4d4d4d", 7.5
CHUNK, COMMIT, NCH = 32, 16, 3


def box(ax, x, y, w, h, title, sub="", ec=GREY, fc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6",
                                lw=1.2, ec=ec, fc=fc, zorder=5))
    ax.text(x + w / 2, y + (h * 0.38 if sub else h / 2), title, ha="center", va="center",
            fontsize=FS, zorder=6)
    if sub:
        ax.text(x + w / 2, y + h * 0.72, sub, ha="center", va="center", fontsize=FS - 1.4,
                style="italic", color=GREY, zorder=6)


def arrow(ax, a, b, lab="", rad=0.0, ls="-", dy=-0.9):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, lw=1.0, color=GREY,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=4))
    if lab:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + dy, lab, ha="center", va="bottom",
                fontsize=FS - 1.6, color=GREY, zorder=7)


def main() -> int:
    fig = plt.figure(figsize=(3.4, 2.9))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_xlim(0, 100); ax.set_ylim(80, 0)

    # ---- (a) the loop -------------------------------------------------------------------------------
    ax.text(0, 4, "(a)", ha="left", va="center", fontsize=FS + 0.5, fontweight="bold")
    box(ax, 10, 0, 34, 9, "action prior", "$K{=}64$ chunks", BLU(0.78), BLU(0.05))
    box(ax, 10, 17, 34, 9, "world model", "frozen", GRN(0.78), GRN(0.05))
    box(ax, 10, 34, 34, 9, "reward head", "vs. the sentence", ORG(0.78), ORG(0.05))
    box(ax, 58, 34, 34, 9, "commit $16$", "then re-plan")
    arrow(ax, (27, 9), (27, 17), "$a_{t:t+32}$", dy=-2.4)
    arrow(ax, (27, 26), (27, 34), "imagined $z$", dy=-2.4)
    arrow(ax, (44, 38.5), (58, 38.5), "$\\arg\\max$", dy=-0.6)
    arrow(ax, (75, 34), (75, 6), "", rad=0.0)
    ax.text(77, 20, "executed actions\nre-enter the context", ha="left", va="center",
            fontsize=FS - 1.6, color=GREY, linespacing=1.15)
    arrow(ax, (75, 6), (44.5, 4.5), "")

    # ---- (b) the chunk schedule --------------------------------------------------------------------
    ax.text(0, 48, "(b)", ha="left", va="center", fontsize=FS + 0.5, fontweight="bold")
    U = 90.0 / (CHUNK + (NCH - 1) * COMMIT)          # panel units per model step
    X0, ROW, H = 6.0, 54.0, 7.0
    for k in range(NCH):
        x = X0 + k * COMMIT * U
        y = ROW + k * (H + 3.5)
        ax.add_patch(Rectangle((x, y), CHUNK * U, H, fc=BLU(0.10), ec=BLU(0.78), lw=1.0))
        ax.add_patch(Rectangle((x, y), COMMIT * U, H, fc=BLU(0.40), ec=BLU(0.78), lw=1.0))
        if k:                                        # the overlap the next draw is conditioned on
            ax.add_patch(Rectangle((x, y), COMMIT * U, H, fill=False, ec="k", lw=0.9,
                                   hatch="////"))
        ax.text(x - 1.2, y + H / 2, f"draw {k + 1}", ha="right", va="center", fontsize=FS - 1.6,
                color=GREY)
        if k == 0:                                   # this half is the tail the next draw continues
            ax.text(x + (COMMIT + CHUNK / 2) * U / 1.0 - COMMIT * U / 2, y + H / 2, "tail",
                    ha="center", va="center", fontsize=FS - 2.0, color=GREY)
    ax.add_patch(Rectangle((X0, ROW - 3.0), COMMIT * U, 1.0, fc=BLU(0.55), ec="none"))
    ax.text(X0 + COMMIT * U / 2, ROW - 4.0, "committed", ha="center", va="bottom", fontsize=FS - 1.6,
            color=BLU(0.9))
    ax.text(X0 + CHUNK * U + 1.0, ROW - 4.0, "chunk $=32$ steps", ha="left", va="bottom",
            fontsize=FS - 1.6, color=GREY)
    ax.text(X0 + COMMIT * U + 1.0, ROW + 3 * (H + 3.5) + 3.0,
            "hatched: the prefix the next draw is\nguided to continue", ha="left", va="top",
            fontsize=FS - 1.6, color="k", linespacing=1.15)

    fig.savefig(OUT, dpi=450, bbox_inches="tight")
    print("  wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
