"""Presentation-ready plot of the BEST trajectory-level OOD separation (per-clip MEAN uncertainty score).
Data = per-clip mean chroma-p99.5 from the ood_gallery run (10 IND val clips vs 10 OOD purple-cube clips).
Threshold = SPLIT-CONFORMAL: calibrate on the IND clips, set the cutoff at the conformal quantile for a nominal
false-positive rate alpha -> a principled, data-calibrated threshold (not the optimistic max-accuracy one).
Out: logs/ood/gallery/best_separation_mean.png
Run: docker compose exec -T app uv run --no-sync python logs/oneoffs/clean_separation_plot.py
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

IND = np.array([18.9, 21.7, 31.3, 28.4, 21.6, 12.8, 15.8, 18.6, 22.5, 23.1])   # per-clip mean Uncertainty score
OOD = np.array([55.7, 37.8, 58.6, 36.6, 57.4, 35.9, 55.5, 45.2, 39.2, 30.5])
ALPHA = 0.1
C_IND, C_OOD = "#000000", "#7e2fb0"
OUT = "logs/ood/gallery/best_separation_mean.png"


def auc(pos, neg):
    lo = np.sort(neg); r = (np.searchsorted(lo, pos, "left") + np.searchsorted(lo, pos, "right")) / 2
    return float(r.mean() / neg.size)


def main():
    n = IND.size
    # split conformal: threshold = the ceil((n+1)(1-alpha))-th smallest IND (calibration) score. By exchangeability
    # a new IND clip exceeds it with prob <= alpha, so flagging score > thr controls the false-positive rate at alpha.
    rank = min(n, int(np.ceil((n + 1) * (1 - ALPHA))))
    thr = float(np.sort(IND)[rank - 1])
    A = auc(OOD, IND); tpr = float((OOD > thr).mean()); fpr = float((IND > thr).mean())

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15})
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    rng = np.random.default_rng(0)
    for x, data, col in [(0, IND, C_IND), (1, OOD, C_OOD)]:
        ax.scatter(x + rng.uniform(-0.09, 0.09, data.size), data, s=130, color=col,
                   edgecolor="white", linewidth=1.3, zorder=3)
        ax.hlines(data.mean(), x - 0.22, x + 0.22, color=col, linewidth=3, zorder=4)
    ax.axhline(thr, color="#555", linestyle="--", linewidth=1.7, zorder=2)

    ax.set_xlim(-0.5, 1.6); ax.set_xticks([])
    ax.set_ylabel("Uncertainty score  (per-clip mean)")
    ax.set_title(f"World model out-of-distribution classification performance\nAUROC={A:.2f} / TPR={tpr:.0%} / FPR={fpr:.0%}", fontsize=15)
    ax.grid(axis="y", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    handles = [
        Line2D([], [], marker="o", color="none", markerfacecolor=C_IND, markeredgecolor="white", markersize=12, label="In-distribution"),
        Line2D([], [], marker="o", color="none", markerfacecolor=C_OOD, markeredgecolor="white", markersize=12, label="Out-of-distribution"),
        Line2D([], [], color="#777", linewidth=3, label="Group mean"),
        Line2D([], [], color="#555", linestyle="--", linewidth=1.7, label=f"Conformal threshold (α={ALPHA})"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True, framealpha=0.92, edgecolor="none", fontsize=12.5)
    fig.tight_layout(); os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=150); plt.close(fig)
    print(f"AUROC {A:.3f} conformal-thr {thr:.1f} (a={ALPHA}) TPR {tpr:.2f} FPR {fpr:.2f} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
