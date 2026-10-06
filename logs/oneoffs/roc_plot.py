"""ROC curve for the trajectory-level OOD detector (per-clip MEAN uncertainty score, 10 IND vs 10 OOD clips).
TPR vs FPR as the decision threshold sweeps; AUROC = area under it. Marks the conformal operating point.
Out: logs/ood/gallery/roc_curve.png
Run: docker compose exec -T app uv run --no-sync python logs/oneoffs/roc_plot.py
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

IND = np.array([18.9, 21.7, 31.3, 28.4, 21.6, 12.8, 15.8, 18.6, 22.5, 23.1])   # per-clip mean Uncertainty score
OOD = np.array([55.7, 37.8, 58.6, 36.6, 57.4, 35.9, 55.5, 45.2, 39.2, 30.5])
ALPHA = 0.1
OUT = "logs/ood/gallery/roc_curve.png"


def main():
    pos, neg = OOD, IND
    thrs = np.concatenate([[np.inf], np.sort(np.concatenate([pos, neg]))[::-1], [-np.inf]])
    tpr = np.array([(pos > t).mean() for t in thrs])
    fpr = np.array([(neg > t).mean() for t in thrs])
    A = float(np.trapz(tpr, fpr))
    n = neg.size; thr_c = float(np.sort(neg)[min(n, int(np.ceil((n + 1) * (1 - ALPHA)))) - 1])  # conformal cutoff
    op = ((neg > thr_c).mean(), (pos > thr_c).mean())

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15})
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    ax.plot([0, 1], [0, 1], color="#bbb", linestyle="--", linewidth=1.4, label="Chance (AUROC 0.5)")
    ax.plot(fpr, tpr, color="#7e2fb0", linewidth=2.8, label=f"World model (AUROC {A:.2f})")
    ax.fill_between(fpr, tpr, alpha=0.12, color="#7e2fb0")
    ax.scatter([op[0]], [op[1]], s=150, color="#000", zorder=5, edgecolor="white", linewidth=1.3,
               label=f"Conformal operating point\n(TPR {op[1]:.0%}, FPR {op[0]:.0%})")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC — out-of-distribution detection", fontsize=16)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect("equal")
    ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="none", fontsize=12)
    fig.tight_layout(); os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=150); plt.close(fig)
    print(f"AUROC {A:.3f} | conformal op TPR {op[1]:.2f} FPR {op[0]:.2f} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
