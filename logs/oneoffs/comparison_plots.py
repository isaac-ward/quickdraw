"""Two matching side-by-side comparison plots: OUR world model vs Cosmos (Predict2-2B).
  params_comparison.png  : model parameters (log)           -- always
  timing_comparison.png  : wall-clock to predict 1 min video (log) -- needs logs/ood/rollout_error/timing.json
Same style so they pair side by side. Out: logs/ood/rollout_error/.
Run: docker compose exec -T app uv run --no-sync python logs/oneoffs/comparison_plots.py
"""
import os, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = "logs/ood/rollout_error"
C_US, C_COS = "#2ca02c", "#9aa0a6"
COSMOS_PARAMS_M = 2000.0        # Predict2-2B
COSMOS_SEC_PER_MIN = 3786.0     # their "one minute of predicted video" = 1.05 h
OURS_PARAMS_M = 9.05


def bar(ax, us, cos, fmt, title, sub):
    xs = [0, 1]
    ax.bar(xs, [us, cos], width=0.62, color=[C_US, C_COS], edgecolor="black", linewidth=0.8, zorder=3)
    for x, v in zip(xs, [us, cos]):
        ax.text(x, v * 1.12, fmt(v), ha="center", va="bottom", fontsize=14, fontweight="bold")
    ax.set_yscale("log"); ax.set_xticks(xs); ax.set_xticklabels(["Ours", "Cosmos\n(Predict2-2B)"], fontsize=13)
    ax.set_ylim(top=cos * 6); ax.set_title(title + f"\n{sub}", fontsize=15)
    ax.grid(axis="y", alpha=0.25, which="both"); ax.spines[["top", "right"]].set_visible(False)


def main():
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14})

    pm = lambda v: (f"{v/1000:.1f} B" if v >= 1000 else f"{v:.2f} M")
    fig, ax = plt.subplots(figsize=(5.2, 5.8))
    bar(ax, OURS_PARAMS_M, COSMOS_PARAMS_M, pm, "Model parameters", f"{COSMOS_PARAMS_M/OURS_PARAMS_M:.0f}x fewer")
    ax.set_ylabel("parameters (log scale)"); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "params_comparison.png"), dpi=150); plt.close(fig)
    print(f"params_comparison.png  (ours {OURS_PARAMS_M}M vs cosmos {COSMOS_PARAMS_M}M = {COSMOS_PARAMS_M/OURS_PARAMS_M:.0f}x)", flush=True)

    tj = os.path.join(OUT, "timing.json")
    if os.path.exists(tj):
        t = json.load(open(tj)); mhz = t["model_rate_hz"]
        ours_spm = (60.0 * mhz) * (t["img_rollout_ms_per_step"] / 1000.0)      # steps for 60s video x s/step
        rtf_us, rtf_cos = ours_spm / 60.0, COSMOS_SEC_PER_MIN / 60.0
        ts = lambda v: (f"{v:.1f} s" if v < 90 else (f"{v/60:.1f} min" if v < 5400 else f"{v/3600:.2f} h"))
        fig, ax = plt.subplots(figsize=(5.2, 5.8))
        bar(ax, ours_spm, COSMOS_SEC_PER_MIN, ts, "Compute to predict 1 min of video",
            f"{COSMOS_SEC_PER_MIN/ours_spm:.0f}x faster")
        ax.axhline(60, color="#c1272d", linestyle="--", linewidth=1.5, zorder=2)
        ax.text(1.46, 60, "real time", color="#c1272d", va="center", ha="left", fontsize=11)
        ax.set_ylabel("wall-clock seconds (log scale)"); fig.tight_layout()
        fig.savefig(os.path.join(OUT, "timing_comparison.png"), dpi=150); plt.close(fig)
        print(f"timing_comparison.png  (ours {ours_spm:.1f}s/min = {rtf_us:.2g}x RT vs cosmos 3786s = 63x RT -> {COSMOS_SEC_PER_MIN/ours_spm:.0f}x faster)", flush=True)
    else:
        print("timing.json not present yet -> timing_comparison.png skipped (rerun after the open-loop run)", flush=True)


if __name__ == "__main__":
    main()
