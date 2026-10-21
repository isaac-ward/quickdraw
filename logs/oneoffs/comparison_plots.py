"""Three ours-vs-Cosmos comparison plots in ONE consistent bar style (Ours = green, Cosmos = grey):
  params_comparison.png : model parameters (log)                       -- always
  timing_comparison.png : wall-clock to predict 1 min of video (log)   -- needs timing.json
  error_comparison.png  : open-loop rollout error, 5 metrics (linear)  -- needs our_metrics.json
All in logs/ood/rollout_error/. Run: docker compose exec -T app uv run --no-sync python logs/oneoffs/comparison_plots.py
"""
import os, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = "logs/ood/rollout_error"
C_US, C_COS = "#2ca02c", "#9aa0a6"
COS_LABEL = "Cosmos\n(Predict2-2B)"
COSMOS_PARAMS_M, COSMOS_SEC_PER_MIN, OURS_PARAMS_M = 2000.0, 3786.0, 9.05
# Cosmos open_loop 30s scene cam, best config (scene+negative): L1, MSE, PSNR, LPIPS, SSIM
COSMOS_ERR = {"l1": 0.0994, "mse": 0.0325, "psnr": 15.25, "lpips": 0.3320, "ssim": 0.5891}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14})


def bar(ax, labels, values, fmt, title, log=False):
    xs = range(len(values))
    ax.bar(xs, values, width=0.62, color=[C_US] + [C_COS] * (len(values) - 1), edgecolor="black", linewidth=0.8, zorder=3)
    for x, v in zip(xs, values):
        ax.text(x, v * 1.12 if log else v + max(values) * 0.02, fmt(v), ha="center", va="bottom", fontsize=13, fontweight="bold")
    if log:
        ax.set_yscale("log"); ax.set_ylim(top=max(values) * 6)
    else:
        ax.set_ylim(0, max(values) * 1.3)
    ax.set_xticks(list(xs)); ax.set_xticklabels(labels, fontsize=12)
    ax.set_title(title, fontsize=14); ax.grid(axis="y", alpha=0.25, which="both" if log else "major")
    ax.spines[["top", "right"]].set_visible(False)


def main():
    os.makedirs(OUT, exist_ok=True)

    # --- params ---
    pm = lambda v: (f"{v/1000:.1f} B" if v >= 1000 else f"{v:.2f} M")
    fig, ax = plt.subplots(figsize=(5.4, 5.8))
    bar(ax, ["Ours", COS_LABEL], [OURS_PARAMS_M, COSMOS_PARAMS_M], pm, f"Model parameters\n{COSMOS_PARAMS_M/OURS_PARAMS_M:.0f}x fewer", log=True)
    ax.set_ylabel("parameters (log scale)"); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "params_comparison.png"), dpi=150); plt.close(fig)
    print("params_comparison.png", flush=True)

    # --- timing ---
    tj = os.path.join(OUT, "timing.json")
    if os.path.exists(tj):
        t = json.load(open(tj)); ours_spm = (60.0 * t["model_rate_hz"]) * (t["img_rollout_ms_per_step"] / 1000.0)
        ts = lambda v: (f"{v:.1f} s" if v < 90 else (f"{v/60:.1f} min" if v < 5400 else f"{v/3600:.2f} h"))
        fig, ax = plt.subplots(figsize=(5.4, 5.8))
        bar(ax, ["Ours", COS_LABEL], [ours_spm, COSMOS_SEC_PER_MIN], ts, f"Compute to predict 1 min of video\n{COSMOS_SEC_PER_MIN/ours_spm:.0f}x faster", log=True)
        ax.axhline(60, color="#c1272d", linestyle="--", linewidth=1.8, zorder=6)
        ax.text(1.46, 60, "real time", color="#c1272d", va="center", ha="left", fontsize=11, zorder=6, clip_on=False)
        ax.set_ylabel("wall-clock seconds (log scale)"); fig.tight_layout()
        fig.savefig(os.path.join(OUT, "timing_comparison.png"), dpi=150); plt.close(fig)
        print(f"timing_comparison.png (ours {ours_spm:.1f}s/min, {COSMOS_SEC_PER_MIN/ours_spm:.0f}x faster)", flush=True)

    # --- error (5 metrics) ---
    mj = os.path.join(OUT, "our_metrics.json")
    if os.path.exists(mj):
        ours = json.load(open(mj))["ours_open_loop_30s"]
        fmt3 = lambda v: f"{v:.3f}"; fmt1 = lambda v: f"{v:.1f}"
        specs = [("L1", "l1", True, fmt3), ("MSE", "mse", True, fmt3), ("PSNR (dB)", "psnr", False, fmt1),
                 ("LPIPS", "lpips", True, fmt3), ("SSIM", "ssim", False, fmt3)]
        fig, axs = plt.subplots(1, 5, figsize=(16, 4.4))
        for ax, (name, key, lower, fmt) in zip(axs, specs):
            bar(ax, ["Ours", "Cosmos"], [ours[key], COSMOS_ERR[key]], fmt, f"{name}  ({'lower' if lower else 'higher'} better {chr(8595) if lower else chr(8593)})")
        fig.suptitle("Open-loop rollout error: ours vs Cosmos  (matched 30s, scene cam; ours also stays bounded to 333s)", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(os.path.join(OUT, "error_comparison.png"), dpi=140); plt.close(fig)
        print("error_comparison.png", flush=True)


if __name__ == "__main__":
    main()
