"""Open-loop rollout ERROR comparison: ours vs Cosmos (Predict2-2B), matched 30s window, scene cam.
Ours: 16 open-loop rollouts (8 starts x 2 val eps), mean over the first 30s (= 90 model-steps @3Hz) of each, via the
repo's image_curves (L1, MSE, PSNR, LPIPS, SSIM). Cosmos: the pasted open_loop numbers (scene-only + scene+negative).
NB the comparison is at matched 30s; ours additionally stays bounded to 333s (see error_over_time.png).
Out: logs/ood/rollout_error/error_comparison.png (+ our_metrics.json)
Run: docker compose exec -T -e CUDA_VISIBLE_DEVICES=1 app uv run --no-sync python logs/oneoffs/error_comparison.py <ckpt>
"""
import os, sys, json
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from quickdraw.data.dataset import DataConfig, load_split_episodes_mm
from quickdraw.evaluation.openloop import image_curves
from quickdraw.training.setup import (build_model, env_cfg, image_head_cams, image_head_sizes,
                                      load_checkpoint, normalizer, resolve_data_root)

HEAD, SECS, STARTS_PER_EP, OUT = "cam_scene", 30.0, 8, "logs/ood/rollout_error"
# Cosmos open_loop (pasted), 30s scene cam:                 L1,     MSE,    PSNR,  LPIPS,  SSIM
COSMOS = {"Cosmos\n(scene)": [0.1418, 0.0439, 14.12, 0.4219, 0.4421],
          "Cosmos\n(+neg)":  [0.0994, 0.0325, 15.25, 0.3320, 0.5891]}
METRICS = [("L1", 0, True), ("MSE", 1, True), ("PSNR", 2, False), ("LPIPS", 3, True), ("SSIM", 4, False)]  # name,idx,lower_better


def main():
    ck = sys.argv[1]; dev = "cuda"
    os.makedirs(OUT, exist_ok=True)
    rc = OmegaConf.load(os.path.join(os.path.dirname(os.path.dirname(ck)), "checkpoints", "config.resolved.yaml"))
    OmegaConf.set_struct(rc, False)
    m = build_model(rc).to(dev).eval(); m = getattr(m, "_orig_mod", m); load_checkpoint(m, ck); m.stochastic_eval = False
    norm = normalizer(rc); P = rc.data.P; s = int(rc.data.subsample)
    model_hz = (1.0 / float(env_cfg(rc).dt)) / s
    H = int(round(SECS * model_hz))                             # 30s -> 90 steps @3Hz
    eps = load_split_episodes_mm(resolve_data_root(rc), "val", dcfg=DataConfig.from_cfg(rc),
                                 img_size=image_head_sizes(rc) or 128,
                                 cam=image_head_cams(rc) or rc.data.get("cam", "fpv"),
                                 repo_id=rc.data.get("repo_id", "torus"))
    fr_keys = list(eps[0][2])

    rows = []                                                   # per-rollout [l1,mse,psnr,lpips,ssim] mean over 30s
    for o, a, fr in eps:
        for st in np.linspace(P, max(P, len(o) - H - P - 1), STARTS_PER_EP).astype(int):
            st = int(st)
            ctx = {"proprio": norm.norm_obs(torch.from_numpy(o[st:st + P])).float()[None].to(dev)}
            for h in fr_keys:
                ctx[h] = torch.from_numpy(fr[h][st:st + P]).float().div(255.0)[None].to(dev)
            acts = norm.norm_act(torch.from_numpy(a[st:st + P + H - 1])).float()[None].to(dev)
            with torch.no_grad():
                pred = m.imagine_eval(ctx, acts, H, heads=[HEAD], decode_chunk=64, norm=norm)[HEAD]
            true = torch.from_numpy(fr[HEAD][st + P:st + P + H]).float().div(255.0)[None].to(dev)
            c = image_curves(pred, true)
            rows.append([np.mean(c["l1"]), np.mean(c["mse"]), np.mean(c["psnr"]),
                         np.mean(c.get("lpips", [np.nan])), np.mean(c["ssim"])])
    ours = np.array(rows); mu = np.nanmean(ours, 0)
    print("ours open-loop 30s mean: L1 %.4f MSE %.4f PSNR %.2f LPIPS %.4f SSIM %.4f  (N=%d)" % (*mu, len(rows)), flush=True)
    json.dump({"ours_open_loop_30s": dict(zip(["l1", "mse", "psnr", "lpips", "ssim"], mu.tolist())), "n": len(rows)},
              open(os.path.join(OUT, "our_metrics.json"), "w"), indent=2)

    bars = [("Ours", mu, "#2ca02c")] + [(k, np.array(v), c) for (k, v), c in zip(COSMOS.items(), ["#b0b4b8", "#7a7f85"])]
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axs = plt.subplots(1, 5, figsize=(16, 4.2))
    for ax, (name, j, lower) in zip(axs, METRICS):
        for i, (lab, vals, col) in enumerate(bars):
            ax.bar(i, vals[j], color=col, edgecolor="black", linewidth=0.7, zorder=3)
            ax.text(i, vals[j], f"{vals[j]:.3f}" if vals[j] < 10 else f"{vals[j]:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(bars))); ax.set_xticklabels([b[0] for b in bars], fontsize=9)
        ax.set_title(f"{name}  ({'lower' if lower else 'higher'} better {'down' if lower else 'up'})", fontsize=11)
        ax.grid(axis="y", alpha=0.25); ax.spines[["top", "right"]].set_visible(False); ax.set_ylim(top=max(b[1][j] for b in bars) * 1.25)
    fig.suptitle("Open-loop rollout error: ours vs Cosmos  (matched 30s, scene cam) — ours also stays bounded to 333s", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(os.path.join(OUT, "error_comparison.png"), dpi=140); plt.close(fig)
    print("wrote error_comparison.png", flush=True)


if __name__ == "__main__":
    main()
