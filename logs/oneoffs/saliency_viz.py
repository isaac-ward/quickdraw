"""Visualize the saliency-loss weight map on real block-stack frames (scene + wrist), so we can eyeball
what the cube-weighted decode loss actually up-weights. Uses the EXACT VisualLoss._saliency_map the loss
uses. Out: logs/oracle_cube/plots/saliency_<cam>.png (rows = frames; cols = GT | weight heatmap | overlay).
Run: docker compose exec -T app uv run --no-sync python logs/oneoffs/saliency_viz.py
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from quickdraw.data.dataset import DataConfig, load_split_episodes_mm
from quickdraw.models.visual_loss import VisualLoss
from quickdraw.training.setup import image_head_cams, image_head_sizes, resolve_data_root

K = 8.0
RUN = "logs/train_world_model_2026_09_28_22_36_57_bs_ss10_cubewire_dyn"  # any current run for cfg + data root


def main():
    rcfg = OmegaConf.load(os.path.join(RUN, "checkpoints", "config.resolved.yaml"))
    OmegaConf.set_struct(rcfg, False)
    P = rcfg.data.P
    img_heads = [m.name for m in [type("x", (), {"name": n})() for n, _ in
                 [(mm.get("name"), mm.get("kind")) for mm in rcfg.model.modalities if mm.get("kind") == "image"]]]
    img_heads = [mm.get("name") for mm in rcfg.model.modalities if mm.get("kind") == "image"]
    eps = load_split_episodes_mm(resolve_data_root(rcfg), "val", dcfg=DataConfig.from_cfg(rcfg),
                                 img_size=image_head_sizes(rcfg) or 128,
                                 cam=image_head_cams(rcfg) or rcfg.data.get("cam", "fpv"),
                                 repo_id=rcfg.data.get("repo_id", "torus"))[:1]
    vl = VisualLoss(saliency="hue", saliency_weight=K)
    os.makedirs("logs/oracle_cube/plots", exist_ok=True)
    _, _, fr = eps[0]
    ts = [P + 4, P + 20, P + 40, P + 56]
    for hd in img_heads:
        frames = torch.from_numpy(fr[hd][ts]).float().div(255.0)      # (T,H,W,3)
        w = vl._saliency_map(frames)[..., 0].numpy()                  # (T,H,W) weights in [1, 1+K]
        m = (w > 1.5)                                                  # cube pixels
        fig, ax = plt.subplots(len(ts), 3, figsize=(9, 2.4 * len(ts)))
        for i in range(len(ts)):
            g = frames[i].numpy()
            ov = g.copy(); ov[m[i]] = ov[m[i]] * 0.35 + np.array([1.0, 0.0, 1.0]) * 0.65   # magenta overlay
            ax[i, 0].imshow(g); ax[i, 0].set_title(f"GT (frame {ts[i]})" if i == 0 else ""); ax[i, 0].axis("off")
            im = ax[i, 1].imshow(w[i], cmap="magma", vmin=1, vmax=1 + K)
            ax[i, 1].set_title("weight map (1..1+K)" if i == 0 else ""); ax[i, 1].axis("off")
            ax[i, 2].imshow(ov); ax[i, 2].set_title(f"cube pixels ({m[i].mean()*100:.1f}%)" if i == 0 else f"{m[i].mean()*100:.1f}%"); ax[i, 2].axis("off")
        fig.suptitle(f"saliency weight (K={K:.0f}) — {hd}")
        fig.tight_layout(); fig.savefig(f"logs/oracle_cube/plots/saliency_{hd}.png", dpi=110); plt.close(fig)
        print(f"[saliency_viz] wrote logs/oracle_cube/plots/saliency_{hd}.png  (cube frac {m.mean()*100:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
