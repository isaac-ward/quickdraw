"""How faithful is the CODEC to the cubes? (no dynamics -- encode(GT)->decode(GT).)
Tight cube mask by HUE band (red/green/blue/purple, high sat) EXCLUDING the tan table.
Outputs plots to logs/oracle_cube/plots/:
  codec_cube_l1.png    -- all-pixel vs cube-only L1 vs horizon, per cam
  codec_size_err.png   -- per-cube-blob: apparent size (px) vs codec L1 in its bbox (grid-limit test)
Run: docker compose exec -T app uv run --no-sync python logs/oneoffs/codec_cube.py <ckpt> <tag>
"""
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from quickdraw.data.dataset import DataConfig, load_split_episodes_mm
from quickdraw.training.setup import (build_model, image_head_cams, image_head_sizes,
                                      load_checkpoint, normalizer, resolve_data_root)

N_EP, H = 2, 64


def rgb2hsv(rgb):                       # (...,3) [0,1] -> h[0,360), s,v [0,1]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(-1); mn = rgb.min(-1); d = mx - mn + 1e-8
    h = np.zeros_like(mx)
    m = (mx == r); h[m] = ((g - b) / d)[m] % 6
    m = (mx == g); h[m] = ((b - r) / d)[m] + 2
    m = (mx == b); h[m] = ((r - g) / d)[m] + 4
    return h * 60, d / (mx + 1e-8), mx


def cube_mask(rgb):                     # (h,w,3)[0,1] -> bool cube pixels (exclude tan table)
    h, s, v = rgb2hsv(rgb)
    red = ((h < 15) | (h > 345)) & (s > 0.5)
    grn = (h > 90) & (h < 165) & (s > 0.4)
    blu = (h > 185) & (h < 255) & (s > 0.4)
    pur = (h > 260) & (h < 325) & (s > 0.35)
    return (red | grn | blu | pur) & (v > 0.25)


def blobs(mask):                        # crude connected comps via flood fill; returns list of (area,bbox)
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if len(ys) >= 8:                # ignore specks
            out.append((len(ys), (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)))
    return out


def main():
    ckpt, tag = sys.argv[1], sys.argv[2]
    run_dir = os.path.dirname(os.path.dirname(ckpt))
    rcfg = OmegaConf.load(os.path.join(run_dir, "checkpoints", "config.resolved.yaml"))
    OmegaConf.set_struct(rcfg, False)
    # pre-cube-dataset runs used obs field 'observation_vector' (now renamed observation.state on disk).
    of = list(rcfg.data.get("obs_fields", ["observation.state"]) or ["observation.state"])
    of = ["observation.state" if f == "observation_vector" else f for f in of]
    rcfg.data.obs_fields = of
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(rcfg).to(dev).eval()
    load_checkpoint(model, ckpt)
    m = getattr(model, "_orig_mod", model)
    norm = normalizer(rcfg)
    P = rcfg.data.P
    img_heads = [n for n, _ in m.layout if n != "proprio"]
    img_size = next((mod.ae.cfg.img_size for mod in m.modalities.values() if hasattr(mod, "ae")), 128)
    eps = load_split_episodes_mm(resolve_data_root(rcfg), "val", dcfg=DataConfig.from_cfg(rcfg),
                                 img_size=image_head_sizes(rcfg) or img_size,
                                 cam=image_head_cams(rcfg) or rcfg.data.get("cam", "fpv"),
                                 repo_id=rcfg.data.get("repo_id", "torus"))[:N_EP]
    gt_future = torch.stack([norm.norm_obs(torch.from_numpy(o[P:P + H])) for o, _, _ in eps]).float().to(dev)
    gt_img = {hd: torch.stack([torch.from_numpy(fr[hd][P:P + H]) for _, _, fr in eps]).float().div(255.0).to(dev)
              for hd in img_heads}
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        floor = m.to_obs(m.encode_state({"proprio": gt_future, **gt_img}), heads=img_heads, commit=True)

    os.makedirs("logs/oracle_cube/plots", exist_ok=True)
    # --- plot 1: all vs cube-only L1 per horizon ---
    fig, axes = plt.subplots(1, len(img_heads), figsize=(6 * len(img_heads), 4), squeeze=False)
    size_err = {hd: ([], []) for hd in img_heads}
    for j, hd in enumerate(img_heads):
        g = gt_img[hd].cpu().numpy(); f = floor[hd].float().cpu().numpy()
        B, T = g.shape[0], g.shape[1]
        hs = list(range(0, T, 2))
        all_l1, cube_l1, cube_frac = [], [], []
        for t in hs:
            a, c, fr = [], [], []
            for b in range(B):
                a.append(np.abs(f[b, t] - g[b, t]).mean())
                msk = cube_mask(g[b, t])
                fr.append(msk.mean())
                if msk.sum() > 0:
                    c.append(np.abs(f[b, t][msk] - g[b, t][msk]).mean())
                for area, (y0, y1, x0, x1) in blobs(msk):
                    e = np.abs(f[b, t][y0:y1, x0:x1] - g[b, t][y0:y1, x0:x1]).mean()
                    size_err[hd][0].append(area); size_err[hd][1].append(e)
            all_l1.append(np.mean(a)); cube_l1.append(np.mean(c) if c else np.nan); cube_frac.append(np.mean(fr))
        ax = axes[0][j]
        ax.plot(hs, all_l1, "-o", ms=3, label="all pixels", color="#888")
        ax.plot(hs, cube_l1, "-o", ms=3, label="cube pixels only", color="#d33")
        ax.set_title(f"{hd}  (cubes ~{np.nanmean(cube_frac)*100:.1f}% of pixels)")
        ax.set_xlabel("rollout horizon (frame)"); ax.set_ylabel("codec L1 to GT"); ax.legend(); ax.grid(alpha=.3)
    fig.suptitle(f"CODEC fidelity (encode(GT)->decode, NO dynamics) — {tag}")
    fig.tight_layout(); fig.savefig("logs/oracle_cube/plots/codec_cube_l1.png", dpi=110); plt.close(fig)

    # --- plot 2: cube apparent size vs codec error (grid-resolution test) ---
    fig, axes = plt.subplots(1, len(img_heads), figsize=(6 * len(img_heads), 4), squeeze=False)
    for j, hd in enumerate(img_heads):
        sz, er = np.array(size_err[hd][0]), np.array(size_err[hd][1])
        ax = axes[0][j]
        if len(sz):
            ax.scatter(sz, er, s=8, alpha=.4, color="#36c")
            # binned mean
            bins = np.linspace(sz.min(), np.percentile(sz, 98), 8)
            idx = np.digitize(sz, bins)
            bm = [er[idx == k].mean() for k in range(1, len(bins))]
            bc = [(bins[k - 1] + bins[k]) / 2 for k in range(1, len(bins))]
            ax.plot(bc, bm, "-o", color="#d33", label="binned mean")
            ax.legend()
        ax.set_title(f"{hd}: cube size vs codec error")
        ax.set_xlabel("cube apparent size (px area)"); ax.set_ylabel("codec L1 in cube bbox"); ax.grid(alpha=.3)
    fig.suptitle(f"Smaller cube -> worse codec? (grid-resolution test) — {tag}")
    fig.tight_layout(); fig.savefig("logs/oracle_cube/plots/codec_size_err.png", dpi=110); plt.close(fig)
    print("[codec_cube] wrote logs/oracle_cube/plots/codec_cube_l1.png + codec_size_err.png", flush=True)
    # console summary
    for hd in img_heads:
        g = gt_img[hd].cpu().numpy(); f = floor[hd].float().cpu().numpy()
        aa = np.abs(f - g).mean()
        cc = np.nanmean([np.abs(f[b, t][cube_mask(g[b, t])] - g[b, t][cube_mask(g[b, t])]).mean()
                         for b in range(g.shape[0]) for t in range(g.shape[1]) if cube_mask(g[b, t]).sum() > 0])
        print(f"   {hd:12s}  all-pixel L1={aa:.4f}   CUBE-only L1={cc:.4f}   ratio={cc/aa:.2f}x", flush=True)


if __name__ == "__main__":
    main()
