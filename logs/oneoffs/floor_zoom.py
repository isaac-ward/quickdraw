"""Is the CODEC (AE-floor) actually faithful to the CUBES? Global L1 is background-dominated, so it
hides cube damage. Here: (1) zoom into the cube region GT-vs-floor, (2) L1 restricted to cube pixels
(HSV saturation mask). No dynamics -- pure encode(GT)->decode(GT). Run same args as oracle_cube."""
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from quickdraw.data.dataset import DataConfig, load_split_episodes_mm
from quickdraw.training.setup import (build_model, image_head_cams, image_head_sizes,
                                      load_checkpoint, normalizer, resolve_data_root)

N_EP, H = 2, 64


def sat_mask(rgb):  # (h,w,3) [0,1] -> bool where saturated (cube) pixels are
    mx = rgb.max(-1); mn = rgb.min(-1)
    sat = (mx - mn) / (mx + 1e-6)
    return sat > 0.35


def main():
    ckpt, arm = sys.argv[1], sys.argv[2]
    run_dir = os.path.dirname(os.path.dirname(ckpt))
    rcfg = OmegaConf.load(os.path.join(run_dir, "checkpoints", "config.resolved.yaml"))
    OmegaConf.set_struct(rcfg, False)
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
    ac = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available())
    with torch.no_grad(), ac:
        gt_obs = {"proprio": gt_future, **gt_img}
        floor = m.to_obs(m.encode_state(gt_obs), heads=img_heads, commit=True)

    out = f"logs/oracle_cube/{arm}"
    os.makedirs(out, exist_ok=True)
    print(f"[floor_zoom] {arm}  L1 all-pixels vs L1 CUBE-pixels-only (GT vs codec-floor)", flush=True)
    for hd in img_heads:
        g = gt_img[hd].cpu().numpy(); f = floor[hd].float().cpu().numpy()
        ih, iw = g.shape[2], g.shape[3]
        # metric
        for q in [8, 24, 40, 56]:
            all_l1 = np.abs(f[:, q] - g[:, q]).mean()
            cvals = []
            for b in range(g.shape[0]):
                msk = sat_mask(g[b, q])
                if msk.sum() > 0:
                    cvals.append(np.abs(f[b, q][msk] - g[b, q][msk]).mean())
            cube_l1 = float(np.mean(cvals)) if cvals else float("nan")
            frac = float(np.mean([sat_mask(g[b, q]).mean() for b in range(g.shape[0])]))
            print(f"   {hd:12s} h{q:2d}: all={all_l1:.4f}  cube-only={cube_l1:.4f}  (cubes are {frac*100:.1f}% of pixels)", flush=True)
        # zoomed filmstrip: GT-vs-floor, 4x nearest, rows=time
        cols = [4, 16, 28, 40, 52, 62]
        rows = []
        for t in cols:
            pair = np.concatenate([(np.clip(g[0, t], 0, 1) * 255).astype(np.uint8),
                                   (np.clip(f[0, t], 0, 1) * 255).astype(np.uint8)], axis=1)
            im = Image.fromarray(pair).resize((pair.shape[1] * 4, pair.shape[0] * 4), Image.NEAREST)
            rows.append(np.asarray(im))
        Image.fromarray(np.concatenate(rows, axis=0)).save(f"{out}/FLOOR_zoom_{hd}.png")
        print(f"   wrote {out}/FLOOR_zoom_{hd}.png  (left=GT  right=codec-floor, 4x, rows=time {cols})", flush=True)


if __name__ == "__main__":
    main()
