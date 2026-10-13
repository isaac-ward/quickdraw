"""ONE-OFF representation diagnostic — is the drift DYNAMICAL or REPRESENTATIONAL/CODEC?

Read-only on a checkpoint. Labeled visual grid (GT / encode->decode / open-loop) x horizon, curve plots,
a COLOR-specific metric (RGB L1 — LPIPS is texture-biased and hides colour shifts), and metrics.json.

Run from /app (quickdraw installed as a package):
  python oneoffs/representation.py '+eval_ckpt="<run>/checkpoints/epoch=NN-step=MM.ckpt"' \
      +eval_out=logs/_repr +horizon=512 +n_ep=4
"""
from __future__ import annotations

import json
import os

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf


def _label(img_u8, text, hpx=16):
    """Stack a white caption bar with black text above an (H,W,3) uint8 image."""
    from PIL import Image, ImageDraw
    bar = Image.new("RGB", (img_u8.shape[1], hpx), (255, 255, 255))
    ImageDraw.Draw(bar).text((3, 2), text, fill=(0, 0, 0))
    return np.concatenate([np.asarray(bar), img_u8], axis=0)


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg):
    import imageio.v2 as imageio
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    from quickdraw.data.dataset import DataConfig, load_split_episodes_mm
    from quickdraw.evaluation.openloop import _lpips_net
    from quickdraw.training.setup import (build_model, image_head_cams, image_head_sizes, load_checkpoint,
                                          normalizer, resolve_data_root)

    ckpt = os.path.expanduser(str(cfg.get("eval_ckpt", "") or ""))
    assert ckpt and os.path.exists(ckpt), "pass +eval_ckpt=<...>.ckpt (quote it)"
    run_dir = os.path.dirname(os.path.dirname(ckpt))
    rcfg = OmegaConf.load(os.path.join(run_dir, "checkpoints", "config.resolved.yaml"))
    OmegaConf.set_struct(rcfg, False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(rcfg).to(dev).eval(); load_checkpoint(model, ckpt)
    m = getattr(model, "_orig_mod", model)
    norm = normalizer(rcfg); cams, sizes = image_head_cams(rcfg), image_head_sizes(rcfg)
    P = int(rcfg.data.P); H = int(cfg.get("horizon", 512)); n_ep = int(cfg.get("n_ep", 4))
    out = str(cfg.get("eval_out", "") or os.path.join(run_dir, "repr")); os.makedirs(out, exist_ok=True)
    img_heads = list(cams); head = img_heads[0]; lp = _lpips_net(dev, net_type="squeeze")

    eps = load_split_episodes_mm(resolve_data_root(rcfg), "val", dcfg=DataConfig.from_cfg(rcfg),
                                 img_size=sizes, cam=cams, repo_id=str(rcfg.data.get("repo_id", "torus")))
    n_ep = min(n_ep, len(eps)); H = min(H, min(len(o) for o, _, _ in eps[:n_ep]) - P - 1)
    HS = [h for h in [1, 8, 16, 32, 64, 128, 256, 512, 1024] if h <= H]

    ctx = {"proprio": torch.stack([norm.norm_obs(torch.from_numpy(o[:P])) for o, _, _ in eps[:n_ep]]).float().to(dev)}
    for h in img_heads:
        ctx[h] = torch.stack([torch.from_numpy(fr[h][:P]).float().div(255.0) for _, _, fr in eps[:n_ep]]).to(dev)
    acts = torch.stack([norm.norm_act(torch.from_numpy(a[np.clip(np.arange(0, P + H - 1), 0, len(a) - 1)]))
                        for _, a, _ in eps[:n_ep]]).float().to(dev)
    fut = {"proprio": torch.stack([norm.norm_obs(torch.from_numpy(o[P:P + H])) for o, _, _ in eps[:n_ep]]).float().to(dev)}
    for h in img_heads:
        fut[h] = torch.stack([torch.from_numpy(fr[h][P:P + H]).float().div(255.0) for _, _, fr in eps[:n_ep]]).to(dev)

    with torch.no_grad():
        res = m.imagine_eval(ctx, acts, H, heads=["proprio"] + img_heads, norm=norm, return_bag=True)
        zc = m.encode_state(fut)
    zr = res["_bag"]
    cos = torch.nn.functional.cosine_similarity(zr.float(), zc.float(), dim=-1).mean(dim=(0, 2)).cpu()

    def pr_of(x):
        f = x.reshape(-1, x.shape[-1]).float(); f = f - f.mean(0, keepdim=True)
        lam = torch.linalg.svdvals(f / (f.shape[0] ** 0.5)) ** 2
        return float((lam.sum() ** 2) / (lam ** 2).sum())
    d = zc.shape[-1]; Dbag = zc.shape[-2] * d
    pr_tok, pr_bag = pr_of(zc), pr_of(zc.reshape(*zc.shape[:2], -1))

    def lpips(a, b):
        return float(lp(a.permute(0, 3, 1, 2).float().clamp(0, 1), b.permute(0, 3, 1, 2).float().clamp(0, 1)).mean())
    def rgb_l1(a, b):                                   # COLOR error: mean |ΔRGB| in [0,1] (LPIPS-blind to this)
        return float((a.clamp(0, 1) - b.clamp(0, 1)).abs().mean())
    def sat(x):                                         # HSV saturation of (B,H,W,C) in [0,1] -> (B,H,W)
        mx = x.clamp(0, 1).amax(-1); mn = x.clamp(0, 1).amin(-1)
        return (mx - mn) / mx.clamp_min(1e-6)
    def cube_rgb_l1(a, b, gt):                          # COLOR error restricted to CUBE pixels (high-saturation
        mask = (sat(gt) > 0.35)                         #   region of the GT; table/arm/bg are low-saturation)
        if mask.sum() == 0: return float("nan")
        diff = (a.clamp(0, 1) - b.clamp(0, 1)).abs().mean(-1)   # (B,H,W)
        return float(diff[mask].mean())

    per_h = {}
    with torch.no_grad():
        for h in HS:
            t = h - 1; gt = fut[head][:, t]
            clean = m.to_obs(zc[:, t:t + 1], heads=[head], commit=True)[head][:, 0]
            rolled = res[head][:, t]
            per_h[h] = dict(cl_lpips=lpips(clean, gt), ro_lpips=lpips(rolled, gt),
                            cl_rgb=rgb_l1(clean, gt), ro_rgb=rgb_l1(rolled, gt),
                            cl_cube=cube_rgb_l1(clean, gt, gt), ro_cube=cube_rgb_l1(rolled, gt, gt))

    print(f"[repr] {os.path.basename(run_dir)}  H={H} n_ep={n_ep}")
    print(f"[repr] participation ratio: per-token {pr_tok:.1f}/{d} ({100*pr_tok/d:.0f}%), full-bag {pr_bag:.1f}/{Dbag}")
    print(f"[repr] {'h':>5} {'cos':>7} | {'LPIPS cl':>9} {'ro':>7} | {'RGBall cl':>10} {'ro':>7} | "
          f"{'RGBcube cl':>11} {'ro':>7}   (cube = high-sat pixels only)")
    for h in HS:
        p = per_h[h]
        print(f"   @+{h:<4}{float(cos[h-1]):+7.3f} | {p['cl_lpips']:9.4f} {p['ro_lpips']:7.4f} | "
              f"{p['cl_rgb']:10.4f} {p['ro_rgb']:7.4f} | {p['cl_cube']:11.4f} {p['ro_cube']:7.4f}")

    # LABELED visual grid (episode 0): rows GT / enc->dec (clean) / open-loop (rolled); cols = horizons
    def row(getter, label):
        cells = [_label((getter(h) * 255).astype(np.uint8), f"@+{h}") for h in HS]
        strip = np.concatenate(cells, axis=1)
        return _label(strip, label, hpx=18)
    with torch.no_grad():
        gt_r = row(lambda h: fut[head][0, h-1].cpu().numpy(), "GROUND TRUTH")
        cl_r = row(lambda h: m.to_obs(zc[:1, h-1:h], heads=[head], commit=True)[head][0,0].clamp(0,1).cpu().numpy(),
                   "ENCODE->DECODE (clean latent, no dynamics = CODEC ceiling)")
        ro_r = row(lambda h: res[head][0, h-1].clamp(0,1).cpu().numpy(), "OPEN-LOOP (rolled latent)")
    W = max(gt_r.shape[1], cl_r.shape[1], ro_r.shape[1])
    grid = np.concatenate([gt_r, cl_r, ro_r], axis=0)
    imageio.imwrite(os.path.join(out, f"grid_{head}.png"), grid)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(HS, [float(cos[h-1]) for h in HS], "o-"); ax[0].set_title("latent cos(rolled,clean)")
    ax[0].set_xscale("log"); ax[0].axhline(0, color="k", lw=.5); ax[0].grid(alpha=.3); ax[0].set_xlabel("horizon")
    ax[1].plot(HS, [per_h[h]['cl_lpips'] for h in HS], "o-", label="clean=codec (A)")
    ax[1].plot(HS, [per_h[h]['ro_lpips'] for h in HS], "s-", label="rolled=OL")
    ax[1].set_title("LPIPS (texture)"); ax[1].set_xscale("log"); ax[1].legend(); ax[1].grid(alpha=.3); ax[1].set_xlabel("horizon")
    ax[2].plot(HS, [per_h[h]['cl_rgb'] for h in HS], "o-", label="clean=codec (A)")
    ax[2].plot(HS, [per_h[h]['ro_rgb'] for h in HS], "s-", label="rolled=OL")
    ax[2].set_title("RGB-L1 (COLOR)"); ax[2].set_xscale("log"); ax[2].legend(); ax[2].grid(alpha=.3); ax[2].set_xlabel("horizon")
    fig.suptitle(f"{os.path.basename(run_dir)} | PR/tok {pr_tok:.0f}/{d}"); fig.tight_layout()
    fig.savefig(os.path.join(out, "curves.png"), dpi=110); plt.close(fig)

    json.dump({"run": os.path.basename(run_dir), "H": H, "n_ep": n_ep,
               "participation_ratio": {"per_token": pr_tok, "d": d, "full_bag": pr_bag, "bag_dim": Dbag},
               "cos": {int(h): float(cos[h-1]) for h in HS}, "per_horizon": {int(h): per_h[h] for h in HS}},
              open(os.path.join(out, "metrics.json"), "w"), indent=2)
    print(f"[repr] wrote grid_{head}.png (labeled), curves.png, metrics.json -> {out}")
    print("[repr] DONE")


if __name__ == "__main__":
    main()
