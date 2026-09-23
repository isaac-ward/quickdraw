"""ONE-OFF representation diagnostic (design/mechanistic_interpretability.md): is the drift DYNAMICAL (the
latent walks off) or REPRESENTATIONAL/CODEC (the decoder can't render the cubes even from the true latent)?

Read-only on a checkpoint. Separates Ceiling B (latent random-walk) from Ceiling A (codec) by PATCHING clean
encoded-true latents into the rollout and decoding, + a superposition (participation-ratio) read + the
latent-cosine drift curve. Saves a VISUAL log folder.

  python -m quickdraw._oneoff_representation '+eval_ckpt="<run>/checkpoints/epoch=NN-step=MM.ckpt"' \
      +eval_out=logs/_repr +horizon=512 +n_ep=4

OUTPUTS in +eval_out:
  grid_<head>.png     rows = GT / decode(CLEAN latent) / decode(ROLLED latent); columns = horizons.
  curves.png          latent cosine + clean(codec) vs rolled(OL) LPIPS, per horizon.
  metrics.json        all numbers (cosine, participation ratio, per-horizon lpips).
"""
from __future__ import annotations

import json
import os

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg):
    import imageio.v2 as imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .data.dataset import DataConfig, load_split_episodes_mm
    from .evaluation.openloop import _lpips_net
    from .training.setup import (build_model, image_head_cams, image_head_sizes, load_checkpoint,
                                 normalizer, resolve_data_root)

    ckpt = os.path.expanduser(str(cfg.get("eval_ckpt", "") or ""))
    assert ckpt and os.path.exists(ckpt), "pass +eval_ckpt=<...>.ckpt (quote it)"
    run_dir = os.path.dirname(os.path.dirname(ckpt))
    rcfg = OmegaConf.load(os.path.join(run_dir, "checkpoints", "config.resolved.yaml"))
    OmegaConf.set_struct(rcfg, False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(rcfg).to(dev).eval()
    load_checkpoint(model, ckpt)
    m = getattr(model, "_orig_mod", model)
    norm = normalizer(rcfg)
    cams, sizes = image_head_cams(rcfg), image_head_sizes(rcfg)
    P = int(rcfg.data.P); H = int(cfg.get("horizon", 512)); n_ep = int(cfg.get("n_ep", 4))
    out = str(cfg.get("eval_out", "") or os.path.join(run_dir, "repr")); os.makedirs(out, exist_ok=True)
    img_heads = list(cams); head = img_heads[0]
    lp = _lpips_net(dev, net_type="squeeze")

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
        zc = m.encode_state(fut)                                  # CLEAN latents of the true future
    zr = res["_bag"]                                              # ROLLED latents

    # 1. latent-cosine drift curve
    cos = torch.nn.functional.cosine_similarity(zr.float(), zc.float(), dim=-1).mean(dim=(0, 2)).cpu()

    # 2. participation ratio — per-token (d) AND full-bag (n_state*d)
    def pr_of(x):                                                 # x: (..., D) -> effective #dims
        f = x.reshape(-1, x.shape[-1]).float(); f = f - f.mean(0, keepdim=True)
        lam = torch.linalg.svdvals(f / (f.shape[0] ** 0.5)) ** 2
        return float((lam.sum() ** 2) / (lam ** 2).sum())
    pr_tok = pr_of(zc)                                            # per-token feature usage (D=d)
    pr_bag = pr_of(zc.reshape(*zc.shape[:2], -1))                 # whole-bag usage (D=n_state*d)
    d = zc.shape[-1]; Dbag = zc.shape[-2] * d

    # 3. per-horizon LPIPS: clean(codec/A) vs rolled(OL); gap = dynamics/B
    def lpips(a, b):
        return float(lp(a.permute(0, 3, 1, 2).float().clamp(0, 1), b.permute(0, 3, 1, 2).float().clamp(0, 1)).mean())
    per_h = {}
    with torch.no_grad():
        for h in HS:
            t = h - 1; gt = fut[head][:, t]
            clean = m.to_obs(zc[:, t:t + 1], heads=[head], commit=True)[head][:, 0]
            rolled = res[head][:, t]
            per_h[h] = (lpips(clean, gt), lpips(rolled, gt))

    # ---- console ----
    print(f"[repr] {os.path.basename(run_dir)}  H={H} n_ep={n_ep}")
    print(f"[repr] participation ratio: per-token {pr_tok:.1f}/{d} ({100*pr_tok/d:.0f}%),  "
          f"full-bag {pr_bag:.1f}/{Dbag} ({100*pr_bag/Dbag:.0f}%)  -> lower % = superposed")
    print(f"[repr] {'h':>6} {'cos':>7} {'clean(A)':>9} {'rolled(OL)':>11} {'gap(B)':>8}")
    for h in HS:
        lc, lr = per_h[h]
        print(f"   @+{h:<4} {float(cos[h-1]):+7.3f} {lc:9.4f} {lr:11.4f} {lr-lc:+8.4f}")

    # ---- visual grid: rows GT / clean / rolled, cols = horizons (episode 0) ----
    def strip(getter):
        return np.concatenate([ (getter(h)*255).astype(np.uint8) for h in HS ], axis=1)   # (Himg, W*len, C)
    with torch.no_grad():
        gt_row = strip(lambda h: fut[head][0, h-1].cpu().numpy())
        cl_row = strip(lambda h: m.to_obs(zc[:1, h-1:h], heads=[head], commit=True)[head][0,0].clamp(0,1).cpu().numpy())
        ro_row = strip(lambda h: res[head][0, h-1].clamp(0,1).cpu().numpy())
    pad = np.full((4, gt_row.shape[1], 3), 255, np.uint8)
    grid = np.concatenate([gt_row, pad, cl_row, pad, ro_row], axis=0)
    imageio.imwrite(os.path.join(out, f"grid_{head}.png"), grid)

    # ---- curves ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot([h for h in HS], [float(cos[h-1]) for h in HS], "o-"); ax[0].set_title("latent cos(rolled,clean)")
    ax[0].set_xlabel("horizon"); ax[0].set_xscale("log"); ax[0].axhline(0, color="k", lw=0.5); ax[0].grid(alpha=.3)
    ax[1].plot(HS, [per_h[h][0] for h in HS], "o-", label="clean = codec (Ceiling A)")
    ax[1].plot(HS, [per_h[h][1] for h in HS], "s-", label="rolled = open-loop")
    ax[1].fill_between(HS, [per_h[h][0] for h in HS], [per_h[h][1] for h in HS], alpha=.2, label="gap = dynamics (B)")
    ax[1].set_title("LPIPS vs horizon"); ax[1].set_xlabel("horizon"); ax[1].set_xscale("log"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.suptitle(f"{os.path.basename(run_dir)}  |  PR/token {pr_tok:.0f}/{d}  PR/bag {pr_bag:.0f}/{Dbag}")
    fig.tight_layout(); fig.savefig(os.path.join(out, "curves.png"), dpi=110); plt.close(fig)

    json.dump({"run": os.path.basename(run_dir), "H": H, "n_ep": n_ep,
               "participation_ratio": {"per_token": pr_tok, "d": d, "full_bag": pr_bag, "bag_dim": Dbag},
               "cos": {int(h): float(cos[h-1]) for h in HS},
               "lpips_clean_codec": {int(h): per_h[h][0] for h in HS},
               "lpips_rolled_openloop": {int(h): per_h[h][1] for h in HS}},
              open(os.path.join(out, "metrics.json"), "w"), indent=2)
    print(f"[repr] wrote grid_{head}.png, curves.png, metrics.json -> {out}")
    print("[repr] DONE")


if __name__ == "__main__":
    main()
