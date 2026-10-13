"""Ceiling B, measured cleanly on the FIXED (g12) codec. With the codec now faithful, all open-loop
cube error is DYNAMICS drift. Decompose it and name the failure mode.

Per rollout horizon h, on the GT cube region (tight hue mask):
  1. cube-only L1: CLEAN (encode(GT)->decode = codec floor) vs ROLLED (open-loop) -- gap = dynamics drift.
  2. cube SURVIVAL: fraction of the rolled frame's GT-cube pixels that are still cube-colored (fade/vanish).
  3. position DRIFT: centroid shift of the rolled cube blob vs GT, per color (drift/merge).
  4. color FIDELITY: mean hue error on surviving cube pixels (recolor).
Plots -> logs/oracle_cube/plots/cube_drift_*.png.  Run: ... python logs/oneoffs/cube_drift.py <ckpt> <tag>
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from quickdraw.data.dataset import DataConfig, load_split_episodes_mm
from quickdraw.models.transformer import pad_block_mask
from quickdraw.training.setup import (build_model, image_head_cams, image_head_sizes,
                                      load_checkpoint, normalizer, resolve_data_root)

N_EP, H = 4, 64
COLORS = {"red": (0, 15, 345, 361), "green": (90, 165), "blue": (185, 255), "purple": (260, 325)}


def rgb2hsv(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(-1); mn = rgb.min(-1); d = mx - mn + 1e-8
    h = np.zeros_like(mx)
    m = (mx == r); h[m] = ((g - b) / d)[m] % 6
    m = (mx == g); h[m] = ((b - r) / d)[m] + 2
    m = (mx == b); h[m] = ((r - g) / d)[m] + 4
    return h * 60, d / (mx + 1e-8), mx


def color_mask(rgb, name):
    h, s, v = rgb2hsv(rgb)
    if name == "red":
        hm = (h < 15) | (h > 345)
    elif name == "green":
        hm = (h > 90) & (h < 165)
    elif name == "blue":
        hm = (h > 185) & (h < 255)
    else:
        hm = (h > 260) & (h < 325)
    smin = 0.35 if name == "purple" else (0.5 if name == "red" else 0.4)
    return hm & (s > smin) & (v > 0.25)


def any_cube_mask(rgb):
    return np.logical_or.reduce([color_mask(rgb, c) for c in COLORS])


def rollout(m, ctx, acts, H):
    # Use the model's OWN rollout so z0 re-injection / attention-sink / DF etc are handled correctly and this
    # probe never drifts out of sync with the model (respects stochastic_eval via predict_next).
    return m._rollout(ctx, acts, H, p_tf=0.0, true_future=None, detach_every=0, use_cache=False)


def centroid(mask):
    ys, xs = np.where(mask)
    return (np.array([ys.mean(), xs.mean()]) if len(ys) else None)


def main():
    ckpt, tag = sys.argv[1], sys.argv[2]
    split = sys.argv[3] if len(sys.argv) > 3 else "val"   # val | eval_purple_play | eval_purple_stack
    run_dir = os.path.dirname(os.path.dirname(ckpt))
    rcfg = OmegaConf.load(os.path.join(run_dir, "checkpoints", "config.resolved.yaml"))
    OmegaConf.set_struct(rcfg, False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = getattr(build_model(rcfg).to(dev).eval(), "_orig_mod", None) or build_model(rcfg).to(dev).eval()
    m = build_model(rcfg).to(dev).eval(); m = getattr(m, "_orig_mod", m)
    load_checkpoint(m, ckpt)
    norm = normalizer(rcfg); P = rcfg.data.P
    img_heads = [n for n, _ in m.layout if n != "proprio"]
    img_size = next((mod.ae.cfg.img_size for mod in m.modalities.values() if hasattr(mod, "ae")), 128)
    eps = load_split_episodes_mm(resolve_data_root(rcfg), split, dcfg=DataConfig.from_cfg(rcfg),
                                 img_size=image_head_sizes(rcfg) or img_size,
                                 cam=image_head_cams(rcfg) or rcfg.data.get("cam", "fpv"),
                                 repo_id=rcfg.data.get("repo_id", "torus"))[:N_EP]
    Hh = min(H, min(len(o) for o, _, _ in eps) - P - 1)
    ctx = {"proprio": torch.stack([norm.norm_obs(torch.from_numpy(o[:P])) for o, _, _ in eps]).float().to(dev)}
    for hd in img_heads:
        ctx[hd] = torch.stack([torch.from_numpy(fr[hd][:P]) for _, _, fr in eps]).float().div(255.0).to(dev)
    acts = torch.stack([norm.norm_act(torch.from_numpy(a[:P + Hh - 1])) for _, a, _ in eps]).float().to(dev)
    gt_img = {hd: torch.stack([torch.from_numpy(fr[hd][P:P + Hh]) for _, _, fr in eps]).float().div(255.0).to(dev)
              for hd in img_heads}
    gt_fut = torch.stack([norm.norm_obs(torch.from_numpy(o[P:P + Hh])) for o, _, _ in eps]).float().to(dev)

    hd = "cam_scene"
    g = gt_img[hd].cpu().numpy(); B = g.shape[0]; hs = list(range(Hh))

    def curves(r):
        l1, sv, dr = [], [], []
        for t in hs:
            e, s, d = [], [], []
            for b in range(B):
                msk = any_cube_mask(g[b, t])
                if msk.sum() < 8:
                    continue
                e.append(np.abs(r[b, t][msk] - g[b, t][msk]).mean())
                s.append((any_cube_mask(r[b, t]) & msk).sum() / max(msk.sum(), 1))
                ds = []
                for col in COLORS:
                    gm = color_mask(g[b, t], col)
                    if gm.sum() < 8:
                        continue
                    rc = centroid(color_mask(r[b, t], col))
                    ds.append(np.linalg.norm(centroid(gm) - rc) if rc is not None else np.nan)
                if ds:
                    d.append(np.nanmean(ds))
            l1.append(np.mean(e) if e else np.nan); sv.append(np.mean(s) if s else np.nan)
            dr.append(np.nanmean(d) if d else np.nan)
        return np.array(l1), np.array(sv), np.array(dr)

    ac = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available())
    with torch.no_grad(), ac:
        clean = m.to_obs(m.encode_state({"proprio": gt_fut, **gt_img}), heads=[hd], commit=True)[hd].float().cpu().numpy()
        m.stochastic_eval = True
        r_stoch = m.to_obs(rollout(m, ctx, acts, Hh), heads=[hd], commit=True)[hd].float().cpu().numpy()
        m.stochastic_eval = False
        r_det = m.to_obs(rollout(m, ctx, acts, Hh), heads=[hd], commit=True)[hd].float().cpu().numpy()

    lc, sc, dc = curves(clean)
    ls, ss, dsd = curves(r_stoch)
    ld, sd, dd = curves(r_det)

    os.makedirs("logs/oracle_cube/plots", exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].plot(hs, lc, "-", color="#2a2", label="codec floor")
    ax[0].plot(hs, ls, "-", color="#d33", label="rollout STOCHASTIC (current default)")
    ax[0].plot(hs, ld, "-", color="#36c", label="rollout DETERMINISTIC (eps=0)")
    ax[0].set_title("cube-only L1"); ax[0].set_xlabel("horizon"); ax[0].set_ylabel("L1"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(hs, ss, "-", color="#d33", label="stochastic"); ax[1].plot(hs, sd, "-", color="#36c", label="deterministic")
    ax[1].set_ylim(0, 1); ax[1].set_title("cube SURVIVAL"); ax[1].set_xlabel("horizon"); ax[1].legend(); ax[1].grid(alpha=.3)
    ax[2].plot(hs, dsd, "-", color="#d33", label="stochastic"); ax[2].plot(hs, dd, "-", color="#36c", label="deterministic")
    ax[2].set_title("position DRIFT (centroid px)"); ax[2].set_xlabel("horizon"); ax[2].set_ylabel("px"); ax[2].legend(); ax[2].grid(alpha=.3)
    fig.suptitle(f"Stochastic vs deterministic latent rollout — {tag} (cam_scene)")
    fig.tight_layout(); fig.savefig("logs/oracle_cube/plots/cube_drift.png", dpi=110); plt.close(fig)
    print("[cube_drift] wrote logs/oracle_cube/plots/cube_drift.png", flush=True)
    print("   horizon |   L1 stoch  det  | survival stoch det | drift stoch det", flush=True)
    for q in [Hh // 4, Hh // 2, 3 * Hh // 4, Hh - 1]:
        print(f"   h{q:2d}: floor={lc[q]:.3f} | L1 {ls[q]:.3f} {ld[q]:.3f} | surv {ss[q]:.2f} {sd[q]:.2f} | drift {dsd[q]:.1f} {dd[q]:.1f}", flush=True)


if __name__ == "__main__":
    main()
