"""PREVIEW: recovered-manifold point clouds from a trained diffusion checkpoint. Pool denoised next-state
end-points over MANY contexts (a sliding window over the TRAINING episodes) -> the union traces the learned
manifold. The image and the collapse video use the SAME points.

  MANIFOLD_STAGE=images uv run python -m quickdraw.smoke.manifold_preview \
      model=diffusion checkpoint=<run_dir_or_ckpt> data.root=<data>           # rectified-flow
  ... model.diffusion.shortcut=true model.diffusion.sampling_steps=1 ...       # shortcut

Stage 'images' (fast): final stills (A position-3D, B UMAP-3D, orthographic, seeded). 'videos'/'both': the
collapse mp4s. Output names are suffixed by the model (flow/shortcut). Writes to logs/viz_preview/."""
from __future__ import annotations

import os
from collections import defaultdict

import hydra
import numpy as np
import torch

from ..logging import viz
from ..training.setup import build_model, env_cfg, eval_episodes, load_checkpoint, normalizer

OUT = "/app/logs/viz_preview"
SPLIT, STRIDE, CUBE, N_POINTS = "train", 1, 3.0, 12000   # train contexts; 1 noise/context; SAME N for img+video
POS_FRAMES, POS_FPS = 480, 60        # position collapse: 8 s @ 60 fps (smooth, eased)
UMAP_FRAMES, UMAP_FPS = 240, 60      # umap collapse: 4 s @ 60 fps
UMAP_FIT_CAP = 24000                 # subsample for the UMAP fit (over ALL denoising steps, not just ends)
CBAR = "speed = |predicted next velocity|"


@hydra.main(config_path="../../../conf", config_name="config", version_base=None)
def main(cfg):
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt

    from ..models.diffusion import Diffusion, _ln
    stage = os.environ.get("MANIFOLD_STAGE", "images")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(device)
    load_checkpoint(model, cfg.checkpoint)
    model.eval()
    m = getattr(model, "_orig_mod", model)
    assert isinstance(m, Diffusion), "manifold preview needs a diffusion model"
    norm, ecfg = normalizer(cfg), env_cfg(cfg)
    R, r, K, dz, P = ecfg.R, ecfg.r, m.sampling_steps, m.cfg.dz, cfg.data.P
    shortcut = bool(m.cfg.shortcut)
    tag = "shortcut" if shortcut else "flow"
    model_name = "shortcut" if shortcut else "rectified-flow"
    ds = eval_episodes(cfg, norm, SPLIT)
    n_ep = len(ds)
    rng = np.random.RandomState(0)
    g = torch.Generator(device=device).manual_seed(0)
    os.makedirs(OUT, exist_ok=True)

    # pick N_POINTS random (episode, step) context slices over the whole training split
    slices = [(ei, t) for ei in range(n_ep) for t in range(P, ds[ei]["obs_seq"].shape[0] - 1, STRIDE)]
    n_avail = len(slices)
    rng.shuffle(slices)
    by_ep = defaultdict(list)
    for ei, t in slices[:N_POINTS]:
        by_ep[ei].append(t)

    # one causal transformer pass per episode gives h at every step; denoise 1 noise/slice, keep the path
    paths6d = []
    with torch.no_grad():
        for ei, ts in by_ep.items():
            ep = ds[ei]
            obs = ep["obs_seq"].to(device)[None].float()
            act = ep["act_seq"].to(device)[None].float()
            z = m.encode_state(obs)
            h_all = m.transformer(m.to_token(z, act))
            ts = np.array(sorted(ts))
            h, zt = h_all[0, ts], z[0, ts]
            eps = (torch.rand(len(ts), dz, generator=g, device=device) * 2 - 1) * CUBE   # uniform hypercube
            _, path = m.flow.sample(h, steps=K, deterministic=False, eps=eps, record_path=True)
            dec = np.stack([norm.denorm_obs(m.to_obs(_ln(zt + x))).cpu().numpy() for x in path])  # (K+1, nt, 6)
            paths6d.extend(np.transpose(dec, (1, 0, 2)))                                          # list of (K+1, 6)
    paths6d = np.stack(paths6d)                                          # (N, K+1, 6) — SAME points for img + video
    N = paths6d.shape[0]
    speed = np.linalg.norm(paths6d[:, -1, 3:], axis=1)                  # color = |predicted next velocity|
    sub = f"{model_name} (K={K}) — {N:,} denoised next-states, one per context (of {n_avail:,} {SPLIT} contexts)"
    print(f"[manifold:{tag}] {N} points; {sub}")

    # ---- A) position 3D ----  (flat torus -> per-axis lims so it fills)
    L, Z = (R + r) * 1.05, r * 1.6
    plims = ((-L, L), (-L, L), (-Z, Z))
    pos = paths6d[..., :3]
    if stage in ("images", "both"):
        f = viz.fig_points_4view(pos[:, -1], color=speed, lims=plims, point_size=2.0, cbar_label=CBAR,
                                 title=f"recovered manifold — position\n{sub}")
        f.savefig(f"{OUT}/manifold_position_final_{tag}.png", dpi=110); plt.close(f)
        print(f"[manifold:{tag}] wrote manifold_position_final_{tag}.png")

    # ---- B) UMAP of the full 6D ----  fit on ALL denoising-step points (not just ends), so the START
    # (noise) occupies its OWN region of the embedding and the collapse genuinely shows noise -> manifold.
    import umap
    flat = paths6d.reshape(-1, 6)
    fit_idx = rng.choice(flat.shape[0], min(UMAP_FIT_CAP, flat.shape[0]), replace=False)
    reducer = umap.UMAP(n_components=3, random_state=0, n_neighbors=30, min_dist=0.05).fit(flat[fit_idx])
    emb_all = reducer.transform(flat).reshape(N, paths6d.shape[1], 3)   # (N, T, 3) over all steps
    emb = emb_all[:, -1]                                                # end-points (image)
    def _ax(a):
        lo, hi = float(emb_all[..., a].min()), float(emb_all[..., a].max()); pad = 0.05 * (hi - lo + 1e-6)
        return (lo - pad, hi + pad)
    elims = (_ax(0), _ax(1), _ax(2))                                    # from ALL steps -> includes the noise region
    if stage in ("images", "both"):
        f = viz.fig_points_4view(emb, color=speed, lims=elims, point_size=2.5, cbar_label=CBAR,
                                 title=f"recovered manifold — UMAP of full 6D (pos+vel), seed=0\n{sub}")
        f.savefig(f"{OUT}/manifold_umap_final_{tag}.png", dpi=110); plt.close(f)
        print(f"[manifold:{tag}] wrote manifold_umap_final_{tag}.png")

    if stage in ("videos", "both"):
        fa = viz.points_collapse_frames(pos, color=speed, lims=plims, n_frames=POS_FRAMES, point_size=2.0,
                                        cbar_label=CBAR, title=f"recovered manifold — position\n{sub}")
        imageio.mimwrite(f"{OUT}/manifold_position_collapse_{tag}.mp4", list(fa), fps=POS_FPS, macro_block_size=2, quality=8)
        fb = viz.points_collapse_frames(emb_all, color=speed, lims=elims, n_frames=UMAP_FRAMES, point_size=2.5,
                                        cbar_label=CBAR, title=f"recovered manifold — UMAP of full 6D (pos+vel)\n{sub}")
        imageio.mimwrite(f"{OUT}/manifold_umap_collapse_{tag}.mp4", list(fb), fps=UMAP_FPS, macro_block_size=2, quality=8)
        print(f"[manifold:{tag}] wrote collapse mp4s")


if __name__ == "__main__":
    main()
