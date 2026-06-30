"""PREVIEW: recovered-manifold point clouds from a trained diffusion checkpoint. Pool denoised samples over
many validation contexts -> the union of next-state end-points traces the learned manifold.

  MANIFOLD_STAGE=images uv run python -m quickdraw.smoke.manifold_preview \
      model=diffusion checkpoint=<run_dir_or_ckpt> data.root=<data>

Stage 'images' (fast): the final-frame stills (A position-3D, B UMAP-3D). Stage 'videos': the collapse mp4s.
Writes to logs/viz_preview/. (Driver for the eventual eval_diffusion/manifold_* routine.)"""
from __future__ import annotations

import os

import hydra
import numpy as np
import torch

from ..logging import viz
from ..training.setup import build_model, env_cfg, eval_episodes, load_checkpoint, normalizer

OUT = "/app/logs/viz_preview"
N_CTX, M_EPS, CUBE = 96, 48, 3.0          # contexts x noise-samples per context; uniform-hypercube half-width (latent)


@hydra.main(config_path="../../../conf", config_name="config", version_base=None)
def main(cfg):
    import imageio.v2 as imageio

    from ..models.diffusion import Diffusion, _ln
    stage = os.environ.get("MANIFOLD_STAGE", "images")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(device)
    load_checkpoint(model, cfg.checkpoint)
    model.eval()
    m = getattr(model, "_orig_mod", model)
    assert isinstance(m, Diffusion), "manifold preview needs a diffusion model"
    norm, ecfg = normalizer(cfg), env_cfg(cfg)
    R, r, K, dz, W, P = ecfg.R, ecfg.r, m.sampling_steps, m.cfg.dz, m.window, cfg.data.P
    eps_ds = eval_episodes(cfg, norm, "val")
    n_ep = len(eps_ds)
    rng = np.random.RandomState(0)
    g = torch.Generator(device=device).manual_seed(0)
    os.makedirs(OUT, exist_ok=True)

    def decode(z_t, x):
        return norm.denorm_obs(m.to_obs(_ln(z_t + x))).cpu().numpy()       # (.,6)

    # pool over many validation contexts: each draws M uniform-hypercube latent noises, denoises K steps
    paths = []
    with torch.no_grad():
        for _ in range(N_CTX):
            ep = eps_ds[rng.randint(n_ep)]
            obs = ep["obs_seq"].to(device)[None].float()
            act = ep["act_seq"].to(device)[None].float()
            t = rng.randint(P, obs.shape[1] - 2)
            z = m.encode_state(obs)
            w = min(W, t + 1)
            h = m.transformer(m.to_token(z[:, t - w + 1:t + 1], act[:, t - w + 1:t + 1]))[:, -1].expand(M_EPS, -1)
            z_t = z[:, t].expand(M_EPS, -1)
            eps = (torch.rand(M_EPS, dz, generator=g, device=device) * 2 - 1) * CUBE   # uniform hypercube
            _, path = m.flow.sample(h, steps=K, deterministic=False, eps=eps, record_path=True)
            dec = np.stack([decode(z_t, x) for x in path])                 # (K+1, M, 6)
            paths.append(np.transpose(dec, (1, 0, 2)))                     # (M, K+1, 6)
    allp = np.concatenate(paths, axis=0)                                   # (Pn, K+1, 6)
    speed = np.linalg.norm(allp[:, -1, 3:], axis=1)                        # end-velocity magnitude (color)
    print(f"[manifold] {allp.shape[0]} points, K={K} steps, contexts={N_CTX}")

    # ---- A) position 3D (the surface should appear) ----
    pos = allp[..., :3]                                                    # (Pn, K+1, 3)
    lim = (R + r) * 1.15
    if stage in ("images", "both"):
        f = viz.fig_points_4view(pos[:, -1], color=speed, lims=(-lim, lim), cbar_label="speed",
                                 title="recovered manifold — position (denoised end-points)")
        f.savefig(f"{OUT}/manifold_position_final.png", dpi=95); import matplotlib.pyplot as plt; plt.close(f)
        print("[manifold] wrote manifold_position_final.png")

    # ---- B) UMAP 3D of the full 6D (one fit, transform all -> consistent video) ----
    import umap
    reducer = umap.UMAP(n_components=3, random_state=0, n_neighbors=30, min_dist=0.05)
    emb = reducer.fit_transform(allp.reshape(-1, 6)).reshape(allp.shape[0], allp.shape[1], 3)  # (Pn, K+1, 3)
    elim = (float(emb.min()), float(emb.max()))
    if stage in ("images", "both"):
        import matplotlib.pyplot as plt
        f = viz.fig_points_4view(emb[:, -1], color=speed, lims=elim, cbar_label="speed",
                                 title="recovered manifold — UMAP of full 6D (position+velocity)")
        f.savefig(f"{OUT}/manifold_umap_final.png", dpi=95); plt.close(f)
        print("[manifold] wrote manifold_umap_final.png")

    if stage in ("videos", "both"):
        sub = rng.choice(allp.shape[0], min(2500, allp.shape[0]), replace=False)   # subsample for render speed
        fa = viz.points_collapse_frames(pos[sub], color=speed[sub], lims=(-lim, lim), n_frames=50,
                                        cbar_label="speed", title="manifold collapse — position")
        imageio.mimwrite(f"{OUT}/manifold_position_collapse.mp4", list(fa), fps=25, macro_block_size=2, quality=8)
        fb = viz.points_collapse_frames(emb[sub], color=speed[sub], lims=elim, n_frames=50,
                                        cbar_label="speed", title="manifold collapse — UMAP 6D")
        imageio.mimwrite(f"{OUT}/manifold_umap_collapse.mp4", list(fb), fps=25, macro_block_size=2, quality=8)
        print("[manifold] wrote collapse mp4s")


if __name__ == "__main__":
    main()
