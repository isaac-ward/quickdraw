"""PREVIEW: recovered-manifold point clouds from a trained diffusion checkpoint. Pool denoised next-state
end-points over MANY contexts (a sliding window over the TRAINING episodes) -> the union traces the learned
manifold. The image and the collapse video use the SAME points.

  MANIFOLD_STAGE=images uv run python -m quickdraw.smoke.manifold_preview \
      model=diffusion checkpoint=<run_dir_or_ckpt> data.root=<data>           # rectified-flow
  ... model.diffusion.shortcut=true model.diffusion.sampling_steps=1 ...       # shortcut

Stage 'images' (fast): the 4 eval_manifold UMAP stills (data/latent space x 2D/3D, seeded, uncolored).
'videos'/'both': the aggregate_denoising mp4. Writes to logs/viz_preview/ (model-agnostic filenames)."""
from __future__ import annotations

import os
from collections import defaultdict

import hydra
import numpy as np
import torch

from ..logging import viz
from ..training.setup import build_model, env_cfg, eval_episodes, load_checkpoint, normalizer

OUT = "/app/logs/viz_preview"
SPLIT, STRIDE, CUBE, N_POINTS = "train", 1, 3.0, 10000   # train contexts; 1 noise/context; SAME N for img+video
POS_FRAMES, POS_FPS = 480, 60        # position collapse: 8 s @ 60 fps (smooth, eased)
CBAR = "speed = |predicted next velocity|"


@hydra.main(config_path="../../../conf", config_name="config", version_base=None)
def main(cfg):
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt

    from ..models.diffusion import Diffusion
    stage = os.environ.get("MANIFOLD_STAGE", "images")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(device)
    load_checkpoint(model, cfg.checkpoint)
    model.eval()
    m = getattr(model, "_orig_mod", model)
    assert isinstance(m, Diffusion), "manifold preview needs a diffusion model"
    norm, ecfg = normalizer(cfg), env_cfg(cfg)
    R, r, K, P = ecfg.R, ecfg.r, m.sampling_steps, cfg.data.P
    model_name = "shortcut" if bool(m.cfg.shortcut) else "rectified-flow"
    ds = eval_episodes(cfg, norm, SPLIT)
    n_ep = len(ds)
    rng = np.random.RandomState(0)
    g = torch.Generator(device=device).manual_seed(0)
    os.makedirs(OUT, exist_ok=True)
    from ..evaluation.manifold import manifold_clouds, manifold_predictions, pad_lims, umap_reduce

    # pick N_POINTS random (episode, step) context slices over the whole training split
    slices = [(ei, t) for ei in range(n_ep) for t in range(P, ds[ei]["obs_seq"].shape[0] - 1, STRIDE)]
    n_avail = len(slices)
    rng.shuffle(slices)
    by_ep = defaultdict(list)
    for ei, t in slices[:N_POINTS]:
        by_ep[ei].append(t)

    # ---- autoregressive-rollout test (a SEPARATE story): seed from real contexts, then roll the model
    # forward under SAMPLED training actions. Does it stay on the torus as the horizon grows, or does
    # compounding error drift it off? One deterministic H=64 rollout per seed -> slice every shorter
    # horizon out of it (readout is deterministic, so the step-h state of a 64-rollout IS an h-rollout). ----
    if stage == "rollout":
        HORIZONS = [1, 2, 4, 8, 16, 32, 64]
        MAXH = HORIZONS[-1]
        rolldir = f"{OUT}/rollout"; os.makedirs(rolldir, exist_ok=True)
        # pooled empirical training-action distribution (marginal; iid samples ignore temporal correlation)
        act_pool = torch.cat([ds[ei]["act_seq"].float() for ei in range(n_ep)], dim=0).to(device)  # (M,2)
        ctxs, racts = [], []                                       # context obs[t-P:t]; real ctx actions a[t-P:t-1]
        for ei, ts in by_ep.items():
            o, a = ds[ei]["obs_seq"].float(), ds[ei]["act_seq"].float()
            for t in sorted(ts):
                ctxs.append(o[t - P:t]); racts.append(a[t - P:t - 1])
        ctx = torch.stack(ctxs).to(device)                         # (N,P,6) real context (normalized)
        ract = torch.stack(racts).to(device)                       # (N,P-1,2) real context actions
        Nr = ctx.shape[0]
        idx = torch.randint(act_pool.shape[0], (Nr, MAXH), generator=g, device=device)
        actions = torch.cat([ract, act_pool[idx]], dim=1)          # (N, P-1+MAXH, 2): real ctx + sampled future
        rolls = []
        with torch.no_grad():
            for i in range(0, Nr, 2000):
                rolls.append(norm.denorm_obs(m.imagine_eval(ctx[i:i + 2000], actions[i:i + 2000], MAXH)).cpu().numpy())
        roll = np.concatenate(rolls, 0)                            # (N, MAXH, 6) physical units
        clouds = {0: norm.denorm_obs(ctx[:, -1]).cpu().numpy()}    # step 0 = on-torus seed
        for h in HORIZONS:
            clouds[h] = roll[:, h - 1]                             # state after h autoregressive steps
        sub_r = f"{model_name} (K={K}) — {Nr:,} seeds, sampled training actions, autoregressive"

        def spd(c):
            return np.linalg.norm(c[:, 3:], axis=1)

        # position: START frame (seed) + an END frame per horizon, shared lims (over all clouds) so drift never clips
        pl = pad_lims(np.vstack([clouds[k][:, :3] for k in clouds]))

        def save_pos(c, name, ttl):
            f = viz.fig_points_4view(c[:, :3], color=spd(c), lims=pl, point_size=2.0, cbar_label=CBAR, title=ttl)
            f.savefig(f"{rolldir}/{name}.png", dpi=110); plt.close(f)
            print(f"[manifold] wrote rollout/{name}.png")
        save_pos(clouds[0], "pos_start", f"AR rollout — position START (seed, on-torus)\n{sub_r}")
        for h in HORIZONS:
            save_pos(clouds[h], f"pos_h{h:02d}", f"AR rollout — position after {h} step(s)\n{sub_r}")

        # embedding: just the END frame per horizon. UMAP(3) of full 6D, fit_transform JOINTLY on the union
        # of ALL clouds (every point is in the fit -> no transform() OOD-folding; drift gets its own region
        # if it exists). One shared embedding + shared lims so frames are directly comparable.
        import umap
        keys = list(clouds)
        stacked = np.vstack([clouds[k] for k in keys])
        joint = umap.UMAP(n_components=3, random_state=0, n_neighbors=30, min_dist=0.05).fit_transform(stacked)
        emb = dict(zip(keys, np.split(joint, len(keys))))                 # equal-size clouds -> clean split
        el = pad_lims(np.vstack([emb[k] for k in keys]))
        for h in HORIZONS:
            f = viz.fig_points_4view(emb[h], color=spd(clouds[h]), lims=el, point_size=2.5, cbar_label=CBAR,
                                     title=f"AR rollout — UMAP(3) full-6D (joint fit) after {h} step(s)\n{sub_r}")
            f.savefig(f"{rolldir}/umap_h{h:02d}.png", dpi=110); plt.close(f)
            print(f"[manifold] wrote rollout/umap_h{h:02d}.png")
        return

    # ---- 6D-embedding experiment: compare 3D reducers on the START (noise) vs END (manifold) frames ----
    # (uses the diffusion-specific stochastic ODE paths: start = noise, end = manifold)
    if stage.startswith("embed_experiment"):
        expdir = f"{OUT}/embed_experiments"; os.makedirs(expdir, exist_ok=True)
        paths6d, speed, _, _ = manifold_clouds(m, norm, ds, P=P, n_points=N_POINTS, cube=CUBE, stride=STRIDE,
                                               seed=0, device=device)
        start, end, flat = paths6d[:, 0], paths6d[:, -1], paths6d.reshape(-1, 6)

        def save(pts, lims, name, ttl):
            f = viz.fig_points_4view(pts, color=speed, lims=lims, point_size=2.5, cbar_label=CBAR, title=ttl)
            f.savefig(f"{expdir}/{name}.png", dpi=110); plt.close(f)
            print(f"[manifold] wrote embed_experiments/{name}.png")

        # PCA(3) on STANDARDIZED 6D, fit on ALL trajectory steps (linear -> no neighbour-squashing, no
        # transform-projection artifact: the noise START projects honestly far from the manifold END).
        from sklearn.decomposition import PCA
        mu, sd = flat.mean(0), flat.std(0) + 1e-6
        pca = PCA(n_components=3).fit((flat - mu) / sd)
        ps, pe = pca.transform((start - mu) / sd), pca.transform((end - mu) / sd)
        pl = pad_lims(np.vstack([ps, pe]))
        save(ps, pl, "pca_start", "PCA(3) std-6D, fit on all steps — START (noise)")
        save(pe, pl, "pca_end", "PCA(3) std-6D, fit on all steps — END (manifold)")
        return

    # ---- the SAME artifact set the in-training evals log (shared helpers) ----
    # images: the 4 eval_manifold UMAP stills, from DETERMINISTIC committed predictions (any-method path).
    if stage in ("images", "both"):
        data6d, latents, _, _ = manifold_predictions(m, norm, ds, P=P, n_points=N_POINTS, stride=STRIDE,
                                                      seed=0, device=device)
        sub = f"{data6d.shape[0]:,} next-state predictions (of {n_avail:,} {SPLIT} contexts)"   # model-agnostic
        for space, label, pts in (("data_space", "data space (full 6D pos+vel)", data6d),
                                  ("latent_space", f"latent space (full {latents.shape[1]}D z)", latents)):
            for nd in (3, 2):
                emb = umap_reduce(pts, n_components=nd, seed=0)
                fig_fn = viz.fig_points_4view if nd == 3 else viz.fig_points_2d
                f = fig_fn(emb, lims=pad_lims(emb), point_size=2.5,    # no color/colorbar (structure only)
                           title=f"recovered manifold — UMAP of {label} to {nd}D, seed=0\n{sub}")
                f.savefig(f"{OUT}/manifold_umap_{space}_to_{nd}d.png", dpi=110); plt.close(f)
        print(f"[manifold] wrote 4 UMAP stills")

    # videos: the eval_diffusion/aggregate_denoising clip, from the stochastic denoising ODE paths.
    if stage in ("videos", "both"):
        L, Z = (R + r) * 1.05, r * 1.6
        plims = ((-L, L), (-L, L), (-Z, Z))
        paths6d, speed, _, _ = manifold_clouds(m, norm, ds, P=P, n_points=N_POINTS, cube=CUBE, stride=STRIDE,
                                               seed=0, device=device)
        sub = f"{model_name} (K={K}) — {paths6d.shape[0]:,} denoised next-states (of {n_avail:,} {SPLIT} contexts)"
        fa = viz.points_collapse_frames(paths6d[..., :3], color=speed, lims=plims, n_frames=POS_FRAMES, point_size=2.0,
                                        cbar_label=CBAR, title=f"aggregate denoising — noise → manifold\n{sub}")
        imageio.mimwrite(f"{OUT}/manifold_aggregate_denoising.mp4", list(fa), fps=POS_FPS, macro_block_size=2, quality=8)
        print(f"[manifold] wrote aggregate_denoising mp4")


if __name__ == "__main__":
    main()
