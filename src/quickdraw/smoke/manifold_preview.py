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
SPLIT, STRIDE, CUBE, N_POINTS = "train", 1, 3.0, 10000   # train contexts; 1 noise/context; SAME N for img+video
POS_FRAMES, POS_FPS = 480, 60        # position collapse: 8 s @ 60 fps (smooth, eased)
UMAP_FRAMES, UMAP_FPS = 240, 60      # umap collapse: 4 s @ 60 fps
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

        def cube_union(arrs):                                      # shared lims over ALL clouds (drift never clips)
            allp = np.vstack(arrs); out = []
            for i in range(allp.shape[1]):
                lo, hi = float(allp[:, i].min()), float(allp[:, i].max()); pad = 0.05 * (hi - lo + 1e-6)
                out.append((lo - pad, hi + pad))
            return tuple(out)

        # position: START frame (seed) + an END frame per horizon, shared lims so drift is visible
        pl = cube_union([clouds[k][:, :3] for k in clouds])

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
        el = cube_union([emb[k] for k in keys])
        for h in HORIZONS:
            f = viz.fig_points_4view(emb[h], color=spd(clouds[h]), lims=el, point_size=2.5, cbar_label=CBAR,
                                     title=f"AR rollout — UMAP(3) full-6D (joint fit) after {h} step(s)\n{sub_r}")
            f.savefig(f"{rolldir}/umap_h{h:02d}.png", dpi=110); plt.close(f)
            print(f"[manifold] wrote rollout/umap_h{h:02d}.png")
        return

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
            dec = np.stack([norm.denorm_obs(m.to_obs(_ln(zt + x))).cpu().numpy()
                            for x in path])                                                       # (K+1, nt, 6)
            paths6d.extend(np.transpose(dec, (1, 0, 2)))                                          # list of (K+1, 6)
    paths6d = np.stack(paths6d)                                          # (N, K+1, 6) — SAME points for img + video
    N = paths6d.shape[0]
    speed = np.linalg.norm(paths6d[:, -1, 3:], axis=1)                  # color = |predicted next velocity|
    sub = f"{model_name} (K={K}) — {N:,} denoised next-states, one per context (of {n_avail:,} {SPLIT} contexts)"
    print(f"[manifold:{tag}] {N} points; {sub}")

    # ---- 6D-embedding experiment: compare 3D reducers on the START (noise) vs END (manifold) frames ----
    if stage.startswith("embed_experiment"):
        expdir = f"{OUT}/embed_experiments"; os.makedirs(expdir, exist_ok=True)
        start, end, flat = paths6d[:, 0], paths6d[:, -1], paths6d.reshape(-1, 6)

        def cube(a, b):                                  # shared per-axis lims covering both clouds
            both = np.vstack([a, b]); out = []
            for i in range(3):
                lo, hi = float(both[:, i].min()), float(both[:, i].max()); pad = 0.05 * (hi - lo + 1e-6)
                out.append((lo - pad, hi + pad))
            return tuple(out)

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
        pl = cube(ps, pe)
        save(ps, pl, "pca_start", "PCA(3) std-6D, fit on all steps — START (noise)")
        save(pe, pl, "pca_end", "PCA(3) std-6D, fit on all steps — END (manifold)")
        return

    # ---- A) position 3D ----  (flat torus -> per-axis lims so it fills)
    L, Z = (R + r) * 1.05, r * 1.6
    plims = ((-L, L), (-L, L), (-Z, Z))
    pos = paths6d[..., :3]
    if stage in ("images", "both"):
        f = viz.fig_points_4view(pos[:, -1], color=speed, lims=plims, point_size=2.0, cbar_label=CBAR,
                                 title=f"recovered manifold — position\n{sub}")
        f.savefig(f"{OUT}/manifold_position_final_{tag}.png", dpi=110); plt.close(f)
        print(f"[manifold:{tag}] wrote manifold_position_final_{tag}.png")

    # ---- B) UMAP of the full 6D ----  fit on the END-points only (the manifold) -> keeps the clean
    # hollow-shell shape; then transform every step for the video.
    import umap
    reducer = umap.UMAP(n_components=3, random_state=0, n_neighbors=30, min_dist=0.05)
    emb = reducer.fit_transform(paths6d[:, -1])                        # (N,3) end-points (manifold)
    emb_all = reducer.transform(paths6d.reshape(-1, 6)).reshape(N, paths6d.shape[1], 3)  # all steps (video)
    def _ax(a):
        lo, hi = float(emb_all[..., a].min()), float(emb_all[..., a].max()); pad = 0.05 * (hi - lo + 1e-6)
        return (lo - pad, hi + pad)
    elims = (_ax(0), _ax(1), _ax(2))
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
