"""Recovered-manifold point clouds from a trained diffusion model: pool denoised next-state PATHS over
many (episode, step) contexts; the union of the committed end-points traces the learned manifold. Shared
by the standalone preview (smoke/manifold_preview.py) and the in-training eval (eval_diffusion_field) so
the SAMPLING is defined in exactly one place; the LOOK lives in logging.viz (fig_points_4view /
points_collapse_frames), which both callers use directly."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch


@torch.no_grad()
def manifold_clouds(m, norm, ds, *, P, n_points, cube, stride, seed, device):
    """One uniform-hypercube noise per context, denoised through the flow to the committed next-state,
    keeping the whole ODE path. `m` is the unwrapped Diffusion model; `ds` a list of episodes with
    obs_seq/act_seq. Returns (paths6d (N, K+1, 6) physical units, speed (N,) = |predicted next velocity|,
    latents (N, dz) = the committed next-state latent _ln(z_t+Δẑ), n_avail). Deterministic given `seed`
    (NumPy shuffle + torch noise both seeded with it)."""
    from ..models.diffusion import _ln
    dz, K = m.cfg.dz, m.sampling_steps
    n_ep = len(ds)
    rng = np.random.RandomState(seed)
    g = torch.Generator(device=device).manual_seed(seed)
    slices = [(ei, t) for ei in range(n_ep) for t in range(P, ds[ei]["obs_seq"].shape[0] - 1, stride)]
    n_avail = len(slices)
    rng.shuffle(slices)
    by_ep = defaultdict(list)
    for ei, t in slices[:n_points]:
        by_ep[ei].append(t)
    paths6d, latents = [], []
    for ei, ts in by_ep.items():
        ep = ds[ei]
        obs = ep["obs_seq"].to(device)[None].float()
        act = ep["act_seq"].to(device)[None].float()
        z = m.encode_state(obs)
        h_all = m.transformer(m.to_token(z, act))                  # one causal pass -> h at every step
        ts = np.array(sorted(ts))
        h, zt = h_all[0, ts], z[0, ts]
        eps = (torch.rand(len(ts), dz, generator=g, device=device) * 2 - 1) * cube   # uniform hypercube
        _, path = m.flow.sample(h, steps=K, deterministic=False, eps=eps, record_path=True)
        lat = [_ln(zt + x) for x in path]                          # K+1 latents along the ODE, each (nt, dz)
        dec = np.stack([norm.denorm_obs(m.to_obs(li)).cpu().numpy() for li in lat])   # (K+1, nt, 6)
        paths6d.extend(np.transpose(dec, (1, 0, 2)))                                  # list of (K+1, 6)
        latents.extend(lat[-1].cpu().numpy())                      # committed latent end-point, (dz,) each
    paths6d, latents = np.stack(paths6d), np.stack(latents)
    speed = np.linalg.norm(paths6d[:, -1, 3:], axis=1)             # color = |predicted next velocity|
    return paths6d, speed, latents, n_avail


def umap3(pts, *, seed=0):
    """UMAP(3) of any (N, D) cloud -> (N, 3). Used for both the decoded data space (6D) and the carried
    latent space (dz). fit_transform directly (no out-of-sample transform), so the embedding is honest."""
    import umap
    return umap.UMAP(n_components=3, random_state=seed, n_neighbors=30, min_dist=0.05).fit_transform(pts)
