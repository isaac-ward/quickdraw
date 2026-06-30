"""Recovered-manifold point clouds: pool the model's next-state predictions over many (episode, step)
contexts; the union traces the learned manifold. Two sources, both seeded/deterministic given `seed`:
  - manifold_predictions: the COMMITTED next-state via the shared forward() path — works for ANY model
    (DSAR/LSAR/diffusion); for diffusion it's the eps=0 readout. Feeds the method-agnostic eval_manifold.
  - manifold_clouds: diffusion-SPECIFIC — one denoised sample per context keeping the whole ODE path, for
    the noise->manifold animation (eval_diffusion/aggregate_denoising).
Shared by the standalone preview (smoke/manifold_preview.py) and the in-training evals so the SAMPLING is
defined in one place; the LOOK lives in logging.viz (fig_points_*/points_collapse_frames)."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch


def _sample_contexts(ds, *, P, n_points, stride, seed):
    """Pick n_points random (episode, step) contexts over `ds` (step in [P, len-2]), grouped as
    {episode_idx: [steps]} for one transformer pass per episode. Returns (by_ep, n_avail)."""
    slices = [(ei, t) for ei in range(len(ds)) for t in range(P, ds[ei]["obs_seq"].shape[0] - 1, stride)]
    np.random.RandomState(seed).shuffle(slices)
    by_ep = defaultdict(list)
    for ei, t in slices[:n_points]:
        by_ep[ei].append(t)
    return by_ep, len(slices)


@torch.no_grad()
def manifold_predictions(m, norm, ds, *, P, n_points, stride, seed, device):
    """METHOD-AGNOSTIC recovered manifold: the model's COMMITTED (deterministic) next-state prediction over
    many contexts, via the shared forward() path (encode -> transformer -> readout). Returns
    (data6d (N, 6) physical units, latents (N, state_dim) the carried next-state, speed (N,), n_avail)."""
    by_ep, n_avail = _sample_contexts(ds, P=P, n_points=n_points, stride=stride, seed=seed)
    data6d, latents = [], []
    for ei, ts in by_ep.items():
        ep = ds[ei]
        obs = ep["obs_seq"].to(device)[None].float()
        act = ep["act_seq"].to(device)[None].float()
        pred = m(obs, act)[0, np.array(sorted(ts))]                # (nt, state) committed next-state per context
        latents.extend(pred.cpu().numpy())
        data6d.extend(norm.denorm_obs(m.to_obs(pred)).cpu().numpy())
    data6d, latents = np.stack(data6d), np.stack(latents)
    speed = np.linalg.norm(data6d[:, 3:], axis=1)                 # color = |predicted next velocity|
    return data6d, latents, speed, n_avail


@torch.no_grad()
def manifold_clouds(m, norm, ds, *, P, n_points, cube, stride, seed, device):
    """Diffusion-SPECIFIC: one uniform-hypercube noise per context, denoised through the flow to the
    committed next-state, keeping the whole ODE path. Returns (paths6d (N, K+1, 6) physical units,
    speed (N,) = |predicted next velocity|, latents (N, dz) = committed latent _ln(z_t+Δẑ), n_avail)."""
    from ..models.diffusion import _ln
    dz, K = m.cfg.dz, m.sampling_steps
    g = torch.Generator(device=device).manual_seed(seed)
    by_ep, n_avail = _sample_contexts(ds, P=P, n_points=n_points, stride=stride, seed=seed)
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


def umap_reduce(pts, *, n_components, seed=0):
    """UMAP of any (N, D) cloud -> (N, n_components). Used for both the decoded data space (6D) and the
    carried latent space (dz), at 2D or 3D. fit_transform directly (no out-of-sample transform), honest."""
    import umap
    return umap.UMAP(n_components=n_components, random_state=seed, n_neighbors=30, min_dist=0.05).fit_transform(pts)


def pad_lims(e, frac=0.05):
    """Per-axis padded (lo, hi) extents of an (N, D) cloud -> tuple of D pairs; feeds the `lims` arg of
    fig_points_4view / fig_points_2d so the box matches the data and the cloud fills each panel. Pass a
    pre-stacked array (np.vstack of several clouds) to get shared lims across them."""
    out = []
    for a in range(e.shape[1]):
        lo, hi = float(e[:, a].min()), float(e[:, a].max()); pad = frac * (hi - lo + 1e-6)
        out.append((lo - pad, hi + pad))
    return tuple(out)
