"""Recovered-manifold point clouds: pool the model's next-state predictions over many (episode, step)
contexts; the union traces the learned manifold. Two sources, both seeded/deterministic given `seed`:
  manifold_predictions: the COMMITTED next-state token bag via the shared forward() path (for diffusion the
  eps=0 readout) — returns decoded-proprio 6D + the flattened latent bag. Feeds the method-agnostic
  eval_manifold. The SAMPLING is defined here in one place; the LOOK lives in logging.viz (fig_points_*)."""
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
def manifold_predictions(m, norm, mm_eps, *, P, n_points, stride, seed, device):
    """Multimodal analogue of manifold_predictions. Committed next-state token bag over many contexts via
    the shared forward(); returns (data6d (N,6) decoded PROPRIO physical, latents (N, n_state*d) = the
    flattened carried token bag, speed (N,), n_avail). mm_eps: list of (obs (T,6), act (T,2), img (T,H,W,3))."""
    n_ep = len(mm_eps)
    rng = np.random.RandomState(seed)
    slices = [(ei, t) for ei in range(n_ep) for t in range(P, len(mm_eps[ei][0]) - 1, stride)]
    n_avail = len(slices)
    rng.shuffle(slices)
    by_ep = defaultdict(list)
    for ei, t in slices[:n_points]:
        by_ep[ei].append(t)
    img_head = next((n for n, _ in m.layout if n != "proprio"), None)   # single FPV feed's head name
    data6d, latents = [], []
    for ei, ts in by_ep.items():
        o, a, im = mm_eps[ei]
        obs = {"proprio": norm.norm_obs(torch.from_numpy(o)).float()[None].to(device),
               img_head: torch.from_numpy(im).float().div(255.0)[None].to(device)}
        act = torch.from_numpy(a).float()[None].to(device)
        pred = m(obs, act)                                   # (1,T,n_state,d)
        sel = pred[0, np.array(sorted(ts))]                  # (nt,n_state,d)
        latents.extend(sel.reshape(sel.shape[0], -1).cpu().numpy())            # flatten bag -> (n_state*d,)
        data6d.extend(norm.denorm_obs(m.to_obs(sel)["proprio"]).cpu().numpy())  # decoded proprio 6-vec
    data6d, latents = np.stack(data6d), np.stack(latents)
    speed = np.linalg.norm(data6d[:, 3:], axis=1)
    return data6d, latents, speed, n_avail


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


@torch.no_grad()
def manifold_clouds(m, norm, mm_eps, *, P, n_points, cube, stride, seed, device):
    """Diffusion-SPECIFIC (token-bag spine): one uniform-hypercube noise per context, denoised through the
    per-token flow to the committed next PROPRIO token, keeping the WHOLE ODE path. Returns (paths6d
    (N, K+1, 6) physical proprio, speed (N,), latents (N, d) = committed proprio token, n_avail). Feeds
    eval_diffusion's `denoising_aggregate` (the swarm collapsing from noise onto the recovered manifold)."""
    import torch.nn.functional as F
    _ln = lambda x: F.layer_norm(x, (x.shape[-1],))
    d, K = m.d, m.sampling_steps
    g = torch.Generator(device=device).manual_seed(seed)
    img_head = next((n for n, _ in m.layout if n != "proprio"), None)
    rng = np.random.RandomState(seed)
    slices = [(ei, t) for ei in range(len(mm_eps)) for t in range(P, len(mm_eps[ei][0]) - 1, stride)]
    n_avail = len(slices); rng.shuffle(slices)
    by_ep = defaultdict(list)
    for ei, t in slices[:n_points]:
        by_ep[ei].append(t)
    dec = m.modalities["proprio"]
    paths6d, latents = [], []
    for ei, ts in by_ep.items():
        o, a, im = mm_eps[ei]
        obs = {"proprio": norm.norm_obs(torch.from_numpy(o)).float()[None].to(device)}
        if img_head is not None:
            obs[img_head] = torch.from_numpy(im).float().div(255.0)[None].to(device)
        act = norm.norm_act(torch.from_numpy(a)).float()[None].to(device)
        z = m.encode_state(obs)                                    # (1, T, n_state, d)
        h_all = m.backbone(m._to_input(z, act))                    # one causal pass -> h at every step
        ts_a = np.array(sorted(ts))
        h_pro, zt_pro = h_all[0, ts_a, 0, :], z[0, ts_a, 0, :]     # (nt, d) proprio-token conditioning + token
        eps = (torch.rand(len(ts_a), d, generator=g, device=device) * 2 - 1) * cube   # uniform hypercube noise
        _, path = m.flow.sample(h_pro, steps=K, deterministic=False, eps=eps, record_path=True)
        decs = np.stack([norm.denorm_obs(dec.decode(_ln(zt_pro + x)[:, None, :].float())).float().cpu().numpy()
                         for x in path])                           # (K+1, nt, 6): decoded proprio along the ODE
        paths6d.extend(np.transpose(decs, (1, 0, 2)))              # list of (K+1, 6)
        latents.extend(_ln(zt_pro + path[-1]).float().cpu().numpy())   # committed proprio token, (d,)
    paths6d, latents = np.stack(paths6d), np.stack(latents)
    speed = np.linalg.norm(paths6d[:, -1, 3:], axis=1)             # |predicted next velocity|
    return paths6d, speed, latents, n_avail
