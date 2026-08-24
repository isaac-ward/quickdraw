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
    the shared forward(); returns (data_phys (N, obs_dim) decoded PROPRIO physical, latents (N, n_state*d) =
    the flattened carried token bag, n_avail). mm_eps: list of (obs (T, obs_dim), act (T, action_dim), img (T,H,W,3))."""
    n_ep = len(mm_eps)
    rng = np.random.RandomState(seed)
    slices = [(ei, t) for ei in range(n_ep) for t in range(P, len(mm_eps[ei][0]) - 1, stride)]
    n_avail = len(slices)
    rng.shuffle(slices)
    by_ep = defaultdict(list)
    for ei, t in slices[:n_points]:
        by_ep[ei].append(t)
    img_head = next((n for n, _ in m.layout if n != "proprio"), None)   # single FPV feed's head name
    data_phys, latents = [], []
    for ei, ts in by_ep.items():
        o, a, im = mm_eps[ei]
        obs = {"proprio": norm.norm_obs(torch.from_numpy(o)).float()[None].to(device),
               img_head: torch.from_numpy(im).float().div(255.0)[None].to(device)}
        act = norm.norm_act(torch.from_numpy(a)).float()[None].to(device)   # NORMALIZE (every other routine does)
        pred = m(obs, act)                                   # (1,T,n_state,d)
        sel = pred[0, np.array(sorted(ts))]                  # (nt,n_state,d)
        latents.extend(sel.reshape(sel.shape[0], -1).cpu().numpy())            # flatten bag -> (n_state*d,)
        data_phys.extend(norm.denorm_obs(m.to_obs(sel)["proprio"]).cpu().numpy())  # decoded proprio vector
    data_phys, latents = np.stack(data_phys), np.stack(latents)
    return data_phys, latents, n_avail


def reduce_dims(pts, method, *, n_components, seed=0, return_reducer=False, y=None, target_weight=0.0):
    """Reduce an (N, D) cloud -> (N, n_components) by 'umap' | 'tsne' | 'pca'. Complementary lenses:
    PCA = linear + deterministic + global-geometry-faithful (the arbiter of whether blobs are REALLY
    connected/separated); UMAP = nonlinear neighborhoods; t-SNE = local cluster structure (t-SNE is fed a
    PCA-50 pre-projection, the standard denoise+speedup). fit_transform directly (no out-of-sample), honest.
    return_reducer=True also returns the FITTED estimator: PCA/UMAP expose .transform() to project NEW points
    into the same embedding repeatably (e.g. MPPI candidates in latent space); t-SNE has no transform.
    y (umap only): integer labels for SUPERVISED UMAP — target_weight in [0,1] blends the data graph (0) with
    the label graph (1), pulling same-label points together (forces separation; >0 => not unsupervised)."""
    if method == "pca":
        from sklearn.decomposition import PCA
        red = PCA(n_components=n_components, random_state=seed); e = red.fit_transform(pts)
    elif method == "umap":
        import umap
        kw = dict(n_components=n_components, random_state=seed, n_neighbors=30, min_dist=0.05)
        if y is not None:
            kw["target_weight"] = float(target_weight)                # supervise toward the labels
            kw["init"] = "random"                                     # spectral init -> NaN on the near-degenerate
            #                                                           fully-supervised graph (target_weight~1); random is robust
        red = umap.UMAP(**kw)
        e = red.fit_transform(pts, y=y) if y is not None else red.fit_transform(pts)
        if y is not None and not np.isfinite(e).all():                # near-full supervision (target_weight >= ~0.99)
            raise ValueError(f"supervised UMAP target_weight={target_weight} gave a non-finite embedding: the "
                             f"fully-supervised graph disconnects into per-class cliques and the layout diverges; "
                             f"use target_weight <= ~0.95")
    elif method == "lda":                                  # supervised LINEAR: max between-class / within-class separation
        from sklearn.decomposition import PCA
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.pipeline import make_pipeline
        if y is None:
            raise ValueError("lda requires y (it is supervised)")
        ncls = len(np.unique(y))
        ncomp = min(n_components, max(1, ncls - 1))        # LDA yields at most n_classes-1 discriminant axes
        npc = min(50, pts.shape[1], max(2, pts.shape[0] - 1))
        steps = ([PCA(n_components=npc, random_state=seed)] if pts.shape[1] > npc else []) + \
                [LinearDiscriminantAnalysis(n_components=ncomp)]   # PCA pre-projection stabilizes LDA in high-D
        red = make_pipeline(*steps)
        e = red.fit_transform(pts, y)
        if ncomp < n_components:                           # e.g. a 3-class factor -> 2 LDA dims; pad for a 3D plot
            e = np.hstack([e, np.zeros((e.shape[0], n_components - ncomp), dtype=e.dtype)])
        return (e, red) if return_reducer else e
    elif method == "tsne":
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        npc = min(50, pts.shape[1], pts.shape[0])          # PCA can't take more components than samples OR features
        x = PCA(n_components=npc, random_state=seed).fit_transform(pts) if pts.shape[1] > 50 else pts
        perp = min(30, max(5, (pts.shape[0] - 1) // 3))   # perplexity must stay below the sample count
        red = TSNE(n_components=n_components, random_state=seed, init="pca", perplexity=perp); e = red.fit_transform(x)
    else:
        raise ValueError(f"unknown reducer {method!r}")
    return (e, red) if return_reducer else e


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
    per-token flow to the committed next PROPRIO token, keeping the WHOLE ODE path. Returns (paths_phys
    (N, K+1, obs_dim) physical proprio, latents (N, d) = committed proprio token, n_avail). Feeds
    eval_flow's `denoising_aggregate` (the swarm collapsing from noise onto the recovered manifold)."""
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
    paths_phys, latents = [], []
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
        paths_phys.extend(np.transpose(decs, (1, 0, 2)))              # list of (K+1, obs_dim)
        latents.extend(_ln(zt_pro + path[-1]).float().cpu().numpy())   # committed proprio token, (d,)
    paths_phys, latents = np.stack(paths_phys), np.stack(latents)
    return paths_phys, latents, n_avail
