"""General latent-space projection + plotting, shared by eval_interpret and train_reward (and, later, the
reward-space animation). Given ANY labeled point cloud — world-model bags, f_z(bags), or any feature — it
fits the standard reducers and plots each projection colored by each factor. It's a plain library: every
workflow calls it independently (no cross-workflow artifact dependency).

Reducers (see manifold.reduce_dims): unsupervised PCA / t-SNE / UMAP (an uncolored `none_` view + one recolor
per factor); supervised LDA and UMAP-sup-<w> (per factor, forced toward that factor's labels). With `save_dir`
it also writes `latents.npy` + each `<key>_{embedding.npy,reducer.pkl}` so a projection can be reused later
(`reducer.transform(new_points)` — pca/umap/lda only; t-SNE has no out-of-sample map)."""

from __future__ import annotations

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from ..logging import viz
from .interpret import point_colors
from .manifold import pad_lims, reduce_dims

PROJECTIONS_GUIDE = r"""# Projection plots guide

Each plot reduces a labeled LATENT point cloud to 2D/3D and colors it by a factor's label. The points are
whatever the producer projected — the world-model token bag (eval_interpret) or f_z(bag) = the reward space
(train_reward). Files: `plots/<method>/<factor>_<nd>d.png` (+ an uncolored `none_<nd>d` for unsupervised
methods). The fitted reducers + embeddings are in `projections/` (`<key>_{reducer.pkl,embedding.npy}`);
pca/lda/umap expose `.transform()` to project NEW points into the SAME embedding (t-SNE has no out-of-sample map).

## The reducers (what each optimizes, how to read it)

### PCA — linear, UNSUPERVISED  (the honest arbiter of global geometry)
- Optimizes VARIANCE: project onto the top eigenvectors of the covariance, max_W Var(Wᵀz) s.t. WᵀW = I.
- Axes: real orthogonal linear directions (axis 1 = most-spread). Unitless, but directions are meaningful.
- Distances: ~faithful to true latent distances (a rotation + truncation, no warping). Trust "are these blobs
  really far apart / connected?" HERE above any nonlinear method.
- Read: global layout; whether classes are linearly separable; how much structure survives in 2–3 dims.

### LDA — linear, SUPERVISED (by the factor's labels)
- Optimizes CLASS SEPARATION: max_W |Wᵀ S_B W| / |Wᵀ S_W W|  (S_B between-class, S_W within-class scatter).
  Fit as a PCA→LDA pipeline. At most (#classes − 1) axes.
- Axes: the most class-discriminative linear directions.
- Distances: OPTIMISTIC — the projection was chosen to pull the LABELED classes apart, so clean separation
  here does NOT prove the raw latent separates. It shows the factor is linearly DECODABLE, not intrinsic structure.
- Read: how linearly separable the factor is; which classes still overlap under the best linear split.

### t-SNE — nonlinear, UNSUPERVISED, LOCAL
- Optimizes NEIGHBORHOODS: match pairwise neighbor probabilities (Gaussian in latent, Student-t in 2D),
  minimize KL(P‖Q). Fed a PCA-50 pre-projection. No `.transform()`.
- Axes: MEANINGLESS. Only local who-is-near-whom is trustworthy.
- Distances: GLOBAL distances, gaps and cluster sizes are NOT meaningful (dense regions inflate; gaps arbitrary).
- Read: fine cluster membership; do NOT read absolute positions or inter-cluster distances.

### UMAP — nonlinear, UNSUPERVISED
- Optimizes a fuzzy-topological graph match (cross-entropy of high-D vs low-D fuzzy simplicial sets); keeps
  more GLOBAL structure than t-SNE but still warps. Has `.transform()`.
- Axes: arbitrary. Neighborhoods trustworthy; distances semi-quantitative at best.
- Read: cluster structure + rough global relations; treat gaps qualitatively.

### umap-sup-<w> — nonlinear, SUPERVISED (target_weight w ∈ [0,1])
- UMAP with the graph blended toward the labels: w=0 is plain UMAP; w→1 forces same-label points together.
- Distances: increasingly OPTIMISTIC as w rises (presentation, not evidence). w≈0.9 = "maximally forced";
  ≥~0.99 degenerates (per-class cliques → NaN layout).
- Read: same caveat as LDA — separation is imposed, not discovered.

## Reading any plot
- SUPERVISED (lda, umap-sup): separation was optimized FOR → shows decodability, not intrinsic structure.
  UNSUPERVISED (pca, tsne, umap): structure the model found on its own.
- For "are two states really similar in the model?", trust PCA distances first; use t-SNE/UMAP only for
  who-clusters-with-whom.
- Color = the factor label; tight same-color islands ⇒ the factor is strongly encoded in the latent.
- Axis numbers are unitless — only RELATIVE positions matter. PCA/LDA axes are linear combinations of latent
  features; t-SNE/UMAP axes carry no meaning.
- `none_<nd>d` = the same embedding with no coloring (the shape of the manifold itself).
"""


def project_and_plot(writer, tag, pts, labels_by_factor, factor_cfgs, *, step, point_size, subtitle="",
                     methods=("pca", "tsne", "umap"), umap_sup_weights=(), n_components=(3, 2),
                     save_dir=None, log=None, plots_name="plots", annotate=None):
    """Project `pts` (N, D) with each reducer and log the plots under `<tag>/<plots_name>/<categorical>/<method>/<Nd>d`.
    `plots_name` names the plot folder so each caller labels it by the SPACE being shown (e.g.
    `world_model_latent_space_plots` for eval_interpret, `joint_latent_space_plots` for the f_z reward space).
    labels_by_factor: {factor: [label per point]} (already broadcast to the N points). factor_cfgs:
    {factor: its interpret config} (buckets/colors/source), for coloring + the supervised label mapping.
    save_dir: if given, persist latents.npy + each fitted embedding/reducer. Returns `transform_ok`
    ({key: reducer-has-.transform}) so the caller can record it (e.g. in its own meta.json)."""
    fig_fn = {nd: (viz.fig_points_9view if nd == 3 else viz.fig_points_2d) for nd in n_components}
    transform_ok = {}
    ann_labels = list(annotate) if annotate else []                # {label: (D,) embedding} -> leader-line labels (2d only)
    ann_embs = np.stack([np.asarray(annotate[k], dtype=np.float32) for k in ann_labels]) if ann_labels else None

    def _ann2d(reducer, nd):                                       # transform the annotation embeddings into THIS reducer (2d, transform-capable)
        if ann_embs is None or nd != 2 or not hasattr(reducer, "transform"):
            return None
        xy = reducer.transform(ann_embs)
        return [{"pos": xy[i], "text": ann_labels[i]} for i in range(len(ann_labels))]
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "latents.npy"), pts)
        open(os.path.join(os.path.dirname(save_dir.rstrip("/")), "README.md"), "w").write(PROJECTIONS_GUIDE)

    def _save(key, e, reducer):
        if save_dir is None:
            return
        np.save(os.path.join(save_dir, f"{key}_embedding.npy"), e)
        try:
            with open(os.path.join(save_dir, f"{key}_reducer.pkl"), "wb") as fh:
                pickle.dump(reducer, fh)
            transform_ok[key] = hasattr(reducer, "transform")   # pca/umap/lda: True; t-SNE: no out-of-sample map
        except Exception:
            transform_ok[key] = False

    def _plot(mdir, method_label, e, factor=None, annotations=None):   # factor=None -> uncolored view (grouped under 'none')
        nd = e.shape[1]
        extra = {"annotations": annotations} if (nd == 2 and annotations) else {}
        if factor is None:                           # <plots_name>/none/<reducer>/<nd>d
            fig = fig_fn[nd](e, lims=pad_lims(e), point_size=point_size,
                             title=f"{tag} — {method_label} of latent to {nd}D (no coloring)\n{subtitle}", **extra)
            writer.figure(f"{tag}/{plots_name}/none/{mdir}/{nd}d", fig, step); plt.close(fig)
        else:                                        # <plots_name>/<categorical>/<reducer>/<nd>d
            rgb, legend = point_colors(labels_by_factor[factor], factor_cfgs[factor])
            fig = fig_fn[nd](e, color=rgb, lims=pad_lims(e), point_size=point_size, legend=legend,
                             title=f"{tag} — {method_label} of latent to {nd}D, colored by {factor} "
                                   f"({factor_cfgs[factor].get('source', '?')})\n{subtitle}", **extra)
            writer.figure(f"{tag}/{plots_name}/{factor}/{mdir}/{nd}d", fig, step); plt.close(fig)

    # ---- unsupervised: one projection per (method, dim); an uncolored view + one recolor per factor ----
    for method in methods:
        for nd in n_components:
            e, reducer = reduce_dims(pts, method, n_components=nd, seed=0, return_reducer=True)
            _save(f"{method}_{nd}d", e, reducer)
            ann = _ann2d(reducer, nd)
            _plot(method, method.upper(), e, annotations=ann)
            for factor in factor_cfgs:
                _plot(method, method.upper(), e, factor, annotations=ann)
        if log is not None:
            log(f"{method} done")

    # ---- supervised: LDA (linear) + UMAP-sup-<w>, fit SEPARATELY per factor (supervised by that factor) ----
    sup_specs = [("lda", "lda", None)] + [(f"umap-sup-{float(w):.2f}", "umap", float(w)) for w in umap_sup_weights]
    for mdir, meth, w in sup_specs:
        for nd in n_components:
            for factor, fc in factor_cfgs.items():
                bmap = {b: k for k, b in enumerate(fc["buckets"])}                    # bucket -> int label
                yv = np.array([bmap[lbl] for lbl in labels_by_factor[factor]])        # supervise by THIS factor
                kw = {"y": yv} if w is None else {"y": yv, "target_weight": w}
                e, reducer = reduce_dims(pts, meth, n_components=nd, seed=0, return_reducer=True, **kw)
                _save(f"{mdir}_{factor}_{nd}d", e, reducer)
                _plot(mdir, mdir, e, factor, annotations=_ann2d(reducer, nd))
        if log is not None:
            log(f"{mdir} done")
    return transform_ok


def animate_joint_space(cloud, traj, t_e, *, method="lda", labels=None, factor_cfg=None,
                        reward_mode=False, n_frames=200, point_size=6.0, tail=60, title="", log=None):
    """Animate an agent's control trajectory through the JOINT latent space (design/language_steering.md P2).
    Fits `method` (lda|umap|pca) on `cloud` (N,D) = f_z(latents), transforms the cloud, the agent `traj`
    (T,D) = f_z(z_t) over control steps, and the goal direction `t_e` (D,) = f_t(goal) into 2D, then renders
    the moving agent (tail-fade) over a STATIC backdrop toward the starred goal. Two colorings:
      reward_mode=False -> by concept (needs `labels` [per cloud point] + `factor_cfg`);
      reward_mode=True  -> by the reward FIELD cos(normalize(cloud), t_e) (a heatmap; how "flat reward far
                           from goal" shows up — uniform vs a gradient toward the star).
    `method` MUST expose .transform() (lda/umap/pca; t-SNE can't project the new trajectory -> excluded).
    Returns (n_frames, H, W, 3) uint8 (feed to viz.save_mp4)."""
    cloud = np.asarray(cloud, dtype=np.float32)
    traj = np.asarray(traj, dtype=np.float32)
    t_e = np.asarray(t_e, dtype=np.float32).reshape(-1)
    y = None
    if method == "lda":                                          # supervised: fit toward the concept labels
        assert labels is not None and factor_cfg is not None, "method='lda' needs labels + factor_cfg"
        bmap = {b: k for k, b in enumerate(factor_cfg["buckets"])}
        y = np.array([bmap[l] for l in labels])
    e, reducer = reduce_dims(cloud, method, n_components=2, seed=0, return_reducer=True,
                             **({"y": y} if y is not None else {}))
    assert hasattr(reducer, "transform"), f"method={method!r} has no .transform() (cannot project the trajectory)"
    traj2 = reducer.transform(traj)
    goal2 = reducer.transform(t_e.reshape(1, -1))[0]
    if reward_mode:                                              # scalar reward field -> colormap + colorbar
        cn = cloud / (np.linalg.norm(cloud, axis=1, keepdims=True) + 1e-9)
        color = cn @ (t_e / (np.linalg.norm(t_e) + 1e-9))
        legend, cbar = None, "reward  cos(f_z(z), f_t(goal))"
    else:                                                        # categorical concept coloring + legend
        assert labels is not None and factor_cfg is not None, "reward_mode=False needs labels + factor_cfg"
        color, legend = point_colors(labels, factor_cfg); cbar = ""
    lims = pad_lims(np.concatenate([e, traj2, goal2[None]], axis=0))
    fig = viz.fig_points_2d(e, color=color, lims=lims, point_size=point_size, legend=legend,
                            cbar_label=cbar, marks=[{"pos": goal2, "text": "G"}], title=title)
    idx = np.unique(np.linspace(1, len(traj2), min(n_frames, len(traj2))).astype(int))
    frames = viz.animate_latent(fig, traj2, is3d=False, idx=idx, log=log, tail=tail)
    plt.close(fig)
    return frames
