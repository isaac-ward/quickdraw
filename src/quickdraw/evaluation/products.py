"""Shared eval-product emission: ONE naming convention + one emit helper per product family, used by
every torus rollout/control routine (open-loop horizon, OOD-split axes, MPPI control). Before this, each
routine hand-wrote its own `writer.figure(f"{routine}/{head}/...")` f-strings and re-implemented the same
filmstrip/rollout/curve/scene emission, which drifted (`rollout` vs `prediction_video`, ...).

Everything routes through `product_tag`, so the folder/label convention is defined once and generalizes to
N image heads for free (callers just loop the heads from `model.layout`):
    <routine>/<name>_<i>            global / proprio product, instance i     (e.g. eval_control/control_video_0)
    <routine>/<head>/<name>_<i>     per-image-head product, instance i       (e.g. eval_ood_horizon/image/rollout_0)
    <routine>/<name>_<scale>        averaged curve                           (e.g. .../error_vs_step_avg_log)
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from ..logging import viz


def product_tag(routine: str, name: str, *, i: int | None = None, head: str | None = None) -> str:
    """Canonical writer tag. `head` nests the product under a per-head subdir; `i` appends the instance index."""
    leaf = name if i is None else f"{name}_{i}"
    return "/".join([routine] + ([head] if head else []) + [leaf])


def torus_scene(R: float, r: float, *, description: str, **fields) -> dict:
    """3D-scene JSON header shared by every torus product (Blender ingest). Callers add their own
    path/agent/action fields via **fields."""
    return {"description": description,
            "coordinate_system": "world xyz, same space as the torus",
            "torus": {"major_radius_R": float(R), "tube_radius_r": float(r)}, **fields}


def log_error_curves(writer, routine, curves, step, *, head=None, split_top=None, colors=None,
                     yscales=("linear", "log"), name="error_vs_step_avg", scalars=True):
    """Emit `fig_error_vs_step` at each y-scale + `<metric>_mean` scalars, under the head prefix if given.
    curves: {metric_name: (H,) array}. Mirrors proprio and image heads with identical structure."""
    for ys in yscales:
        f = viz.fig_error_vs_step(curves, yscale=ys, split_top=split_top, colors=colors)
        writer.figure(product_tag(routine, f"{name}_{ys}", head=head), f, step)
        plt.close(f)
    if scalars:
        writer.scalars({product_tag(routine, f"{k}_mean", head=head): float(np.mean(v))
                        for k, v in curves.items()}, step)


def log_image_head(writer, routine, head, i, true_full, pred, step, fps, *,
                   context_len, filmstrip=True, n_cols=8, title=None):
    """Per-head image pred-vs-actual products for instance i: `<head>/rollout_i` (video) and, if
    `filmstrip`, `<head>/filmstrip_i` (figure). true_full: (context_len+H, s, s, 3) in [0,1] (the shared
    context frames precede the fork); pred: (H, s, s, 3) in [0,1]. The filmstrip compares pred against the
    post-fork future `true_full[context_len:]`."""
    writer.video(product_tag(routine, "rollout", head=head, i=i),
                 viz.image_rollout_video(true_full, pred, context_len=context_len).astype(np.uint8), fps, step)
    if filmstrip:
        ff = viz.fig_image_filmstrip(pred, true_full[context_len:], n_cols=n_cols,
                                     title=title or f"{head} #{i} pred(top)/GT(bottom)")
        writer.figure(product_tag(routine, "filmstrip", head=head, i=i), ff, step)
        plt.close(ff)
        # raw frames behind the filmstrip -> logs/epoch_<step>/<routine>/<head>/raw_frames_<i>.npz (pred + GT,
        # float32 [0,1]). Keeps native pixels so sharpness can be judged later (PSNR can't; a figure is downscaled).
        writer.array(product_tag(routine, "raw_frames", head=head, i=i), step,
                     pred=np.asarray(pred, np.float32), gt=np.asarray(true_full[context_len:], np.float32))


def emit_openloop(writer, routine, step, *, R, r, coloring, fps, P, smooth_window, description,
                  ctx_xyz, p_true_xyz, p_hat_xyz, actions, curves, n_plot, images=None, title_fn=None, log=None):
    """The ONE open-loop product orchestrator, shared by eval_ood_horizon (multimodal) and _openloop_split
    (vector). Given a COMPLETED rollout's per-episode positions + aggregate curves, it emits everything:
    proprio `error_vs_step_avg` + each head's image `error_vs_step_avg`, then per-episode `trajectory_{plot,
    video}_i` (+scene) and each head's `rollout_i`/`filmstrip_i`. The rollout itself stays modality-specific
    (dict-obs image decode vs vector tensor) — only the emission is unified here.
      ctx_xyz (N,P,3); p_true_xyz/p_hat_xyz (N,H,3); actions[i] -> (T,2) applied along the true path;
      curves {metric:(H,)}; images (optional) {head: {"icurves": {m:(H,)}, "full_true": (N,P+H,s,s,3),
      "ipred": (N,H,s,s,3)}}. title_fn(i) -> a plot title (default '<routine> #i')."""
    title_fn = title_fn or (lambda i: f"{routine} #{i}")
    log_error_curves(writer, routine, curves, step)                        # proprio averaged curves + scalars
    for head, d in (images or {}).items():                                 # per image head, mirrored (PSNR split top)
        log_error_curves(writer, routine, d["icurves"], step, head=head, split_top={"psnr"}, colors={"psnr": "red"})
    for i in range(n_plot):
        if log is not None:
            log(f"episode {i + 1}/{n_plot} visuals")
        anchor = ctx_xyz[i][-1:]                                           # shared launch state; branches fork here
        log_torus_paths(writer, routine, i, R=R, r=r, coloring=coloring, ctx_xyz=ctx_xyz[i],
                        true_xyz=np.concatenate([anchor, p_true_xyz[i]]),
                        pred_xyz=np.concatenate([anchor, p_hat_xyz[i]]),
                        actions=actions[i], P=P, step=step, fps=fps, smooth_window=smooth_window,
                        title=title_fn(i), description=description, log=log)
        for head, d in (images or {}).items():
            log_image_head(writer, routine, head, i, d["full_true"][i], d["ipred"][i], step, fps,
                           context_len=P, title=f"{head} #{i} pred(top)/GT(bottom)")


def log_torus_paths(writer, routine, i, *, R, r, coloring, ctx_xyz, true_xyz, pred_xyz, actions, P,
                    step, fps, smooth_window, title, description, log=None):
    """Proprio open-loop products for instance i: `trajectory_plot_i` (PNG), `trajectory_video_i` (MP4) +
    its scene JSON. ctx_xyz (P,3) shared context; true_xyz/pred_xyz ((H+1),3) each START with the shared
    launch state so truth and prediction branch from the same anchor. actions: applied actions along the
    true path (len == context+branch). The two branches fork at step P."""
    trajs = [{"xyz": ctx_xyz, "color": "lightgray", "start_sphere": True, "end_sphere": False,
              "start_scale": 0.5, "marker_color": "black"},
             {"xyz": true_xyz, "color": "black", "start_sphere": False, "end_sphere": True},
             {"xyz": pred_xyz, "color": "dimgray", "start_sphere": False, "end_sphere": True}]
    fp = viz.fig_torus_atlas(R, r, trajs=trajs, coloring=coloring, title=title,
                             view_pad=viz.EVAL_VIEW_PAD, torus_opacity=viz.TORUS_OPACITY)
    writer.figure(product_tag(routine, "trajectory_plot", i=i), fp, step); plt.close(fp)
    true_full = np.concatenate([ctx_xyz, true_xyz[1:]], axis=0)
    pred_full = np.concatenate([ctx_xyz, pred_xyz[1:]], axis=0)
    avec = viz.action_ambient(true_full, actions, R, r)
    frames = viz.traj_compare_frames(R, r, coloring, true_full, pred_full, avec, P,
                                     n_frames=len(true_full), title=title, smooth_window=smooth_window, log=log)
    writer.video(product_tag(routine, "trajectory_video", i=i), frames, fps, step)
    writer.scene(product_tag(routine, "trajectory_video", i=i),
                 torus_scene(R, r, description=description, true_path_xyz=true_full, predicted_path_xyz=pred_full,
                             fork_step_index=int(P),
                             action_arrow_per_step={"origins_xyz": true_full[:len(avec)], "vectors_xyz": avec}), step)


def load_latent_projection(interpret_run, method, factor, dim, *, fc, reward, request, hull_frac=0.8):
    """Prepare a latent-space-animation backdrop from an eval_interpret run's saved projection for
    (method, factor, dim): the point cloud + per-point colors/legend/lims (colored by `factor`), plus the two
    target crosses X_c (centroid of the request-labeled points) and X_r (the highest-reward point). Returns
    None if the reducer has no out-of-sample map (tsne) — we can't project the agent's new latents then.
      fc: the factor's config (buckets/colors) for point_colors; reward: a LanguageReward; request: e.g. 'red'."""
    import glob
    import os
    import pickle

    import torch

    from . import interpret as I
    from .manifold import pad_lims
    base = glob.glob(os.path.join(interpret_run, "logs", "epoch_*", "eval_interpret"))
    assert base, f"no eval_interpret outputs under {interpret_run}/logs/epoch_*/"
    pdir = os.path.join(base[0], "saved_projections")
    sup = method == "lda" or method.startswith("umap-sup")             # supervised reducers include the factor in the name
    key = f"{method}_{factor}_{dim}d" if sup else f"{method}_{dim}d"
    with open(os.path.join(pdir, f"{key}_reducer.pkl"), "rb") as fh:
        reducer = pickle.load(fh)
    if not hasattr(reducer, "transform"):
        return None                                                    # tsne: no out-of-sample projection
    emb = np.load(os.path.join(pdir, f"{key}_embedding.npy"))[:, :dim]  # (N, dim)
    clip_idx = np.load(os.path.join(pdir, "clip_index.npy"))
    latents = np.load(os.path.join(pdir, "latents.npy"))               # (N, D) — for the reward argmax (X_r)
    recs = json.load(open(os.path.join(base[0], "labels.json")))
    labels = np.array([recs[int(c)]["label"][factor] for c in clip_idx])   # per-point label for THIS factor
    rgb, legend = I.point_colors(list(labels), fc)
    # a compound request ("top red") mentions >=1 of THIS factor's buckets; C/hull = those points (color plot ->
    # 'red', positioning plot -> 'top'). If the request names none of this factor's buckets, fall back to all.
    rq = str(request).lower()
    relevant = [b for b in fc["buckets"] if b.lower() in rq]
    mask = np.isin(labels, relevant) if relevant else np.ones(len(labels), bool)
    cluster = emb[mask]
    xc = cluster.mean(0) if len(cluster) else emb.mean(0)             # centroid of the request's relevant-bucket points
    t_e = reward.text_embedding(request)                             # full compound direction (all factors)
    with torch.no_grad():
        R = reward.score(torch.from_numpy(latents.astype(np.float32)), t_e).cpu().numpy()   # (N,)
    xr = emb[int(R.argmax())]                                          # highest-(compound-)reward point
    red = cluster
    if len(red):                                                      # trim to the inner hull_frac by distance from
        d = np.linalg.norm(red - red.mean(0), axis=1)                 # the centroid, so stray members don't balloon it
        red = red[d <= np.quantile(d, hull_frac)]
    return {"emb": emb, "reducer": reducer, "rgb": rgb, "legend": legend, "lims": pad_lims(emb), "dim": dim,
            "marks": [{"pos": xc, "text": "C"}, {"pos": xr, "text": "M"}],   # C=centroid, M=max-reward
            "hull": red if len(red) >= (4 if dim == 3 else 3) else None}


def render_latent_video(proj, agent_latents, *, n_frames=None, point_size=2.5, title="", log=None):
    """Animate the agent moving through a prepared projection backdrop (from load_latent_projection): project
    its per-step latents with the SAME reducer, then blit the agent (sphere + trail) over the SAME figure the
    eval_interpret PNG uses (fig_points_9view for 3D — all 9 views — or fig_points_2d), built ONCE with the
    cloud + C/M circles + request hull. 3D uses a translucent cloud (alpha 0.5) so the agent shows through.
    n_frames=None -> one frame per control step (syncs 1:1 with the control video at the same fps). (T,H,W,3)."""
    traj = np.asarray(proj["reducer"].transform(agent_latents))                    # (T, ncomp)
    if traj.shape[1] < proj["dim"]:                                                # LDA gives only n_classes-1 axes;
        traj = np.hstack([traj, np.zeros((len(traj), proj["dim"] - traj.shape[1]))])   # zero-pad to match the (padded) embedding
    traj = traj[:, :proj["dim"]]                                                    # (T, dim), SAME space as the backdrop
    T = len(traj)
    idx = list(range(1, T + 1)) if n_frames is None else list(np.linspace(1, T, min(n_frames, T)).astype(int))
    is3d = proj["dim"] == 3
    common = dict(color=proj["rgb"], lims=proj["lims"], point_size=point_size, legend=proj["legend"],
                  title=title, marks=proj["marks"], hull=proj.get("hull"))
    fig = (viz.fig_points_9view(proj["emb"], alpha=0.5, **common) if is3d       # translucent cloud so the agent shows
           else viz.fig_points_2d(proj["emb"], **common))
    fig.set_dpi(viz.VIDEO_DPI)
    frames = viz.animate_latent(fig, traj, is3d, idx, log=log)                   # backdrop rasterized ONCE, agent blitted
    plt.close(fig)
    return frames
