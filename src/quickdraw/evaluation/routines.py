"""Eval routines — one definition each, used both as in-training subscriptions (on a cadence) and
as standalone post-hoc steps. REGISTRY maps name -> routine.

A routine: (cfg, model, norm, ecfg, writer, device, step) -> summary dict. It logs every scalar and
plot through `writer` (one call -> local + wandb identically). Routines are read-only (no grad).
"""

from __future__ import annotations

import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from ..controller.run import _plog, run_and_log_control
from ..logging import viz
from ..training.setup import eval_episodes
import torch

from .openloop import eval_batched


def _is_mm(model):
    return hasattr(getattr(model, "_orig_mod", model), "layout")


def _openloop_split(cfg, model, norm, writer, device, split, R, r, v_scale, prefix, step, coloring="rainbow", fps=60):
    if _is_mm(model):                      # multimodal models use eval_vision for rollouts (dict obs) — skip
        return {}
    t0 = time.perf_counter()
    eps = eval_episodes(cfg, norm, split)  # whole split; one batched rollout for all of it
    obs = torch.stack([eps[i]["obs_seq"] for i in range(len(eps))]).to(device)
    act = torch.stack([eps[i]["act_seq"] for i in range(len(eps))]).to(device)
    n_eval = cfg.eval.get("n_episodes", None)  # cap the eval N (config knob); null -> whole split
    if n_eval is not None:
        obs, act = obs[: int(n_eval)], act[: int(n_eval)]
    P, win = cfg.data.P, int(cfg.data.action_smooth_window)
    n_plot = min(int(cfg.eval.n_plot), obs.shape[0])
    _plog(writer, f"[{prefix} @ep{step}] start: {obs.shape[0]} episodes, {obs.shape[1]}-step open-loop rollout, "
                  f"{n_plot} plot/video episodes")
    res = eval_batched(model, norm, R, r, v_scale, P, obs, act)
    _plog(writer, f"[{prefix} @ep{step}] rollout done in {time.perf_counter() - t0:.1f}s; rendering...")

    for i in range(n_plot):
        ctx_xyz = res["ctx_xyz"][i]
        anchor = ctx_xyz[-1:]  # shared launch state o_{P-1}; truth & prediction branch from here
        true_xyz = np.concatenate([anchor, res["p_true_xyz"][i]], axis=0)
        pred_xyz = np.concatenate([anchor, res["p_hat_xyz"][i]], axis=0)
        # PNG: context (light grey) -> ground truth (black) -> prediction (dark grey), all solid.
        # Half-size start sphere on the context (matches summary plots), end spheres on truth + pred.
        trajs = [{"xyz": ctx_xyz, "color": "lightgray", "start_sphere": True, "end_sphere": False,
                  "marker_color": "black", "start_scale": 0.5},
                 {"xyz": true_xyz, "color": "black", "start_sphere": False, "end_sphere": True},
                 {"xyz": pred_xyz, "color": "dimgray", "start_sphere": False, "end_sphere": True}]
        f_traj = viz.fig_torus_atlas(R, r, trajs=trajs, coloring=coloring, title=f"{split} #{i}",
                                     view_pad=viz.EVAL_VIEW_PAD, torus_opacity=viz.TORUS_OPACITY)
        writer.figure(f"{prefix}/trajectory_plot_{i}", f_traj, step)   # per-rollout error curve dropped (avg-only)
        plt.close(f_traj)
        # MP4 mirror: full true/pred paths (context + branch) + the true applied-action arrow
        true_full = np.concatenate([ctx_xyz, true_xyz[1:]], axis=0)
        pred_full = np.concatenate([ctx_xyz, pred_xyz[1:]], axis=0)
        avec = viz.action_ambient(true_full, res["actions"][i], R, r)
        _plog(writer, f"[{prefix} @ep{step}]   episode {i + 1}/{n_plot} video ({len(true_full)} frames)")
        frames = viz.traj_compare_frames(R, r, coloring, true_full, pred_full, avec, P,
                                         n_frames=len(true_full), title=f"{split} #{i}", smooth_window=win,
                                         log=lambda m, i=i: _plog(writer, f"[{prefix} @ep{step}]     ep{i} {m}"))
        writer.video(f"{prefix}/trajectory_video_{i}", frames, fps, step)
        writer.scene(f"{prefix}/trajectory_video_{i}", {  # 3D geometry for Blender (plain-language keys)
            "description": "Open-loop long-horizon rollout on the torus: a BLACK agent on the TRUE path and a "
                           "GREY agent on the model's PREDICTED path. They share the context, then diverge at "
                           "the fork step. The action arrow is the applied action along the true path.",
            "coordinate_system": "world xyz, same space as the torus",
            "torus": {"major_radius_R": float(R), "tube_radius_r": float(r)},
            "true_path_xyz": true_full,                 # (T,3)
            "predicted_path_xyz": pred_full,            # (T,3)
            "fork_step_index": int(P),                  # prediction diverges from truth at this index
            "action_arrow_per_step": {"origins_xyz": true_full[:len(avec)], "vectors_xyz": avec},
        }, step)

    # dataset-aggregated error vs rollout step (mean of each metric over all episodes), both y-scales
    for ys in ("linear", "log"):
        f_avg = viz.fig_error_vs_step(res["agg"], yscale=ys)
        writer.figure(f"{prefix}/error_vs_step_avg_{ys}", f_avg, step)
        plt.close(f_avg)
    summary = {m: float(res["agg"][m].mean()) for m in res["agg"]}  # mean over the rollout
    writer.scalars({f"{prefix}/{m}_mean": v for m, v in summary.items()}, step)
    _plog(writer, f"[{prefix} @ep{step}] done in {time.perf_counter() - t0:.1f}s")
    return summary


@torch.no_grad()
def _mm_openloop(cfg, m, norm, ecfg, writer, device, step):
    """Multimodal long-horizon open loop: PROPRIO rollout (images stay in the latent, not decoded — cheap)
    over held-out val episodes. Logs error-vs-step (avg, linear+log) + a torus pred-vs-true trajectory plot."""
    import numpy as _np

    from ..data.dataset import load_split_episodes_mm
    was = m.training
    m.eval()
    t0 = time.perf_counter()
    P = cfg.data.P
    img_heads = [n for n, _ in m.layout if n != "proprio"]
    img_size = next((mod.ae.cfg.img_size for mod in m.modalities.values() if hasattr(mod, "ae")), 128)
    eps = load_split_episodes_mm(cfg.data.root, "val", img_size=img_size)
    n_ep = min(int(cfg.eval.get("n_episodes", 32) or 32), len(eps))
    eps = eps[:n_ep]
    H = min(int(cfg.eval.get("horizon", 2048)), min(len(o) for o, _, _ in eps) - P - 1)
    pro = torch.stack([norm.norm_obs(torch.from_numpy(o[:P])) for o, _, _ in eps]).float().to(device)
    ctx = {"proprio": pro}
    for h in img_heads:
        ctx[h] = torch.stack([torch.from_numpy(im[:P]) for _, _, im in eps]).float().div(255.0).to(device)
    acts = torch.stack([torch.from_numpy(a[:P + H - 1]) for _, a, _ in eps]).float().to(device)
    pred = m.imagine_eval(ctx, acts, H, heads=["proprio"])["proprio"]            # (n_ep,H,6)
    p_hat = torch.nan_to_num(norm.denorm_obs(pred), nan=10.0, posinf=10.0, neginf=-10.0)
    p_true = torch.stack([torch.from_numpy(o[P:P + H]) for o, _, _ in eps]).float().to(device)
    err = (p_hat[..., :3] - p_true[..., :3]).norm(dim=-1).mean(0).cpu().numpy()  # position L2 per step
    curves = {"pointwise_error": err}
    for ys in ("linear", "log"):
        f = viz.fig_error_vs_step(curves, yscale=ys)
        writer.figure(f"eval_ood_horizon/error_vs_step_avg_{ys}", f, step); plt.close(f)
    # torus trajectory plot (episode 0): context (light) -> true (black) -> pred (grey)
    ctx_xyz = norm.denorm_obs(pro[0]).cpu().numpy()[:, :3]
    true_xyz = _np.concatenate([ctx_xyz[-1:], p_true[0, :, :3].cpu().numpy()])
    pred_xyz = _np.concatenate([ctx_xyz[-1:], p_hat[0, :, :3].cpu().numpy()])
    trajs = [{"xyz": ctx_xyz, "color": "lightgray", "start_sphere": True, "end_sphere": False, "start_scale": 0.5},
             {"xyz": true_xyz, "color": "black", "start_sphere": False, "end_sphere": True},
             {"xyz": pred_xyz, "color": "dimgray", "start_sphere": False, "end_sphere": True}]
    f = viz.fig_torus_atlas(ecfg.R, ecfg.r, trajs=trajs, title=f"eval_ood_horizon (proprio) H={H}",
                            view_pad=viz.EVAL_VIEW_PAD, torus_opacity=viz.TORUS_OPACITY)
    writer.figure("eval_ood_horizon/trajectory_plot_0", f, step); plt.close(f)
    writer.scalars({"eval_ood_horizon/pointwise_error_mean": float(err.mean())}, step)
    if was:
        m.train()
    _plog(writer, f"[ood_horizon-mm @ep{step}] {n_ep} eps, H={H}, done in {time.perf_counter() - t0:.1f}s")
    return {"eval_ood_horizon": float(err.mean())}


def eval_ood_horizon(cfg, model, norm, ecfg, writer, device, step=0):
    """Long-horizon open-loop rollout on the base geometry. OOD because the rollout is far longer
    than the short horizon trained on; same env, so scored on ecfg's geometry."""
    m = getattr(model, "_orig_mod", model)
    if hasattr(m, "layout"):               # multimodal: proprio long-horizon (images stay latent)
        return _mm_openloop(cfg, m, norm, ecfg, writer, device, step)
    s = _openloop_split(cfg, model, norm, writer, device, "eval_ood_horizon", ecfg.R, ecfg.r,
                        ecfg.init_speed, "eval_ood_horizon", step, fps=round(1.0 / ecfg.dt))
    return {"eval_ood_horizon": s}


def _ood_axis(cfg, model, norm, ecfg, writer, device, step, split):
    """Open-loop on one OOD split, scored on its own geometry + drawn with its coloring (from the
    dataset card). Shared by the visual/geometric/dynamics axes."""
    card = json.load(open(os.path.join(cfg.data.root, "dataset_card.json")))
    split_env, coloring = card.get("split_env", {}), card.get("coloring", {})
    se = split_env.get(split, {"R": ecfg.R, "r": ecfg.r})
    s = _openloop_split(cfg, model, norm, writer, device, split, se["R"], se["r"],
                        se.get("init_speed", ecfg.init_speed), split, step,
                        coloring.get(split, "rainbow"), fps=round(1.0 / ecfg.dt))
    return {split: s}


def eval_ood_visual(cfg, model, norm, ecfg, writer, device, step=0):
    return _ood_axis(cfg, model, norm, ecfg, writer, device, step, "eval_ood_visual")


def eval_ood_geometric(cfg, model, norm, ecfg, writer, device, step=0):
    return _ood_axis(cfg, model, norm, ecfg, writer, device, step, "eval_ood_geometric")


def eval_ood_dynamics(cfg, model, norm, ecfg, writer, device, step=0):
    return _ood_axis(cfg, model, norm, ecfg, writer, device, step, "eval_ood_dynamics")


def eval_control(cfg, model, norm, ecfg, writer, device, step=0):
    """Dual MPPI control (oracle vs learned) through a random sequence of 8 goals. Multimodal models plan
    with an FPV context rendered in the loop (run_and_log_control handles it)."""
    return {"control": run_and_log_control(cfg, model, norm, ecfg, writer, device, step)}


def _quiver_frames_data(swarm, n_frames=120):
    """Per-frame geometry for the quiver: each decoded swarm ODE path (list of (k,3)) is interpolated to
    `n_frames` so the grey swarm flows smoothly from off-surface noise onto the torus over the animation,
    each member leaving a growing tail. Returns [{swarm: [{particle, trail}, ...]}, ...]."""
    swarm = [np.asarray(s) for s in (swarm or [])]

    def _along(path, s):                                              # interp position + growing trail at fraction s
        Kp = path.shape[0] - 1
        fc = s * Kp; i0 = int(np.floor(fc)); i1 = min(i0 + 1, Kp); w = fc - i0
        p = (1 - w) * path[i0] + w * path[i1]
        trail = np.vstack([path[: i0 + 1], p[None]]) if w > 1e-6 else path[: i0 + 1]
        return {"particle": p, "trail": trail}

    return [{"swarm": [_along(sp, f / (n_frames - 1) if n_frames > 1 else 0.0) for sp in swarm]}
            for f in range(n_frames)]


def _quiver_round_data(swarm, grow=10, collapse=5):
    """One round of a sequential swarm: a GROW phase (swarm flows noise->surface, trails growing) then a
    COLLAPSE phase (each tail RETRACTS onto its convergence endpoint, so the swarm ends as a tight knot at
    the predicted point) before the next round begins. grow + collapse frames total."""
    swarm = [np.asarray(s) for s in (swarm or [])]
    if not swarm:
        return [{"swarm": []}]
    K = swarm[0].shape[0] - 1

    def along(path, s):                                # particle at fraction s + growing trail
        fc = s * K; i0 = int(np.floor(fc)); i1 = min(i0 + 1, K); w = fc - i0
        p = (1 - w) * path[i0] + w * path[i1]
        return {"particle": p, "trail": np.vstack([path[: i0 + 1], p[None]]) if w > 1e-6 else path[: i0 + 1]}

    pf = [{"swarm": [along(sp, f / (grow - 1) if grow > 1 else 1.0) for sp in swarm]} for f in range(grow)]
    for f in range(collapse):                          # retract each tail onto its endpoint (the convergence)
        j = int(round((f + 1) / collapse * K))         # trail-start index advances 0 -> K (leaves a tight knot)
        pf.append({"swarm": [{"particle": sp[K], "trail": sp[min(j, K):]} for sp in swarm]})
    return pf


def _ssim(a, b):
    """Windowed SSIM over (N,H,W,3) images in [0,1] (uniform 7x7 window via avg_pool — pooling, not a
    learned conv). Returns mean SSIM scalar."""
    import torch.nn.functional as F
    a, b = a.permute(0, 3, 1, 2), b.permute(0, 3, 1, 2)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_a, mu_b = F.avg_pool2d(a, 7, 1), F.avg_pool2d(b, 7, 1)
    va = F.avg_pool2d(a * a, 7, 1) - mu_a ** 2
    vb = F.avg_pool2d(b * b, 7, 1) - mu_b ** 2
    cab = F.avg_pool2d(a * b, 7, 1) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + C1) * (2 * cab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2))
    return float(s.mean())


@torch.no_grad()
def eval_vision(cfg, model, norm, ecfg, writer, device, step=0):
    """Vision eval (SELF-SKIPS unless the model has image head(s)): autoregressive FPV rollout on held-out
    val episodes. Logs, under eval_ood_horizon/<image-head>/: a filmstrip (pred top / GT bottom, 8 steps),
    a synced rollout video (pred top black-through-context / GT bottom), and per-step psnr/ssim/mse curves
    (avg over episodes, linear + log). Reuses viz.fig_image_filmstrip / image_rollout_video."""
    import numpy as _np

    from ..data.dataset import load_split_episodes_mm
    m = getattr(model, "_orig_mod", model)
    if not hasattr(m, "layout"):
        return {}
    img_heads = [n for n, _ in m.layout if n != "proprio"]
    if not img_heads:
        return {}
    was = m.training
    m.eval()
    t0 = time.perf_counter()
    P = cfg.data.P
    H = min(int(cfg.data.F), 24)
    n_ep = 8
    img_size = m.modalities[img_heads[0]].ae.cfg.img_size
    eps = load_split_episodes_mm(cfg.data.root, "val", img_size=img_size)
    eps = [e for e in eps if len(e[0]) >= P + H][:n_ep]
    # batched context + real actions across the chosen episodes
    pro = torch.stack([norm.norm_obs(torch.from_numpy(o[:P])) for o, _, _ in eps]).float().to(device)
    imgs = {h: torch.stack([torch.from_numpy(im[:P]) for _, _, im in eps]).float().div(255.0).to(device) for h in img_heads}
    ctx = {"proprio": pro, **imgs}
    acts = torch.stack([torch.from_numpy(a[:P + H - 1]) for _, a, _ in eps]).float().to(device)
    out = m.imagine_eval(ctx, acts, H)                                  # {head:(n_ep,H,...)}
    _plog(writer, f"[vision @ep{step}] {len(eps)} episodes, H={H}, heads={img_heads}")

    for head in img_heads:
        pred = out[head].clamp(0, 1)                                    # (n_ep,H,size,size,3)
        true = torch.stack([torch.from_numpy(im[P:P + H]) for _, _, im in eps]).float().div(255.0).to(device)
        # per-step metrics averaged over episodes
        psnr_s, ssim_s, mse_s = [], [], []
        for t in range(H):
            mse = float(torch.mean((pred[:, t] - true[:, t]) ** 2))
            mse_s.append(mse)
            psnr_s.append(-10.0 * _np.log10(max(mse, 1e-12)))
            ssim_s.append(_ssim(pred[:, t], true[:, t]))
        curves = {"psnr": _np.array(psnr_s), "ssim": _np.array(ssim_s), "mse": _np.array(mse_s)}
        for ys in ("linear", "log"):
            f = viz.fig_error_vs_step(curves, yscale=ys)
            writer.figure(f"eval_ood_horizon/{head}/metric_vs_step_{ys}", f, step); plt.close(f)
        writer.scalars({f"val/metric/{head}/psnr": float(_np.mean(psnr_s)),
                        f"val/metric/{head}/ssim": float(_np.mean(ssim_s)),
                        f"val/metric/{head}/mse": float(_np.mean(mse_s))}, step)
        # filmstrip + synced rollout video for episode 0
        p0, t0f = pred[0].cpu().numpy(), true[0].cpu().numpy()
        full_true = eps[0][2][: P + H].astype(_np.float32) / 255.0
        ff = viz.fig_image_filmstrip(p0, t0f, n_cols=8,
                                     title=f"{head} rollout — pred (top) vs GT (bottom), H={H}, PSNR {_np.mean(psnr_s):.1f}dB")
        writer.figure(f"eval_ood_horizon/{head}/filmstrip", ff, step); plt.close(ff)
        vid = viz.image_rollout_video(full_true, p0, context_len=P)
        writer.video(f"eval_ood_horizon/{head}/rollout", vid.astype(_np.uint8), 60, step)   # 60 fps (matches dt)

    if was:
        m.train()
    _plog(writer, f"[vision @ep{step}] done in {time.perf_counter() - t0:.1f}s")
    return {}


@torch.no_grad()
def eval_manifold(cfg, model, norm, ecfg, writer, device, step=0):
    """Recovered-manifold UMAPs — works for ANY method. Pool the model's COMMITTED next-state prediction
    (deterministic forward() readout; for diffusion the eps=0 prediction) over many VAL contexts, then UMAP
    both the decoded DATA space (6D pos+vel) and the carried LATENT space to 3D and 2D (4 stills). The
    union traces the learned manifold; speed colors |predicted next velocity|."""
    from .manifold import manifold_predictions, manifold_predictions_mm, pad_lims, umap_reduce
    m = getattr(model, "_orig_mod", model)
    was = m.training
    m.eval()
    t0 = time.perf_counter()
    if hasattr(m, "layout"):               # multimodal: latent = flattened token bag, data = decoded proprio
        from ..data.dataset import load_split_episodes_mm
        img_size = next((mod.ae.cfg.img_size for mod in m.modalities.values() if hasattr(mod, "ae")), 128)
        mm_eps = load_split_episodes_mm(cfg.data.root, "val", img_size=img_size)
        data6d, latents, _, n_avail = manifold_predictions_mm(m, norm, mm_eps, P=cfg.data.P, n_points=2000,
                                                              stride=1, seed=0, device=device)
    else:
        eps_ds = eval_episodes(cfg, norm, "val")
        data6d, latents, _, n_avail = manifold_predictions(m, norm, eps_ds, P=cfg.data.P, n_points=5000,
                                                           stride=1, seed=0, device=device)
    sub = f"{data6d.shape[0]:,} next-state predictions (of {n_avail:,} val contexts)"   # model-agnostic
    for space, label, pts in (("data_space", "data space (full 6D pos+vel)", data6d),
                              ("latent_space", f"latent space (full {latents.shape[1]}D z)", latents)):
        for nd in (3, 2):
            e = umap_reduce(pts, n_components=nd, seed=0)
            fig_fn = viz.fig_points_4view if nd == 3 else viz.fig_points_2d
            f = fig_fn(e, lims=pad_lims(e), point_size=2.5,            # no color/colorbar (structure only)
                       title=f"recovered manifold — UMAP of {label} to {nd}D, seed=0\n{sub}")
            writer.figure(f"eval_manifold/umap_{space}_to_{nd}d", f, step); plt.close(f)
    if was:
        m.train()
    _plog(writer, f"[manifold @ep{step}] done in {time.perf_counter() - t0:.1f}s")
    return {}


@torch.no_grad()
def eval_diffusion_field(cfg, model, norm, ecfg, writer, device, step=0):
    """The headline diffusion artifacts (design/models/diffusion.md), both diffusion-SPECIFIC:
    (a) `denoising_multistep` (8 s): a FIXED agent (history + future lines static) while 32 SEQUENTIAL swarms
    each denoise — a grey swarm of decoded ODE paths flowing off-surface onto the torus, each leaving a tail
    that traces the field — then collapse their tails onto the next convergence point along the fixed future
    line, one after another. (b) `denoising_aggregate` (4 s): the denoising paths pooled over many val
    contexts collapsing from noise onto the recovered manifold. Diffusion-specific scalars:
    eval_diffusion/std_of_samples (std of the swarm's FINAL positions = predicted uncertainty) and
    eval_diffusion/time/{sample_s, sample_ms_per_euler_step}. (Pointwise accuracy lives in val/train
    pointwise_error; the static manifold UMAPs are the method-agnostic eval_manifold.) Self-SKIPS for non-diffusion."""
    from ..models.diffusion import Diffusion, _ln
    m = getattr(model, "_orig_mod", model)            # unwrap torch.compile
    if not isinstance(m, Diffusion):
        return {}
    t0 = time.perf_counter()
    R, r, v_scale = ecfg.R, ecfg.r, ecfg.init_speed
    coloring = "rainbow"
    was = m.training
    m.eval()
    eps_ds = eval_episodes(cfg, norm, "val")
    ep = eps_ds[0]
    obs = ep["obs_seq"].to(device)[None].float()      # (1, Tlen, 6) normalized
    act = ep["act_seq"].to(device)[None].float()      # (1, Tlen, 2) normalized
    Tlen, P, W, dz = obs.shape[1], cfg.data.P, m.window, m.cfg.dz
    z = m.encode_state(obs)                            # (1, Tlen, dz) LN'd
    K, n_swarm = m.sampling_steps, 16                  # SAME K as the model's real inference (fair metric) + swarm size
    g = torch.Generator(device=device).manual_seed(1234)   # reproducible swarm -> golden-able

    def decode_xyz(z_t, x):                            # latent residual x -> physical xyz (endpoint == committed metric pred)
        return norm.denorm_obs(m.to_obs(_ln(z_t + x)))[..., :3]

    def step_data(t, swarm_steps=None):                # per-step geometry; swarm uses swarm_steps (default K)
        ks = swarm_steps or K
        w = min(W, t + 1)
        h_t = m.transformer(m.to_token(z[:, t - w + 1:t + 1], act[:, t - w + 1:t + 1]))[:, -1]
        z_t = z[:, t]
        cur = norm.denorm_obs(obs[:, t])[0, :3].cpu().numpy()
        nxt = norm.denorm_obs(obs[:, t + 1])[0, :3].cpu().numpy()
        a_tail = norm.denorm_obs(obs[0, max(0, t - 60):t + 1])[:, :3].cpu().numpy()        # history (last ~60)
        a_fut = norm.denorm_obs(obs[0, t:t + 2])[:, :3].cpu().numpy()                      # FUTURE = current -> next
        #   (ends exactly where the denoising converges; the black line does not run past the predicted step)
        a_amb = viz.action_ambient(cur, norm.denorm_act(act[:, t])[0].cpu().numpy(), R, r)
        ts = time.perf_counter()
        m.flow.sample(h_t, steps=K, deterministic=True)            # the deterministic readout (timed; matches rollout)
        sample_s = time.perf_counter() - ts
        swarm, ends = [], []
        for _ in range(n_swarm):
            e = torch.randn(1, dz, generator=g, device=device)
            _, pth = m.flow.sample(h_t, steps=ks, deterministic=False, eps=e, record_path=True)
            sp = np.stack([decode_xyz(z_t, x)[0].cpu().numpy() for x in pth])
            swarm.append(sp); ends.append(sp[-1])
        return {"current": cur, "true_next": nxt, "agent_tail": a_tail, "future_path": a_fut,
                "action_amb": a_amb, "swarm": swarm, "ends": ends, "sample_s": sample_s}

    # (a) denoising_multistep: agent + history + future lines all STATIC; N SEQUENTIAL swarms — each predicts
    # the next consecutive step and denoises, then COLLAPSES its tails onto the convergence point before the
    # next swarm. The target advances along the fixed future line (which spans the N steps). 128 x 15 = 1920
    # frames @ 60 fps = 32 s. Also yields the diffusion scalars (std_of_samples + sample timing) over its steps.
    ms_t0 = max(P, 60)                                 # start late enough that the agent has a full ~60-step history tail
    n_ms = min(128, Tlen - 2 - ms_t0)                  # 4x longer than before; stay in-episode
    ms_cur = norm.denorm_obs(obs[:, ms_t0])[0, :3].cpu().numpy()
    ms_tail = norm.denorm_obs(obs[0, max(0, ms_t0 - 60):ms_t0 + 1])[:, :3].cpu().numpy()
    ms_future = norm.denorm_obs(obs[0, ms_t0:ms_t0 + n_ms + 1])[:, :3].cpu().numpy()   # current -> ms_t0+n_ms (fixed)
    ms_act = viz.action_ambient(ms_cur, norm.denorm_act(act[:, ms_t0])[0].cpu().numpy(), R, r)
    _plog(writer, f"[diffusion_field @ep{step}] denoising_multistep {n_ms} steps from {ms_t0}, K={K} swarm={n_swarm}")
    ms_steps, spreads, sample_times = [], [], []
    for i in range(n_ms):
        d = step_data(ms_t0 + i)                       # predict step ms_t0+i+1
        spreads.append(float(np.linalg.norm(np.stack(d["ends"]).std(axis=0))))   # std of the swarm's FINAL positions
        sample_times.append(d["sample_s"])
        ms_steps.append({"per_frame": _quiver_round_data(d["swarm"], grow=10, collapse=5),
                         "true_next": norm.denorm_obs(obs[:, ms_t0 + i + 1])[0, :3].cpu().numpy()})
    ms_frames = viz.diffusion_quiver_sequential_frames(R, r, coloring, ms_cur, ms_act, ms_tail, ms_future,
                                                       ms_steps, title="denoising multistep")
    writer.video("eval_diffusion/denoising_multistep", ms_frames, 60, step)  # N swarms x 15 frames @ 60 fps
    writer.scene("eval_diffusion/denoising_multistep", {
        "description": "Sequential swarms at a FIXED agent: each swarm denoises, then its tails collapse "
                       "onto the convergence point along the (fixed) black future line, before the next "
                       "swarm; agent, history and future do not move.",
        "coordinate_system": "world xyz, same space as the torus",
        "torus": {"major_radius_R": float(R), "tube_radius_r": float(r)},
        "current_position_xyz": ms_cur, "history_tail_xyz": ms_tail, "future_path_xyz": ms_future,
        "swarm_target_per_step_xyz": [s["true_next"] for s in ms_steps]}, step)

    # (b) denoising aggregate (diffusion-SPECIFIC): pool the denoising ODE paths over many VAL contexts and
    # animate the swarm collapsing from noise onto the recovered manifold. The static, method-agnostic
    # manifold UMAPs live in the separate eval_manifold routine.
    from .manifold import manifold_clouds
    MAN_N, MAN_VID = 5000, 240
    paths6d, _, _, n_avail = manifold_clouds(m, norm, eps_ds, P=P, n_points=MAN_N, cube=3.0,
                                             stride=1, seed=0, device=device)
    msub = (f"{'shortcut' if m.cfg.shortcut else 'rectified-flow'} (K={K}) — "
            f"{paths6d.shape[0]:,} next-states (of {n_avail:,} val contexts)")
    Lm, Zm = (R + r) * 1.05, r * 1.6
    plims = ((-Lm, Lm), (-Lm, Lm), (-Zm, Zm))
    mframes = viz.points_collapse_frames(paths6d[..., :3], lims=plims, n_frames=MAN_VID,    # flat purple
                                         point_size=2.0, title=f"denoising aggregate — noise → manifold\n{msub}")
    writer.video("eval_diffusion/denoising_aggregate", mframes, 60, step)  # 240 frames @ 60 fps = 4 s

    writer.scalars({"eval_diffusion/std_of_samples": float(np.mean(spreads)),   # uncertainty (no val equivalent)
                    "eval_diffusion/time/sample_s": float(np.mean(sample_times)),
                    "eval_diffusion/time/sample_ms_per_euler_step": float(1000.0 * np.mean(sample_times) / max(1, K))},
                   step)
    if was:
        m.train()
    _plog(writer, f"[diffusion_field @ep{step}] done in {time.perf_counter() - t0:.1f}s "
                  f"std_of_samples={np.mean(spreads):.4f}")
    return {"eval_diffusion_std_of_samples": float(np.mean(spreads))}


REGISTRY = {"ood_horizon": eval_ood_horizon, "ood_visual": eval_ood_visual,
            "ood_geometric": eval_ood_geometric, "ood_dynamics": eval_ood_dynamics,
            "control": eval_control, "diffusion_field": eval_diffusion_field, "manifold": eval_manifold,
            "vision": eval_vision}
