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
from ..environments import torus as T
from ..logging import viz
from ..training.setup import eval_episodes
import torch

from .openloop import eval_batched


def _openloop_split(cfg, model, norm, writer, device, split, R, r, v_scale, prefix, step, coloring="rainbow", fps=60):
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
        f_err = viz.fig_error_vs_step({m: res["per_step"][m][i] for m in res["per_step"]})
        writer.figure(f"{prefix}/trajectory_plot_{i}", f_traj, step)
        writer.figure(f"{prefix}/error_vs_step_{i}", f_err, step)
        plt.close(f_traj)
        plt.close(f_err)
        # MP4 mirror: full true/pred paths (context + branch) + the true applied-action arrow
        true_full = np.concatenate([ctx_xyz, true_xyz[1:]], axis=0)
        pred_full = np.concatenate([ctx_xyz, pred_xyz[1:]], axis=0)
        avec = viz.action_ambient(true_full, res["actions"][i], R, r)
        _plog(writer, f"[{prefix} @ep{step}]   episode {i + 1}/{n_plot} video ({len(true_full)} frames)")
        frames = viz.traj_compare_frames(R, r, coloring, true_full, pred_full, avec, P,
                                         n_frames=len(true_full), title=f"{split} #{i}", smooth_window=win,
                                         log=lambda m, i=i: _plog(writer, f"[{prefix} @ep{step}]     ep{i} {m}"))
        writer.video(f"{prefix}/trajectory_video_{i}", frames, fps, step)

    # dataset-aggregated error vs rollout step (mean of each metric over all episodes)
    f_avg = viz.fig_error_vs_step(res["agg"])
    writer.figure(f"{prefix}/error_vs_step_avg", f_avg, step)
    plt.close(f_avg)
    summary = {m: float(res["agg"][m].mean()) for m in res["agg"]}  # mean over the rollout
    writer.scalars({f"{prefix}/{m}_mean": v for m, v in summary.items()}, step)
    _plog(writer, f"[{prefix} @ep{step}] done in {time.perf_counter() - t0:.1f}s")
    return summary


def eval_ood_horizon(cfg, model, norm, ecfg, writer, device, step=0):
    """Long-horizon open-loop rollout on the base geometry. OOD because the rollout is far longer
    than the short horizon trained on; same env, so scored on ecfg's geometry."""
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
    """Dual MPPI control (oracle vs learned) through a random sequence of 8 goals."""
    return {"control": run_and_log_control(cfg, model, norm, ecfg, writer, device, step)}


def _torus_point(th, ph, R, r):
    rho = R + r * np.cos(ph)
    return np.array([rho * np.cos(th), rho * np.sin(th), r * np.sin(ph)])


def _quiver_frames_data(m, norm, h_t, z_t, cur_xyz, cur_vel, committed, R, r,
                        n_frames=14, grid=(-1, 0, 1), delta=0.05):
    """Per-tau-frame geometry for the quiver animation: at each tau (1->0) probe the field on a grid of
    positions near the agent (v=v_theta(enc(p), tau, h), obs-space arrow dec(z+delta*v) - dec(z)), and
    place the committed PARTICLE at its decoded ODE position for that frame (moving along its path)."""
    from ..models.diffusion import _ln  # noqa: F401 (decode of z_p uses to_obs directly; no LN drift here)
    import torch
    device, dz = z_t.device, z_t.shape[-1]
    th0, ph0 = T.angles_from_point(torch.as_tensor(cur_xyz, dtype=torch.float32)[None], R)
    th0, ph0 = float(th0), float(ph0)
    # grid of nearby positions on the surface (perturb the current angles), each encoded with the
    # current velocity -> a latent "guess" z_p the field acts on.
    zps, bases = [], []
    for dth in grid:
        for dph in grid:
            p = _torus_point(th0 + 0.25 * dth, ph0 + 0.5 * dph, R, r)
            obs_p = np.concatenate([p, cur_vel])                       # [position, current velocity]
            zp = m.encode_state(norm.norm_obs(torch.as_tensor(obs_p, dtype=torch.float32, device=device))[None])
            zps.append(zp)
            bases.append(p)
    zps = torch.cat(zps, dim=0)                                        # (G, dz)
    bases = np.stack(bases)
    d = (zps.new_full((zps.shape[0], 1), 1.0 / max(2, committed.shape[0] - 1))
         if m.flow.shortcut else None)                                # shortcut viz uses the FINE field
    h_g = h_t.expand(zps.shape[0], -1)
    K = committed.shape[0] - 1                                        # committed has K+1 decoded points
    per_frame = []
    for f in range(n_frames):
        tau = zps.new_full((zps.shape[0], 1), 1.0 - f / (n_frames - 1))
        v = m.flow.velocity(zps, tau, h_g, d)                         # (G, dz) latent velocity at this tau
        base = norm.denorm_obs(m.to_obs(zps))[:, :3].detach().cpu().numpy()
        tip = norm.denorm_obs(m.to_obs(zps + delta * v))[:, :3].detach().cpu().numpy()
        arrows = [(bases[i], tip[i] - base[i]) for i in range(bases.shape[0])]
        idx = int(round(f / (n_frames - 1) * K))                      # particle rides the committed path
        per_frame.append({"arrows": arrows, "particle": committed[idx], "trail": committed[: idx + 1]})
    return per_frame


@torch.no_grad()
def eval_diffusion_field(cfg, model, norm, ecfg, writer, device, step=0):
    """The headline diffusion artifact (design/models/diffusion.md): at ~4 FIXED prediction steps of
    episode 0, render the latent flow field through the decoder onto the torus as (a) streamline PNGs —
    a swarm of decoded ODE paths flowing off-surface onto the manifold, the bright committed (eps=0,
    metric) path, and the true-next marker — and (b) a tau-sweep quiver animation with the committed
    particle riding its decoded path. Logs diffusion/{streamline,quiver}/example_{0..3} +
    diffusion/{flow_endpoint_error, sample_spread}. Self-SKIPS (returns {}) for non-diffusion models."""
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
    n_steps = 4
    lo, hi = P, Tlen - 2
    steps_idx = [int(round(lo + (hi - lo) * k / (n_steps - 1))) for k in range(n_steps)]  # fixed, comparable across epochs
    K, n_swarm = 16, 16                                # viz path resolution (decoupled from sampling_steps) + swarm size
    g = torch.Generator(device=device).manual_seed(1234)   # reproducible swarm -> golden-able

    def decode_xyz(z_t, x):                            # latent residual x -> physical xyz (endpoint == committed metric pred)
        return norm.denorm_obs(m.to_obs(_ln(z_t + x)))[..., :3]

    _plog(writer, f"[diffusion_field @ep{step}] {n_steps} steps {steps_idx}, K={K} swarm={n_swarm}")
    endpoint_errs, spreads = [], []
    for si, t in enumerate(steps_idx):
        w = min(W, t + 1)
        s_win, a_win = z[:, t - w + 1:t + 1], act[:, t - w + 1:t + 1]   # token i consumes a_i
        h_t = m.transformer(m.to_token(s_win, a_win))[:, -1]            # (1, d) teacher-forced context
        z_t = z[:, t]                                                   # (1, dz)
        cur_xyz = norm.denorm_obs(obs[:, t])[0, :3].cpu().numpy()
        cur_vel = norm.denorm_obs(obs[:, t])[0, 3:].cpu().numpy()
        nxt_xyz = norm.denorm_obs(obs[:, t + 1])[0, :3].cpu().numpy()
        act_amb = viz.action_ambient(cur_xyz, norm.denorm_act(act[:, t])[0].cpu().numpy(), R, r)
        _, cpath = m.flow.sample(h_t, steps=K, deterministic=True, record_path=True)
        committed = np.stack([decode_xyz(z_t, x)[0].cpu().numpy() for x in cpath])        # (K+1, 3)
        swarm, ends = [], []
        for _ in range(n_swarm):
            e = torch.randn(1, dz, generator=g, device=device)
            _, pth = m.flow.sample(h_t, steps=K, deterministic=False, eps=e, record_path=True)
            sp = np.stack([decode_xyz(z_t, x)[0].cpu().numpy() for x in pth])
            swarm.append(sp)
            ends.append(sp[-1])
        endpoint_errs.append(float(np.linalg.norm(committed[-1] - nxt_xyz) / r))          # tube-radii
        spreads.append(float(np.linalg.norm(np.stack(ends).std(axis=0))))
        f_s = viz.fig_diffusion_streamline(R, r, coloring, swarm, committed, cur_xyz, act_amb, nxt_xyz,
                                           title=f"diffusion flow  step {t}  @ep{step}")
        writer.figure(f"diffusion/streamline/example_{si}", f_s, step)
        plt.close(f_s)
        per_frame = _quiver_frames_data(m, norm, h_t, z_t, cur_xyz, cur_vel, committed, R, r)
        frames = viz.diffusion_quiver_frames(R, r, coloring, cur_xyz, act_amb, per_frame,
                                             title=f"diffusion quiver step {t}")
        writer.video(f"diffusion/quiver/example_{si}", frames, 8, step)
    writer.scalars({"diffusion/flow_endpoint_error": float(np.mean(endpoint_errs)),
                    "diffusion/sample_spread": float(np.mean(spreads))}, step)
    if was:
        m.train()
    _plog(writer, f"[diffusion_field @ep{step}] done in {time.perf_counter() - t0:.1f}s "
                  f"endpoint_err={np.mean(endpoint_errs):.3f} spread={np.mean(spreads):.4f}")
    return {"diffusion_flow_endpoint_error": float(np.mean(endpoint_errs))}


REGISTRY = {"ood_horizon": eval_ood_horizon, "ood_visual": eval_ood_visual,
            "ood_geometric": eval_ood_geometric, "ood_dynamics": eval_ood_dynamics,
            "control": eval_control, "diffusion_field": eval_diffusion_field}
