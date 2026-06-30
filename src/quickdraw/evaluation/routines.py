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


def _quiver_frames_data(committed, swarm=None, n_frames=120):
    """Per-frame geometry for the quiver animation (a denoising FLOW animation; no field arrows). Each
    decoded ODE path is interpolated to `n_frames` so playback is smooth at 60 fps. The RED committed
    particle rides its decoded path (`committed`, (k,3)) leaving a tail; the grey `swarm` (the streamline's
    decoded stochastic paths, list of (k,3)) rides alongside as tail-less points so you watch the flow land
    from off-surface noise onto the torus."""
    committed = np.asarray(committed)
    swarm = [np.asarray(s) for s in (swarm or [])]

    def _along(path, s):                                              # interp position + growing trail at fraction s
        Kp = path.shape[0] - 1
        fc = s * Kp; i0 = int(np.floor(fc)); i1 = min(i0 + 1, Kp); w = fc - i0
        p = (1 - w) * path[i0] + w * path[i1]
        trail = np.vstack([path[: i0 + 1], p[None]]) if w > 1e-6 else path[: i0 + 1]
        return {"particle": p, "trail": trail}

    per_frame = []
    for f in range(n_frames):
        s = f / (n_frames - 1) if n_frames > 1 else 0.0              # 0->1 as tau goes 1->0
        fr = _along(committed, s)                                     # red committed particle + trail
        fr["swarm"] = [_along(sp, s) for sp in swarm]                 # grey swarm points
        per_frame.append(fr)
    return per_frame


@torch.no_grad()
def eval_diffusion_field(cfg, model, norm, ecfg, writer, device, step=0):
    """The headline diffusion artifact (design/models/diffusion.md). At 3 FIXED prediction steps of episode
    0, render the latent flow field through the decoder onto the torus as a tau-sweep quiver ATLAS animation
    (2 s): a grey swarm of decoded ODE paths flowing off-surface onto the manifold (each leaving a tail that
    traces the field), the agent's black history tail AND future path, and a black truth ring (no committed/
    red particle). ALSO a `quiver_multistep` (4 s): 16 consecutive denoising predictions as the agent walks
    forward. The metric uses the model's NATIVE sampling_steps K, so it matches the rollout. Logs
    eval_diffusion/quiver/example_{0..2} + eval_diffusion/quiver_multistep + eval_diffusion/{pointwise_error, sample_spread,
    time/sample_s, time/sample_ms_per_euler_step}. Self-SKIPS (returns {}) for non-diffusion models."""
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
    n_steps = 3
    lo, hi = P, Tlen - 2
    steps_idx = [int(round(lo + (hi - lo) * k / (n_steps - 1))) for k in range(n_steps)]  # fixed, comparable across epochs
    K, n_swarm = m.sampling_steps, 16                  # SAME K as the model's real inference (fair metric) + swarm size
    g = torch.Generator(device=device).manual_seed(1234)   # reproducible swarm -> golden-able

    def decode_xyz(z_t, x):                            # latent residual x -> physical xyz (endpoint == committed metric pred)
        return norm.denorm_obs(m.to_obs(_ln(z_t + x)))[..., :3]

    def step_data(t):                                  # all per-step geometry + the K-step sample wall time
        w = min(W, t + 1)
        h_t = m.transformer(m.to_token(z[:, t - w + 1:t + 1], act[:, t - w + 1:t + 1]))[:, -1]
        z_t = z[:, t]
        cur = norm.denorm_obs(obs[:, t])[0, :3].cpu().numpy()
        nxt = norm.denorm_obs(obs[:, t + 1])[0, :3].cpu().numpy()
        a_tail = norm.denorm_obs(obs[0, max(0, t - 60):t + 1])[:, :3].cpu().numpy()        # history (last ~60)
        a_fut = norm.denorm_obs(obs[0, t + 1:t + 61])[:, :3].cpu().numpy()                 # FUTURE (next ~60)
        a_amb = viz.action_ambient(cur, norm.denorm_act(act[:, t])[0].cpu().numpy(), R, r)
        ts = time.perf_counter()
        _, cpath = m.flow.sample(h_t, steps=K, deterministic=True, record_path=True)       # the metric prediction
        sample_s = time.perf_counter() - ts
        committed = np.stack([decode_xyz(z_t, x)[0].cpu().numpy() for x in cpath])         # (K+1, 3)
        swarm, ends = [], []
        for _ in range(n_swarm):
            e = torch.randn(1, dz, generator=g, device=device)
            _, pth = m.flow.sample(h_t, steps=K, deterministic=False, eps=e, record_path=True)
            sp = np.stack([decode_xyz(z_t, x)[0].cpu().numpy() for x in pth])
            swarm.append(sp); ends.append(sp[-1])
        return {"current": cur, "true_next": nxt, "agent_tail": a_tail, "future_path": a_fut,
                "action_amb": a_amb, "committed": committed, "swarm": swarm, "ends": ends, "sample_s": sample_s}

    _plog(writer, f"[diffusion_field @ep{step}] {n_steps} steps {steps_idx}, K={K} swarm={n_swarm}")
    endpoint_errs, spreads, sample_times = [], [], []
    for si, t in enumerate(steps_idx):                 # (a) the 3 single-step quivers (2 s denoising each)
        d = step_data(t)
        mean_end = np.stack(d["ends"]).mean(axis=0)    # posterior-mean prediction: average over n_swarm fixed-seed
        endpoint_errs.append(float(np.linalg.norm(mean_end - d["true_next"]) / r))   # samples (fairer than eps=0); tube-radii
        spreads.append(float(np.linalg.norm(np.stack(d["ends"]).std(axis=0))))
        sample_times.append(d["sample_s"])
        per_frame = _quiver_frames_data(d["committed"], swarm=d["swarm"])
        frames = viz.diffusion_quiver_frames(R, r, coloring, d["current"], d["action_amb"], per_frame,
                                             agent_tail=d["agent_tail"], future_path=d["future_path"],
                                             true_next=d["true_next"], title=f"diffusion quiver step {t}")
        writer.video(f"eval_diffusion/quiver/example_{si}", frames, 60, step)  # 120 frames @ 60 fps = 2 s
        writer.scene(f"eval_diffusion/quiver/example_{si}", {  # 3D geometry for Blender (plain-language keys)
            "description": "Diffusion flow-field quiver at one fixed prediction step. The torus is the manifold "
                           "the agent moves on. The grey swarm are noise samples the model denoises ONTO the "
                           "surface; the committed path is the model's single best-guess prediction "
                           "(current -> predicted next); the black ring marks the TRUE next position; the agent's "
                           "history tail and future path are both black.",
            "coordinate_system": "world xyz, same space as the torus",
            "torus": {"major_radius_R": float(R), "tube_radius_r": float(r)},
            "moving_agent": {"history_tail_xyz": d["agent_tail"], "future_path_xyz": d["future_path"],
                             "current_position_xyz": d["current"]},
            "action_arrow": {"origin_xyz": d["current"], "vector_xyz": np.asarray(d["action_amb"])},
            "true_next_position_xyz": d["true_next"],
            "committed_prediction_path_xyz": d["committed"],       # (K+1,3): current -> predicted next
            "swarm_paths_xyz": [np.asarray(sp) for sp in d["swarm"]],  # each (K+1,3): decoded noise -> surface
        }, step)

    # (b) multistep quiver: 16 CONSECUTIVE denoising predictions, each ~0.25 s @ 60 fps -> 4 s, agent walking
    ms_ts = list(range(steps_idx[0], min(steps_idx[0] + 16, Tlen - 2)))
    ms_steps = [{**(d := step_data(t)), "per_frame": _quiver_frames_data(d["committed"], swarm=d["swarm"])}
                for t in ms_ts]
    ms_frames = viz.diffusion_quiver_multistep_frames(R, r, coloring, ms_steps, frames_per_step=15,
                                                      title="diffusion quiver multistep")
    writer.video("eval_diffusion/quiver_multistep", ms_frames, 60, step)  # 16 x 15 = 240 frames @ 60 fps = 4 s
    writer.scene("eval_diffusion/quiver_multistep", {
        "description": "16 consecutive denoising predictions as the agent walks forward along its path; each "
                       "step shows the grey swarm denoising onto the surface to predict the next position.",
        "coordinate_system": "world xyz, same space as the torus",
        "torus": {"major_radius_R": float(R), "tube_radius_r": float(r)},
        "steps": [{"current_position_xyz": s["current"], "history_tail_xyz": s["agent_tail"],
                   "future_path_xyz": s["future_path"], "true_next_position_xyz": s["true_next"],
                   "committed_prediction_path_xyz": s["committed"],
                   "swarm_paths_xyz": [np.asarray(sp) for sp in s["swarm"]]} for s in ms_steps],
    }, step)

    writer.scalars({"eval_diffusion/pointwise_error": float(np.mean(endpoint_errs)),
                    "eval_diffusion/sample_spread": float(np.mean(spreads)),
                    "eval_diffusion/time/sample_s": float(np.mean(sample_times)),               # wall time of one K-step prediction
                    "eval_diffusion/time/sample_ms_per_euler_step": float(1000.0 * np.mean(sample_times) / max(1, K))},
                   step)
    if was:
        m.train()
    _plog(writer, f"[diffusion_field @ep{step}] done in {time.perf_counter() - t0:.1f}s "
                  f"endpoint_err={np.mean(endpoint_errs):.3f} spread={np.mean(spreads):.4f}")
    return {"eval_diffusion_pointwise_error": float(np.mean(endpoint_errs))}


REGISTRY = {"ood_horizon": eval_ood_horizon, "ood_visual": eval_ood_visual,
            "ood_geometric": eval_ood_geometric, "ood_dynamics": eval_ood_dynamics,
            "control": eval_control, "diffusion_field": eval_diffusion_field}
