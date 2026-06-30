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


@torch.no_grad()
def eval_diffusion_field(cfg, model, norm, ecfg, writer, device, step=0):
    """The headline diffusion artifact (design/models/diffusion.md). At 3 FIXED prediction steps of episode
    0, render the latent flow field through the decoder onto the torus as a quiver ATLAS animation (2 s):
    a grey swarm of decoded ODE paths flowing off-surface onto the manifold (each leaving a tail that
    traces the field), the agent's black history tail AND future path, and a black truth ring. ALSO a
    `quiver_multistep` (4 s) at ONE fixed step (agent + history do NOT move): the SAME denoising shown with
    a fine 16-step integration so each denoising step reads clearly. Diffusion-specific scalars:
    eval_diffusion/std_of_samples (std of the swarm's FINAL positions = predicted uncertainty) and
    eval_diffusion/time/{sample_s, sample_ms_per_euler_step}. (Pointwise accuracy lives in val/train
    pointwise_error — the shared rollout metric — so it's NOT duplicated here.) Self-SKIPS for non-diffusion."""
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

    def step_data(t, swarm_steps=None):                # per-step geometry; swarm uses swarm_steps (default K)
        ks = swarm_steps or K
        w = min(W, t + 1)
        h_t = m.transformer(m.to_token(z[:, t - w + 1:t + 1], act[:, t - w + 1:t + 1]))[:, -1]
        z_t = z[:, t]
        cur = norm.denorm_obs(obs[:, t])[0, :3].cpu().numpy()
        nxt = norm.denorm_obs(obs[:, t + 1])[0, :3].cpu().numpy()
        a_tail = norm.denorm_obs(obs[0, max(0, t - 60):t + 1])[:, :3].cpu().numpy()        # history (last ~60)
        a_fut = norm.denorm_obs(obs[0, t + 1:t + 61])[:, :3].cpu().numpy()                 # FUTURE (next ~60)
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

    def scene_dict(d, desc):                           # shared Blender scene payload for a single prediction
        return {"description": desc, "coordinate_system": "world xyz, same space as the torus",
                "torus": {"major_radius_R": float(R), "tube_radius_r": float(r)},
                "moving_agent": {"history_tail_xyz": d["agent_tail"], "future_path_xyz": d["future_path"],
                                 "current_position_xyz": d["current"]},
                "action_arrow": {"origin_xyz": d["current"], "vector_xyz": np.asarray(d["action_amb"])},
                "true_next_position_xyz": d["true_next"],
                "swarm_paths_xyz": [np.asarray(sp) for sp in d["swarm"]]}  # each (k,3): decoded noise -> surface

    _plog(writer, f"[diffusion_field @ep{step}] {n_steps} steps {steps_idx}, K={K} swarm={n_swarm}")
    spreads, sample_times = [], []
    for si, t in enumerate(steps_idx):                 # (a) the 3 single-step quivers (2 s denoising each)
        d = step_data(t)
        spreads.append(float(np.linalg.norm(np.stack(d["ends"]).std(axis=0))))   # std of the swarm's FINAL positions
        sample_times.append(d["sample_s"])
        frames = viz.diffusion_quiver_frames(R, r, coloring, d["current"], d["action_amb"],
                                             _quiver_frames_data(d["swarm"]), agent_tail=d["agent_tail"],
                                             future_path=d["future_path"], true_next=d["true_next"],
                                             title=f"diffusion quiver step {t}")
        writer.video(f"eval_diffusion/quiver/example_{si}", frames, 60, step)  # 120 frames @ 60 fps = 2 s
        writer.scene(f"eval_diffusion/quiver/example_{si}", scene_dict(
            d, "Diffusion flow-field quiver at one fixed prediction step. The grey swarm are noise samples the "
               "model denoises ONTO the torus; the black ring marks the TRUE next position; the agent's black "
               "history tail and future path show where it came from and where it's going."), step)

    # (b) multistep quiver: ONE fixed step (agent + history do NOT move), the swarm denoised with a fine
    # 16-step integration shown over 4 s so each denoising step reads clearly.
    dm = step_data(steps_idx[0], swarm_steps=16)
    ms_frames = viz.diffusion_quiver_frames(R, r, coloring, dm["current"], dm["action_amb"],
                                            _quiver_frames_data(dm["swarm"], n_frames=240),
                                            agent_tail=dm["agent_tail"], future_path=dm["future_path"],
                                            true_next=dm["true_next"], title="diffusion quiver multistep")
    writer.video("eval_diffusion/quiver_multistep", ms_frames, 60, step)  # 240 frames @ 60 fps = 4 s
    writer.scene("eval_diffusion/quiver_multistep", scene_dict(
        dm, "Diffusion denoising at ONE fixed prediction step (agent stationary), the swarm integrated with a "
            "fine 16-step ODE so each denoising step is visible, played over 4 s."), step)

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
            "control": eval_control, "diffusion_field": eval_diffusion_field}
