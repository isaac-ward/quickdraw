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


REGISTRY = {"ood_horizon": eval_ood_horizon, "ood_visual": eval_ood_visual,
            "ood_geometric": eval_ood_geometric, "ood_dynamics": eval_ood_dynamics,
            "control": eval_control}
