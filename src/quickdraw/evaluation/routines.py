"""Eval routines — one definition each, used both as in-training subscriptions (on a cadence) and
as standalone post-hoc steps. REGISTRY maps name -> routine.

A routine: (cfg, model, norm, ecfg, run_dir, device, wandb_run) -> summary dict. It saves plots to
run_dir and logs scalars/media to wandb_run (if given). Routines are read-only (no grad).
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt

from ..controller.run import run_and_log_control
from ..logging import viz
from ..training.setup import eval_episodes
from .openloop import eval_episode

OOD_SPLITS = ["eval_ood_visual", "eval_ood_geometric", "eval_ood_dynamics"]


def _openloop_split(cfg, model, norm, run_dir, device, split, R, r, prefix, wandb_run, coloring="rainbow"):
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    b = eval_episodes(cfg, norm, split)[0]
    res = eval_episode(model, norm, R, r, cfg.data.P, b["obs_seq"][None].to(device), b["act_seq"][None].to(device))
    trajs = [{"xyz": res["true_xyz"], "color": "tab:green", "label": "true"},
             {"xyz": res["pred_xyz"], "color": "tab:red", "label": "pred"}]
    f_traj = viz.fig_torus_atlas(R, r, trajs=trajs, coloring=coloring, title=split, legend=True)
    f_err = viz.fig_error_vs_step(res["curves"])
    f_traj.savefig(os.path.join(run_dir, "plots", f"{split}_traj.png"), bbox_inches="tight", dpi=viz.DPI)
    f_err.savefig(os.path.join(run_dir, "plots", f"{split}_err.png"), bbox_inches="tight", dpi=viz.DPI)
    if wandb_run is not None:
        import wandb

        wandb_run.log({f"{prefix}/trajectory_plot": wandb.Image(f_traj),
                       f"{prefix}/error_vs_step": wandb.Image(f_err),
                       **{f"{prefix}/{k}": v for k, v in res["summary"].items()}})
    plt.close(f_traj)
    plt.close(f_err)
    return res["summary"]


def eval_long_horizon(cfg, model, norm, ecfg, run_dir, device, wandb_run=None):
    """In-distribution long-horizon open-loop rollout (eval_ind)."""
    s = _openloop_split(cfg, model, norm, run_dir, device, "eval_ind", ecfg.R, ecfg.r, "eval/long_horizon", wandb_run)
    return {"eval_ind": s}


def eval_ood(cfg, model, norm, ecfg, run_dir, device, wandb_run=None):
    """Open-loop on the 3 OOD splits, each scored on its own geometry + drawn with its coloring."""
    card = json.load(open(os.path.join(cfg.data.root, "dataset_card.json")))
    split_env, coloring = card.get("split_env", {}), card.get("coloring", {})
    out = {}
    for split in OOD_SPLITS:
        se = split_env.get(split, {"R": ecfg.R, "r": ecfg.r})
        out[split] = _openloop_split(cfg, model, norm, run_dir, device, split, se["R"], se["r"],
                                     f"eval/ood/{split}", wandb_run, coloring.get(split, "rainbow"))
    return out


def eval_control(cfg, model, norm, ecfg, run_dir, device, wandb_run=None):
    """MPPI control to the 16 torus targets."""
    res = run_and_log_control(cfg, model, norm, ecfg, run_dir, device, wandb_run)
    return {"control": {k: res[k] for k in ("hz", "success_rate", "mean_time_to_completion")}}


REGISTRY = {"long_horizon": eval_long_horizon, "ood": eval_ood, "control": eval_control}
