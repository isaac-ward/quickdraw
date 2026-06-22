"""Run MPPI control and log it (atlas + per-target scalars/videos). Shared by `eval` and `control`."""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt

from ..logging import viz
from .mppi import MPPIConfig, run_control


def run_and_log_control(cfg, model, normalizer, ecfg, run_dir: str, device, wandb_run=None) -> dict:
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    res = run_control(model, normalizer, ecfg, MPPIConfig(**cfg.control), device=device)
    trajs = [{"xyz": res["paths"][n], "color": "k"} for n, _ in res["targets"]]
    atlas = viz.fig_torus_atlas(ecfg.R, ecfg.r, trajs=trajs, targets=res["targets"],
                                title="control targets + reached paths")
    atlas.savefig(os.path.join(run_dir, "plots", "target_atlas.png"), bbox_inches="tight", dpi=viz.DPI)
    with open(os.path.join(run_dir, "control_summary.json"), "w") as f:
        json.dump({k: res[k] for k in ("hz", "success_rate", "mean_time_to_completion", "per_target")}, f, indent=2)
    if wandb_run is not None:
        import wandb

        wandb_run.log({
            "eval/control/control_hz": res["hz"],
            "eval/control/success_rate": res["success_rate"],
            "eval/control/mean_time_to_completion": res["mean_time_to_completion"],
            "eval/control/targets": wandb.Image(atlas),
        })
        for name, m in res["per_target"].items():
            video = viz.rollout_video(res["paths"][name], res["paths"][name], ecfg.R, ecfg.r, n_frames=60)
            wandb_run.log({
                f"eval/control/{name}/time_to_completion": m["time_to_completion"],
                f"eval/control/{name}/success": float(m["success"]),
                f"eval/control/{name}/final_distance": m["final_distance"],
                f"eval/control/{name}/control_video": wandb.Video(video.transpose(0, 3, 1, 2), fps=15),
            })
    plt.close(atlas)
    return res
