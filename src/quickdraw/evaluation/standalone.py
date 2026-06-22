"""Shared post-hoc runner for the standalone eval entrypoints (best-checkpoint)."""

from __future__ import annotations

import json
import os

import torch

from ..training.setup import build_model, env_cfg, load_checkpoint, normalizer
from ..utils.logging import make_run_dir
from .routines import REGISTRY


def run_standalone(cfg, routine_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(device)
    load_checkpoint(model, cfg.checkpoint)  # .ckpt file or train run dir (-> best.ckpt)
    model.eval()
    run_dir = make_run_dir(f"eval_{routine_name}", cfg.experiment)

    try:
        import wandb

        run = wandb.init(project=cfg.logging.project, group=cfg.logging.group, job_type=f"eval_{routine_name}")
    except Exception:
        run = None
    summary = REGISTRY[routine_name](cfg, model, normalizer(cfg), env_cfg(cfg), run_dir, device, run)
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if run is not None:
        run.finish()
    print(f"[eval_{routine_name}] {json.dumps(summary, indent=2)}  run_dir={run_dir}")
