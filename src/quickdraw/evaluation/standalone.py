"""Shared post-hoc runner for the standalone eval entrypoints (best-checkpoint)."""

from __future__ import annotations

import json
import os

import torch

from ..logging.writer import make_writer
from ..training.setup import build_model, env_cfg, load_checkpoint, normalizer
from ..utils.logging import make_run_dir
from .routines import REGISTRY


def run_standalone(cfg, routines, label: str | None = None):
    """Run one or more eval routines under a single run dir/writer. `routines` is a name or a list of
    names (REGISTRY keys); `label` names the run dir (defaults to the sole routine name)."""
    if isinstance(routines, str):
        routines = [routines]
    label = label or routines[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(device)
    load_checkpoint(model, cfg.checkpoint)  # .ckpt file or train run dir (-> best.ckpt)
    model.eval()
    run_dir = make_run_dir(f"eval_{label}", cfg.experiment)

    writer = make_writer(run_dir, cfg, job_type=f"eval_{label}")
    norm, ecfg = normalizer(cfg), env_cfg(cfg)
    summary = {}
    for name in routines:
        summary.update(REGISTRY[name](cfg, model, norm, ecfg, writer, device, 0))
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    writer.finalize()
    print(f"[eval_{label}] {json.dumps(summary, indent=2)}  run_dir={run_dir}")
