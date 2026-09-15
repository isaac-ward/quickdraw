"""WHERE does pit_delta's level go wrong? Per-lead, not just at the ends.

The run logs w1 at lead 0 and lead 31 only, and those two say pit_delta is BETTER at 0 (0.054 vs 0.113)
and 18x WORSE at 31 (1.588 vs 0.088). The shape in between decides whether it matters in practice: the
planner commits `commit` steps of each chunk and re-plans, so if commit is 16 it never executes leads
16-31 and a blowup that only arrives at 31 is irrelevant to it.

Measured per lead against the RECORDED marginal at that lead: mean |a|, std, and the accumulated drift of
the mean (which is what a random walk in the level produces).

    CUDA_VISIBLE_DEVICES=1 python scratch/check_lead_drift.py <run> [<run> ...]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

from quickdraw.controller.mppi import _h_ctx
from quickdraw.data.dataset import load_split_episodes_mm, set_action_aggregate, set_subsample
from quickdraw.evaluation.steering import PriorProposal
from quickdraw.training.setup import (build_model, effective_action_dim, image_head_cams, image_head_sizes,
                                      load_checkpoint, normalizer, resolve_data_root)

NA = 4


def main(*runs: str) -> int:
    for run in runs:
        cfg = OmegaConf.create(json.load(open(os.path.join(run, "logs", "config.json"))))
        set_subsample(4); set_action_aggregate("concat")
        m = build_model(cfg).cuda(); load_checkpoint(m, os.path.join(run, "checkpoints", "last.ckpt")); m.eval()
        core = getattr(m, "_orig_mod", m); norm = normalizer(cfg)
        P, A, K = int(cfg.data.P), effective_action_dim(cfg), int(core.action_head_chunk)
        img = next((n for n, _ in core.layout if n != "proprio"), None)
        eps = load_split_episodes_mm(resolve_data_root(cfg), "val", img_size=image_head_sizes(cfg),
                                     cam=image_head_cams(cfg), repo_id="starling-2")
        ctxs = [(e, t) for e in range(len(eps)) for t in range(40, 380, 40)]
        with torch.no_grad():
            h = torch.cat([_h_ctx(core, norm,
                                  torch.from_numpy(eps[e][0][t - P:t]).float()[None].cuda(),
                                  torch.from_numpy(eps[e][2][img][t - P:t]).float().div(255.)[None].cuda(),
                                  torch.from_numpy(eps[e][1][t - P:t - 1]).float()[None].cuda(), img)
                           for e, t in ctxs], 0)
            g = torch.Generator(device="cuda"); g.manual_seed(0)
            d = PriorProposal(core, a_max=1e9, norm=norm, prefix_guidance=False).sample(
                torch.zeros(len(h), K, A, device="cuda"), 32, ctx=h, g=g)
            d = d.reshape(-1, K, A).cpu().numpy().reshape(-1, K, A // NA, NA).mean(axis=2)   # (n,K,NA)
        # the recorded marginal is the same at every lead (a stationary flight), so one reference suffices
        rec = np.concatenate([a for _, a, _ in eps]).reshape(-1, A // NA, NA).mean(axis=1)
        print(f"\n=== {os.path.basename(run)}   ({core.action_head_target_transform}, {len(ctxs)} contexts x 32 draws)")
        print(f"  RECORDED   |a| {np.abs(rec).mean():.3f}   std {rec.std():.3f}")
        print(f"  {'lead':>5s} {'|a|':>7s} {'std':>7s} {'|mean drift|':>13s}   ratio to recorded std")
        for k in (0, 1, 2, 4, 8, 12, 16, 24, 31):
            if k >= K:
                continue
            x = d[:, k]
            print(f"  {k:>5d} {np.abs(x).mean():>7.3f} {x.std():>7.3f} "
                  f"{np.abs(x.mean(axis=0) - rec.mean(axis=0)).mean():>13.3f}   "
                  f"{x.std() / rec.std():>6.2f}x")
    print("\n  A level that random-walks shows std growing with lead. A level that is predicted in absolute")
    print("  terms does not. Whether it MATTERS depends on how many leads the planner actually commits.")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
