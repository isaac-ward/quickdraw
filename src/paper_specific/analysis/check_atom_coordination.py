"""Does the head fail at the MARGINAL or at the JOINT? The answer picks the fix.

PIT already gives the prior the right FRACTION of actions exactly at rest -- that was the whole point of
mapping the atom to a slab a smooth flow can land in. Yet holds collapse (2.0 consecutive steps against a
pilot's 13.4). Two very different causes:

  MARGINAL failure   the model does not even put the right fraction of single slots on the atom.
                     -> capacity / training, and neither a transformer nor increment-PIT is the fix.
  JOINT failure      single slots hit the atom at the right rate, but CONSECUTIVE slots do not hit it
                     TOGETHER -- the model treats chunk positions as if independent. A hold needs a RUN
                     of slots to agree, so this is a coordination failure.
                     -> the two candidate fixes both attack it, differently:
                        a transformer over chunk positions lets slots see each other while denoising;
                        increment-PIT moves the atom onto the DIFFERENCE, so "hold" becomes one draw in
                        one slab instead of several slots independently agreeing on a level.

The test is P(rest at t+1 | rest at t) against P(rest). Independent slots give a ratio of 1.

    CUDA_VISIBLE_DEVICES=0 python scratch/check_atom_coordination.py <train_action_run>
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from quickdraw.controller.mppi import _h_ctx
from quickdraw.data.dataset import load_split_episodes_mm, set_action_aggregate, set_subsample
from quickdraw.evaluation.steering import PriorProposal
from quickdraw.training.setup import (build_model, effective_action_dim, image_head_cams, image_head_sizes,
                                      load_checkpoint, normalizer, resolve_data_root)

AXES = [a["name"] for a in yaml.safe_load(open("conf/interpret/starling.yaml"))["action_axes"]]
NA = len(AXES)


def stats(x, atom, tol=1e-4):
    """x (n, T, NA) raw stick -> per-axis P(rest), P(rest|rest), and the ratio."""
    out = []
    for j in range(NA):
        r = np.abs(x[:, :, j] - atom[j]) < tol
        p = r.mean()
        both = (r[:, :-1] & r[:, 1:]).sum()
        prev = r[:, :-1].sum()
        cond = both / max(1, prev)
        out.append((p, cond, cond / max(1e-9, p)))
    return out


def main(run: str) -> int:
    cfg = OmegaConf.create(json.load(open(os.path.join(run, "logs", "config.json"))))
    set_subsample(4); set_action_aggregate("concat")
    m = build_model(cfg).to("cuda"); load_checkpoint(m, os.path.join(run, "checkpoints", "last.ckpt")); m.eval()
    core = getattr(m, "_orig_mod", m); norm = normalizer(cfg)
    P, A, K = int(cfg.data.P), effective_action_dim(cfg), int(core.action_head_chunk)
    img = next((n for n, _ in core.layout if n != "proprio"), None)
    eps = load_split_episodes_mm(resolve_data_root(cfg), "val", img_size=image_head_sizes(cfg),
                                 cam=image_head_cams(cfg), repo_id="starling-2")
    real = np.stack([a[i:i + K] for _, a, _ in eps for i in range(0, len(a) - K, 8)])
    real = real.reshape(len(real), K, -1, NA).mean(axis=2)
    # the atom = the most common value on each axis, which is what PIT's slab inverts to
    atom = np.array([float(np.bincount(np.digitize(real[:, :, j].ravel(),
                                                   np.linspace(-1, 1, 2001)))[1:].argmax() / 1000.0 - 1.0)
                     for j in range(NA)])
    print(f"\n  rest value per axis (mode of the recorded stick): "
          + "  ".join(f"{n} {v:+.3f}" for n, v in zip(AXES, atom)))
    ctxs = [(4, 55), (5, 200), (3, 259), (3, 367), (0, 120), (1, 300), (2, 200), (6, 150)]
    with torch.no_grad():
        h = torch.cat([_h_ctx(core, norm,
                              torch.from_numpy(eps[e][0][t - P:t]).float()[None].cuda(),
                              torch.from_numpy(eps[e][2][img][t - P:t]).float().div(255.)[None].cuda(),
                              torch.from_numpy(eps[e][1][t - P:t - 1]).float()[None].cuda(), img)
                      for e, t in ctxs], 0)
        pr = PriorProposal(core, prefix_guidance=False, norm=norm)
        g = torch.Generator(device="cuda"); g.manual_seed(0)
        d = pr.sample(torch.zeros(len(h), K, A, device="cuda"), 128, ctx=h, g=g)
        d = d.reshape(-1, K, A).cpu().numpy().reshape(-1, K, A // NA, NA).mean(axis=2)
    print(f"\n  {'axis':10s} {'P(rest)':>18s}   {'P(rest | prev rest)':>22s}   {'ratio (1 = independent)':>24s}")
    print(f"  {'':10s} {'recorded   prior':>18s}   {'recorded    prior':>22s}   {'recorded   prior':>24s}")
    print("  " + "-" * 82)
    for j, (rs, ps) in enumerate(zip(stats(real, atom), stats(d, atom))):
        print(f"  {AXES[j]:10s} {rs[0]:>9.3f} {ps[0]:>8.3f}   {rs[1]:>11.3f} {ps[1]:>9.3f}   "
              f"{rs[2]:>12.1f} {ps[2]:>9.1f}")
    print("\n  P(rest) close but ratio collapsed => the marginal is learned and the JOINT is not: the head")
    print("  treats chunk positions as near-independent, which is precisely what a run of held steps needs")
    print("  it not to do.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
