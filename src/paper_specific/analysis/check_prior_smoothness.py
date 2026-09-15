"""Is the JERK the action head's, or the planner's?

The planner's plans command stick changes 2.5-4x faster than a real flight. Two very different causes:
the head draws jerky chunks, or the head draws fine chunks and argmax-over-64 SELECTS the jerky ones
(a jerkier chunk visits more varied states, so it has more chances at a high reward). Only the first is a
modelling defect, and only the first is worth fixing in training rather than in the planner.

So take the head alone: draw chunks from many real contexts, no reward, no selection, and compare their
step-to-step |da| with the recorded sticks -- PER AXIS, because the axes are wildly different (the recorded
fore/aft stick barely moves between steps while yaw swings).

THE CALIBRATION THAT MAKES IT READABLE is the shuffled null: take real chunks and permute their timesteps.
That destroys temporal correlation while keeping every marginal exactly right, so it is what a head that
learned the per-step marginals and NOTHING about their joint would produce. A prior sitting at the shuffled
level has learned no smoothness at all; sitting at the recorded level has learned it perfectly.

    CUDA_VISIBLE_DEVICES=0 python scratch/check_prior_smoothness.py <train_action_run> [temps...]
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


def jerk(x):                       # x (n, T, NA) raw stick -> per-axis mean |da|
    return np.abs(np.diff(x, axis=1)).mean(axis=(0, 1))


def main(run: str, *_ignored: str) -> int:
    """`temperature` was removed from PriorProposal (it bought steering by making commands unflyable --
    every axis past the time-shuffled null at T=1.8), so the sweep is gone and extra args are ignored."""
    cfg = OmegaConf.create(json.load(open(os.path.join(run, "logs", "config.json"))))
    set_subsample(int(cfg.data.get("subsample", 1) or 1)); set_action_aggregate("concat")
    m = build_model(cfg).to("cuda"); load_checkpoint(m, os.path.join(run, "checkpoints", "last.ckpt")); m.eval()
    core = getattr(m, "_orig_mod", m); norm = normalizer(cfg)
    P, A = int(cfg.data.P), effective_action_dim(cfg)
    K = int(core.action_head_chunk)
    img = next((n for n, _ in core.layout if n != "proprio"), None)
    eps = load_split_episodes_mm(resolve_data_root(cfg), "val", img_size=image_head_sizes(cfg),
                                 cam=image_head_cams(cfg), repo_id="starling-2")

    # ---- the data: real chunks of the same length, and the same chunks with time permuted ----
    real = []
    for _, a, _ in eps:
        f = a.reshape(len(a), -1, NA).mean(axis=1)              # fold the concat sub-steps, as everywhere
        for i in range(0, len(f) - K, K):
            real.append(f[i:i + K])
    real = np.stack(real)
    rng = np.random.RandomState(0)
    shuf = np.stack([c[rng.permutation(K)] for c in real])      # marginals kept, time destroyed
    print(f"\n  chunk {K} | {len(real)} real chunks from {len(eps)} val episodes")
    print(f"  {'source':28s} " + "  ".join(f"{n:>9s}" for n in AXES) + f"  {'mean':>8s}")
    print("  " + "-" * 76)

    def row(name, j):
        print(f"  {name:28s} " + "  ".join(f"{v:>9.4f}" for v in j) + f"  {j.mean():>8.4f}")

    jr, js = jerk(real), jerk(shuf)
    row("RECORDED (the target)", jr)
    row("recorded, time-shuffled", js)

    # ---- the head, from real contexts, no reward and no selection ----
    ctxs = [(4, 55), (5, 200), (3, 259), (3, 367), (0, 120), (1, 300), (2, 200), (6, 150)]
    with torch.no_grad():
        hs = []
        for ei, t in ctxs:
            o, a, fr = eps[ei]
            hs.append(_h_ctx(core, norm,
                             torch.from_numpy(o[t - P:t]).float()[None].cuda(),
                             torch.from_numpy(fr[img][t - P:t]).float().div(255.)[None].cuda(),
                             torch.from_numpy(a[t - P:t - 1]).float()[None].cuda(), img))
        h = torch.cat(hs, 0)                                     # (G, d)
        pr = PriorProposal(core, a_max=1e9, norm=norm, prefix_guidance=False)   # norm -> RAW stick units
        g = torch.Generator(device="cuda"); g.manual_seed(0)
        d = pr.sample(torch.zeros(len(h), K, A, device="cuda"), 64, ctx=h, g=g)      # (G,64,K,A)
        d = d.reshape(-1, K, A).cpu().numpy().reshape(-1, K, A // NA, NA).mean(axis=2)
        row(f"prior draws ({core.action_head_target_transform})", jerk(d))
        # HOLDS. On the two near-constant axes the recorded pilot does not move the stick at all for long
        # stretches (fore/aft changes by 0.005 per step on average), so "smooth" there means HOLDING, not
        # moving gently. Mean run length of |da| < tol says whether the head reproduces that behaviour or
        # merely reproduces the right marginal while scattering the values through time.
        def holds(x, tol=0.02):
            out = []
            for ax in range(x.shape[-1]):
                runs, cur = [], 1
                for c in x[..., ax]:
                    cur = 1
                    for i in range(1, len(c)):
                        if abs(c[i] - c[i - 1]) < tol:
                            cur += 1
                        else:
                            runs.append(cur); cur = 1
                    runs.append(cur)
                out.append(float(np.mean(runs)))
            return np.array(out)

        print()
        row("HOLD length, recorded", holds(real))
        row("HOLD length, shuffled", holds(shuf))
        row(f"HOLD length, prior ({core.action_head_target_transform})", holds(d))
        print("  (mean consecutive steps with |da| < 0.02 -- how long the stick is HELD. "
              f"chunk length {K} is the cap.)")
    print("\n  A prior at the SHUFFLED level has learned the per-step marginals and nothing about their")
    print("  joint -- that is a training defect and no planner penalty can repair it, it can only pick the")
    print("  least-bad of a bad set. A prior at the RECORDED level is already smooth and any remaining")
    print("  jerk in a plan is the planner's (selection, and the seams between chunks).")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
