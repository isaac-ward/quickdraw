"""Grade the continuity harnesses on the PROPOSAL alone -- no planner, no world model, no reward head.

Three things a harness must be judged on, and the third is the one that catches a bad idea:
  AGREEMENT  how closely the drawn chunk matches the prefix it was told to continue (the seam).
  JERK       step-to-step |da| AFTER the overlap, against the recorded 0.055. A harness that fixes the
             seam by making the rest of the chunk wilder has moved the problem, not solved it.
  HOLDS      mean consecutive steps with the stick unmoved, against the recorded 13.4 on fore/aft.

The prefix is a REAL chunk tail from the data, so "agreement" is measured against something a pilot
actually flew rather than against another sample.

    CUDA_VISIBLE_DEVICES=0 python scratch/check_harnesses.py <train_action_run>
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
from quickdraw.evaluation import steering as S
from quickdraw.training.setup import (build_model, effective_action_dim, image_head_cams, image_head_sizes,
                                      load_checkpoint, normalizer, resolve_data_root)

AXES = [a["name"] for a in yaml.safe_load(open("conf/interpret/starling.yaml"))["action_axes"]]
NA = len(AXES)


def fold(x):                       # (..., T, A_wide) -> (..., T, NA) the per-axis stick, as everywhere
    return x.reshape(*x.shape[:-1], -1, NA).mean(axis=-2)


def main(run: str) -> int:
    cfg = OmegaConf.create(json.load(open(os.path.join(run, "logs", "config.json"))))
    set_subsample(4); set_action_aggregate("concat")
    m = build_model(cfg).to("cuda"); load_checkpoint(m, os.path.join(run, "checkpoints", "last.ckpt")); m.eval()
    core = getattr(m, "_orig_mod", m); norm = normalizer(cfg)
    P, A, K = int(cfg.data.P), effective_action_dim(cfg), int(core.action_head_chunk)
    img = next((n for n, _ in core.layout if n != "proprio"), None)
    eps = load_split_episodes_mm(resolve_data_root(cfg), "val", img_size=image_head_sizes(cfg),
                                 cam=image_head_cams(cfg), repo_id="starling-2")
    p_over = K // 2                      # commit half the chunk -> half of the next one is prefixed
    ctxs = [(4, 55), (5, 200), (3, 259), (3, 367), (0, 120), (1, 300), (2, 200), (6, 150)]
    hs, prefixes = [], []
    with torch.no_grad():
        for ei, t in ctxs:
            o, a, fr = eps[ei]
            hs.append(_h_ctx(core, norm,
                             torch.from_numpy(o[t - P:t]).float()[None].cuda(),
                             torch.from_numpy(fr[img][t - P:t]).float().div(255.)[None].cuda(),
                             torch.from_numpy(a[t - P:t - 1]).float()[None].cuda(), img))
            prefixes.append(norm.norm_act(torch.from_numpy(a[t:t + p_over]).float()).cuda())  # REAL tail
        h = torch.cat(hs, 0)                                   # (G, d)
        pre = torch.stack(prefixes)                            # (G, p, A) normalized
        G, k = len(h), 64
        mean = torch.zeros(G, K, A, device="cuda")
        bank = torch.stack([torch.from_numpy(a[i:i + K]).float()
                            for _, a, _ in eps for i in range(0, len(a) - K, max(1, K // 4))])

        def grade(name, prop, prefix):
            g = torch.Generator(device="cuda"); g.manual_seed(0)
            d = prop.sample(mean, k, ctx=h, g=g, prefix=prefix)            # (G,k,K,A) normalized
            raw = fold(norm.denorm_act(d.cpu()).numpy())                   # (G,k,K,NA) raw stick
            tgt = fold(norm.denorm_act(pre.cpu()).numpy())                 # (G,p,NA)
            agree = np.abs(raw[:, :, :p_over] - tgt[:, None]).mean()
            # SPREAD over the overlap: how much the k candidates still DIFFER there. Agreement is trivially
            # perfect if you just overwrite the draw with the prefix -- but then all 64 candidates share
            # those steps and the planner has no choice over them, which is not continuity, it is turning
            # the planner off for half the chunk. A harness has to buy agreement without spending this.
            spread = raw[:, :, :p_over].std(axis=1).mean()
            jerk = np.abs(np.diff(raw[:, :, p_over:], axis=2)).mean()
            hold = []
            flat = raw.reshape(-1, K, NA)
            for ax in range(NA):
                runs, cur = [], 1
                for c in flat[:, :, ax]:
                    cur = 1
                    for i in range(1, K):
                        cur = cur + 1 if abs(c[i] - c[i - 1]) < 0.02 else (runs.append(cur) or 1)
                    runs.append(cur)
                hold.append(np.mean(runs))
            print(f"  {name:44s} {agree:>9.4f} {spread:>8.4f} {jerk:>8.4f} ({jerk / 0.0545:>4.1f}x) "
                  f"{np.mean(hold):>7.2f} {hold[3]:>8.2f}")

        print(f"\n  chunk {K}, overlap {p_over} (commit half), {G} contexts x {k} candidates")
        print(f"  {'proposal':44s} {'agree':>9s} {'spread':>8s} {'jerk':>8s}         {'hold':>7s} "
              f"{'f-a hold':>8s}")
        print("  " + "-" * 97)
        rec = fold(np.stack([a[i:i + K] for _, a, _ in eps for i in range(0, len(a) - K, K // 4)]))
        print(f"  {'RECORDED (the target)':44s} {0.0:>9.4f} {'--':>8s} "
              f"{np.abs(np.diff(rec, axis=1)).mean():>8.4f} ( 1.0x) {'':>7s} {'13.41':>8s}")
        pri_off = S.PriorProposal(core, prefix_guidance=False)
        pri_on = S.PriorProposal(core, prefix_guidance=True)
        grade("prior (no harness)", pri_off, pre)
        grade("prior + prefix guidance", pri_on, pre)
        grade("prior + crossfade", S.Crossfade(pri_off, 0.5), pre)
        grade("prior + held", S.Held(pri_off, 0.02), pre)
        grade("prior + prefix guidance + held", S.Held(pri_on, 0.02), pre)
        # the REAL bank (train at stride 1), not the 365-chunk val slice the first verdict was based on
        from quickdraw.data.dataset import load_split_episodes
        be = load_split_episodes(resolve_data_root(cfg), "train", repo_id="starling-2")
        big = torch.stack([torch.from_numpy(a[i:i + K]).float() for _, a in be
                           for i in range(0, len(a) - K, 1)])
        print(f"  (data bank: {len(big)} chunks from train at stride 1)")
        dat = S.DataProposal(norm.norm_act(big).cuda(), prefix_retrieval=False)
        ret = S.DataProposal(norm.norm_act(big).cuda(), prefix_retrieval=True)
        grade("data (no harness)", dat, pre)
        grade("data + prefix retrieval", ret, pre)
        grade("data + crossfade", S.Crossfade(dat, 0.5), pre)
        # FAIRNESS SWEEPS. crossfade at decay 0.5 has faded to 0.08 by position 5, so it only ever touches
        # the first few of a 16-long overlap -- a slower decay is the version that competes. And `held` at
        # the 0.02 tolerance is a no-op if the draws contain no near-holds to snap, so raising it says HOW
        # FAR from holding they are, which is the number that decides whether rounding can substitute for
        # learning the atom.
        print()
        for dec in (0.1, 0.02):
            grade(f"prior + crossfade(decay={dec:g})", S.Crossfade(pri_off, dec), pre)
        for tol in (0.05, 0.1, 0.2):
            grade(f"prior + held(tol={tol:g})", S.Held(pri_off, tol), pre)
        grade("prior + prefix guidance + crossfade(0.1)", S.Crossfade(pri_on, 0.1), pre)
    print("\n  agree = mean |stick - the real tail it was told to continue| over the overlap (0 = perfect)")
    print("  jerk  = step-to-step |da| AFTER the overlap, so a harness cannot hide roughness there")
    print("  spread= how much the 64 candidates still differ over the overlap. Agreement bought by")
    print("          driving this to zero is agreement bought by having nothing left to choose between.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
