"""Is the planner choosing the best CANDIDATE, or the luckiest ROLLOUT?

`stochastic_eval=True`, so each candidate is rolled ONCE and its score is a single noisy sample of its
value. argmax over 64 such samples maximises value PLUS noise. That matters twice over: it overstates the
chosen candidate's worth, and -- because a world model's sampling noise is largest where it is least
certain -- it systematically prefers candidates that drive the imagination into the model's high-variance
region, which is exactly the blurry off-manifold place the plan videos end up.

The test is a variance decomposition at ONE decision point:
  ACROSS   roll K different candidates once each      -> var(value) + var(noise)
  WITHIN   roll ONE candidate m times                 -> var(noise) alone
If WITHIN is comparable to ACROSS, the ranking the planner consumes is mostly noise, and both `more
samples` and `argmax` make it worse rather than better.

    CUDA_VISIBLE_DEVICES=0 python scratch/check_selection_noise.py <train_action_run> <reward_head.pt>
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

from quickdraw.data.dataset import load_split_episodes_mm, set_action_aggregate, set_subsample
from quickdraw.evaluation.steering import PriorProposal
from quickdraw.language.reward import LanguageReward
from quickdraw.training.setup import (build_model, effective_action_dim, image_head_cams, image_head_sizes,
                                      load_checkpoint, normalizer, resolve_data_root)

REQ = "climb"


def main(run: str, head: str) -> int:
    cfg = OmegaConf.create(json.load(open(os.path.join(run, "logs", "config.json"))))
    set_subsample(int(cfg.data.get("subsample", 1) or 1)); set_action_aggregate(str(cfg.data.get("action_aggregate", "sum")))
    m = build_model(cfg).to("cuda"); load_checkpoint(m, os.path.join(run, "checkpoints", "last.ckpt")); m.eval()
    core = getattr(m, "_orig_mod", m); norm = normalizer(cfg)
    P, A = int(cfg.data.P), effective_action_dim(cfg)
    img = next((n for n, _ in core.layout if n != "proprio"), None)
    lang = LanguageReward(head, device="cuda")
    t_e = lang.text_embedding(REQ).reshape(-1)
    eps = load_split_episodes_mm(resolve_data_root(cfg), "val", img_size=image_head_sizes(cfg),
                                 cam=image_head_cams(cfg), repo_id=cfg.data.get("repo_id", "torus"))
    o, a, fr = eps[4]
    t = 55
    ctx = {"proprio": norm.norm_obs(torch.from_numpy(o[t - P:t])).float()[None].cuda(),
           img: torch.from_numpy(fr[img][t - P:t]).float().div(255.)[None].cuda()}
    ca = norm.norm_act(torch.from_numpy(a[t - P:t])).float()[None].cuda()
    K, M = 64, 64
    with torch.no_grad():
        bags = core.encode_state(ctx)
        buf = list(bags.unbind(dim=1)); acts = list(ca.unbind(dim=1))
        W = int(core.window)
        zs = torch.stack(buf[-W:], dim=1); aa = torch.stack(acts[-zs.shape[1]:], dim=1)
        h = core.pool_context(core.backbone(core._to_input(zs, aa)))[:, -1]
        pr = PriorProposal(core, a_max=1e9)
        g = torch.Generator(device="cuda"); g.manual_seed(0)
        steps = pr.K
        cand = pr.sample(torch.zeros(1, steps, A, device="cuda"), K, ctx=h, g=g)[0]      # (K,steps,A)

        def roll_and_score(c):                                   # c: (n,steps,A) -> (n,) chunk return
            n = c.shape[0]
            bb = [b.expand(n, -1, -1) for b in buf]
            hist = torch.cat([x.expand(n, -1).unsqueeze(1) for x in acts], dim=1)
            r = core._rollout_from(bb, torch.cat([hist, c], dim=1), steps, 0.0, None, 0)
            return lang.score(r.reshape(n * steps, -1), t_e).reshape(n, steps).mean(1)

        across = roll_and_score(cand)                            # K candidates, one roll each
        best = int(across.argmax())
        within = roll_and_score(cand[best:best + 1].expand(M, -1, -1).contiguous())   # ONE candidate, M rolls
        # and the same for a middling candidate, so the finding isn't special to the winner
        mid = int(across.argsort()[K // 2])
        within_mid = roll_and_score(cand[mid:mid + 1].expand(M, -1, -1).contiguous())

    a_sd, w_sd, wm_sd = float(across.std()), float(within.std()), float(within_mid.std())
    print(f"\n  request {REQ!r} at ep4 t55, one decision point, chunk {steps}, K={K}, M={M}")
    print(f"  ACROSS {K} candidates (1 roll each)  mean {float(across.mean()):+.4f}  sd {a_sd:.4f}"
          f"  range {float(across.min()):+.4f}..{float(across.max()):+.4f}")
    print(f"  WITHIN the CHOSEN candidate, {M} rolls  mean {float(within.mean()):+.4f}  sd {w_sd:.4f}")
    print(f"  WITHIN a median candidate,   {M} rolls  mean {float(within_mid.mean()):+.4f}  sd {wm_sd:.4f}")
    print(f"\n  noise / total spread = {w_sd / max(1e-9, a_sd):.2f}   (1.0 = the ranking is pure noise)")
    # what argmax actually bought: the winner's ONE sample vs its own re-measured mean
    print(f"  the winner scored {float(across[best]):+.4f} on its single roll, but re-rolled it averages "
          f"{float(within.mean()):+.4f}")
    print(f"  -> argmax overstated it by {float(across[best] - within.mean()):+.4f}")
    print(f"  true-value spread (var_across - var_within, floored) sd ~ "
          f"{np.sqrt(max(0.0, a_sd ** 2 - w_sd ** 2)):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
