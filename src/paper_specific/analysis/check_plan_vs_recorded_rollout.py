"""Why does a PLAN look blurrier than our open-loop rollouts? Hold everything fixed except the actions.

The rollout machinery is the same spine either way (`_rollout_from`, p_tf=0, stochastic_eval on, no
relativization anchor since relative_position=False, no physics hook since there is no dynamics_prior), and
the horizon is the same 128 the world model is characterised at. The one thing a plan changes is WHICH
ACTIONS get rolled: the recorded sequence a real pilot flew, versus 8-step chunks drawn from the prior and
stitched together by argmax of a learned reward.

So roll both from the SAME context and look. Note what CANNOT be scored: a plan is a counterfactual, so
there is no ground-truth future for it and no LPIPS to report -- only the recorded-action rollout has a
target. That asymmetry is why the plan videos have no GT panel beside them.

    CUDA_VISIBLE_DEVICES=0 python scratch/check_plan_vs_recorded_rollout.py <train_action_run> <plan_dir>
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from quickdraw.data.dataset import load_split_episodes_mm, set_action_aggregate, set_subsample
from quickdraw.training.setup import (build_model, image_head_cams, image_head_sizes, load_checkpoint,
                                      normalizer, resolve_data_root)


def main(run: str, plan_dir: str, out: str = "logs/paper_icra_2027/_actions_control.png") -> int:
    d = json.load(open(os.path.join(plan_dir, "plan.json")))
    ei, t, H = d["episode"], d["start"], d["horizon"]
    cfg = OmegaConf.create(json.load(open(os.path.join(run, "logs", "config.json"))))
    set_subsample(int(cfg.data.get("subsample", 1) or 1)); set_action_aggregate(str(cfg.data.get("action_aggregate", "sum")))
    m = build_model(cfg).to("cuda"); load_checkpoint(m, os.path.join(run, "checkpoints", "last.ckpt")); m.eval()
    core = getattr(m, "_orig_mod", m); norm = normalizer(cfg)
    P = int(cfg.data.P)
    img = next((n for n, _ in core.layout if n != "proprio"), None)
    eps = load_split_episodes_mm(resolve_data_root(cfg), "val", img_size=image_head_sizes(cfg),
                                 cam=image_head_cams(cfg), repo_id=cfg.data.get("repo_id", "torus"))
    o, a, fr = eps[ei]
    H = min(H, len(o) - t - 1)
    ctx = {"proprio": norm.norm_obs(torch.from_numpy(o[t - P:t])).float()[None].cuda(),
           img: torch.from_numpy(fr[img][t - P:t]).float().div(255.)[None].cuda()}
    rec = norm.norm_act(torch.from_numpy(a[t - P:t + H]).float())[None].cuda()      # P-1+H+1 -> trim below
    rec = rec[:, : P - 1 + H]
    with torch.no_grad():
        outr = core.imagine_eval(ctx, rec, H, heads=[img])
        imr = (outr[img][0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        # the TRUE frames, which exist only for the recorded actions
        tru = fr[img][t:t + H]
    # the plan's own frames come from the video the routine already wrote
    import cv2
    cap = cv2.VideoCapture(os.path.join(plan_dir, "image.mp4")); pl = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        pl.append(f[..., ::-1])
    cap.release()
    pl = np.stack(pl)[:, :imr.shape[1]]                                     # drop the caption bar
    ks = [1, H // 4, H // 2, (3 * H) // 4, H - 1]
    rows = [np.concatenate([x[k] for k in ks], axis=1) for x in (tru, imr, pl)]
    lab = ["TRUE frames (recorded actions)", "ROLLOUT on recorded actions", f"PLAN: {d['request']!r} ({d['proposal']})"]
    tiles = []
    for r, l in zip(rows, lab):
        pad = np.zeros((26, r.shape[1], 3), np.uint8)
        cv2.putText(pad, l + f"   steps {ks}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles += [pad, r]
    img_out = np.concatenate(tiles, axis=0)
    Image.fromarray(img_out).save(out)
    print(f"  ep{ei} t{t} H={H} -> {out}  {img_out.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
