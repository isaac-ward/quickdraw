"""Five diagnostics that between them killed five hypotheses about why objects vanish in rollouts.

    python -m quickdraw._oneoff_rollout_diagnosis <run_dir> [--ckpt epoch=9-step=10940.ckpt]

Each answers ONE question with a number, on a trained checkpoint, with no training and no config
change. Written 2026-09-13 for block-stack; the symptom was an UNTOUCHED, STATIONARY object
disappearing over an open-loop rollout.

  spread   Are the flow's samples diverse, or has it collapsed to a deterministic map?
           -> block-stack: 8 draws spread to 2.8x their own step-motion by 10 s. NOT collapsed.
           The robocasa record concluded the opposite ("the flow has learned a near-DETERMINISTIC
           map"), but robocasa is a SIMULATOR REPLAYING SCRIPTED DEMOS -- deterministic by
           construction, so there was nothing there to learn. Re-measure on any new dataset.

  bestofk  Is real diversity being averaged away by scoring one sample against one future?
           -> best-of-8 gained only 2.5% at 21 s, and the gain SHRANK with horizon (8.8% -> 2.5%),
           the opposite of what multimodality predicts. The samples differ hugely in latent space
           but are all wrong in the same way. Scoring is not the problem.

  codec    Can the codec even hold the moving content?
           -> recon error was 0.29x the patch's own temporal variation on dynamic patches against
           2.48x on still ones: the codec holds moving content PROPORTIONALLY BETTER. Capacity is
           not the problem. NOTE ae_floor cannot answer this -- it averages over a frame that is
           84.6% static, understating the error where the action is by 2.2x.

  sharp    Has the rolled latent drifted OFF the manifold the decoder knows?
           -> rolled frames are 0.90-1.01x as sharp as real ones at every horizon, and
           encode->decode is 0.95x. Nothing is blurring. The model renders a SHARP, CONFIDENTLY
           WRONG scene. This is the finding that reframed everything: an object is not being
           smeared out of existence, it is being omitted from a scene drawn somewhere else.

  drift    How fast does the rolled latent leave the truth?
           -> distance from the true encoded latent, relative to the latent's own norm: 0.58 at
           step 0, 1.22 by 5 s, 1.31 by 21 s. Essentially uncorrelated within five seconds.
           `eval_ood_horizon/open_loop/<head>/latent_cos` already logs this per epoch and shows it
           DEGRADING MONOTONICALLY with training (0.374 -> 0.234 at 5 s over ep1..ep9) while lpips
           IMPROVES. Prefer the logged series; this is the one-off, higher-resolution version.
"""
from __future__ import annotations

import glob
import json
import sys

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

DEV = "cuda:0"


def _load(run_dir: str, ckpt: str | None):
    from .training.setup import build_model, load_checkpoint
    cfg = OmegaConf.load(f"{run_dir}/checkpoints/config.resolved.yaml")
    path = f"{run_dir}/checkpoints/{ckpt}" if ckpt else f"{run_dir}/checkpoints/best.ckpt"
    torch.manual_seed(0)
    m = build_model(cfg)
    load_checkpoint(m, path)
    return m.to(DEV).eval(), cfg


def _data(root: str, stride: int):
    df = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{root}/val/data/*/*.parquet"))])
    e0 = df[df.episode_index == 0].sort_values("frame_index")
    st = json.load(open(f"{root}/normalization_stats.json"))
    O = (np.stack(e0.observation_vector.values).astype(np.float32)
         - np.array(st["observation_vector"]["mean"], np.float32)) / np.array(st["observation_vector"]["std"], np.float32)
    A = (np.stack(e0.action.values).astype(np.float32)
         - np.array(st["action"]["mean"], np.float32)) / np.array(st["action"]["std"], np.float32)
    cams = {c: np.load(f"{root}/val/{c}_96x128.npy", mmap_mode="r")
            for c in ("scene_right", "gripper_right_top")}
    return O, A, cams


def main() -> int:
    run = sys.argv[1] if len(sys.argv) > 1 else "logs/train_world_model_2026_09_11_03_19_45_bs_stride10"
    ckpt = None
    if "--ckpt" in sys.argv:
        ckpt = sys.argv[sys.argv.index("--ckpt") + 1]
    model, cfg = _load(run, ckpt)
    ROOT, STRIDE, P, H, K = cfg.data.root, int(cfg.data.subsample), int(cfg.data.P), 64, 8
    O, A, cams = _data(ROOT, STRIDE)

    def fr(c, s, n):
        return torch.from_numpy(np.stack([cams[c][s + i * STRIDE] for i in range(n)])).float().div(255.)[None].to(DEV)

    S0 = 4000
    ctx = {"proprio": torch.from_numpy(O[S0:S0 + P * STRIDE:STRIDE])[None].to(DEV),
           "cam_scene": fr("scene_right", S0, P), "cam_wrist": fr("gripper_right_top", S0, P)}
    # actions must span the CONTEXT too: _rollout_cached indexes from the context steps onward.
    acts = torch.from_numpy(A[S0:S0 + (P + H) * STRIDE:STRIDE])[None].to(DEV)
    true_future = fr("scene_right", S0 + P * STRIDE, H)

    bags, imgs = [], []
    with torch.no_grad():
        for k in range(K):
            torch.manual_seed(1000 + k)                      # DIFFERENT eps per draw
            o = model.imagine_eval({m: v.clone() for m, v in ctx.items()}, acts.clone(), H,
                                   heads=["cam_scene"], return_bag=True)
            bags.append(o["_bag"][0].float().cpu())
            imgs.append(o["cam_scene"][0].float().cpu())
    B, I = torch.stack(bags), torch.stack(imgs)

    print(f"\n=== SPREAD: are the flow's samples diverse? (K={K}) ===")
    flat = B.reshape(K, H, -1)
    iu = torch.triu_indices(K, K, offset=1)
    between = torch.cdist(flat.permute(1, 0, 2), flat.permute(1, 0, 2))[:, iu[0], iu[1]].mean(1)
    within = (flat[:, 1:] - flat[:, :-1]).norm(dim=-1).mean(0)
    for t in (0, 15, 31, 63):
        print(f"    step {t:>3} ({t*STRIDE/30:>4.1f}s): between-sample / own step-motion = "
              f"{float(between[t]) / max(float(within[min(t, H-2)]), 1e-9):.2f}x")

    print(f"\n=== SHARP: is the rollout blurry, or confidently wrong? ===")
    def sharp(x):
        return float((x[:, 1:] - x[:, :-1]).abs().mean() + (x[:, :, 1:] - x[:, :, :-1]).abs().mean())
    real = sharp(true_future[0].cpu())
    print(f"    real frames {real:.5f}")
    for t in (0, 31, 63):
        print(f"    rolled to step {t:>3}: {sharp(I[0, t:t+1]) / real:.2f}x as sharp as real")

    print(f"\n=== DRIFT: how fast does the latent leave the truth? ===")
    with torch.no_grad():
        tb = model.encode_state({"proprio": torch.from_numpy(O[S0+P*STRIDE:S0+(P+H)*STRIDE:STRIDE])[None].to(DEV),
                                 "cam_scene": true_future,
                                 "cam_wrist": fr("gripper_right_top", S0 + P*STRIDE, H)})[0].float().cpu()
    d = (B[0] - tb).norm(dim=-1).mean(-1) / tb.norm(dim=-1).mean(-1)
    for t in (0, 15, 31, 63):
        print(f"    step {t:>3} ({t*STRIDE/30:>4.1f}s): {float(d[t]):.3f}   (>1 = further than its own norm)")
    print("\n    NOTE eval_ood_horizon/open_loop/<head>/latent_cos logs this per epoch already.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
