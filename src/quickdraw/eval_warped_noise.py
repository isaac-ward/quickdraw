"""Diagnostic: iid vs shared vs WARPED (∫-noise) stochastic decode on a short open-loop clip.

Renders a side-by-side MP4 (GT | iid | shared | warped) + a noise-field MP4, so temporal coherence is VISIBLE
(record §8.29) -- LPIPS cannot see it. Warped uses option (c): Farnebäck optical flow between the two
PREVIOUSLY-decoded frames (a constant-velocity estimate, since at inference there is no given video), accumulated
and used to advect ONE ∫-noise field (models/noise_transport, the paper's Eq-5 transport). This is a standalone
diagnostic: it does NOT touch the training/eval rollout path, only reads a checkpoint.

  python -m quickdraw.eval_warped_noise '+eval_ckpt="<run>/checkpoints/epoch=NN-step=MM.ckpt"' \
      +eval_out=logs/_warped +clip=64 +up=8
"""
from __future__ import annotations

import os

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf


def _farneback(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(B,H,W,3) in [0,1] -> (B,H,W,2) pixel flow a->b (Farnebäck grayscale, per batch item)."""
    import cv2
    out = []
    for i in range(a.shape[0]):
        ga = (a[i].mean(-1).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        gb = (b[i].mean(-1).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        fl = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 3, 15, 3, 5, 1.2, 0)   # (H,W,2)
        out.append(torch.from_numpy(fl))
    return torch.stack(out).to(a.device, a.dtype)


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg):
    import imageio.v2 as imageio

    from .data.dataset import DataConfig, load_split_episodes_mm
    from .models.noise_transport import fine_noise, warp_integral
    from .models.vision import img_hw
    from .training.setup import (build_model, image_head_cams, image_head_sizes, load_checkpoint,
                                 normalizer, resolve_data_root)

    ckpt = os.path.expanduser(str(cfg.get("eval_ckpt", "") or ""))
    assert ckpt and os.path.exists(ckpt), "pass +eval_ckpt=<...>.ckpt (quote it)"
    run_dir = os.path.dirname(os.path.dirname(ckpt))
    rcfg = OmegaConf.load(os.path.join(run_dir, "checkpoints", "config.resolved.yaml"))
    OmegaConf.set_struct(rcfg, False)
    for mod in rcfg.model.modalities:                                # image heads must SAMPLE for this to mean anything
        if str(mod.get("kind", "")) == "image":
            mod["decode_stochastic"] = True
    # data config flows explicitly: load_split_episodes_mm(dcfg=...) below + normalizer(rcfg) build it from rcfg.

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(rcfg).to(dev).eval()
    load_checkpoint(model, ckpt)
    m = getattr(model, "_orig_mod", model)
    norm = normalizer(rcfg)
    cams, sizes = image_head_cams(rcfg), image_head_sizes(rcfg)
    P, clip, up = int(rcfg.data.P), int(cfg.get("clip", 64)), int(cfg.get("up", 8))
    out_dir = str(cfg.get("eval_out", "") or os.path.join(run_dir, "warped_noise"))
    os.makedirs(out_dir, exist_ok=True)

    eps_data = load_split_episodes_mm(resolve_data_root(rcfg), "val", dcfg=DataConfig.from_cfg(rcfg),
                                      img_size=sizes, cam=cams, repo_id=str(rcfg.data.get("repo_id", "torus")))
    o, a, fr = eps_data[0]                                            # one val episode
    clip = min(clip, len(o) - P - 1)
    img_heads = list(cams)
    ctx = {"proprio": norm.norm_obs(torch.from_numpy(o[:P])).float()[None].to(dev)}   # (1,P,obs) NORMALIZED
    for h in img_heads:
        ctx[h] = torch.from_numpy(fr[h][:P]).float().div(255.0)[None].to(dev)         # (1,P,H,W,3) in [0,1]
    idx = np.clip(np.arange(0, P + clip - 1), 0, len(a) - 1)
    acts = norm.norm_act(torch.from_numpy(a[idx])).float()[None].to(dev)              # (1,P+clip-1,act)

    with torch.no_grad():
        res = m.imagine_eval(ctx, acts, clip, heads=["proprio"] + img_heads, norm=norm, return_bag=True)
    bag = res["_bag"]                                                 # (1, clip, n_state, d) — the LATENT rollout
    off, c = {}, 0
    for name, n in m.layout:
        off[name] = (c, n); c += n

    for h in img_heads:
        o0, n = off[h]; tok = bag[..., o0:o0 + n, :]                  # (1, clip, n, d)
        mod = m.modalities[h]
        H, W = img_hw(sizes[h])
        with torch.no_grad():
            mod.decode_stochastic, mod.decode_shared_noise = True, False
            iid = mod.decode(tok)[0]                                  # (clip,H,W,3) — independent noise per frame
            mod.decode_shared_noise = True
            shared = mod.decode(tok)[0]                               # one fixed field, all frames
            mod.decode_shared_noise = False
            # WARPED (c): sequential; flow from the two previously-decoded frames, accumulated.
            fine = fine_noise(1, 3, H, W, up, device=bag.device, dtype=bag.dtype)
            cum = torch.zeros(1, H, W, 2, device=bag.device, dtype=bag.dtype)
            prev2 = prev = None
            wf, nzf = [], []
            for t in range(clip):
                if t >= 2:
                    cum = cum + _farneback(prev2[None], prev[None])   # (c): last step's flow as this step's estimate
                epst = warp_integral(fine, cum, up).permute(0, 2, 3, 1)   # (1,H,W,3)
                f = mod.decode(tok[:, t:t + 1], eps=epst)[0, 0]       # (H,W,3)
                wf.append(f); nzf.append(epst[0])
                prev2, prev = prev, f
            warped = torch.stack(wf); noise = torch.stack(nzf)        # (clip,H,W,3)

        gt = torch.from_numpy(fr[h][P:P + clip]).float().div(255.0).to(bag.device)
        panel = torch.cat([gt, iid.clamp(0, 1), shared.clamp(0, 1), warped.clamp(0, 1)], dim=2)  # (clip,H,4W,3)
        imageio.mimsave(os.path.join(out_dir, f"{h}_GT-iid-shared-warped.mp4"),
                        list((panel.cpu().numpy() * 255).astype(np.uint8)), fps=10)
        nv = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)
        imageio.mimsave(os.path.join(out_dir, f"{h}_warped_noisefield.mp4"),
                        list((nv.cpu().numpy() * 255).astype(np.uint8)), fps=10)
        print(f"[warped] {h}: wrote GT|iid|shared|warped ({clip} frames) + noise field", flush=True)

    print("[warped] DONE ->", out_dir, flush=True)


if __name__ == "__main__":
    main()
