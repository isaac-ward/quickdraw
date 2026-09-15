"""LATENT-space anomaly, drawn in IMAGE space. Two routes, both measured against the pixel-space map.

The latent surprise (1 - cos of predicted vs encoded bag) detects well but is a scalar -- the image head's
32 tokens are a bag, not a grid, so there is nothing to lay over the frame. Two ways to get a map anyway:

  recon_diff   Decode BOTH bags and difference the decodes: |decode(z_true) - decode(z_pred)|. The
               pixel-space map compares the prediction against the RAW frame, so it is charged for the
               codec's own blur -- everything the tokenizer cannot represent shows up as "surprise" in
               every frame, which is exactly the standing edge response we could not remove. Differencing
               two decodes puts the codec on both sides, so what remains is what the DYNAMICS got wrong.

  token_attrib A real latent->image attribution. For each image token k, substitute z_true's token k into
               the predicted bag, decode, and measure where the image changed: that is token k's spatial
               footprint. Weight each footprint by how much that token actually disagreed and sum. 32
               decodes per frame, no gradients, and it answers "which parts of the picture are explained
               by the tokens the model got wrong".

Scored the same way as before: pixel AUC against the colour mask, so the three are directly comparable.

    CUDA_VISIBLE_DEVICES=0 python scratch/latent_ood_maps.py <wm_ckpt> [out_dir]
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(__file__))   # sibling analyses in this package
from localise_ood_pixels import PATCH, SPLIT, SUB, auc, ensemble, maps_from, pink_mask   # noqa: E402

from quickdraw.data.dataset import load_split_episodes_mm, set_action_aggregate, set_subsample
from quickdraw.data.ood_windows import kept, window_steps
from quickdraw.training.setup import (build_model, image_head_cams, image_head_sizes, load_checkpoint,
                                      normalizer, resolve_data_root)

MAPS = ["surprise_patch", "recon_diff", "token_attrib"]


def pool(x):
    return F.avg_pool2d(x[None, None], PATCH, stride=1, padding=PATCH // 2)[0, 0][:x.shape[0], :x.shape[1]]


@torch.no_grad()
def recon_diff_map(core, norm, o, a, fr, key, P, t, dev, patch=4):
    """|decode(z_true) - decode(z_pred)|, pooled. The codec sits on BOTH sides so its blur cancels.

    Split out of `latent_maps` when token_attrib was dropped (2026-09-15): it scored 0.777 pixel AUC, the
    weakest of the four maps, and cost 32 extra decodes per frame -- the substitution footprints are too
    diffuse to localise with, because each of the 32 image tokens influences a broad region of the frame."""
    ctx = {"proprio": norm.norm_obs(torch.from_numpy(o[t - P:t])).float()[None].to(dev),
           key: torch.from_numpy(fr[t - P:t]).float().div(255.0)[None].to(dev)}
    acts = norm.norm_act(torch.from_numpy(a[t - P:t])).float()[None].to(dev)
    out = core.imagine_eval(ctx, acts, 1, heads=[key], norm=norm, return_bag=True)
    z_pred = out["_bag"][:, 0]
    z_true = core.encode_state({"proprio": norm.norm_obs(torch.from_numpy(o[t:t + 1])).float()[None].to(dev),
                                key: torch.from_numpy(fr[t:t + 1]).float().div(255.0)[None].to(dev)})[:, 0]
    dec = lambda z: core.to_obs(z[:, None], heads=[key], commit=True)[key][0, 0].clamp(0, 1)
    dp, dt = dec(z_pred), dec(z_true)
    diff = (dp - dt).abs().mean(-1)
    pooled = F.avg_pool2d(diff[None, None], patch, 1, patch // 2)[0, 0][:diff.shape[0], :diff.shape[1]]
    return pooled, dp


@torch.no_grad()
def latent_maps(core, norm, o, a, fr, key, P, t, dev):
    """-> (recon_diff, token_attrib, decoded predicted mean) all in image space."""
    ctx = {"proprio": norm.norm_obs(torch.from_numpy(o[t - P:t])).float()[None].to(dev),
           key: torch.from_numpy(fr[t - P:t]).float().div(255.0)[None].to(dev)}
    acts = norm.norm_act(torch.from_numpy(a[t - P:t])).float()[None].to(dev)
    out = core.imagine_eval(ctx, acts, 1, heads=[key], norm=norm, return_bag=True)
    z_pred = out["_bag"][:, 0]                                            # (1, n_state, d)
    z_true = core.encode_state({"proprio": norm.norm_obs(torch.from_numpy(o[t:t + 1])).float()[None].to(dev),
                                key: torch.from_numpy(fr[t:t + 1]).float().div(255.0)[None].to(dev)})[:, 0]
    dec = lambda z: core.to_obs(z[:, None], heads=[key], commit=True)[key][0, 0].clamp(0, 1)
    dp, dt = dec(z_pred), dec(z_true)
    recon_diff = pool((dp - dt).abs().mean(-1))

    off = 0                                                               # image tokens' slice in the bag
    for nm, n in core.layout:
        if nm == key:
            sl = slice(off, off + n); break
        off += n
    ntok = sl.stop - sl.start
    # per-token disagreement, and per-token spatial footprint by substitution
    dis = (1.0 - F.cosine_similarity(z_pred[0, sl], z_true[0, sl], dim=-1)).clamp(min=0)   # (ntok,)
    Z = z_pred.repeat(ntok, 1, 1)
    for k in range(ntok):
        Z[k, sl.start + k] = z_true[0, sl.start + k]
    decs = core.to_obs(Z[:, None], heads=[key], commit=True)[key][:, 0].clamp(0, 1)         # (ntok,H,W,3)
    foot = (decs - dp[None]).abs().mean(-1)                                                 # (ntok,H,W)
    attrib = pool((foot * dis[:, None, None]).sum(0))
    return recon_diff, attrib, dp


def main(ckpt: str, out_root: str = "logs/paper_icra_2027") -> int:
    run = os.path.dirname(os.path.dirname(ckpt)) if ckpt.endswith(".ckpt") else ckpt
    cfg = OmegaConf.create(json.load(open(os.path.join(run, "logs", "config.json"))))
    dev = "cuda"
    set_subsample(SUB); set_action_aggregate("concat")
    m = build_model(cfg).to(dev); load_checkpoint(m, ckpt); m.eval()
    core = getattr(m, "_orig_mod", m); norm = normalizer(cfg)
    P = int(cfg.data.P)
    key = next((n for n, _ in core.layout if n != "proprio"))
    eps = load_split_episodes_mm(resolve_data_root(cfg), SPLIT, img_size=image_head_sizes(cfg),
                                 cam=image_head_cams(cfg), repo_id="starling-2")
    d = os.path.join(out_root, "eval_ood", SPLIT, "viz"); os.makedirs(d, exist_ok=True)
    pooled = {k: ([], []) for k in MAPS}
    shown = 0
    print(f"  {SPLIT}: pixel-space surprise vs two latent-space routes, same frames as before")
    print(f"  {'ep':>3s} {'step':>5s} " + " ".join(f"{k:>15s}" for k in MAPS))
    for i in kept(SPLIT):
        o, a, fr = eps[i]
        w0, w1 = window_steps(SPLIT, i, SUB)
        for t in [x for x in range(max(P, w0), min(w1, len(o)))][:3]:
            gt = pink_mask(fr[key][t])
            if gt.sum() < 50:
                continue
            obs = torch.from_numpy(fr[key][t]).float().div(255.0).to(dev)
            mp, mu, _ = maps_from(ensemble(core, norm, o, a, fr[key], key, P, t, dev, n=32), obs)
            rd, at, dp = latent_maps(core, norm, o, a, fr[key], key, P, t, dev)
            cur = {"surprise_patch": mp["surprise_patch"], "recon_diff": rd, "token_attrib": at}
            line = ""
            for k in MAPS:
                v = cur[k].cpu().numpy()
                pooled[k][0].extend(v[gt].tolist()); pooled[k][1].extend(v[~gt].tolist())
                line += f" {auc(v[gt], v[~gt]):>15.3f}"
            print(f"  {i:>3d} {t:>5d}{line}")
            if shown < 3:
                fig, ax = plt.subplots(1, 5, figsize=(17, 3.1))
                ax[0].imshow(fr[key][t]); ax[0].set_title(f"observed (ep{i:02d} t{t})", fontsize=8)
                ax[1].imshow((dp.cpu().numpy() * 255).astype(np.uint8))
                ax[1].set_title("decode(z_pred)", fontsize=8)
                for A, k in ((ax[2], "surprise_patch"), (ax[3], "recon_diff"), (ax[4], "token_attrib")):
                    arr = cur[k].cpu().numpy()
                    A.imshow(arr, cmap="inferno", vmin=np.percentile(arr, 50), vmax=np.percentile(arr, 99))
                    A.set_title(f"{k}  AUC {auc(arr[gt], arr[~gt]):.3f}", fontsize=8)
                for A in ax:
                    A.axis("off")
                fig.tight_layout()
                fig.savefig(os.path.join(d, f"ep{i:02d}_step{t:03d}_latent_maps.png"), dpi=110)
                plt.close(fig); shown += 1
    print("\n  POOLED " + " ".join(
        f"{auc(np.array(pooled[k][0]), np.array(pooled[k][1])):>15.3f}" for k in MAPS))
    json.dump({k: auc(np.array(pooled[k][0]), np.array(pooled[k][1])) for k in MAPS},
              open(os.path.join(out_root, "ood_latent_maps.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
