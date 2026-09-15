"""Per split: open-loop image errors on top, heading on the bottom, turn-aligned and averaged.

Alignment is the piecewise-linear warp -- real step offsets before the turn, the turn rescaled to the
split's mean duration, real offsets after the return -- so x=0 is always "leaves the scene" and the dashed
line is always "back facing it", with no smearing at either event.

Averaged over whatever episodes have data at each x, with no minimum-count clipping; the thin grey line at
the bottom of the top panel reports how many episodes that is, since the ends of the axis are supported by
fewer of them.

Heading is plotted TRUE (solid) and PREDICTED (dashed) because the pair answers a question the error
curves cannot: whether the model is even tracking the rotation it was commanded, or whether it has lost
the turn entirely.

    CUDA_VISIBLE_DEVICES=0 python scratch/memory_curves.py <wm_ckpt> [out_dir]
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
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(__file__))   # sibling analyses in this package
from detect_anomalies_wm import MEM, SUB, yaw_deg                                      # noqa: E402
from memory_turn_warped import PRE, POST, warp                                         # noqa: E402

from quickdraw.data.dataset import load_split_episodes_mm, set_action_aggregate, set_subsample
from quickdraw.data.ood_windows import kept
from quickdraw.evaluation.openloop import image_curves
from quickdraw.training.setup import (build_model, image_head_cams, image_head_sizes, load_checkpoint,
                                      normalizer, resolve_data_root)


@torch.no_grad()
def rollout(core, norm, o, a, fr, key, P, dev):
    """One open-loop rollout -> steps, per-step {lpips,l1,l2}, and the PREDICTED heading."""
    H = len(o) - P
    ctx = {"proprio": norm.norm_obs(torch.from_numpy(o[:P])).float()[None].to(dev),
           key: torch.from_numpy(fr[:P]).float().div(255.0)[None].to(dev)}
    acts = norm.norm_act(torch.from_numpy(a[:P - 1 + H])).float()[None].to(dev)
    out = core.imagine_eval(ctx, acts, H, heads=["proprio", key], norm=norm)
    pred = out[key][0].clamp(0, 1)
    true = torch.from_numpy(fr[P:P + H]).float().div(255.0).to(dev)
    lp, l1, l2 = [], [], []
    for i in range(H):
        c = image_curves(pred[i:i + 1].unsqueeze(1), true[i:i + 1].unsqueeze(1))
        lp.append(c["lpips"][0]); l1.append(c["l1"][0]); l2.append(float(np.sqrt(c["mse"][0])))
    pro = norm.denorm_obs(out["proprio"][0].cpu()).numpy()
    return (np.arange(P, P + H), np.array(lp), np.array(l1), np.array(l2), yaw_deg(pro[:, 6:10]))


def main(ckpt: str, out_root: str = "logs/paper_icra_2027") -> int:
    run = os.path.dirname(os.path.dirname(ckpt)) if ckpt.endswith(".ckpt") else ckpt
    cfg = OmegaConf.create(json.load(open(os.path.join(run, "logs", "config.json"))))
    dev = "cuda"
    set_subsample(SUB); set_action_aggregate("concat")
    m = build_model(cfg).to(dev); load_checkpoint(m, ckpt); m.eval()
    core = getattr(m, "_orig_mod", m); norm = normalizer(cfg)
    P = int(cfg.data.P)
    key = next((n for n, _ in core.layout if n != "proprio"))
    root = resolve_data_root(cfg)
    kw = dict(img_size=image_head_sizes(cfg), cam=image_head_cams(cfg), repo_id="starling-2")
    summary = {}
    for sp in MEM:
        eps = load_split_episodes_mm(root, sp, **kw)
        recs = []
        for i in [x for x in kept(sp) if x < len(eps)]:
            o, a, fr = eps[i]
            yt = yaw_deg(o[:, 6:10]); dv = yt - yt[0]
            aw = np.where(np.abs(dv) > 45.0)[0]
            if len(aw) == 0:
                continue
            s = int(aw[0]); bk = np.where(np.abs(dv[s:]) < 25.0)[0]
            e = s + int(bk[0]) if len(bk) else len(dv) - 1
            st, lp, l1, l2, yp = rollout(core, norm, o, a, fr[key], key, P, dev)
            recs.append({"ep": i, "steps": st, "lpips": lp, "l1": l1, "l2": l2, "away": s, "back": e,
                         "yaw_true": yt[st] - yt[0], "yaw_pred": yp - yp[0]})
        D = float(np.mean([r["back"] - r["away"] for r in recs]))
        grid = np.arange(-PRE, D + POST + 1e-9, 1.0)
        S = {}
        for kk in ("lpips", "l1", "l2", "yaw_true", "yaw_pred"):
            M = np.full((len(recs), len(grid)), np.nan)
            for j, r in enumerate(recs):
                x = warp(r["steps"], r["away"], r["back"], D)
                ins = (grid >= x.min()) & (grid <= x.max())
                M[j, ins] = np.interp(grid[ins], x, r[kk])
            S[kk] = M
        cnt = np.sum(~np.isnan(S["lpips"]), axis=0)

        fig, ax = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
        for kk, c in (("lpips", "crimson"), ("l1", "tab:blue"), ("l2", "tab:green")):
            mu = np.nanmean(S[kk], axis=0)
            sem = np.nanstd(S[kk], axis=0) / np.sqrt(np.maximum(1, cnt))
            ax[0].plot(grid, mu, color=c, lw=1.9, label=kk)
            ax[0].fill_between(grid, mu - sem, mu + sem, color=c, alpha=0.18)
        a2 = ax[0].twinx()
        a2.plot(grid, cnt, color="0.6", lw=0.8, ls=":")
        a2.set_ylabel("episodes averaged", color="0.5", fontsize=8)
        a2.set_ylim(0, len(recs) + 0.5); a2.tick_params(labelsize=7, colors="0.5")
        ax[0].set_ylabel("open-loop image error"); ax[0].legend(fontsize=8, loc="upper left")
        for kk, ls, lab in (("yaw_true", "-", "recorded heading"), ("yaw_pred", "--", "predicted heading")):
            mu = np.nanmean(S[kk], axis=0)
            sem = np.nanstd(S[kk], axis=0) / np.sqrt(np.maximum(1, cnt))
            ax[1].plot(grid, mu, color="k", ls=ls, lw=1.6, label=lab)
            ax[1].fill_between(grid, mu - sem, mu + sem, color="k", alpha=0.12)
        ax[1].set_ylabel("heading change (deg)"); ax[1].legend(fontsize=8)
        for A in ax:
            A.axvspan(0, D, color="tab:orange", alpha=0.10, lw=0)
            A.axvline(0, color="k", ls="--", lw=1); A.axvline(D, color="k", ls="--", lw=1)
            A.grid(alpha=0.25)
        ax[1].set_xlabel(f"warped model step — 0 = leaves the scene, {D:.0f} = back facing it "
                         f"(turn rescaled to the mean; shaded)")
        fig.suptitle(f"{sp} — {len(recs)} episodes, open-loop (only the first {P} frames are real)",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        f = os.path.join(out_root, "eval_memory", sp, "_curves.png")
        fig.savefig(f, dpi=120); plt.close(fig)
        seg = lambda kk, lo, hi: float(np.nanmean(S[kk][:, (grid >= lo) & (grid < hi)]))
        summary[sp] = {"n": len(recs), "mean_turn_steps": D,
                       **{f"{kk}_{w}": seg(kk, *b) for kk in ("lpips", "l1", "l2")
                          for w, b in (("before", (-PRE, 0)), ("during", (0, D)), ("after", (D, D + POST)))}}
        print(f"  {sp:24s} n={len(recs)} turn={D:.1f}  " + "  ".join(
            f"{kk} {summary[sp][f'{kk}_before']:.3f}/{summary[sp][f'{kk}_during']:.3f}/"
            f"{summary[sp][f'{kk}_after']:.3f}" for kk in ("lpips", "l1", "l2")) + f"   -> {f}")
    json.dump(summary, open(os.path.join(out_root, "memory_curves.json"), "w"), indent=1)
    print("  (each triple is before / during / after the turn)")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
