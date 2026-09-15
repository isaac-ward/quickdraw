"""ONE turn-aligned curve per split: piecewise time-warp so both events land in the same figure.

Aligning on the turn START and on the RETURN gave two plots, because the turn's DURATION varies across
episodes (8-21 model steps) -- align on the start and the return smears; align on the return and the start
smears. The standard fix for event-aligned averaging with a variable-duration event is a piecewise-linear
warp of the time axis:

    t < away          x = t - away                      real steps before the turn (aligned on the start)
    away <= t <= back x = (t-away)/(back-away) * D       the turn, rescaled to the split's MEAN duration D
    t > back          x = D + (t - back)                real steps after the return (aligned on the return)

So x=0 is always the moment the drone leaves the scene and x=D is always the moment it is back, with the
turn itself stretched or squeezed to the average. Only the interior is warped; the parts either side keep
their true step spacing, which is where the claim lives.

Two rows, as before: raw open-loop error, and EXCESS over val at the same horizon -- the second is the one
that means anything, because open-loop error rises with horizon whether or not there is a turn.

    CUDA_VISIBLE_DEVICES=0 python scratch/memory_turn_warped.py <wm_ckpt> [out_dir]
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
from detect_anomalies_wm import MEM, SUB, open_loop, yaw_deg                           # noqa: E402

import torch as _torch

from quickdraw.data.dataset import load_split_episodes_mm, set_action_aggregate, set_subsample
from quickdraw.evaluation.openloop import image_curves
from quickdraw.data.ood_windows import kept
from quickdraw.training.setup import (build_model, image_head_cams, image_head_sizes, load_checkpoint,
                                      normalizer, resolve_data_root)

PRE, POST = 10, 14          # real steps kept either side of the warped turn
METRICS = [("l1", "open-loop $L_1$"), ("l2", "open-loop $L_2$"), ("lpips", "open-loop LPIPS")]


@_torch.no_grad()
def open_loop_multi(core, norm, o, a, fr, key, P, dev, want_frames=False):
    """The same rollout open_loop() does, but keeping L1 and L2 as well as LPIPS -> (steps, {metric: (H,)}).

    open_loop() returns LPIPS alone because that is what the detector needs; the memory figure plots all
    three, so the per-step curves come from one image_curves call over the whole rollout rather than three
    separate rollouts."""
    H = len(o) - P
    ctx = {"proprio": norm.norm_obs(_torch.from_numpy(o[:P])).float()[None].to(dev),
           key: _torch.from_numpy(fr[:P]).float().div(255.0)[None].to(dev)}
    acts = norm.norm_act(_torch.from_numpy(a[:P - 1 + H])).float()[None].to(dev)
    pred = core.imagine_eval(ctx, acts, H, heads=[key], norm=norm)[key][0].clamp(0, 1)
    true = _torch.from_numpy(fr[P:P + H]).float().div(255.0).to(dev)
    c = image_curves(pred.unsqueeze(0), true.unsqueeze(0))
    cur = {"l1": np.asarray(c["l1"]), "l2": np.sqrt(np.asarray(c["mse"])),
           "lpips": np.asarray(c["lpips"])}
    if not want_frames:
        return np.arange(P, P + H), cur
    fr8 = ((pred.cpu().numpy() * 255).astype(np.uint8), (true.cpu().numpy() * 255).astype(np.uint8))
    return np.arange(P, P + H), cur, fr8


def warp(steps, away, back, D):
    """Piecewise-linear map from model step to the common warped axis."""
    x = np.empty(len(steps), float)
    for j, t in enumerate(steps):
        if t < away:
            x[j] = t - away
        elif t <= back:
            x[j] = (t - away) / max(1, back - away) * D
        else:
            x[j] = D + (t - back)
    return x


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

    veps = load_split_episodes_mm(root, "val", **kw)
    CAP = 48
    cur = [open_loop(core, norm, o[:CAP], a[:CAP], fr[key][:CAP], key, P, dev)[1] for o, a, fr in veps]
    n = min(len(c) for c in cur)
    base = np.mean([c[:n] for c in cur], axis=0)
    print(f"  val horizon baseline: {n} steps, lpips {base[0]:.3f} (h=0) -> {base[-1]:.3f} (h={n - 1})")

    allsp, summary = {}, {}
    for sp in MEM:
        eps = load_split_episodes_mm(root, sp, **kw)
        recs = []
        for i in [x for x in kept(sp) if x < len(eps)]:
            o, a, fr = eps[i]
            dv = yaw_deg(o[:, 6:10]) - yaw_deg(o[:, 6:10])[0]
            aw = np.where(np.abs(dv) > 45.0)[0]
            if len(aw) == 0:
                continue
            s = int(aw[0]); bk = np.where(np.abs(dv[s:]) < 25.0)[0]
            e = s + int(bk[0]) if len(bk) else len(dv) - 1
            st, cur_m, frames = open_loop_multi(core, norm, o, a, fr[key], key, P, dev, want_frames=True)
            lp = cur_m["lpips"]
            h = st - P
            ok = h < len(base)
            exc = np.full(len(lp), np.nan); exc[ok] = lp[ok] - base[h[ok]]
            recs.append({"ep": i, "steps": st, "excess": exc, "away": s, "back": e,
                         "heading": dv[st], "frames": frames, **cur_m})
        D = float(np.mean([r["back"] - r["away"] for r in recs]))
        grid = np.arange(-PRE, D + POST + 1e-9, 1.0)
        stacks = {}
        for kk in ("lpips", "excess", "l1", "l2", "heading"):   # `frames` is per-episode, not stacked
            S = np.full((len(recs), len(grid)), np.nan)
            for j, r in enumerate(recs):
                x = warp(r["steps"], r["away"], r["back"], D)
                good = np.isfinite(r[kk])
                if good.sum() < 2:
                    continue
                inside = (grid >= x[good].min()) & (grid <= x[good].max())
                S[j, inside] = np.interp(grid[inside], x[good], np.asarray(r[kk])[good])
            stacks[kk] = S
        # FEATURE AN EPISODE WHOSE PRE-TURN MOMENT EXISTS IN THE ROLLOUT. In most of these the drone is
        # already turning inside the 8-frame context (away as low as 2), so "before the turn" clipped to
        # the first predicted step and the three frames came out nearly identical. Require away to sit at
        # least two steps into the rollout, then take the turn duration closest to the split mean.
        ok_feat = [r for r in recs if r["away"] >= P + 2 and r["back"] < r["steps"][-1]]
        feat = min(ok_feat or recs, key=lambda r: abs((r["back"] - r["away"]) - D))
        print(f"  featured ep{feat['ep']}: away {feat['away']} back {feat['back']} "
              f"(turn {feat['back'] - feat['away']} steps vs mean {D:.1f})")
        allsp[sp] = {"grid": grid, "D": D, "n": len(recs), "feat": feat, **stacks}
        ex = stacks["excess"]
        seg = lambda lo, hi: float(np.nanmean(ex[:, (grid >= lo) & (grid < hi)]))
        summary[sp] = {"n_episodes": len(recs), "mean_turn_steps": D,
                       "excess_before_turn": seg(-PRE, 0), "excess_during_turn": seg(0, D),
                       "excess_after_return": seg(D, D + POST)}
        print(f"  {sp:24s} {len(recs)} eps | mean turn {D:.1f} steps | EXCESS before {summary[sp]['excess_before_turn']:+.4f} "
              f"during {summary[sp]['excess_during_turn']:+.4f} after {summary[sp]['excess_after_return']:+.4f}")

    # ---- one figure: both splits, both rows, single warped axis ----
    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    cols = {MEM[0]: "tab:blue", MEM[1]: "tab:red"}
    for sp in MEM:
        d = allsp[sp]; g, D = d["grid"], d["D"]
        for row, kk in enumerate(("lpips", "excess")):
            S = d[kk]
            mu = np.nanmean(S, axis=0)
            cnt = np.sum(~np.isnan(S), axis=0)
            sem = np.nanstd(S, axis=0) / np.sqrt(np.maximum(1, cnt))
            keep = cnt >= max(3, d["n"] // 2)          # only plot where at least half the episodes contribute
            ax[row].plot(g[keep], mu[keep], color=cols[sp], lw=1.9,
                         label=f"{sp.replace('eval_memory_', '')} (n={d['n']}, turn {D:.0f} steps)")
            ax[row].fill_between(g[keep], (mu - sem)[keep], (mu + sem)[keep], color=cols[sp], alpha=0.18)
            ax[row].axvspan(0, D, color="tab:orange", alpha=0.08, lw=0)
    for row, lab in enumerate(("open-loop LPIPS", "EXCESS over val at the same horizon")):
        ax[row].axvline(0, color="k", ls="--", lw=1)
        ax[row].set_ylabel(lab); ax[row].grid(alpha=0.25); ax[row].legend(fontsize=8)
    ax[1].axhline(0, color="k", ls=":", lw=0.9)
    for sp in MEM:
        ax[0].axvline(allsp[sp]["D"], color=cols[sp], ls="--", lw=1)
        ax[1].axvline(allsp[sp]["D"], color=cols[sp], ls="--", lw=1)
    ax[1].set_xlabel("warped model step — 0 = drone leaves the scene, dashed = back facing it "
                     "(turn rescaled to each split's mean duration; shaded = the turn)")
    fig.suptitle("Memory: does an already-seen scene come back cheaper?\n"
                 "open-loop, only the first 8 frames real. Excess of 0 = no worse than an ordinary step at "
                 "the same horizon.\nIf the scene survived the turn, the red curve should FALL back toward "
                 "0 after the dashed line.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    f = os.path.join(out_root, "eval_memory", "_turn_warped_both_splits.png")
    fig.savefig(f, dpi=120); plt.close(fig)

    # ---- the PAPER figure: one split, the three image errors, the heading, and what it looks like ----
    # BACKWALL2 ONLY. Averaging two splits with opposite turn directions and different approaches put two
    # stories in one axis; the author asked for the cleaner split, and every claim in the text is stated
    # per split anyway.
    SP = MEM[1]
    d = allsp[SP]
    g, D, feat = d["grid"], d["D"], d["feat"]
    pred_fr, true_fr = feat["frames"]
    # The three moments, in the FEATURED episode's own step index: last step before it turns away, the
    # middle of its turn, and the first step after it is facing the scene again.
    off = feat["steps"][0]
    EV = [("before the turn", feat["away"]), ("during the turn", (feat["away"] + feat["back"]) // 2),
          ("back facing it", feat["back"])]
    fig = plt.figure(figsize=(7.1, 4.6))
    outer = fig.add_gridspec(3, 1, height_ratios=(2.0, 1.25, 0.95), hspace=0.34)
    # NESTED, so the Truth frame touches the Predicted frame above it (hspace 0) while the blocks below
    # keep their own spacing -- one flat gridspec cannot do both.
    gim = outer[0].subgridspec(2, 3, hspace=0.0, wspace=0.16)
    gmet = outer[1].subgridspec(1, 3, wspace=0.30)
    for c, (lab, t) in enumerate(EV):
        k = int(np.clip(t - off, 0, len(pred_fr) - 1))
        for r, (img, nm) in enumerate(((pred_fr[k], "Predicted"), (true_fr[k], "Truth"))):
            A = fig.add_subplot(gim[r, c])
            A.imshow(img, interpolation="bilinear"); A.set_xticks([]); A.set_yticks([])
            for sp_ in A.spines.values():
                sp_.set_linewidth(1.4); sp_.set_color("black")
            if c == 0:
                A.set_ylabel(nm, fontsize=7.5)
            if r == 0:
                A.set_title(f"{lab}   (step $+${k + 1})", fontsize=7.5)
    for c, (kk, name) in enumerate(METRICS):
        A = fig.add_subplot(gmet[0, c])
        S = d[kk]
        mu = np.nanmean(S, axis=0)
        cnt = np.sum(~np.isnan(S), axis=0)
        sem = np.nanstd(S, axis=0) / np.sqrt(np.maximum(1, cnt))
        keep = cnt >= 1                          # EVERY step with data, however few episodes reach it
        A.plot(g[keep], mu[keep], color="tab:red", lw=1.5)
        A.fill_between(g[keep], (mu - sem)[keep], (mu + sem)[keep], color="tab:red", alpha=0.18)
        A.axvspan(0, D, color="tab:orange", alpha=0.12, lw=0)
        A.axvline(0, color="k", ls="--", lw=0.8); A.axvline(D, color="k", ls="--", lw=0.8)
        A.set_title(name, fontsize=8); A.grid(alpha=0.25); A.tick_params(labelsize=6.5)
        A2 = A.twinx()                           # how many episodes each point averages
        A2.plot(g[keep], cnt[keep], color="0.55", lw=0.7, ls=":")
        A2.set_ylim(0, d["n"] + 0.5)
        if c == 2:
            A2.tick_params(labelsize=5.5, colors="0.45")
            A2.set_ylabel("episodes averaged", fontsize=6, color="0.45")
        else:
            A2.set_yticks([])
    A = fig.add_subplot(outer[2])
    S = d["heading"]
    mu = np.nanmean(S, axis=0); cnt = np.sum(~np.isnan(S), axis=0); keep = cnt >= 1
    A.plot(g[keep], mu[keep], color="tab:blue", lw=1.6)
    A.axvspan(0, D, color="tab:orange", alpha=0.12, lw=0)
    A.axvline(0, color="k", ls="--", lw=0.8); A.axvline(D, color="k", ls="--", lw=0.8)
    A.axhline(0, color="k", ls=":", lw=0.7)
    A.set_ylabel("heading change (deg)", fontsize=8); A.grid(alpha=0.25); A.tick_params(labelsize=6.5)
    A.set_xlabel(f"warped model step: $0$ = the scene leaves the frame, "
                 f"{D:.0f} = the drone faces it again", fontsize=8)
    f2 = os.path.join(out_root, "eval_memory", "_memory_paper.png")
    fig.savefig(f2, dpi=450, bbox_inches="tight"); plt.close(fig)
    print("  wrote", f2)
    json.dump(summary, open(os.path.join(out_root, "memory_turn_warped.json"), "w"), indent=1)
    print(f"\n  -> {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
