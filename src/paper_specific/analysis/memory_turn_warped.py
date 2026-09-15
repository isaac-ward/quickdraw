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
from matplotlib.patches import PathPatch
from matplotlib.path import Path
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
FS = 6.4                    # ONE font size for every label; the figure is ONE COLUMN wide
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


def rounded_path(pts, r):
    """Open polyline with interior corners rounded to radius ~r -- copied from
    paper_specific/figures/make_sequence_figs.py, which is where the paper's braces come from. A stroked
    polyline's `round` joinstyle only rounds by half the line width, so it cannot honour a radius; this
    trims each corner back by r along both edges and joins them with a quadratic through the vertex."""
    pts = [np.asarray(q, float) for q in pts]
    rs = [float(r)] * max(0, len(pts) - 2)
    verts, codes = [pts[0]], [Path.MOVETO]
    for i in range(1, len(pts) - 1):
        prev, cur, nxt = pts[i - 1], pts[i], pts[i + 1]
        u_in, u_out = cur - prev, nxt - cur
        l_in, l_out = np.linalg.norm(u_in) or 1.0, np.linalg.norm(u_out) or 1.0
        dd = min(rs[i - 1], 0.5 * l_in, 0.5 * l_out)
        verts += [cur - dd * u_in / l_in, cur, cur + dd * u_out / l_out]
        codes += [Path.LINETO, Path.CURVE3, Path.CURVE3]
    verts.append(pts[-1]); codes.append(Path.LINETO)
    return Path(verts, codes)


def brace_up(fig, xa, xb, y, up_to, x_stem, r=0.010, inset=0.004, **kw):
    """ONE brace: a flat span from xa to xb at y, ends turned down, and a stem from its middle up to
    (x_stem, up_to). Figure coordinates.

    `inset` pulls each end in so two adjacent braces do not touch -- consecutive periods share a
    boundary in x, and without it the pair reads as one continuous line. The stem is VERTICAL, then
    angled, then VERTICAL again, so it leaves the brace and meets the image square on."""
    xa, xb = xa + inset, xb - inset
    mid = 0.5 * (xa + xb)
    for xe in (xa, xb):
        pts = [(xe, y - 0.007), (xe, y), (mid, y), (mid, y + 0.005)]
        fig.add_artist(PathPatch(rounded_path(pts, r), fill=False, transform=fig.transFigure, **kw))
    rise = up_to - (y + 0.005)
    pts = [(mid, y + 0.005), (mid, y + 0.005 + 0.34 * rise),
           (x_stem, y + 0.005 + 0.72 * rise), (x_stem, up_to)]
    fig.add_artist(PathPatch(rounded_path(pts, r * 0.8), fill=False, transform=fig.transFigure, **kw))


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
    # BACKWALL2 ONLY, as asked: averaging two splits with opposite turn directions put two stories in one
    # axis. The three errors SHARE an axis (they are all in 0..0.5) and sit directly above the heading
    # they are explained by, on one x axis that starts at 0.
    SP = MEM[1]
    d = allsp[SP]
    g, D = d["grid"], d["D"]
    # THE FEATURED EPISODE IS CHOSEN SO THE RETURN IS REAL. "back" is the first step where the heading is
    # within 25 deg of its start, which is not the same as being back in front of the wall -- in the first
    # version the final Truth frame was still facing away. Score every candidate by how closely SOME
    # post-return truth frame matches its pre-turn truth frame, and feature the episode that returns
    # best; the frame shown is that matching step, not `back` itself.
    cands = []
    for r in recs:
        if r["away"] < P + 2 or r["back"] >= r["steps"][-1]:
            continue
        pf, tf = r["frames"]
        kb = int(r["away"] - P)                                   # the pre-turn step, in rollout index
        post = range(int(r["back"] - P), len(tf))
        if kb < 0 or not len(post):
            continue
        dif = [(float(np.abs(tf[k].astype(np.float32) - tf[kb].astype(np.float32)).mean()), k)
               for k in post]
        best_d, best_k = min(dif)
        cands.append((best_d, r, kb, best_k))
    cands.sort(key=lambda x: x[0])
    _, feat, k_before, k_after = cands[0]
    pred_fr, true_fr = feat["frames"]
    k_during = int((feat["away"] + feat["back"]) // 2 - P)
    print(f"  featured ep{feat['ep']}: away {feat['away']} back {feat['back']} | frames at rollout steps "
          f"{k_before}, {k_during}, {k_after} | return match {cands[0][0]:.1f}/255 mean abs")
    EV = [("before the turn", k_before), ("during the turn", k_during), ("facing it again", k_after)]

    # RE-ZERO ON THE FIRST DATA POINT, not on the grid: the grid starts before any episode contributes,
    # which is where the leading empty stretch and the negative ticks came from.
    have = np.sum(~np.isnan(d[METRICS[0][0]]), axis=0) >= 1
    x0 = float(g[have][0])
    gx = g - x0
    t_away, t_back, t_end = -x0, D - x0, float(gx[have][-1])
    # HAND-PLACED, so the six frames are COMPLETELY FLUSH. In a gridspec the cell is not the frame's
    # aspect, so imshow (adjustable="box") shrinks the axes inside its cell and centres it -- the slack
    # became whitespace that no wspace/hspace could close, which is why setting them to zero never did
    # anything. Here the figure height is SOLVED from the frame width, so each cell IS the frame. No
    # frame is rescaled and no aspect is touched.
    FIG_W, L, R = 3.4, 0.60, 0.01     # L clears the plots' two-line ylabel and their tick labels
    IW = (FIG_W - L - R) / 3.0        # one frame, and there are three columns
    IH = IW * pred_fr[0].shape[0] / pred_fr[0].shape[1]
    # BR is the band between the frames and the plots: it holds nothing but the brace and its stem, so
    # it is HALF what it was, at the author's ask.
    TOP_IN, BR, PL, BOT = 0.22, 0.16, 1.55, 0.34
    FIG_H = TOP_IN + 2 * IH + BR + PL + BOT
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    xl, xr = L / FIG_W, 1.0 - R / FIG_W
    iw, ih = IW / FIG_W, IH / FIG_H
    y_r0 = 1.0 - (TOP_IN + IH) / FIG_H
    gcur = fig.add_gridspec(2, 1, hspace=0.0, height_ratios=(1.15, 1.0),
                            left=xl, right=xr, top=(BOT + PL) / FIG_H, bottom=BOT / FIG_H)
    im_axes, top_axes = [], []
    for c, (lab, k) in enumerate(EV):
        k = int(np.clip(k, 0, len(pred_fr) - 1))
        for r, (img, nm) in enumerate(((pred_fr[k], "Predicted"), (true_fr[k], "Truth"))):
            A = fig.add_axes([xl + c * iw, y_r0 - r * ih, iw, ih])
            A.imshow(img, interpolation="bilinear")   # equal aspect: never stretch a frame
            A.set_xticks([]); A.set_yticks([])
            for sp_ in A.spines.values():
                sp_.set_linewidth(0.45); sp_.set_color("black")
            if c == 0:
                A.set_ylabel(nm, fontsize=FS)
            if r == 1:
                im_axes.append(A)
            else:
                top_axes.append(A)
                # THE STEP TAG GOES INSIDE, on a translucent plate: the design the author prefers. Only
                # the period's name is left above, which is what the brace has to point at.
                A.text(0.035, 0.94, f"$+${int(k) + 1}", transform=A.transAxes, ha="left", va="top",
                       fontsize=FS - 1.0, color="black",
                       bbox=dict(boxstyle="square,pad=0.14", fc="white", ec="none", alpha=0.74))
    AC = fig.add_subplot(gcur[0])
    for (kk, name), col in zip(METRICS, ("tab:blue", "tab:green", "tab:red")):
        S = d[kk]
        mu = np.nanmean(S, axis=0); cnt = np.sum(~np.isnan(S), axis=0)
        sem = np.nanstd(S, axis=0) / np.sqrt(np.maximum(1, cnt))
        keep = cnt >= 1                          # EVERY step with data, however few episodes reach it
        AC.plot(gx[keep], mu[keep], color=col, lw=1.5, label=name.replace("open-loop ", ""))
        AC.fill_between(gx[keep], (mu - sem)[keep], (mu + sem)[keep], color=col, alpha=0.16)
    AC.set_ylabel("Prediction\nerror", fontsize=FS)
    AC.legend(fontsize=FS - 1.0, ncol=1, loc="lower right", handlelength=1.2, borderpad=0.35,
              labelspacing=0.25, framealpha=0.85)
    AH = fig.add_subplot(gcur[1], sharex=AC)
    S = d["heading"]
    mu = np.nanmean(S, axis=0); cnt = np.sum(~np.isnan(S), axis=0); keep = cnt >= 1
    AH.plot(gx[keep], mu[keep], color="0.25", lw=1.6)
    AH.set_ylabel("Heading ($^\circ$)", fontsize=FS)
    AH.set_xlabel("Open-loop prediction step", fontsize=FS)
    for A in (AC, AH):
        A.axvspan(t_away, t_back, color="tab:orange", alpha=0.12, lw=0)
        A.axvline(t_away, color="k", ls="--", lw=0.8); A.axvline(t_back, color="k", ls="--", lw=0.8)
        A.grid(alpha=0.25); A.tick_params(labelsize=FS - 1.5)
        A.set_xlim(0, t_end)
    AC.tick_params(labelbottom=False)
    # ...and TIE EACH PERIOD TO ITS IMAGE COLUMN with the paper's own brace: a flat span over the period,
    # its ends turned down, and a stem from the middle up to the frames drawn from it. The period's name
    # sits between the brace and the plot.
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    # the brace sits high enough that its two-line label clears the axes below it
    y_br = AC.get_position().y1 + 0.012
    steps = [int(np.clip(k, 0, len(pred_fr) - 1)) + 1 for _, k in EV]
    for i, ((xa, xb), A, txt) in enumerate(zip(((0.0, t_away), (t_away, t_back), (t_back, t_end)),
                                               im_axes,
                                               ("Looking at\nOOD region",
                                                "Looking away from\nOOD region",
                                                "Looking back at\nOOD region"))):
        fa = inv.transform(AC.transData.transform((xa, 0)))[0]
        fb = inv.transform(AC.transData.transform((xb, 0)))[0]
        col = A.get_position()
        brace_up(fig, fa, fb, y_br, col.y0, 0.5 * (col.x0 + col.x1), color="0.35", lw=0.9)
        # the period's name goes ABOVE its image pair, not under the brace
        fig.text(0.5 * (col.x0 + col.x1), top_axes[i].get_position().y1 + 0.006, txt, ha="center",
                 va="bottom", fontsize=FS, color="0.25", linespacing=1.2)
    f2 = os.path.join(out_root, "eval_memory", "_memory_paper.png")
    fig.savefig(f2, dpi=450); plt.close(fig)          # NO bbox_inches: tight would re-trim the margins
    print("  wrote", f2)
    json.dump(summary, open(os.path.join(out_root, "memory_turn_warped.json"), "w"), indent=1)
    print(f"\n  -> {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
