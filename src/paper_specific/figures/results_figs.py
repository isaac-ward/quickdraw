"""The paper's result figures, generated from the runs -- no hand assembly.

    python -m paper_specific.figures.results_figs <paper_repo> [which ...]

`which` selects figures by name (longhorizon, memory, ood, curves, data, overview); default is all.
Everything lands in <paper_repo>/figures/ as PNG at print resolution.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
import torch

DPI = 450          # figures whose content is text and line art: export high so print stays crisp
DPI_IMG = 300      # figures that are mostly decoded 112x192 frames -- past this, DPI only upscales blur
#                    and the file grows without carrying more information
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis"))

from quickdraw.evaluation.openloop import image_curves
from quickdraw.data.dataset import load_split_episodes_mm, set_action_aggregate, set_subsample
from quickdraw.training.setup import (build_model, image_head_cams, image_head_sizes, load_checkpoint,
                                      normalizer, resolve_data_root)

WM = "logs/paper_icra_2027/model_backups/train_world_model_2026_09_11_03_17_47_s2_sub4_concat"
CKPT = WM + "/checkpoints/ah_base_ep38.ckpt"
SUB = 4


def _model(dev="cuda"):
    cfg = OmegaConf.create(json.load(open(os.path.join(WM, "logs", "config.json"))))
    set_subsample(SUB); set_action_aggregate("concat")
    m = build_model(cfg).to(dev); load_checkpoint(m, CKPT); m.eval()
    core = getattr(m, "_orig_mod", m)
    key = next((n for n, _ in core.layout if n != "proprio"))
    return cfg, core, normalizer(cfg), int(cfg.data.P), key


@torch.no_grad()
def longhorizon(paper, dev="cuda", n_show=8):
    """Open-loop rollout on RECORDED actions: Predicted above, Truth directly below, and under each pair
    the three image errors over the WHOLE rollout, not just the eight frames shown.

    Recorded actions, not planned ones: this figure is about the dynamics, so the action sequence has to
    be one the drone actually flew.

    HOW LONG: the horizon is capped by the shortest val episode, not by the model -- 7 episodes of
    1763-1796 frames, which at stride 4 leaves 440 steps, minus P=8 of context and the 8 steps of takeoff
    that a rollout should not start from. That is 423, which is 113 s, and it is what the long pair uses.
    The short pair is at +128 (34 s), the horizon the project is characterised at."""
    H_SHORT, H_LONG, T0 = 128, 423, 8
    PLAN = [(0, H_SHORT, 40), (2, H_LONG, T0)]
    cfg, core, norm, P, key = _model(dev)
    eps = load_split_episodes_mm(resolve_data_root(cfg), "val", img_size=image_head_sizes(cfg),
                                 cam=image_head_cams(cfg), repo_id="starling-2")
    rows = []
    for ei, H, t0 in PLAN:
        o, a, fr = eps[ei]
        h = min(H, len(o) - t0 - P - 1)
        ctx = {"proprio": norm.norm_obs(torch.from_numpy(o[t0:t0 + P])).float()[None].to(dev),
               key: torch.from_numpy(fr[key][t0:t0 + P]).float().div(255.0)[None].to(dev)}
        acts = norm.norm_act(torch.from_numpy(a[t0:t0 + P - 1 + h])).float()[None].to(dev)
        pr = core.imagine_eval(ctx, acts, h, heads=[key], norm=norm)[key][0].clamp(0, 1)
        gt = fr[key][t0 + P:t0 + P + h]
        gt_t = torch.from_numpy(gt).float().div(255.0).to(dev)
        c = image_curves(pr.unsqueeze(0), gt_t.unsqueeze(0))     # every step, not just the shown ones
        cur = {"$L_1$": np.asarray(c["l1"]), "$L_2$": np.sqrt(np.asarray(c["mse"])),
               "LPIPS": np.asarray(c["lpips"])}
        ks = np.unique(np.linspace(0, h - 1, n_show).round().astype(int))
        rows.append(([(pr[k].cpu().numpy() * 255).astype(np.uint8) for k in ks], [gt[k] for k in ks],
                     ks, h, cur))
        print(f"    episode {ei}: {h} steps ({h * 4 / 15.0:.0f} s), columns "
              + ", ".join(f"+{k + 1}" for k in ks))

    ih, iw = rows[0][0][0].shape[:2]
    NC = len(rows[0][2])
    # ONE GRID, BY HAND: gridspec with hspace=0 is the only way the Truth row touches the Predicted row.
    # Per pair: a label strip, the two image rows, then a full-width curve panel.
    fig = plt.figure(figsize=(14, 14 * (2 * len(rows) * ih * 1.62) / (NC * iw)))
    # FIVE rows per pair: label strip, the two images, the curve panel, and a SPACER -- with hspace=0
    # (which the flush image pair needs) the next pair's step labels otherwise land on this pair's ticks.
    gs = fig.add_gridspec(5 * len(rows), NC, hspace=0.0, wspace=0.0,
                          height_ratios=[0.34, 1.0, 1.0, 0.95, 0.30] * len(rows))
    for r, (pred, true, ks, h, cur) in enumerate(rows):
        for c in range(NC):
            lab = fig.add_subplot(gs[5 * r, c]); lab.axis("off")
            lab.text(0.5, 0.12, f"$+${ks[c] + 1}", ha="center", va="bottom", fontsize=13)
            for k, img in ((1, pred[c]), (2, true[c])):
                A = fig.add_subplot(gs[5 * r + k, c])
                A.imshow(img, interpolation="bilinear", aspect="auto")
                A.set_xticks([]); A.set_yticks([])
                for sp in A.spines.values():                 # the frame outline: black, and the same
                    sp.set_linewidth(2.0); sp.set_color("black")   # weight as the green one in Fig. 4
                if c == 0:                                   # 13pt made the two labels touch
                    A.set_ylabel("Predicted" if k == 1 else "Truth", fontsize=10.5, labelpad=2)
        # THE ERRORS OVER THE WHOLE ROLLOUT, under the pair they belong to and on the same x
        AC = fig.add_subplot(gs[5 * r + 3, :])
        x = np.arange(1, h + 1)
        for (nm, v), col in zip(cur.items(), ("tab:blue", "tab:green", "tab:red")):
            AC.plot(x, v, lw=1.3, color=col, label=nm)
        for k in ks:                                          # which steps the frames above came from
            AC.axvline(k + 1, color="0.75", lw=0.6, ls=":")
        AC.set_xlim(1, h); AC.set_ylim(0, 1.0)
        AC.set_ylabel("error", fontsize=12); AC.tick_params(labelsize=10)
        AC.grid(alpha=0.25)
        AC.legend(fontsize=10, ncol=3, loc="upper left", framealpha=0.85)
        # EACH PAIR HAS ITS OWN X AXIS -- the rollouts are different lengths, so they never shared one --
        # but the NAME of that axis is written once, under the arrow of time, because a label per panel
        # lands on the next pair's step numbers.
        AC.tick_params(labelbottom=True)
    # THE ARROW OF TIME, matched to the width of the sequences rather than the whole figure
    fig.canvas.draw()
    first = fig.axes[1].get_position(); last = None
    for A in fig.axes:                                        # the last image axes in the top row
        pos = A.get_position()
        if abs(pos.y0 - first.y0) < 1e-6:
            last = pos
    fig.subplots_adjust(bottom=0.052)
    x0, x1 = first.x0, (last or first).x1
    fig.patches.append(FancyArrowPatch((x0, 0.018), (x1, 0.018), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=22, lw=1.6, color="#333333"))
    fig.text(x1 + 0.006, 0.018, "$t$", ha="left", va="center", fontsize=15, color="#333333")
    fig.text(0.5 * (x0 + x1), 0.012, "Open-loop prediction step", ha="center", va="top", fontsize=12,
             color="#333333")
    f = os.path.join(paper, "figures", "longhorizon.png")
    fig.savefig(f, dpi=DPI_IMG, bbox_inches="tight"); plt.close(fig)
    print(f"  figures/longhorizon.png  {len(rows)} pairs, horizons "
          + ", ".join(str(r[3]) for r in rows))


def memory(paper):
    """The turn-warped memory curve: excess error over val at matched horizon, both splits."""
    j = json.load(open("logs/paper_icra_2027/memory_curves.json"))
    src = "logs/paper_icra_2027/eval_memory/_memory_paper.png"   # the clean two-row version
    dst = os.path.join(paper, "figures", "memory.png")
    if os.path.exists(src):
        import shutil
        shutil.copy(src, dst)
        print(f"  figures/memory.png  (turn-warped, from {src})")
    print("  memory numbers:", {k: {kk: round(vv, 4) for kk, vv in v.items() if isinstance(vv, float)}
                                for k, v in j.items()})




@torch.no_grad()
def _scene_photo(paper, name, shape=None):
    """The third-person shot out of the author's own composite figure (figures/leaf-blower.png etc).

    Only the TOP panel is wanted -- the scene with the disturbance in it -- and the panels in those files
    are separated by a black rule, so the cut is found rather than hard-coded: the first almost-black row
    below the halfway point of the top third."""
    f = os.path.join(paper, "figures", name)
    if not os.path.exists(f):
        return None
    im = cv2.imread(f)[:, :, ::-1]
    h = im.shape[0]
    dark = (im.max(axis=(1, 2)) < 40)
    rows = [r for r in range(int(h * 0.20), int(h * 0.60)) if dark[r]]
    cut = rows[0] if rows else int(h * 0.42)
    im = im[2:cut - 2, 2:-2]
    if shape is not None:
        # MATCH THE CAMERA FRAMES' ASPECT by cropping off the TOP, so the photo sits in the row at the
        # same size as the frames beside it instead of being taller than all of them.
        want = shape[0] / shape[1]
        hh, ww = im.shape[:2]
        keep = int(min(hh, ww * want))
        im = im[hh - keep:, :]
    return im


def ood(paper, dev="cuda"):
    """TWO SEPARATE FIGURES, one per anomaly kind, because they are not the same figure and the author
    reads them separately: ood-visual.png and ood-dynamical.png.

    Each carries the third-person scene (what was physically done to the drone, from the author's own
    composite figures), the evidence in the sensor the anomaly actually lives in, and the detector's
    channel with its conformal threshold. For the noodle that sensor is the camera, so the panel shows
    frames and the per-pixel surprise map; for the blower it is the IMU, because two camera frames of an
    invisible disturbance prove nothing.

    Every trace marks WHERE PREDICTION STARTS: the first P frames are context, so there is no prediction
    before then and the empty stretch at the left is not missing data."""
    from quickdraw.data.ood_windows import kept, window_steps
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis"))
    from detect_anomalies_wm import one_step
    from localise_ood_pixels import ensemble, maps_from, pink_mask
    cfg, core, norm, P, key = _model(dev)
    FS = 8.5

    def pick(split, need_out=4, min_width=3, prefer="central"):
        """The kept episode that reads best for a figure.

        `prefer="end"` wants the disturbance to ARRIVE and stay -- a long clean stretch first, then the
        window running to the end of the clip -- which is what the noodle needs, since every reviewed
        noodle window touches an edge anyway and starting inside it shows the anomaly before the baseline.
        `prefer="central"` wants a window with clean flight either side, which the blower has."""
        eps = load_split_episodes_mm(resolve_data_root(cfg), split, img_size=image_head_sizes(cfg),
                                     cam=image_head_cams(cfg), repo_id="starling-2")
        cands = []
        for ep in kept(split):
            o = eps[ep][0]
            w0, w1 = window_steps(split, ep, SUB)
            inside = [t for t in range(max(P, w0), min(w1, len(o)))]
            outside = [t for t in range(P, len(o)) if not (w0 <= t < w1)]
            n_bef = len([t for t in outside if t < w0])
            n_aft = len([t for t in outside if t >= w1])
            if not inside or len(outside) < need_out or (w1 - w0) < min_width:
                continue
            cands.append(dict(ep=ep, w0=w0, w1=w1, bef=n_bef, aft=n_aft, n=len(o),
                              ctr=abs(0.5 * (w0 + w1) / max(1, len(o)) - 0.5)))
        assert cands, f"no usable episode in {split}"
        if prefer == "end":
            c = max(cands, key=lambda c: (c["bef"], -c["aft"]))    # longest clean run BEFORE the window
        else:
            ok = [c for c in cands if min(c["bef"], c["aft"]) >= 3]
            c = min(ok, key=lambda c: c["ctr"]) if ok else max(cands, key=lambda c: min(c["bef"], c["aft"]))
        print(f"    {split}: ep{c['ep']} window {c['w0']}-{c['w1']} of {c['n']} | before {c['bef']} "
              f"after {c['aft']} | prefer={prefer}")
        return eps, c

    def mark_context(A):
        # a SOLID line named in the legend, not text on the plot: the first P frames are context, so
        # there is no prediction to the left of it
        A.axvline(P, color="tab:blue", ls="-", lw=1.2, label="Prediction starts")

    # ================= (a) VISUAL =====================================================================
    split, chan, chan_lab = "eval_ood_noodle", "latent_cos", "Latent surprise"
    eps, c = pick(split, prefer="end")
    ep, w0, w1 = c["ep"], c["w0"], c["w1"]
    o, a, fr = eps[ep]
    r = one_step(core, norm, o, a, fr[key], key, P, dev)
    inside = [t for t in range(max(P, w0), min(w1, len(o)))]
    # SEVERAL CANDIDATES, and the one whose surprise map is sharpest wins. Pink fraction alone picks the
    # frame with the MOST noodle in it, which is often the one where it fills the frame and the map has
    # nothing to contrast against; the ratio of the map's 99th percentile to its median says how much the
    # object stands out from the rest of the scene, which is what makes the panel legible.
    top_pink = [t for _, t in sorted(((pink_mask(fr[key][t]).mean(), t) for t in inside),
                                     reverse=True)[:5]]
    best_t, best_sur, best_q = None, None, -1.0
    for t in top_pink:
        ob = torch.from_numpy(fr[key][t]).float().div(255.0).to(dev)
        mm, _, _ = maps_from(ensemble(core, norm, o, a, fr[key], key, P, t, dev, n=32), ob)
        sm = mm["surprise_patch"].cpu().numpy()
        q = float(np.percentile(sm, 99) / max(1e-6, np.median(sm)))
        print(f"      candidate t={t}: surprise contrast {q:.2f}")
        if q > best_q:
            best_t, best_sur, best_q = t, sm, q
    t_in = best_t
    cand = [t for t in range(P, len(o)) if not (w0 <= t < w1)]
    t_out = max(cand, key=lambda t: abs(t - t_in)) if cand else P
    sur = best_sur
    photo = _scene_photo(paper, "pool-noodle.png", shape=fr[key][0].shape)
    fig = plt.figure(figsize=(7.2, 3.6))
    outer = fig.add_gridspec(2, 1, height_ratios=(1.05, 1.0), hspace=0.24)
    top = outer[0].subgridspec(1, 4 if photo is not None else 3, wspace=0.06)
    panes = [(photo, "How it was applied")] if photo is not None else []
    panes += [(fr[key][t_out], f"In distribution ($t{{=}}{t_out}$)"),
              (fr[key][t_in], f"Anomalous ($t{{=}}{t_in}$)"), (None, "Per-pixel surprise")]
    for jx, (img, lab) in enumerate(panes):
        A = fig.add_subplot(top[0, jx])
        if img is None:
            A.imshow(sur, cmap="inferno", vmin=np.percentile(sur, 50), vmax=np.percentile(sur, 99),
                     aspect="auto")
        else:
            A.imshow(img, aspect="auto")
        A.set_title(lab, fontsize=FS)
        A.set_xticks([]); A.set_yticks([])
        for sp_ in A.spines.values():                    # every image in the paper carries this border
            sp_.set_visible(True); sp_.set_linewidth(1.2); sp_.set_color("black")
    A = fig.add_subplot(outer[1])
    v = np.asarray(r[chan])
    A.plot(r["steps"], v, color="crimson", lw=1.4)
    A.axvspan(w0, w1, color="tab:purple", alpha=0.16, lw=0, label="Anomaly window")
    out = (r["steps"] < w0) | (r["steps"] >= w1)
    if out.any():
        A.axhline(float(np.quantile(v[out], 0.90)), color="k", ls=":", lw=1.1,
                  label="90% threshold")
    A.set_xlim(0, len(o) - 1)
    A.set_ylabel(chan_lab, fontsize=FS); A.set_xlabel("Prediction step", fontsize=FS)
    A.tick_params(labelsize=FS - 1.5); A.grid(alpha=0.25)
    mark_context(A)
    A.legend(fontsize=FS - 1.5, loc="upper right")
    f = os.path.join(paper, "figures", "ood-visual.png")
    fig.savefig(f, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print("  figures/ood-visual.png")

    # ================= (b) DYNAMICAL ==================================================================
    split, chan, chan_lab = "eval_ood_leafblower", "angvel_err", "$\\omega$ error"
    eps, c = pick(split, prefer="central")
    ep, w0, w1 = c["ep"], c["w0"], c["w1"]
    o, a, fr = eps[ep]
    r = one_step(core, norm, o, a, fr[key], key, P, dev)
    photo = _scene_photo(paper, "leaf-blower.png", shape=fr[key][0].shape)
    fig = plt.figure(figsize=(7.2, 3.3))
    gb = fig.add_gridspec(2, 2, width_ratios=(1.0, 1.7), wspace=0.20, hspace=0.14)
    if photo is not None:
        A = fig.add_subplot(gb[:, 0]); A.imshow(photo, aspect="auto")
        A.set_xticks([]); A.set_yticks([])
        for sp_ in A.spines.values():
            sp_.set_visible(True); sp_.set_linewidth(1.2); sp_.set_color("black")
        A.set_title("How it was applied", fontsize=FS)
    AV = fig.add_subplot(gb[0, 1])
    for ci, lab in zip(range(10, 13), ("$\\omega_x$", "$\\omega_y$", "$\\omega_z$")):
        AV.plot(np.arange(len(o)), o[:, ci], lw=1.1, label=lab)
    AV.axvspan(w0, w1, color="tab:purple", alpha=0.16, lw=0)
    AV.set_ylabel("Observed $\\omega$\n(rad/s)", fontsize=FS, labelpad=2)
    AV.legend(fontsize=FS - 2, ncol=3, loc="upper left", frameon=False)
    AV.tick_params(labelsize=FS - 1.5, labelbottom=False); AV.grid(alpha=0.25)
    AV.set_xlim(0, len(o) - 1)
    AE = fig.add_subplot(gb[1, 1], sharex=AV)
    v = np.asarray(r[chan])
    AE.plot(r["steps"], v, color="crimson", lw=1.4)
    AE.axvspan(w0, w1, color="tab:purple", alpha=0.16, lw=0, label="Anomaly window")
    out = (r["steps"] < w0) | (r["steps"] >= w1)
    if out.any():
        AE.axhline(float(np.quantile(v[out], 0.90)), color="k", ls=":", lw=1.1,
                   label="90% threshold")
    AE.set_ylabel(chan_lab, fontsize=FS); AE.set_xlabel("Prediction step", fontsize=FS)
    AE.tick_params(labelsize=FS - 1.5); AE.grid(alpha=0.25)
    mark_context(AE)
    AE.legend(fontsize=FS - 1.5, loc="upper right", ncol=1)
    f = os.path.join(paper, "figures", "ood-dynamical.png")
    fig.savefig(f, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print("  figures/ood-dynamical.png")


def curves(paper):
    """Train/validation curves for the three models, stacked, epochs below and wall-clock hours above."""
    # TAG NAMES DIFFER PER MODEL, so they are discovered rather than assumed: the world and action models
    # log `train|val/loss/total`, the reward model logs `train|val/loss/contrastive`. Guessing wrong here
    # produced an empty panel and a matplotlib legend warning rather than an error, which is exactly the
    # kind of silent hole a figure should not have.
    RUNS = [("logs/paper_icra_2027/model_backups/train_world_model_2026_09_11_03_17_47_s2_sub4_concat",
             "World Model", None),
            ("logs/paper_icra_2027/model_backups/train_action_2026_09_14_04_41_17_s2_ah_chunk32_full",
             "Action Model", None),
            # The reward model logs no epoch timing: it trains in 91 s wall clock (08:19:40 -> 08:21:11 in
            # its progress.log) over 300 epochs, so the per-epoch figure is passed in rather than averaged
            # from a series that does not exist.
            ("logs/paper_icra_2027/result_backups/train_reward_model_2026_09_14_08_19_38_reward_starling_v5",
             "Reward Model", 91.0 / 3600.0 / 300.0)]
    fig, ax = plt.subplots(3, 1, figsize=(3.4, 5.4))
    for i, (run, name, fallback_h) in enumerate(RUNS):
        p = os.path.join(run, "logs", "metrics.jsonl")
        rows = []
        if os.path.exists(p):
            for line in open(p):
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
        tags = {r["tag"] for r in rows if r.get("tag")}
        # THE REWARD MODEL IS SCORED ON ITS PROBE, NOT ITS LOSS. train_reward_model.py calls
        # probe/<factor>_acc "what steering actually queries, so it's the headline", and it is the only
        # reward series that tells a story: the contrastive val loss bottoms at 6.22 against ln(512)=6.24,
        # i.e. held-out retrieval is at chance, while the probe reaches 2.4-4.6x chance and train and val
        # track each other. Chance differs per factor (6, 10 and 9 buckets), so the mean of the three is
        # plotted against the mean of their chance levels, drawn.
        if any(t.startswith("val/probe/") for t in tags):
            facs = sorted({t.split("/")[-1][:-4] for t in tags if t.startswith("val/probe/")
                           and t.endswith("_acc")})
            NB = {"facing": 6, "object_in_view": 10, "motion_dominant": 9}
            chance = float(np.mean([1.0 / NB[f] for f in facs]))
            acc = {"train": {}, "val": {}}
            for r in rows:
                t = r.get("tag") or ""
                for sp in ("train", "val"):
                    if t.startswith(f"{sp}/probe/") and t.endswith("_acc"):
                        acc[sp].setdefault(r.get("step", 0), []).append(r["value"])
            A = ax[i]
            for k, c in (("train", "tab:blue"), ("val", "tab:red")):
                x = sorted(acc[k])
                A.plot(x, [float(np.mean(acc[k][v])) for v in x], color=c, lw=1.3, label=k)
            A.axhline(chance, color="k", ls=":", lw=0.9)
            A.text(0.98, chance, "chance", ha="right", va="bottom", fontsize=6, color="k",
                   transform=A.get_yaxis_transform())
            A.set_title(f"{name}  (steering probe)", fontsize=8)
            A.set_xlabel("epoch", fontsize=7); A.set_ylabel("probe accuracy", fontsize=7)
            A.tick_params(labelsize=6); A.grid(alpha=0.25); A.legend(fontsize=6)
            A.set_ylim(0.0, None)
            if fallback_h:
                span = fallback_h * max(max(acc["train"] or [0]), max(acc["val"] or [0]))
                mul, unit = (60.0, "min") if span < 0.2 else (1.0, "h")
                tw = A.twiny(); tw.set_xlim(*[x * fallback_h * mul for x in A.get_xlim()])
                tw.set_xlabel(f"wall clock ({unit})", fontsize=7); tw.tick_params(labelsize=6)
            continue
        pair = None
        for cand in ("loss/total", "loss/contrastive"):
            if f"train/{cand}" in tags and f"val/{cand}" in tags:
                pair = cand; break
            if f"val/{cand}" in tags:                       # reward model logs val only under this name
                pair = cand; break
        assert pair, f"no train/val loss pair found in {run}: {sorted(tags)[:6]}"
        cur = {"train": {}, "val": {}}
        hrs = []
        for r in rows:
            t = r.get("tag")
            if t == f"train/{pair}":
                cur["train"][r.get("step", 0)] = r["value"]
            elif t == f"val/{pair}":
                cur["val"][r.get("step", 0)] = r["value"]
            elif t == "time/avg_epoch_hours":
                hrs.append(r["value"])
        A = ax[i]
        for k, c in (("train", "tab:blue"), ("val", "tab:red")):
            if cur[k]:
                x = sorted(cur[k])
                A.plot(x, [cur[k][v] for v in x], color=c, lw=1.3, label=k)
        A.set_title(f"{name}  ({pair.split('/')[-1]})", fontsize=8)
        A.set_xlabel("epoch", fontsize=7); A.tick_params(labelsize=6); A.grid(alpha=0.25)
        if cur["train"] or cur["val"]:
            A.legend(fontsize=6)
        A.set_ylabel("loss", fontsize=7)
        h = float(np.mean(hrs)) if hrs else fallback_h
        if h:
            # UNIT PER PANEL. The world model took 45 h and the Reward Model 91 s; one axis in hours makes
            # the third panel read 0.000 to 0.007, which says nothing. Switch to minutes below 12 min.
            span = h * max(max(cur["train"] or [0]), max(cur["val"] or [0]))
            mul, unit = (60.0, "min") if span < 0.2 else (1.0, "h")
            tw = A.twiny(); tw.set_xlim(*[x * h * mul for x in A.get_xlim()])
            tw.set_xlabel(f"wall clock ({unit})", fontsize=7); tw.tick_params(labelsize=6)
    fig.tight_layout(h_pad=1.6)
    f = os.path.join(paper, "figures", "training-curves.png")
    fig.savefig(f, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print(f"  figures/training-curves.png")


def main(paper: str, *which: str) -> int:
    os.makedirs(os.path.join(paper, "figures"), exist_ok=True)
    todo = which or ("longhorizon", "memory", "ood", "curves")
    if "longhorizon" in todo:
        longhorizon(paper)
    if "memory" in todo:
        memory(paper)
    if "ood" in todo:
        ood(paper)
    if "curves" in todo:
        curves(paper)
    return 0


if __name__ == "__main__":
    sys.exit(main(*(sys.argv[1:] or ["/tmp/paper"])))
