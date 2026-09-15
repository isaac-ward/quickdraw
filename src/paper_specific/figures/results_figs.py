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
    fig = plt.figure(figsize=(14, 14 * (2 * len(rows) * ih * 1.50) / (NC * iw)))
    # FOUR rows per pair: the two images, the curve panel, and a SPACER. The step-label strip is GONE --
    # the tag sits INSIDE its own frame now, which is the design the author prefers and which buys the
    # strip's height back.
    gs = fig.add_gridspec(4 * len(rows), NC, hspace=0.0, wspace=0.0,
                          height_ratios=[1.0, 1.0, 0.95, 0.34] * len(rows))
    for r, (pred, true, ks, h, cur) in enumerate(rows):
        for c in range(NC):
            for k, img in ((0, pred[c]), (1, true[c])):
                A = fig.add_subplot(gs[4 * r + k, c])
                A.imshow(img, interpolation="bilinear", aspect="auto")
                A.set_xticks([]); A.set_yticks([])
                for sp in A.spines.values():                 # the frame outline: black, and the same
                    sp.set_linewidth(1.0); sp.set_color("black")   # half weight, at the author's ask
                if c == 0:                                   # 13pt made the two labels touch
                    A.set_ylabel("Predicted" if k == 0 else "Truth", fontsize=10.5, labelpad=2)
                if k == 0:
                    A.text(0.022, 0.96, f"$+${ks[c] + 1}", transform=A.transAxes, ha="left", va="top",
                           fontsize=11, color="black",
                           bbox=dict(boxstyle="square,pad=0.14", fc="white", ec="none", alpha=0.72))
        # THE ERRORS OVER THE WHOLE ROLLOUT, under the pair they belong to and on the same x
        AC = fig.add_subplot(gs[4 * r + 2, :])
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
        # CROP THE TOP to the frames' own aspect, so the photo renders at exactly their height without
        # any image being rescaled or stretched -- the ceiling is what gets cut, and it carries nothing.
        hh, ww = im.shape[:2]
        keep = int(min(hh, ww * shape[0] / shape[1]))
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
        # A DASHED LINE plus the hatched span it bounds, both named in the legend rather than written on
        # the plot. Everything left of the line is context, so there is no prediction there at all --
        # hatching it says that, where an unmarked white gap read as a flat score.
        A.axvline(P, color="black", ls="--", lw=1.2, label="Prediction begins")
        # SPARSE AND PALE. At "///" in mid grey the hatch was the loudest thing in the panel; it is
        # background, so it reads as background. hatch.linewidth is an rcParam, not a patch property.
        plt.rcParams["hatch.linewidth"] = 0.6
        A.axvspan(0, P, facecolor="none", edgecolor="0.72", hatch="////", lw=0.0,
                  label="Context frames")

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
    # A SECOND ANOMALOUS FRAME, later than the first: the pair shows the noodle moving through the scene
    # and the map following it. PINNED by the author (t=27) rather than picked by contrast -- the
    # contrast rule kept landing on t=23, where the noodle reads less clearly to the eye.
    T_LATE = 26
    later = [t for t in inside if t > t_in and (T_LATE is None or t == T_LATE)]
    assert later, f"pinned T_LATE={T_LATE} is not an anomaly-window step after t_in={t_in}"
    t_late, sur_late, q_late = t_in, best_sur, -1.0
    for t in later:
        ob = torch.from_numpy(fr[key][t]).float().div(255.0).to(dev)
        mm, _, _ = maps_from(ensemble(core, norm, o, a, fr[key], key, P, t, dev, n=32), ob)
        sm = mm["surprise_patch"].cpu().numpy()
        q = float(np.percentile(sm, 99) / max(1e-6, np.median(sm)))
        print(f"      later candidate t={t}: contrast {q:.2f}")
        if q > q_late:
            t_late, sur_late, q_late = t, sm, q
    # ...and the CLEAN frame's map too, which should be near-empty -- that is the control for the pair
    ob = torch.from_numpy(fr[key][t_out]).float().div(255.0).to(dev)
    mm, _, _ = maps_from(ensemble(core, norm, o, a, fr[key], key, P, t_out, dev, n=32), ob)
    sur_out = mm["surprise_patch"].cpu().numpy()
    print(f"      second frame t={t_late} (contrast {q_late:.2f})")
    # ONE BRIGHTNESS SCALE FOR ALL THREE MAPS. Per-panel percentiles made the control LIE: an
    # in-distribution map whose values are uniformly low got stretched to its own 50th-99th range, so its
    # noise floor rendered as bright as a real detection. The scale is now shared, taken from the
    # anomalous maps, and the control is plotted on it -- if it looks dark, it IS dark.
    # FLOOR FROM THE CLEAN FRAME, CEILING FROM THE ANOMALY. The surprise field is never zero anywhere --
    # the decoder ensemble disagrees about ordinary texture too (clean median 3.5 vs anomalous 4.0-5.1) --
    # so what separates an anomaly is the PEAK, not the floor (clean p99 11.5 vs 57.1). Anchoring vmin on
    # the clean map's own median puts that floor at black by construction and leaves the peak to carry
    # the signal, which is the claim the figure is actually making.
    SUR_LO = float(np.median(sur_out))
    SUR_HI = float(np.percentile(np.concatenate([sur.ravel(), sur_late.ravel()]), 99))
    for nm, m in (("in-distribution", sur_out), (f"anomalous t={t_in}", sur),
                  (f"anomalous t={t_late}", sur_late)):
        print(f"      {nm:22s} median {np.median(m):.3f}  p99 {np.percentile(m, 99):.3f}  "
              f"frac above shared vmin {float((m > SUR_LO).mean()):.2f}")
    photo = _scene_photo(paper, "pool-noodle.png", shape=fr[key][0].shape)
    ih, iw = fr[key][0].shape[:2]                     # 112x192; the patch map is 7x12, the SAME aspect
    # ONE GRID BY HAND. The images must line up with the TRACE -- leftmost left edge and rightmost right
    # edge flush with its axes box -- and a gridspec cannot do that, because the trace's box is inset by
    # its own tick labels and ylabel while the image grid is not. So every panel is placed with add_axes
    # in figure coordinates: the 2x4 block of images is CONTIGUOUS (no gaps, as in the dynamical figure)
    # and spans exactly [L, 1-R], and the trace below spans the same. The figure height is then SOLVED
    # from the image width so that equal aspect fills each box exactly -- never aspect="auto".
    FIG_W, L, R = 3.4, 0.175, 0.008      # L must clear the trace's ylabel AND its tick labels
    W = 1.0 - L - R
    cw_in = W * FIG_W / 4.0                           # one image column, in inches
    ch_in = cw_in * ih / iw
    # THE LEGEND COMES OUT OF THE AXES. Four entries will not fit beside the data at 3.4in -- inside
    # the panel it covered the rise that is the whole point of the figure -- so it sits under the trace
    # as a two-column strip and the plot area is left clear.
    TOP_IN, PAD_IN, TR_IN, BOT_IN, LEG_IN = 0.34, 0.0, 0.78, 0.34, 0.30
    FIG_H = TOP_IN + 2 * ch_in + PAD_IN + TR_IN + BOT_IN + LEG_IN
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    cw, ch = W / 4.0, ch_in / FIG_H
    y0 = 1.0 - (TOP_IN + ch_in) / FIG_H               # top image row
    y1 = y0 - ch                                      # surprise row, touching it
    # Row one: how it was applied, a clean frame, the anomalous frame, and a LATER anomalous frame.
    # Row two: the per-pixel surprise under each, with the first cell blank -- the map belongs under the
    # frame it explains, and the in-distribution map is the control that shows it stays dark.
    # THE STEP OFFSET COMES INSIDE, in +X form, and NOTHING ELSE: the descriptive label stays a title,
    # at the author's ask. `None` for the scene photo, which is not a step of the rollout.
    panes = [(0, 0, photo, "Disturbance\nis applied", None),
             (0, 1, fr[key][t_out], "In distribution", t_out),
             (0, 2, fr[key][t_in], "Anomalous", t_in),
             (0, 3, fr[key][t_late], "Anomalous", t_late),
             (1, 1, None, None, None), (1, 2, None, None, None), (1, 3, None, None, None)]
    for r_, c_, img, lab, step in panes:
        A = fig.add_axes([L + c_ * cw, y0 if r_ == 0 else y1, cw, ch])
        if img is None:
            m = {1: sur_out, 2: sur, 3: sur_late}[c_]
            A.imshow(m, cmap="inferno", vmin=SUR_LO, vmax=SUR_HI)
        else:
            A.imshow(img)
        if lab:
            A.set_title(lab, fontsize=FS - 2.2, pad=1.8, linespacing=1.15)
        if step is not None:
            A.text(0.03, 0.95, f"$+${step}", transform=A.transAxes, ha="left", va="top",
                   fontsize=FS - 2.4, color="black",
                   bbox=dict(boxstyle="square,pad=0.16", fc="white", ec="none", alpha=0.74))
        A.set_xticks([]); A.set_yticks([])
        for sp_ in A.spines.values():
            sp_.set_visible(True); sp_.set_linewidth(0.6); sp_.set_color("black")
    A = fig.add_axes([L, (BOT_IN + LEG_IN) / FIG_H, W, TR_IN / FIG_H])   # the trace, on the images' span
    v = np.asarray(r[chan])
    A.plot(r["steps"], v, color="tab:purple", lw=1.6, label="OOD score")
    A.axvspan(w0, w1, color="#c62828", alpha=0.20, lw=0, label="Anomaly frames")
    A.set_xlim(0, len(o) - 1)
    A.set_ylabel(chan_lab, fontsize=FS); A.set_xlabel("Prediction step", fontsize=FS)
    A.tick_params(labelsize=FS - 1.5); A.grid(alpha=0.25)
    mark_context(A)
    _h, _l = A.get_legend_handles_labels()
    fig.legend(_h, _l, loc="lower center", bbox_to_anchor=(L + W / 2, 0.0), ncol=2,
               fontsize=FS - 2.2, frameon=False, handlelength=1.4, columnspacing=1.2,
               borderaxespad=0.0)
    f = os.path.join(paper, "figures", "ood-visual.png")
    fig.savefig(f, dpi=DPI); plt.close(fig)            # NO bbox_inches: tight would re-trim the margins
    print("  figures/ood-visual.png")

    # ================= (b) DYNAMICAL =================================================================
    # A 2x2: what the drone SAW while it was pushed, the angular velocity it recorded, how the push was
    # applied, and what the detector made of it. The two plots share an x axis; the two images are
    # cropped to a common shape and never rescaled anisotropically.
    split, chan, chan_lab = "eval_ood_leafblower", "angvel_err", "$\\omega$ error"
    eps, c = pick(split, prefer="central")
    ep, w0, w1 = c["ep"], c["w0"], c["w1"]
    o, a, fr = eps[ep]
    r = one_step(core, norm, o, a, fr[key], key, P, dev)
    print(f"    (b) {split} ep{ep}: window {w0}-{w1} of {len(o)} steps")
    v = np.asarray(r[chan])
    t_peak = int(r["steps"][int(np.argmax(np.where((r["steps"] >= w0) & (r["steps"] < w1), v, -1)))])
    pov = fr[key][t_peak]
    photo = _scene_photo(paper, "leaf-blower.png", shape=pov.shape)
    ph, pw = pov.shape[:2]                             # 112x192; the photo is cropped to the same aspect
    # HAND-PLACED, for the same reason the visual figure is: in a gridspec the image column's cell is
    # wider than the image's own aspect, so imshow (adjustable="box") shrinks the axes and centres it,
    # leaving a gap between the plots and the images that no wspace can close. Here the image block is
    # butted straight against the plot block, and the figure height is SOLVED so the two plots together
    # are exactly as tall as the two images -- no frame is ever rescaled anisotropically.
    FIG_W, L, GAP, R = 3.4, 0.56, 0.0, 0.01            # inches: ylabel+ticks, plot-to-image gap, margin
    WI = 1.06                                          # image width; the plots take whatever is left
    # TOP_IN is not zero even with the captions inside: the rotated two-line "Observed omega (rad/s)" is
    # taller than its own panel, so it overflows both ends of it and the top end needs somewhere to go.
    TOP_IN, BOT_IN, LEG_IN = 0.30, 0.34, 0.30          # 2-line image title, xlabel+ticks, legend
    hi = WI * ph / pw                                  # one image, and therefore one plot, in inches
    PW = FIG_W - L - GAP - WI - R
    FIG_H = TOP_IN + 2 * hi + BOT_IN + LEG_IN
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    hr, y_top = hi / FIG_H, 1.0 - TOP_IN / FIG_H       # ...so the block's bottom clears the strip
    xp, wp = L / FIG_W, PW / FIG_W
    xi, wi_ = (L + PW + GAP) / FIG_W, WI / FIG_W
    # THE TWO PLOTS SHARE AN X AXIS, so they are joined with no gap and only the lower one is labelled;
    # the two images are joined the same way. The POV caption is gone -- the figure's own caption says it.
    AI = fig.add_axes([xi, y_top - hr, wi_, hr]); AI.imshow(photo)
    AI.set_xticks([]); AI.set_yticks([])
    for sp_ in AI.spines.values():
        sp_.set_visible(True); sp_.set_linewidth(0.6); sp_.set_color("black")
    AI.set_title("Disturbance\nis applied", fontsize=FS - 2.2, pad=1.8, linespacing=1.15)
    A = fig.add_axes([xi, y_top - 2 * hr, wi_, hr]); A.imshow(pov)
    # THE LABEL BELONGS UNDER THIS ONE: the frame is the whole argument for the dynamical case, and the
    # argument is that there is nothing in it to see. It sits in the band the legend strip occupies on
    # the plot side, which is free out here.
    # the caption stays OUTSIDE, under the frame; only the step offset comes inside
    A.annotate("Disturbance is visually\nundetectable", xy=(0.5, 0.0), xycoords="axes fraction",
               ha="center", va="top", fontsize=FS - 2.2, linespacing=1.15,
               xytext=(0, -3), textcoords="offset points")
    A.text(0.03, 0.95, f"$+${t_peak}", transform=A.transAxes, ha="left", va="top",
           fontsize=FS - 2.4, color="black",
           bbox=dict(boxstyle="square,pad=0.16", fc="white", ec="none", alpha=0.74))
    A.set_xticks([]); A.set_yticks([])
    for sp_ in A.spines.values():
        sp_.set_visible(True); sp_.set_linewidth(0.6); sp_.set_color("black")
    AV = fig.add_axes([xp, y_top - hr, wp, hr])
    for ci, lab in zip(range(10, 13), ("$\\omega_x$", "$\\omega_y$", "$\\omega_z$")):
        AV.plot(np.arange(len(o)), o[:, ci], lw=1.1, label=lab)
    AV.axvspan(w0, w1, color="#c62828", alpha=0.20, lw=0)
    AV.set_ylabel("Observed $\\omega$\n(rad/s)", fontsize=FS, labelpad=2)
    AV.legend(fontsize=FS - 2, ncol=1, loc="lower right", frameon=False, handlelength=1.1,
              labelspacing=0.25)
    AV.tick_params(labelsize=FS - 1.5, labelbottom=False); AV.grid(alpha=0.25)
    AV.set_xlim(0, len(o) - 1)
    AE = fig.add_axes([xp, y_top - 2 * hr, wp, hr], sharex=AV)
    AE.set_ylim(top=1.0)                               # a round ceiling, at the author's ask
    AE.set_yticks([0.0, 0.5])                          # ...left unlabelled: the two panels share that
    #                                                    boundary, so a 1.0 here landed on the -1 above
    AE.plot(r["steps"], v, color="tab:purple", lw=1.6, label="OOD score")
    AE.axvspan(w0, w1, color="#c62828", alpha=0.20, lw=0, label="Anomaly frames")
    AE.set_ylabel(chan_lab, fontsize=FS); AE.set_xlabel("Prediction step", fontsize=FS)
    AE.tick_params(labelsize=FS - 1.5); AE.grid(alpha=0.25)
    mark_context(AE)
    # OUT OF THE AXES, same reason as the visual panel: in here the legend sat squarely on the peak.
    _h, _l = AE.get_legend_handles_labels()
    fig.legend(_h, _l, loc="lower center", bbox_to_anchor=(xp + wp / 2, 0.0), ncol=2,
               fontsize=FS - 2.2, frameon=False, handlelength=1.4, columnspacing=1.2,
               borderaxespad=0.0)
    f = os.path.join(paper, "figures", "ood-dynamical.png")
    fig.savefig(f, dpi=DPI); plt.close(fig)             # NO bbox_inches: tight would re-trim the margins
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
            # THE DENSE RE-RUN, report_every=1. The deployed head logged every 10th epoch, so inside the
            # first twenty the curve had three points and read as two straight segments. Same config,
            # same seed; it lands at val 6.211 @ ep 8 against the deployed head's 6.233 @ ep 10, because
            # evaluating every epoch draws from the same CUDA generator the batching does and the
            # trajectories diverge. The head that SHIPS is still the original -- the steering evals ran
            # against it -- this run exists to draw the curve.
            # It logs no epoch timing either: 93 s wall clock (20:17:15 -> 20:18:48) over 89 epochs.
            ("logs/train_reward_model_2026_09_15_20_17_13_rw_dense",
             "Reward Model", 93.0 / 3600.0 / 89.0)]
    # ONE ROW, NOT THREE. Stacked, three panels with their own x axis and a twin wall-clock axis
    # each cost 5.4 inches of page; side by side they cost 1.9, and nothing about the curves needs the
    # extra width -- they are each a single decaying line.
    CF = 1.5                                         # every font in this figure, at the author's ask
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.55))
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
        # THE REWARD HEAD IS PLOTTED ON ITS LOSS, like the other two. The probe was the honest
        # headline but it is not a training curve: it peaks and drifts, which reads as a fault. The
        # contrastive loss shows the thing that actually happened -- train falls and val does not
        # follow -- and the wall clock is in SECONDS, because 91 s is the number worth seeing.
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
        A.set_title(name, fontsize=8 * CF)
        A.set_xlabel("Epoch", fontsize=7 * CF); A.tick_params(labelsize=6 * CF); A.grid(alpha=0.25)
        # ZOOMED PAST THE FLAT TAIL, per panel. Each model's curve is over well before its last epoch,
        # and the wall-clock axis follows because it is derived from this limit.
        XLIM = {"World Model": 30, "Action Model": 50, "Reward Model": 20}
        if name in XLIM:
            A.set_xlim(0, XLIM[name])
        if cur["train"] or cur["val"]:
            A.legend(fontsize=6 * CF, loc="lower left" if name == "Reward Model" else "best")
        if i == 0:
            A.set_ylabel("Loss", fontsize=7 * CF)
        h = float(np.mean(hrs)) if hrs else fallback_h
        if h:
            # UNIT PER PANEL. The world model took 45 h and the Reward Model 91 s; one axis in hours makes
            # the third panel read 0.000 to 0.007, which says nothing. Switch to minutes below 12 min.
            span = h * max(max(cur["train"] or [0]), max(cur["val"] or [0]))
            mul, unit = (3600.0, "s") if span < 0.05 else (60.0, "min") if span < 0.2 else (1.0, "h")
            tw = A.twiny(); tw.set_xlim(*[x * h * mul for x in A.get_xlim()])
            tw.set_xlabel(f"Wall clock ({unit})", fontsize=7 * CF); tw.tick_params(labelsize=6 * CF)
    fig.tight_layout(w_pad=0.5, pad=0.4)
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
