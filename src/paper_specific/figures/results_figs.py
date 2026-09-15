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
import torch

DPI = 450          # figures whose content is text and line art: export high so print stays crisp
DPI_IMG = 300      # figures that are mostly decoded 112x192 frames -- past this, DPI only upscales blur
#                    and the file grows without carrying more information
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis"))

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
def longhorizon(paper, dev="cuda", n_traj=4, H=128, n_show=8):
    """Open-loop rollout on RECORDED actions: prediction above, ground truth below, time left to right.

    Recorded actions, not planned ones: this figure is about the dynamics, so the action sequence must be
    one the drone actually flew. Four different val episodes, each a row-pair."""
    cfg, core, norm, P, key = _model(dev)
    eps = load_split_episodes_mm(resolve_data_root(cfg), "val", img_size=image_head_sizes(cfg),
                                 cam=image_head_cams(cfg), repo_id="starling-2")
    rows = []
    for ei in range(min(n_traj, len(eps))):
        o, a, fr = eps[ei]
        t0 = 40 + ei * 60
        h = min(H, len(o) - t0 - P - 1)
        ctx = {"proprio": norm.norm_obs(torch.from_numpy(o[t0:t0 + P])).float()[None].to(dev),
               key: torch.from_numpy(fr[key][t0:t0 + P]).float().div(255.0)[None].to(dev)}
        acts = norm.norm_act(torch.from_numpy(a[t0:t0 + P - 1 + h])).float()[None].to(dev)
        pr = core.imagine_eval(ctx, acts, h, heads=[key], norm=norm)[key][0].clamp(0, 1)
        gt = fr[key][t0 + P:t0 + P + h]
        ks = np.unique(np.linspace(0, h - 1, n_show).round().astype(int))
        pred = np.concatenate([(pr[k].cpu().numpy() * 255).astype(np.uint8) for k in ks], axis=1)
        true = np.concatenate([gt[k] for k in ks], axis=1)
        rows.append((pred, true, ks, h))
    fig, ax = plt.subplots(2 * len(rows), 1, figsize=(14, 1.35 * 2 * len(rows)))
    for i, (pred, true, ks, h) in enumerate(rows):
        for j, (img, lab) in enumerate(((pred, "predicted"), (true, "ground truth"))):
            A = ax[2 * i + j]
            A.imshow(img); A.axis("off")
            A.text(-0.008, 0.5, lab, transform=A.transAxes, ha="right", va="center", fontsize=7,
                   rotation=90)
        ax[2 * i].set_title(f"episode {i}   open-loop steps " + ", ".join(f"$+${k + 1}" for k in ks),
                            fontsize=7, loc="left")
    fig.tight_layout(h_pad=0.15)
    f = os.path.join(paper, "figures", "longhorizon.png")
    fig.savefig(f, dpi=DPI_IMG, bbox_inches="tight"); plt.close(fig)
    print(f"  figures/longhorizon.png  {len(rows)} episodes x {H} steps")


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
def ood(paper, dev="cuda"):
    """Two blocks, one per anomaly kind: frames plus the per-pixel surprise map, over the trace of the
    channel that detects it with the conformal threshold drawn.

    The channels differ ON PURPOSE and that is the figure's point -- the noodle is invisible to proprio and
    the blower is invisible in the image -- so each block shows the channel carrying its own anomaly.

    FRAMES ARE CHOSEN BY PEAK, not by window midpoint. The midpoint of a reviewed window is often a moment
    when the object is edge-on or half out of shot, which made the first version of this figure show two
    near-identical frames and prove nothing. The anomalous frame is the one where the ground-truth evidence
    is strongest (pink fraction for the noodle, the detection channel for the blower); the in-distribution
    frame is taken as far from the window as the clip allows."""
    from quickdraw.data.ood_windows import kept, window_steps
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis"))
    from detect_anomalies_wm import one_step
    from localise_ood_pixels import ensemble, maps_from, pink_mask
    cfg, core, norm, P, key = _model(dev)
    BLOCK = [("eval_ood_noodle", "latent_cos", "latent surprise",
              "Visual anomaly: a pool noodle enters frame"),
             ("eval_ood_leafblower", "angvel_err", "angular velocity error",
              "Dynamical anomaly: an off-camera leaf blower pushes the drone")]
    fig = plt.figure(figsize=(7.2, 6.9))
    outer = fig.add_gridspec(2, 1, hspace=0.30)
    for bi, (split, chan, chan_lab, title) in enumerate(BLOCK):
        eps = load_split_episodes_mm(resolve_data_root(cfg), split, img_size=image_head_sizes(cfg),
                                     cam=image_head_cams(cfg), repo_id="starling-2")
        # pick the episode/step with the strongest ground-truth evidence inside its window
        best = None
        for ep in kept(split):
            o, a, fr = eps[ep]
            w0, w1 = window_steps(split, ep, SUB)
            inside = [t for t in range(max(P, w0), min(w1, len(o)))]
            outside = [t for t in range(P, len(o)) if not (w0 <= t < w1)]
            # BOTH CLASSES REQUIRED: an episode whose window covers every predicted step has no
            # in-distribution baseline, so the panel would show shading everywhere and no threshold --
            # which is exactly what the first version of this figure did.
            if not inside or len(outside) < 4:
                continue
            if split.endswith("noodle"):
                sc = [(pink_mask(fr[key][t]).mean(), t) for t in inside]
            else:
                r0 = one_step(core, norm, o, a, fr[key], key, P, dev)
                m = {int(t): float(v) for t, v in zip(r0["steps"], np.asarray(r0[chan]))}
                sc = [(m.get(t, 0.0), t) for t in inside]
            v, t = max(sc)
            if best is None or v > best[0]:
                best = (v, ep, t, w0, w1)
        _, ep, t_in, w0, w1 = best
        o, a, fr = eps[ep]
        r = one_step(core, norm, o, a, fr[key], key, P, dev)
        cand = [t for t in range(P, len(o)) if not (w0 <= t < w1)]
        t_out = max(cand, key=lambda t: abs(t - t_in)) if cand else P
        obs = torch.from_numpy(fr[key][t_in]).float().div(255.0).to(dev)
        mp, _, _ = maps_from(ensemble(core, norm, o, a, fr[key], key, P, t_in, dev, n=32), obs)
        sur = mp["surprise_patch"].cpu().numpy()

        gs = outer[bi].subgridspec(2, 3, height_ratios=[1.35, 1.0], hspace=0.30, wspace=0.05)
        for jx, (img, lab) in enumerate((
                (fr[key][t_out], f"in distribution ($t{{=}}{t_out}$)"),
                (fr[key][t_in], f"anomalous ($t{{=}}{t_in}$)"), (None, "per-pixel surprise"))):
            A = fig.add_subplot(gs[0, jx])
            if img is None:
                A.imshow(sur, cmap="inferno", vmin=np.percentile(sur, 50), vmax=np.percentile(sur, 99))
            else:
                A.imshow(img)
            A.set_title(lab, fontsize=7); A.axis("off")
        fig.add_subplot(gs[0, :]).set_axis_off()
        fig.text(0.02, {0: 0.955, 1: 0.475}[bi], f"({'ab'[bi]}) {title}   [episode {ep}]",
                 fontsize=8.5, weight="bold", ha="left")
        A = fig.add_subplot(gs[1, :])
        v = np.asarray(r[chan])
        A.plot(r["steps"], v, color="crimson", lw=1.4)
        A.axvspan(w0, w1, color="tab:green", alpha=0.16, lw=0, label="reviewed anomaly window")
        out = (r["steps"] < w0) | (r["steps"] >= w1)
        if out.any():
            A.axhline(float(np.quantile(v[out], 0.90)), color="k", ls=":", lw=1.1,
                      label="90% conformal threshold")
        A.set_ylabel(chan_lab, fontsize=7.5); A.set_xlabel("model step", fontsize=7.5)
        A.tick_params(labelsize=6.5); A.grid(alpha=0.25); A.legend(fontsize=6.5, loc="best")
    f = os.path.join(paper, "figures", "ood-detection.png")
    fig.savefig(f, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print("  figures/ood-detection.png")


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
