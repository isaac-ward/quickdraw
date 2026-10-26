"""Three-way OOD attribution for the Grapevine cave, from the two world models' own one-step surprise.

Replicates the method validated in wizard/records/starling.md §4 and src/quickdraw/sim/mbari, with the
cave's TWO-MODEL split -- which is the whole reason the user asked for a vision model AND a dynamics-only
one. The DYNAMICS WM (cave_proprio) scores the proprio channels; the VISION WM (cave_fpv) scores the
image channels. The anomaly is attributed by WHICH model is surprised:

    ood_thruster    erratic plant command, frames normal     -> proprio channels fire, image quiet
    ood_person      red figure in frame, dynamics normal      -> image channels fire, proprio quiet
    ood_appearance  white-balance + haze, dynamics normal     -> image channels fire, proprio quiet

(the OOD semantics are exactly what prospector/cave_wm/record.py injects.)

NOMINAL REFERENCE is the held-out `val` split. The cave ships an explicit val split, so unlike mbari we
do not reproduce the packer's seed draw -- val IS the held-out nominal set, and calibrating OOD
thresholds on it (not on train, which the models were fit to) is what keeps every threshold honest.

SCORE: one-step Violation-of-Expectation per channel, episode score = 90th percentile over steps (an
intruder is in frame for only part of an episode; a mean would dilute it). CALIBRATION, DETECTION and
the nearest-centroid 3-way confusion are reused verbatim from sim/mbari/uncertainty.py -- that math is
generic and already validated; only the data reader and the two-model routing are new here.

obs layout (record.py): 0:3 pos | 3:7 quat wxyz | 7:10 vel world | 10:13 body rates -- same slices as mbari.
"""
from __future__ import annotations

import pathlib

import numpy as np
import torch

POS, QUAT, VEL, ANGVEL = slice(0, 3), slice(3, 7), slice(7, 10), slice(10, 13)

IMAGE_CHANNELS = ("image_lpips", "image_l1")
DYN_CHANNELS = ("proprio_pos", "proprio_vel", "proprio_angvel", "proprio_rot_deg")
CHANNELS = IMAGE_CHANNELS + DYN_CHANNELS
PCTL = 90.0


# --------------------------------------------------------------------------- one-step surprise
@torch.no_grad()
def one_step_dyn(core, norm, o, a, P, dev, chunk=256):
    """Teacher-forced one-step proprio prediction at every step -> per-step dynamics channel errors.

    Proprio-only: the DYNAMICS WM has no image head, so the context carries proprio alone."""
    T = len(o)
    ts = list(range(P, T))
    pred = []
    for i0 in range(0, len(ts), chunk):
        part = ts[i0:i0 + chunk]
        ctx = {"proprio": torch.stack([norm.norm_obs(torch.from_numpy(o[t - P:t])) for t in part]).float().to(dev)}
        acts = torch.stack([norm.norm_act(torch.from_numpy(a[t - P:t])) for t in part]).float().to(dev)
        out = core.imagine_eval(ctx, acts, 1, heads=["proprio"], norm=norm)
        pred.append(norm.denorm_obs(out["proprio"][:, 0].cpu()).numpy())
    pp = np.concatenate(pred)
    pt = np.stack([o[t] for t in ts])
    qp = pp[:, QUAT] / (np.linalg.norm(pp[:, QUAT], axis=1, keepdims=True) + 1e-9)
    qt = pt[:, QUAT] / (np.linalg.norm(pt[:, QUAT], axis=1, keepdims=True) + 1e-9)
    return {"proprio_pos": np.linalg.norm(pp[:, POS] - pt[:, POS], axis=1),
            "proprio_vel": np.linalg.norm(pp[:, VEL] - pt[:, VEL], axis=1),
            "proprio_angvel": np.linalg.norm(pp[:, ANGVEL] - pt[:, ANGVEL], axis=1),
            "proprio_rot_deg": np.degrees(2.0 * np.arccos(np.clip(np.abs((qp * qt).sum(1)), 0, 1)))}


@torch.no_grad()
def one_step_image(core, norm, o, a, fr, key, P, dev, chunk=48):
    """Teacher-forced one-step image prediction at every step -> per-step image channel errors.

    The VISION WM decodes `key`; the proprio head rides along in the context (it was trained jointly)."""
    from ...evaluation.openloop import image_curves

    T = len(o)
    ts = list(range(P, T))
    lp, l1 = [], []
    for i0 in range(0, len(ts), chunk):
        part = ts[i0:i0 + chunk]
        ctx = {"proprio": torch.stack([norm.norm_obs(torch.from_numpy(o[t - P:t])) for t in part]).float().to(dev),
               key: torch.stack([torch.from_numpy(fr[t - P:t]) for t in part]).float().div(255.0).to(dev)}
        acts = torch.stack([norm.norm_act(torch.from_numpy(a[t - P:t])) for t in part]).float().to(dev)
        out = core.imagine_eval(ctx, acts, 1, heads=[key], norm=norm)
        pred = out[key][:, 0].clamp(0, 1)
        true = torch.stack([torch.from_numpy(fr[t]) for t in part]).float().div(255.0).to(dev)
        for i in range(len(part)):
            c = image_curves(pred[i:i + 1].unsqueeze(1), true[i:i + 1].unsqueeze(1))
            lp.append(float(c["lpips"][0]))
            l1.append(float(c["l1"][0]))
    return {"image_lpips": np.array(lp), "image_l1": np.array(l1)}


# --------------------------------------------------------------------------- per-split scoring
def _episode_score(per_step):
    return {k: float(np.percentile(np.asarray(v, np.float64), PCTL)) for k, v in per_step.items()}


def score_split(episodes, core_dyn, norm_dyn, core_vis, norm_vis, key, P, dev, log=print):
    """Per-episode channel scores for one split, routing each channel to its model."""
    rows = []
    for n, (o, a, frames) in enumerate(episodes):
        fr = frames[key]
        e = {**one_step_dyn(core_dyn, norm_dyn, o, a, P, dev),
             **one_step_image(core_vis, norm_vis, o, a, fr, key, P, dev)}
        rows.append(_episode_score(e))
        if (n + 1) % 5 == 0:
            log(f"      {n + 1}/{len(episodes)} episodes")
    return {c: np.array([r[c] for r in rows]) for c in CHANNELS}


# --------------------------------------------------------------------------- orchestration
def _load(ckpt, dev):
    from ...evaluation.standalone import load_checkpoint
    from ...training.setup import build_model, normalizer
    from ..mbari.run_uncertainty import _cfg_for

    cfg = _cfg_for(ckpt)
    core = build_model(cfg).to(dev)
    load_checkpoint(core, ckpt)
    core.eval()
    return core, normalizer(cfg), cfg


def run(fpv_ckpt, proprio_ckpt, root, out="/app/logs/cave/ood",
        splits=("ood_thruster", "ood_person", "ood_appearance"), nominal="val",
        key="fpv", repo_id="cave-grapevine", img_hw=(144, 192), limit=None, dev="cuda", log=print):
    """Score nominal + every OOD split with the two models, calibrate on nominal, classify, plot."""
    import json

    from ...data.dataset import DataConfig, load_split_episodes_mm
    from ..mbari import uncertainty as U

    log(f"  vision WM   {fpv_ckpt}")
    core_vis, norm_vis, cfg_vis = _load(fpv_ckpt, dev)
    log(f"  dynamics WM {proprio_ckpt}")
    core_dyn, norm_dyn, _ = _load(proprio_ckpt, dev)
    P = int(cfg_vis.data.P)

    scores = {}
    for sp in (nominal, *splits):
        if not (pathlib.Path(root) / sp).exists():
            log(f"  !! split {sp} missing under {root}, skipping")
            continue
        eps = load_split_episodes_mm(root, sp, img_size=img_hw, cam=key, repo_id=repo_id,
                                     dcfg=DataConfig.from_cfg(cfg_vis))  # loaders take the data config explicitly (2026-09-21 refactor)
        if limit:
            eps = eps[:limit]
        log(f"  scoring {sp:14s} ({len(eps)} episodes)")
        scores[sp] = score_split(eps, core_dyn, norm_dyn, core_vis, norm_vis, key, P, dev, log=log)

    nom = scores[nominal]
    report = {"channels": list(CHANNELS), "nominal": nominal,
              "image_channels": list(IMAGE_CHANNELS), "dyn_channels": list(DYN_CHANNELS),
              "n": {k: len(v["image_l1"]) for k, v in scores.items()}, "detection": {}}
    for sp in splits:
        if sp in scores:
            report["detection"][sp] = {c: U.detect(nom[c], scores[sp][c]) for c in CHANNELS}
    ood_only = {k: v for k, v in scores.items() if k != nominal}
    report["three_way"] = _jsonable(U.confusion_3way(ood_only, channels=CHANNELS))

    outp = pathlib.Path(out)
    outp.mkdir(parents=True, exist_ok=True)
    (outp / "scores.json").write_text(json.dumps(
        {k: {c: v[c].tolist() for c in CHANNELS} for k, v in scores.items()}, indent=1))
    (outp / "report.json").write_text(json.dumps(report, indent=1, default=str))
    plot(scores, report, nominal, outp, log=log)
    return report, scores


def _jsonable(d):
    return {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in d.items()}


# --------------------------------------------------------------------------- plots (the "needs eyes" output)
def plot(scores, report, nominal, outp, log=print):
    """Four interpretable figures, so a human can see the two-model attribution at a glance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ood = [s for s in scores if s != nominal]
    colors = {nominal: "#4c72b0", "ood_thruster": "#c44e52", "ood_person": "#dd8452", "ood_appearance": "#8172b3"}
    clabel = {"image_lpips": "img LPIPS", "image_l1": "img L1", "proprio_pos": "pos",
              "proprio_vel": "vel", "proprio_angvel": "angvel", "proprio_rot_deg": "rot(deg)"}

    # 1) per-channel score distributions: nominal vs each OOD, with the 90% nominal threshold.
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, c in zip(axes.ravel(), CHANNELS):
        thr = float(np.percentile(scores[nominal][c], PCTL))
        groups = [nominal] + ood
        data = [scores[g][c] for g in groups]
        parts = ax.violinplot(data, showmeans=False, showextrema=True)
        for pc, g in zip(parts["bodies"], groups):
            pc.set_facecolor(colors.get(g, "#999")); pc.set_alpha(0.65)
        ax.axhline(thr, color="k", ls="--", lw=1, label=f"nominal p{int(PCTL)}")
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels([g.replace("ood_", "") for g in groups], rotation=20, ha="right", fontsize=9)
        band = "IMAGE" if c in IMAGE_CHANNELS else "DYNAMICS"
        ax.set_title(f"{clabel[c]}  [{band}]", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        if c == CHANNELS[0]:
            ax.legend(fontsize=8)
    fig.suptitle("One-step surprise per channel: nominal (blue) vs each OOD condition\n"
                 "image channels from the VISION WM, dynamics channels from the DYNAMICS WM", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outp / "channel_distributions.png", dpi=110); plt.close(fig)

    # 2) AUC heatmap: channel x OOD condition (how well each channel separates each condition from nominal).
    auc = np.array([[report["detection"][sp][c]["auc"] for sp in ood] for c in CHANNELS])
    fig, ax = plt.subplots(figsize=(1.6 * len(ood) + 3, 5))
    im = ax.imshow(auc, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(ood))); ax.set_xticklabels([s.replace("ood_", "") for s in ood])
    ax.set_yticks(range(len(CHANNELS))); ax.set_yticklabels([clabel[c] for c in CHANNELS])
    for i in range(len(CHANNELS)):
        for j in range(len(ood)):
            ax.text(j, i, f"{auc[i, j]:.2f}", ha="center", va="center",
                    color="black", fontsize=10, fontweight="bold")
    ax.axhline(1.5, color="k", lw=2)  # separates the 2 image channels from the 4 dynamics channels
    ax.set_title("Detection AUC (nominal vs OOD)\nimage channels above the line, dynamics below")
    fig.colorbar(im, ax=ax, label="AUC"); fig.tight_layout()
    fig.savefig(outp / "auc_heatmap.png", dpi=110); plt.close(fig)

    # 3) 3-way confusion matrix (which OOD type the channel vector is classified as).
    tw = report["three_way"]
    cm = np.array(tw["confusion"], float); classes = tw["classes"]
    cmn = cm / np.maximum(1, cm.sum(1, keepdims=True))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels([c.replace("ood_", "") for c in classes], rotation=20)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels([c.replace("ood_", "") for c in classes])
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{int(cm[i, j])}\n{cmn[i, j]*100:.0f}%", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=10)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"3-way OOD attribution (nearest-centroid, LOO)\nbalanced acc {tw['balanced_acc']*100:.0f}%")
    fig.tight_layout(); fig.savefig(outp / "confusion_3way.png", dpi=110); plt.close(fig)

    # 4) attribution signature: mean z-scored channel vector per OOD condition (image band vs dynamics band).
    allv = {c: np.concatenate([scores[s][c] for s in scores]) for c in CHANNELS}
    mu = {c: allv[c].mean() for c in CHANNELS}; sd = {c: allv[c].std() + 1e-9 for c in CHANNELS}
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(CHANNELS)); w = 0.8 / max(1, len(ood))
    for k, sp in enumerate(ood):
        z = [(scores[sp][c].mean() - mu[c]) / sd[c] for c in CHANNELS]
        ax.bar(x + k * w, z, w, label=sp.replace("ood_", ""), color=colors.get(sp, "#999"))
    ax.axvline(1.5, color="k", ls=":", lw=1)
    ax.text(0.5, ax.get_ylim()[1] * 0.92, "IMAGE", ha="center", fontsize=10, color="#555")
    ax.text(3.5, ax.get_ylim()[1] * 0.92, "DYNAMICS", ha="center", fontsize=10, color="#555")
    ax.set_xticks(x + w * (len(ood) - 1) / 2); ax.set_xticklabels([clabel[c] for c in CHANNELS])
    ax.axhline(0, color="k", lw=0.8); ax.set_ylabel("mean surprise (z vs all data)")
    ax.set_title("OOD signature per condition: dynamics faults light the DYNAMICS band,\n"
                 "visual anomalies light the IMAGE band -- this is the two-model attribution")
    ax.legend(); ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    fig.savefig(outp / "attribution_signature.png", dpi=110); plt.close(fig)

    log(f"  wrote 4 figures to {outp}:")
    log("    channel_distributions.png  -- per-channel nominal vs OOD, with threshold")
    log("    auc_heatmap.png            -- which channel detects which condition")
    log("    confusion_3way.png         -- 3-way attribution accuracy")
    log("    attribution_signature.png  -- the image-band vs dynamics-band fingerprint")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Cave 3-way OOD attribution from the vision + dynamics WMs.")
    ap.add_argument("--fpv_ckpt", required=True, help="cave_fpv run dir or .ckpt (image channels)")
    ap.add_argument("--proprio_ckpt", required=True, help="cave_proprio run dir or .ckpt (dynamics channels)")
    ap.add_argument("--root", required=True, help="packaged dataset dir (has val/ and ood_*/)")
    ap.add_argument("--out", default="/app/logs/cave/ood")
    ap.add_argument("--limit", type=int, default=None, help="cap episodes per split (smoke test)")
    a = ap.parse_args()
    run(a.fpv_ckpt, a.proprio_ckpt, a.root, out=a.out, limit=a.limit,
        log=lambda s: print(s, flush=True))
