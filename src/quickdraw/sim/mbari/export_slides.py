"""Camera-ready export for the MBARI presentation: one folder per slide, generated not curated.

    python -m quickdraw.sim.mbari.export_slides slide1 [--out DIR]
    python -m quickdraw.sim.mbari.export_slides slide2 --fpv-ckpt R --proprio-ckpt R
    python -m quickdraw.sim.mbari.export_slides slide3 --proprio-ckpt R

EVERY ARTEFACT IS OVER-PRODUCED ON PURPOSE. The brief was "when I say a couple, give me six" -- a
figure that is nearly right is worthless at slide-making time and re-rendering costs a round trip,
so each view gets several framings and each video several trajectories. Pick at the slide, not here.

NO TEXT BURNED INTO SLIDE 1. Layout renders carry no captions or overlays: a presentation puts its
own labels on, and baked-in text is the one thing that cannot be undone in Keynote.
"""
from __future__ import annotations

import argparse
import os
import pathlib

import numpy as np

W, H = 1152, 864                 # 4:3, large enough to crop or scale down without resampling mush
FPS = 30


def _scene(textures=True, name="tank_export.xml", **kw):
    import mujoco
    from .tank import build
    p, _ = build(textures=textures, name=name, **kw)
    return mujoco.MjModel.from_xml_path(str(p))


# --------------------------------------------------------------------------- slide 1: the tank
# CAMERAS GO INSIDE THE TANK. It is an enclosed 15 x 10 x 10 box with only the top open, so an eye
# placed outside sees the outside of a wall -- the first attempt orbited at dist 15-19 and produced
# six renders of flat fog for "side on" and six of a dark box for "isometric". Only the plan view
# survived, because looking straight down it sees in through the open surface.
#
# So views are specified as (eye, lookat) and converted, which is the only way to reason about
# whether the eye is in the water. MuJoCo places a free camera at
#   eye = lookat - dist * (cos(elev)cos(azim), cos(elev)sin(azim), sin(elev))
# so the inverse is azim = atan2(dy, dx), elev = asin(dz / dist) on delta = lookat - eye.
#
# `fovy` is a MODEL field, not a camera one, so it is set per view before rendering. The default 45
# cannot frame a 15 m tank from inside it.
TANK_X, TANK_Y, TANK_Z = 7.5, 5.0, -10.0


def cam_from_eye(eye, lookat):
    e, t = np.asarray(eye, float), np.asarray(lookat, float)
    d = t - e
    dist = float(np.linalg.norm(d))
    return dict(lookat=tuple(t), dist=dist,
                azim=float(np.degrees(np.arctan2(d[1], d[0]))),
                elev=float(np.degrees(np.arcsin(d[2] / max(1e-9, dist)))))


def _inside(eye):
    x, y, z = eye
    return abs(x) < TANK_X - 0.3 and abs(y) < TANK_Y - 0.3 and TANK_Z + 0.3 < z < -0.7


LAYOUTS = {
    # Plan view through the open surface -- the one framing that works from outside the box.
    # Plan view through the open surface. The azimuth matters even looking straight down: at azim 0
    # the 15 m length runs up the frame and a 4:3 image wastes both sides on nothing, so half of
    # these put the long axis across the frame instead.
    "topdown": [dict(fovy=62, azim_override=90, **cam_from_eye((0, 0, 9.0), (0, 0, -5.0))),
                dict(fovy=70, azim_override=90, **cam_from_eye((0, 0, 8.0), (0, 0, -5.0))),
                dict(fovy=56, azim_override=90, **cam_from_eye((0, 0, 10.5), (0, 0, -5.0))),
                dict(fovy=62, **cam_from_eye((0, 0, 9.0), (0, 0, -5.0))),
                dict(fovy=70, **cam_from_eye((0, 0, 8.0), (0, 0, -5.0))),
                dict(fovy=62, azim_override=90, **cam_from_eye((0, 0, 9.0), (0.8, 0, -5.0)))],
    # Down the LONG axis from just inside an end wall: shows the 15 m run, both buoy tethers and
    # the window rings in one shot.
    "sideon": [dict(fovy=90, **cam_from_eye((-6.9, 0, -4.6), (3.0, 0, -5.2))),
               dict(fovy=90, **cam_from_eye((6.9, 0, -4.6), (-3.0, 0, -5.2))),
               dict(fovy=100, **cam_from_eye((-6.9, -1.5, -3.2), (3.0, 0.5, -5.6))),
               dict(fovy=90, **cam_from_eye((0, -4.4, -4.6), (0, 3.0, -5.2))),
               dict(fovy=100, **cam_from_eye((-6.9, 0, -7.6), (3.0, 0, -5.6))),
               dict(fovy=82, **cam_from_eye((-6.9, 2.0, -4.0), (3.0, -1.0, -5.4)))],
    # Corner-to-corner, eye high and inside, looking down the diagonal.
    # Corner-to-corner down the diagonal. The eyes are pulled IN from the corner: at (-6.6, -4.2)
    # the near wall's window frames sit a metre from the lens and fill a third of the frame, which
    # ruined half the first batch. ~1.5 m of standoff from each wall fixes it.
    "iso": [dict(fovy=88, **cam_from_eye((-5.4, -3.4, -2.6), (1.5, 0.8, -6.2))),
            dict(fovy=88, **cam_from_eye((5.4, -3.4, -2.6), (-1.5, 0.8, -6.2))),
            dict(fovy=88, **cam_from_eye((-5.4, 3.4, -2.6), (1.5, -0.8, -6.2))),
            dict(fovy=88, **cam_from_eye((5.4, 3.4, -2.6), (-1.5, -0.8, -6.2))),
            dict(fovy=96, **cam_from_eye((-5.0, -3.0, -1.8), (1.0, 0.5, -6.8))),
            dict(fovy=80, **cam_from_eye((-4.6, -2.8, -3.6), (2.0, 1.0, -6.4)))],
}


def slide1(out, n_fpv=6, steps=900, log=print):
    """Tank layout stills (no text) + FPV clips of the MOLA flying."""
    import imageio
    import mujoco

    from . import record, trajectories as tj
    from .water import UnderwaterRenderer

    d = pathlib.Path(out) / "slide1"
    (d / "layout").mkdir(parents=True, exist_ok=True)
    (d / "fpv").mkdir(parents=True, exist_ok=True)

    model = _scene()
    data = mujoco.MjData(model)
    data.qpos[:3] = [1.5, -1.0, -4.6]                 # the vehicle in frame, off-centre, not posed flat
    data.qpos[3:7] = [0.94, 0.0, 0.0, 0.34]
    mujoco.mj_forward(model, data)
    rend = UnderwaterRenderer(model, H, W)
    for kind, views in LAYOUTS.items():
        for i, v in enumerate(views):
            eye = np.asarray(v["lookat"], float) - v["dist"] * np.array([
                np.cos(np.radians(v["elev"])) * np.cos(np.radians(v["azim"])),
                np.cos(np.radians(v["elev"])) * np.sin(np.radians(v["azim"])),
                np.sin(np.radians(v["elev"]))])
            if kind != "topdown" and not _inside(eye):
                log(f"  !! {kind}_{i}: eye {np.round(eye, 1)} is OUTSIDE the tank -- skipped")
                continue
            model.vis.global_.fovy = float(v.get("fovy", 45))
            # azim is degenerate looking straight down (dx=dy=0 -> atan2(0,0)=0), so a plan view
            # needs it stated: it chooses which way the tank's long axis runs across the frame.
            rend.free_cam(f"{kind}{i}", v["lookat"], v["dist"],
                          v.get("azim_override", v["azim"]), v["elev"])
            imageio.imwrite(d / "layout" / f"{kind}_{i}.png", rend.render(data, f"{kind}{i}"))
        log(f"  layout/{kind}: {len(views)} renders")
    model.vis.global_.fovy = 45
    rend.close()

    # FPV clips at presentation resolution rather than the dataset's 144x192 -- the stored frames
    # are training data, not figures, and upscaling them looks like upscaling them.
    rend2 = UnderwaterRenderer(model, 544, 720, cams=("fpv",))   # 544: divisible by 16 for the codec
    made, seed = 0, 7000
    while made < n_fpv:
        seed += 1
        rng = np.random.default_rng(seed)
        kind = ("circle", "circle_yaw", "random")[made % 3]
        if kind != "random" and tj.fit_circle(rng, steps / FPS, 1.0 / FPS) is None:
            continue
        r = record.rollout(model, kind, seed, steps=steps, size=(544, 720),
                           cams=("fpv",), renderer=rend2)
        if r is None or len(r["obs"]) < steps:
            continue
        imageio.mimwrite(d / "fpv" / f"fpv_{kind}_{made}.mp4", r["frames"]["fpv"], fps=FPS, quality=8)
        made += 1
        log(f"  fpv/{kind}_{made - 1}: {len(r['obs'])} steps")
    rend2.close()
    log(f"wrote {d}")


def _load_model(ckpt, dev="cuda"):
    from ...evaluation.standalone import load_checkpoint
    from ...training.setup import build_model, normalizer
    from .run_uncertainty import _cfg_for
    cfg = _cfg_for(ckpt)
    core = build_model(cfg).to(dev).eval()
    load_checkpoint(core, ckpt)
    return cfg, core, normalizer(cfg)


def prediction_rows(core, norm, o, a, fr, key, P, T, dev="cuda", lead=16, chunk=48):
    """Ground truth, AE floor, 1-step-ahead and `lead`-step-ahead for one trajectory.

    THE THREE ROWS ANSWER DIFFERENT QUESTIONS and belong on one figure because each bounds the next.
    The AE floor is encode->decode with NO dynamics: whatever it loses, no predictor can recover, so
    it is the ceiling the other two are measured against. One-step is the dynamics given a fresh
    frame every step. `lead`-step is the dynamics given a frame every `lead` steps -- the same
    sliding target-time forecast the eval reports, so the video and the number describe one thing.
    """
    import torch

    # FIRST VALID TARGET. The lead-step row is grounded `lead` steps back and needs P frames before
    # that, so t - lead >= P, i.e. t >= P + lead. Starting one earlier (P + lead - 1) asks for
    # o[-1:7], which numpy happily returns EMPTY rather than erroring, and torch.stack then reports
    # a shape mismatch several frames later. All four rows are aligned to this t so they are the
    # same moments side by side.
    ts = list(range(P + lead, T))
    gt, ae, p1, pk = [], [], [], []
    for i0 in range(0, len(ts), chunk):
        part = ts[i0:i0 + chunk]
        true = torch.stack([torch.from_numpy(fr[t]) for t in part]).float().div(255.0).to(dev)
        gt.append(true.cpu().numpy())
        # AE floor: encode and decode the REAL frame, no dynamics (see routines.eval_ae_floor)
        sub = {"proprio": torch.stack([norm.norm_obs(torch.from_numpy(o[t]))[None] for t in part]).float().to(dev),
               key: true[:, None]}
        with torch.no_grad():
            rec = core.to_obs(core.encode_state(sub), heads=[key])
        ae.append(rec[key][:, 0].clamp(0, 1).float().cpu().numpy())
        for horizon, acc in ((1, p1), (lead, pk)):
            st = [t - horizon for t in part]          # context ENDS here; the target is `horizon` later
            ctx = {"proprio": torch.stack([norm.norm_obs(torch.from_numpy(o[u - P:u])) for u in st]).float().to(dev),
                   key: torch.stack([torch.from_numpy(fr[u - P:u]) for u in st]).float().div(255.0).to(dev)}
            acts = torch.stack([norm.norm_act(torch.from_numpy(a[u - P:u + horizon - 1])) for u in st]).float().to(dev)
            with torch.no_grad():
                out = core.imagine_eval(ctx, acts, horizon, heads=[key], norm=norm,
                                        **({"decode_last": True} if horizon > 1 else {}))
            acc.append(out[key][:, -1].clamp(0, 1).float().cpu().numpy())
    return [np.concatenate(x) for x in (gt, ae, p1, pk)]


def slide2_predictions(ckpt, out, n=6, steps=900, lead=16, root=None, dev="cuda", log=print):
    import glob

    from ...logging import viz

    d = pathlib.Path(out) / "slide2" / "prediction"
    d.mkdir(parents=True, exist_ok=True)
    cfg, core, norm = _load_model(ckpt, dev)
    P, key = int(cfg.data.P), str(cfg.data.cam)
    root = root or "/app/logs/mbari/data/ood_controlled"
    files = sorted(glob.glob(f"{root}/nominal_test/ep_*.npz"))[:n]
    titles = ["ground truth", "AE floor (encode->decode, no dynamics)",
              "1-step ahead", f"{lead}-step ahead"]
    for i, f in enumerate(files):
        z = np.load(f)
        T = min(steps, len(z["obs"]))
        rows = prediction_rows(core, norm, z["obs"], z["act"], z[f"frames_{key}"], key, P, T,
                               dev=dev, lead=lead)
        strips = [viz.label_panels([r], [t], panel_wh=(768, 576), header_px=44, font_px=24)
                  for r, t in zip(rows, titles)]
        canvas = np.concatenate(strips, axis=1)
        viz.save_mp4(str(d / f"predict_{i}.mp4"), canvas, FPS, quality=8)
        log(f"  prediction/predict_{i}: {len(rows[0])} frames from {pathlib.Path(f).name}")
    log(f"wrote {d}")


SPLIT_LABEL = {"nominal_test": "nominal", "ood_thruster": "OOD: thruster glitch",
               "ood_fish": "OOD: shark", "ood_buoy": "OOD: purple buoy"}
VISUAL = ("ood_fish", "ood_buoy")


def slide2_uncertainty(ckpt, out, n=6, onset=300, root=None, dev="cuda", log=print):
    """Per-episode surprise traces with the onset marked -- static PNG and an animated version.

    THE ONSET LINE IS THE POINT. Every episode is 10 s clean then 30 s anomalous, so the trace shows
    the detector firing against a boundary the viewer can see, on the same episode -- which is a
    much stronger claim than a bar chart of episode scores, because no cross-episode confound can
    produce a step at exactly t=10 s in every episode of a condition and in none of the nominals.

    Channels are split by modality rather than all plotted together: the visual conditions move the
    image channels and the thruster moves proprio, so one shared y-axis would hide whichever is
    smaller. The non-visual case is exactly why the trace exists -- it has no pixel signature at all.
    """
    import glob

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .run_uncertainty import one_step

    d = pathlib.Path(out) / "slide2" / "uncertainty"
    (d / "timeseries").mkdir(parents=True, exist_ok=True)
    cfg, core, norm = _load_model(ckpt, dev)
    P, key = int(cfg.data.P), str(cfg.data.cam)
    root = root or "/app/logs/mbari/data/ood_controlled"
    traces = {}
    for split in ("nominal_test", *VISUAL, "ood_thruster"):
        files = sorted(glob.glob(f"{root}/{split}/ep_*.npz"))[:n]
        traces[split] = []
        for i, f in enumerate(files):
            z = np.load(f)
            e = one_step(core, norm, z["obs"], z["act"], z[f"frames_{key}"], key, P, dev)
            traces[split].append(e)
            steps = np.asarray(e["steps"])
            fig, ax = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True)
            for a_, chans, lab in ((ax[0], ("image_lpips", "image_l1"), "image channels"),
                                   (ax[1], ("proprio_vel", "proprio_angvel"), "proprio channels")):
                for c in chans:
                    v = np.asarray(e[c], float)
                    if np.isfinite(v).any():
                        a_.plot(steps / FPS, v, lw=0.9, label=c.replace("image_", "").replace("proprio_", ""))
                a_.axvline(onset / FPS, color="crimson", lw=1.6, ls="--")
                a_.set_ylabel(lab, fontsize=8)
                a_.legend(fontsize=7, loc="upper left", framealpha=0.6)
                a_.grid(alpha=0.25)
            ax[0].set_title(f"{SPLIT_LABEL[split]} — anomaly begins at {onset / FPS:.0f} s (dashed)", fontsize=10)
            ax[1].set_xlabel("time (s)")
            fig.tight_layout()
            fig.savefig(d / "timeseries" / f"{split}_{i}.png", dpi=150)
            plt.close(fig)
        log(f"  timeseries/{split}: {len(files)}")
    np.savez_compressed(d / "traces.npz",
                        **{f"{s}_{i}_{c}": np.asarray(t[c], float)
                           for s, ts in traces.items() for i, t in enumerate(ts) for c in t})
    log(f"wrote {d}")
    return traces


def slide2_scores(out, fpv_report, proprio_report, log=print):
    """Episode scores by condition with the conformal threshold drawn -- nominals green, OOD purple."""
    import json

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = pathlib.Path(out) / "slide2"
    d.mkdir(parents=True, exist_ok=True)
    for tag, rep_dir, chans in (("fpv", fpv_report, ("image_lpips", "proprio_vel")),
                                ("proprio", proprio_report, ("proprio_angvel", "proprio_vel"))):
        sc = json.loads((pathlib.Path(rep_dir) / "scores.json").read_text())
        rep = json.loads((pathlib.Path(rep_dir) / "conformal.json").read_text())
        for c in chans:
            det = rep["detection"].get(c, {})
            if det.get("threshold") is None:
                continue
            order = [("nominal_calib", "calibration"), ("nominal_test", "nominal (test)"),
                     ("ood_thruster", "thruster"), ("ood_fish", "shark"), ("ood_buoy", "buoy")]
            fig, ax = plt.subplots(figsize=(7.2, 4.0))
            for xi, (sp, lab) in enumerate(order):
                v = np.asarray(sc[sp]["post"][c], float)
                v = v[np.isfinite(v)]
                col = "#2e8b57" if sp.startswith("nominal") else "#7b2ff7"
                ax.scatter(np.full(len(v), xi) + np.random.default_rng(0).uniform(-.13, .13, len(v)),
                           v, s=22, alpha=0.75, color=col, edgecolors="none")
                ax.scatter([xi], [np.median(v)], marker="_", s=900, color="k", lw=1.6, zorder=5)
            ax.axhline(det["threshold"], color="crimson", lw=1.5, ls="--",
                       label=f"conformal threshold (alpha=0.1)  FPR={det['fpr_test']:.2f}")
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels([l for _, l in order], fontsize=9)
            ax.set_ylabel(f"episode score — {c}", fontsize=9)
            ax.set_title(f"{tag} model · {c} · green = nominal, purple = OOD", fontsize=10)
            ax.legend(fontsize=8, loc="upper left")
            ax.grid(alpha=0.25, axis="y")
            fig.tight_layout()
            fig.savefig(d / f"scores_{tag}_{c}.png", dpi=160)
            plt.close(fig)
            log(f"  scores_{tag}_{c}.png  thr={det['threshold']:.4f} fpr={det['fpr_test']:.2f}")


ARMS = [("true", "true dynamics"), ("analytic", "analytic model"), ("proprio", "learned world model")]


def slide3(out, bench="/app/logs/mbari/control/bench32",
           compare="/app/logs/mbari/control/compare", log=print):
    """Pose-to-pose control: the side-by-side clips, and the aggregate across every metric.

    THE CLIPS ARE COPIED, NOT RE-RENDERED. They already exist from the benchmark and re-running the
    episodes would produce different noise draws, so the video and the table would describe
    different trials. Eight are copied rather than the three asked for, and they deliberately span
    the result -- two of them are the FAILURES, because a slide showing three successes from a 94%
    arm is a selected slide.
    """
    import json
    import shutil

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = pathlib.Path(out) / "slide3"
    (d / "clips").mkdir(parents=True, exist_ok=True)
    for f in sorted(pathlib.Path(compare).glob("compare_trial*.mp4")):
        shutil.copy2(f, d / "clips" / f.name)
    log(f"  clips: {len(list((d / 'clips').glob('*.mp4')))} side-by-side videos")

    trials = {}
    for arm, _ in ARMS:
        rows = []
        for f in sorted(pathlib.Path(bench).glob(f"{arm}_s*/trials.json")):
            rows += json.loads(f.read_text())
        if rows:
            trials[arm] = rows
    if not trials:
        log("  !! no trials.json found"); return

    # SIX PANELS, NOT ONE BAR CHART. These metrics have different units and wildly different scales
    # -- a 0.012 m distance beside a 94% rate beside 0.8 degrees on shared axes is unreadable, and
    # normalising them to a common scale would invent a comparison nobody asked for.
    panels = [("success", "sustained-hold success", "rate", True),
              ("final_m", "final distance (m)", "m", False),
              ("att_deg", "final attitude error (deg)", "deg", False),
              ("wander", "wander over last 2 s (m)", "m", False),
              ("a_end", "command magnitude at rest", "frac of full thrust", False),
              ("crashed", "collision rate", "rate", True)]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.0))
    cols = {"true": "#2e8b57", "analytic": "#3b76c4", "proprio": "#7b2ff7"}
    for ax, (k, title, unit, is_rate) in zip(axes.ravel(), panels):
        xs, labels = [], []
        for xi, (arm, lab) in enumerate(ARMS):
            if arm not in trials:
                continue
            v = np.array([float(t[k]) for t in trials[arm]])
            labels.append(lab)
            if is_rate:
                ax.bar(xi, v.mean(), color=cols[arm], width=0.6, alpha=0.85)
                ax.text(xi, v.mean(), f" {v.mean()*100:.0f}%", ha="center", va="bottom", fontsize=9)
            else:
                ax.scatter(np.full(len(v), xi) + np.random.default_rng(1).uniform(-.12, .12, len(v)),
                           v, s=16, alpha=0.6, color=cols[arm], edgecolors="none")
                ax.scatter([xi], [np.median(v)], marker="_", s=700, color="k", lw=1.6, zorder=5)
                ax.set_yscale("log" if v.max() / max(1e-9, np.median(v)) > 25 else "linear")
            xs.append(xi)
        ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=10); ax.set_ylabel(unit, fontsize=8)
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle("Pose-to-pose control, 32 trials per arm — dots are trials, bar is the median "
                 "(log scale where one outlier dominates)", fontsize=11)
    fig.tight_layout()
    fig.savefig(d / "aggregate_metrics.png", dpi=160)
    plt.close(fig)
    log(f"  aggregate_metrics.png  ({', '.join(f'{a}={len(t)}' for a, t in trials.items())} trials)")
    (d / "aggregate.json").write_text(json.dumps(
        {a: {k: {"mean": float(np.mean([t[k] for t in ts])),
                 "median": float(np.median([t[k] for t in ts]))}
             for k, *_ in panels} for a, ts in trials.items()}, indent=1))
    log(f"wrote {d}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("slide1"); s1.add_argument("--out", default="/app/logs/mbari/export")
    s1.add_argument("--n-fpv", type=int, default=6)
    s2 = sub.add_parser("slide2-pred"); s2.add_argument("--ckpt", required=True)
    s2.add_argument("--out", default="/app/logs/mbari/export"); s2.add_argument("--n", type=int, default=6)
    s2.add_argument("--steps", type=int, default=900); s2.add_argument("--lead", type=int, default=16)
    s2u = sub.add_parser("slide2-unc"); s2u.add_argument("--ckpt", required=True)
    s2u.add_argument("--out", default="/app/logs/mbari/export"); s2u.add_argument("--n", type=int, default=6)
    s2s = sub.add_parser("slide2-scores"); s2s.add_argument("--out", default="/app/logs/mbari/export")
    s2s.add_argument("--fpv-report", default="/app/logs/mbari/uncertainty/conformal_fpv")
    s2s.add_argument("--proprio-report", default="/app/logs/mbari/uncertainty/conformal_proprio")
    s3 = sub.add_parser("slide3"); s3.add_argument("--out", default="/app/logs/mbari/export")
    a = ap.parse_args()
    if a.cmd == "slide3":
        slide3(a.out)
    elif a.cmd == "slide2-unc":
        slide2_uncertainty(a.ckpt, a.out, n=a.n)
    elif a.cmd == "slide2-scores":
        slide2_scores(a.out, a.fpv_report, a.proprio_report)
    elif a.cmd == "slide1":
        slide1(a.out, n_fpv=a.n_fpv)
    elif a.cmd == "slide2-pred":
        slide2_predictions(a.ckpt, a.out, n=a.n, steps=a.steps, lead=a.lead)



# ------------------------------------------------------------------- slide 2: prediction + uncertainty


if __name__ == "__main__":
    main()
