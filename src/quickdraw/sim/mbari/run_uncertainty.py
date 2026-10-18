"""Score every episode's one-step VoE per channel, classify the three OOD types, render pixel maps.

Adapted from `paper_specific/analysis/detect_anomalies_wm.one_step`, with the MBARI state layout:

    starling   0:3 pos | 3:6 vel | 6:10 quat | 10:13 imu angvel
    mbari      0:3 pos | 3:7 quat | 7:10 vel  | 10:13 angvel (BODY)

The indices are NOT the same and silently reusing starling's would compare velocity against a
quaternion component and report it as a velocity error. They are written out once, here.
"""
from __future__ import annotations

import glob
import json
import pathlib

import numpy as np
import torch

POS, QUAT, VEL, ANGVEL = slice(0, 3), slice(3, 7), slice(7, 10), slice(10, 13)


def one_step(core, norm, o, a, fr, key, P, dev, chunk=64):
    """Teacher-forced one-step prediction at every step -> per-step channel errors."""
    from ...evaluation.openloop import image_curves

    T = len(o)
    ts = list(range(P, T))
    out_lp, out_l1, pro_p = [], [], []
    for i0 in range(0, len(ts), chunk):
        part = ts[i0:i0 + chunk]
        ctx = {"proprio": torch.stack([norm.norm_obs(torch.from_numpy(o[t - P:t])) for t in part]).float().to(dev),
               key: torch.stack([torch.from_numpy(fr[t - P:t]) for t in part]).float().div(255.0).to(dev)}
        acts = torch.stack([norm.norm_act(torch.from_numpy(a[t - P:t])) for t in part]).float().to(dev)
        with torch.no_grad():
            out = core.imagine_eval(ctx, acts, 1, heads=["proprio", key], norm=norm)
        pred = out[key][:, 0].clamp(0, 1)
        true = torch.stack([torch.from_numpy(fr[t]) for t in part]).float().div(255.0).to(dev)
        for i in range(len(part)):
            c = image_curves(pred[i:i + 1].unsqueeze(1), true[i:i + 1].unsqueeze(1))
            out_lp.append(float(c["lpips"][0]))
            out_l1.append(float(c["l1"][0]))
        pro_p.append(norm.denorm_obs(out["proprio"][:, 0].cpu()).numpy())
    pro_p = np.concatenate(pro_p)
    pro_t = np.stack([o[t] for t in ts])
    qp = pro_p[:, QUAT] / (np.linalg.norm(pro_p[:, QUAT], axis=1, keepdims=True) + 1e-9)
    qt = pro_t[:, QUAT] / (np.linalg.norm(pro_t[:, QUAT], axis=1, keepdims=True) + 1e-9)
    return {"steps": np.array(ts),
            "image_lpips": np.array(out_lp), "image_l1": np.array(out_l1),
            "proprio_pos": np.linalg.norm(pro_p[:, POS] - pro_t[:, POS], axis=1),
            "proprio_vel": np.linalg.norm(pro_p[:, VEL] - pro_t[:, VEL], axis=1),
            "proprio_angvel": np.linalg.norm(pro_p[:, ANGVEL] - pro_t[:, ANGVEL], axis=1),
            "proprio_rot_deg": np.degrees(2.0 * np.arccos(np.clip(np.abs((qp * qt).sum(1)), 0, 1)))}


CHANNELS = ("image_lpips", "image_l1", "proprio_pos", "proprio_vel", "proprio_angvel", "proprio_rot_deg")


def score_split(core, norm, root, split, key="fpv", P=8, dev="cuda", limit=None, log=print):
    """Per-episode channel scores (90th percentile over steps) for one split."""
    import ast

    files = sorted(glob.glob(str(pathlib.Path(root) / split / "ep_*.npz")))[:limit]
    rows = []
    for n, f in enumerate(files):
        z = np.load(f)
        o, a, fr = z["obs"], z["act"], z[f"frames_{key}"]
        e = one_step(core, norm, o, a, fr, key, P, dev)
        rows.append({c: float(np.percentile(e[c], 90)) for c in CHANNELS})
        if (n + 1) % 10 == 0:
            log(f"    {split} {n+1}/{len(files)}")
    return {c: np.array([r[c] for r in rows]) for c in CHANNELS}


def run(ckpt, root="/app/logs/mbari/data/v4", out="/app/logs/mbari/uncertainty",
        splits=("train", "ood_thruster", "ood_fish", "ood_buoy"), key="fpv",
        limit=30, dev="cuda", log=print):
    """Score every split, calibrate on nominals, classify the three conditions, write JSON."""
    import json

    from ...evaluation.standalone import load_checkpoint
    from ...training.setup import build_model, normalizer
    from . import uncertainty as U

    cfg = _cfg_for(ckpt)
    core = build_model(cfg).to(dev)
    load_checkpoint(core, ckpt)
    core.eval()
    norm = normalizer(cfg)

    scores = {}
    for sp in splits:
        if not (pathlib.Path(root) / sp).exists():
            continue
        scores[sp] = score_split(core, norm, root, sp, key=key, P=int(cfg.data.P), dev=dev,
                                 limit=limit, log=log)
        log(f"  {sp:14s} scored {len(next(iter(scores[sp].values())))} episodes")

    nominal = scores["train"]
    report = {"channels": list(CHANNELS), "detection": {}, "n": {k: len(v["image_l1"]) for k, v in scores.items()}}
    for sp in [s for s in scores if s != "train"]:
        report["detection"][sp] = {c: U.detect(nominal[c], scores[sp][c]) for c in CHANNELS}
    report["three_way"] = _jsonable(U.confusion_3way({k: v for k, v in scores.items() if k != "train"},
                                                     channels=CHANNELS))
    report["four_way"] = _jsonable(U.confusion_3way(scores, channels=CHANNELS))
    outp = pathlib.Path(out)
    outp.mkdir(parents=True, exist_ok=True)
    (outp / "scores.json").write_text(json.dumps(
        {k: {c: v[c].tolist() for c in CHANNELS} for k, v in scores.items()}, indent=1))
    (outp / "report.json").write_text(json.dumps(report, indent=1, default=str))
    return report, scores


def _jsonable(d):
    return {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in d.items()}


def _cfg_for(ckpt):
    """The run's own saved config, so the architecture matches the weights exactly."""
    import json as _json
    import os

    from omegaconf import OmegaConf

    run = os.path.dirname(os.path.dirname(ckpt)) if str(ckpt).endswith(".ckpt") else ckpt
    p = os.path.join(run, "logs", "config.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"no saved config at {p}")
    return OmegaConf.create(_json.load(open(p)))
