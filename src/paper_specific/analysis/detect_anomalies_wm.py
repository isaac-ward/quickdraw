"""Does the WORLD MODEL detect the anomalies? And does it hold a scene across a turn?

THE CHOICE OF SIGNAL IS THE WHOLE DESIGN. Open-loop error compounds monotonically with horizon, so
comparing "inside the anomaly window" against "outside" on an open-loop rollout would mostly measure WHEN
the window falls in the clip -- a late window looks anomalous whatever is in it. The detector signal is
therefore the ONE-STEP-AHEAD error: at every step the model is given the true last P frames and asked for
the next one. No compounding, so every step is comparable to every other, and the only thing that varies
is whether the thing being predicted is anomalous.

Reported per step, per episode:
    lpips, l1, psnr   image prediction error          -> should catch the VISUAL anomaly (pink noodle)
    proprio error     position + velocity error       -> should catch the DYNAMIC anomaly (leafblower)
    latent_cos        1 - cos(predicted bag, encoded true bag). Decoder-free "surprise": it asks whether
                      the model's internal next-state prediction matched the state that arrived, which is
                      the quantity an anomaly detector would actually deploy (no decode needed).

Scored against the reviewed windows in data/ood_windows.py with AUC -- the probability that a randomly
chosen in-window step scores higher than a randomly chosen out-of-window one. 0.5 is chance, and it needs
no threshold, which matters because the error scale differs per episode.

MEMORY is a different question, so it gets the OPEN-LOOP rollout: the model must carry the scene through
its own predictions and put it back when the drone turns around. Error is reported against the away/return
spans, with val at matched horizon as the reference.

    CUDA_VISIBLE_DEVICES=1 python scratch/detect_anomalies_wm.py <wm_ckpt> [out_dir]
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

from quickdraw.data.dataset import load_split_episodes_mm, set_action_aggregate, set_subsample
from quickdraw.data.ood_windows import WINDOWS, kept, window_steps
from quickdraw.evaluation.openloop import image_curves
from quickdraw.training.setup import (build_model, env_cfg, image_head_cams, image_head_sizes,
                                      load_checkpoint, normalizer, resolve_data_root)

SUB = 4
METRIC_DOC = {
    "lpips": "perceptual distance between the predicted and true next FRAME (lower=better). Rises with "
             "blur, so it separates 'hedging toward the mean frame' from 'confidently wrong'.",
    "l1": "mean absolute pixel difference between predicted and true next frame. Pixelwise, so a model "
          "that just copied the previous frame would score respectably -- kept as the plain baseline.",
    "pos_err": "euclidean error (metres) of the predicted next POSITION against the recorded one.",
    "vel_err": "euclidean error (m/s) of the predicted next VELOCITY. The channel a force disturbance "
               "shows up in first, since a push changes velocity before it has moved the drone far.",
    "l2": "pixel RMSE between predicted and true next frame. Same information as l1 but weighting large "
          "errors more, so a single badly-wrong region counts for more than diffuse blur.",
    "rot_err": "angle (degrees) between the predicted and true next ORIENTATION quaternion. Unlike "
               "position it is bounded and has no drift, so a disturbance that spins the drone shows here "
               "even when the position error is still small.",
    "angvel_err": "euclidean error of the predicted IMU ANGULAR VELOCITY. The rotational sibling of "
                  "vel_err, and the channel an off-axis push shows up in first.",
    "latent_cos": "1 - cos(predicted latent bag, encoder's bag for the true next frame). DECODER-FREE "
                  "surprise: did the model's internal next-state prediction match the state that arrived. "
                  "This is the form an anomaly detector would actually deploy.",
}

OOD = {"eval_ood_noodle": "visual", "eval_ood_leafblower": "dynamic"}
MEM = ["eval_memory_backwall1", "eval_memory_backwall2"]


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def yaw_deg(q):
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.degrees(np.unwrap(np.arctan2(2 * (w * z + x * y), 1.0 - 2 * (y * y + z * z))))


@torch.no_grad()
def open_loop(core, norm, o, a, fr, key, P, dev):
    """ONE open-loop rollout from the first P frames to the end of the clip. -> (steps, lpips).

    This is the signal memory needs and one-step cannot give: under teacher forcing the model is handed the
    true last P frames at every step, so when the drone turns back the wall is already in its context and
    nothing about retention is being tested. Open-loop, only the first P frames are real -- everything the
    model attends to at the return is its OWN earlier prediction, so the scene has to have survived inside
    the rolled state."""
    T = len(o)
    H = T - P
    ctx = {"proprio": norm.norm_obs(torch.from_numpy(o[:P])).float()[None].to(dev),
           key: torch.from_numpy(fr[:P]).float().div(255.0)[None].to(dev)}
    acts = norm.norm_act(torch.from_numpy(a[:P - 1 + H])).float()[None].to(dev)
    out = core.imagine_eval(ctx, acts, H, heads=[key], norm=norm)
    pred = out[key][0].clamp(0, 1)
    true = torch.from_numpy(fr[P:P + H]).float().div(255.0).to(dev)
    lp = [image_curves(pred[i:i + 1].unsqueeze(1), true[i:i + 1].unsqueeze(1))["lpips"][0] for i in range(H)]
    return np.arange(P, P + H), np.array(lp)


@torch.no_grad()
def one_step(core, norm, o, a, fr, key, P, dev):
    """Teacher-forced one-step-ahead prediction at EVERY step. -> dict of (T-P,) per-step errors."""
    T = len(o)
    ts = list(range(P, T))
    ctx = {"proprio": torch.stack([norm.norm_obs(torch.from_numpy(o[t - P:t])) for t in ts]).float().to(dev),
           key: torch.stack([torch.from_numpy(fr[t - P:t]) for t in ts]).float().div(255.0).to(dev)}
    acts = torch.stack([norm.norm_act(torch.from_numpy(a[t - P:t])) for t in ts]).float().to(dev)
    out = core.imagine_eval(ctx, acts, 1, heads=["proprio", key], norm=norm, return_bag=True)
    pred_img = out[key][:, 0].clamp(0, 1)                                      # (N,H,W,3)
    true_img = torch.stack([torch.from_numpy(fr[t]) for t in ts]).float().div(255.0).to(dev)
    ic = image_curves(pred_img.unsqueeze(1), true_img.unsqueeze(1))            # per-step, N=1 along dim 1
    # image_curves averages over dim 0, so call it per step instead
    lp, l1 = [], []
    for i in range(len(ts)):
        c = image_curves(pred_img[i:i + 1].unsqueeze(1), true_img[i:i + 1].unsqueeze(1))
        lp.append(c["lpips"][0] if "lpips" in c else np.nan); l1.append(c["l1"][0])
    pro_p = norm.denorm_obs(out["proprio"][:, 0].cpu()).numpy()
    pro_t = np.stack([o[t] for t in ts])
    lat = None
    if "_bag" in out:
        zp = out["_bag"][:, 0]
        zt = core.encode_state({"proprio": ctx["proprio"][:, -1:], key: ctx[key][:, -1:]})[:, 0] * 0
        # encode the TRUE next state for the surprise term
        zt = core.encode_state({"proprio": torch.stack([norm.norm_obs(torch.from_numpy(o[t:t + 1])) for t in ts]).float().to(dev),
                                key: torch.stack([torch.from_numpy(fr[t:t + 1]) for t in ts]).float().div(255.0).to(dev)})[:, 0]
        a_ = torch.nn.functional.normalize(zp.reshape(len(ts), -1), dim=-1)
        b_ = torch.nn.functional.normalize(zt.reshape(len(ts), -1), dim=-1)
        lat = (1.0 - (a_ * b_).sum(-1)).cpu().numpy()
    l2 = torch.sqrt(((pred_img - true_img) ** 2).mean(dim=(1, 2, 3))).cpu().numpy()
    # quaternion angular distance, |<q1,q2>| so q and -q are the same rotation
    qp = pro_p[:, 6:10] / (np.linalg.norm(pro_p[:, 6:10], axis=1, keepdims=True) + 1e-9)
    qt = pro_t[:, 6:10] / (np.linalg.norm(pro_t[:, 6:10], axis=1, keepdims=True) + 1e-9)
    rot = np.degrees(2.0 * np.arccos(np.clip(np.abs((qp * qt).sum(1)), 0.0, 1.0)))
    return {"steps": np.array(ts), "lpips": np.array(lp), "l1": np.array(l1), "l2": l2,
            "pos_err": np.linalg.norm(pro_p[:, 0:3] - pro_t[:, 0:3], axis=1),
            "vel_err": np.linalg.norm(pro_p[:, 3:6] - pro_t[:, 3:6], axis=1),
            "rot_err": rot,
            "angvel_err": np.linalg.norm(pro_p[:, 10:13] - pro_t[:, 10:13], axis=1),
            "latent_cos": lat if lat is not None else np.full(len(ts), np.nan)}


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
    report = {}

    METRICS = ["lpips", "l1", "l2", "pos_err", "vel_err", "rot_err", "angvel_err", "latent_cos"]
    for sp, kind in OOD.items():
        eps = load_split_episodes_mm(root, sp, **kw)
        d = os.path.join(out_root, "eval_ood", sp); os.makedirs(d, exist_ok=True)
        print(f"\n=== {sp}  ({kind} anomaly)  one-step-ahead detection, {len(kept(sp))} kept episodes")
        print(f"  {'ep':>3s} {'in/out steps':>13s} " + " ".join(f"{k:>11s}" for k in METRICS))
        pooled = {k: ([], []) for k in METRICS}
        rows = []
        for i in kept(sp):
            o, a, fr = eps[i]
            r = one_step(core, norm, o, a, fr[key], key, P, dev)
            # THE WINDOW IS IN FRAMES, THE PREDICTIONS ARE IN MODEL STEPS. ood_windows.window_steps does
            # the conversion; comparing the two directly (as this first did) silently gave episodes with
            # zero in-window or zero out-of-window steps and a nan AUC, which looked like missing data
            # rather than a units error.
            w0, w1 = window_steps(sp, i, SUB)
            inw = (r["steps"] >= w0) & (r["steps"] < w1)
            line, per = "", {}
            for k in METRICS:
                v = r[k]
                pooled[k][0].extend(v[inw].tolist()); pooled[k][1].extend(v[~inw].tolist())
                per[k] = auc(v[inw], v[~inw])
                line += f" {per[k]:>11.3f}"
            rows.append({"episode": i, "auc": per, "n_in": int(inw.sum()), "n_out": int((~inw).sum())})
            print(f"  {i:>3d} {f'{int(inw.sum())}/{int((~inw).sum())}':>13s}{line}")
            # per-episode inspectable plot
            fig, ax = plt.subplots(len(METRICS), 1, figsize=(9, 10), sharex=True)
            for j, k in enumerate(METRICS):
                ax[j].plot(r["steps"], r[k], color="crimson" if k in ("lpips", "l1") else "tab:blue")
                ax[j].set_ylabel(k); ax[j].grid(alpha=0.25)
                ax[j].axvspan(w0, w1, color="tab:green", alpha=0.15, lw=0)
                ax[j].set_title(f"AUC {per[k]:.3f}", fontsize=8, loc="right")
            ax[-1].set_xlabel("model step (stride 4); green = reviewed anomaly window")
            fig.suptitle(f"{sp} ep{i:02d} — ONE-STEP-AHEAD prediction error\n"
                         f"AUC = P(in-window step scores higher than out-of-window); 0.5 is chance",
                         fontsize=10)
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            fig.savefig(os.path.join(d, f"ep{i:02d}_wm_detect.png"), dpi=100); plt.close(fig)
        print(f"  {'POOLED':>3s} {f'{len(pooled[METRICS[0]][0])}/{len(pooled[METRICS[0]][1])}':>13s}"
              + "".join(f" {auc(*pooled[k]):>11.3f}" for k in METRICS))
        report[sp] = {"kind": kind, "per_episode": rows,
                      "pooled_auc": {k: auc(*pooled[k]) for k in METRICS},
                      "pooled_mean_in": {k: float(np.mean(pooled[k][0])) for k in METRICS},
                      "pooled_mean_out": {k: float(np.mean(pooled[k][1])) for k in METRICS}}

    for sp in MEM:
        eps = load_split_episodes_mm(root, sp, **kw)
        d = os.path.join(out_root, "eval_memory", sp); os.makedirs(d, exist_ok=True)
        keep_m = [x for x in kept(sp) if x < len(eps)]
        print(f"\n=== {sp}  one-step-ahead error across the turn, {len(keep_m)} of {len(eps)} episodes")
        print(f"  {'ep':>3s} {'away':>5s} {'back':>5s} {'lpips before':>13s} {'during away':>12s} "
              f"{'after return':>13s}   (OPEN-LOOP)")
        rows = []
        for i in keep_m:
            o, a, fr = eps[i]
            # `o` is already strided to model steps by set_subsample, so the heading trace and the
            # away/return indices are in MODEL steps here -- no conversion needed, unlike the OOD windows
            # above which are stored as raw 15 Hz frames.
            yw = yaw_deg(o[:, 6:10]); dv = yw - yw[0]
            aw = np.where(np.abs(dv) > 45.0)[0]
            if len(aw) == 0:
                continue
            s = int(aw[0]); bk = np.where(np.abs(dv[s:]) < 25.0)[0]
            e = s + int(bk[0]) if len(bk) else len(yw) - 1
            st, lp = open_loop(core, norm, o, a, fr[key], key, P, dev)
            pre, dur, post = st < s, (st >= s) & (st < e), st >= e
            rows.append({"episode": i, "away": s, "back": e,
                         "lpips_before": float(np.mean(lp[pre])) if pre.any() else None,
                         "lpips_during": float(np.mean(lp[dur])) if dur.any() else None,
                         "lpips_after": float(np.mean(lp[post])) if post.any() else None})
            f = lambda x: f"{x:.4f}" if x is not None else "-"
            print(f"  {i:>3d} {s:>5d} {e:>5d} {f(rows[-1]['lpips_before']):>13s} "
                  f"{f(rows[-1]['lpips_during']):>12s} {f(rows[-1]['lpips_after']):>13s}")
            fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
            ax[0].plot(st, lp, color="crimson"); ax[0].set_ylabel("lpips (OPEN-LOOP)")
            ax[1].plot(np.arange(len(yw)), dv, color="k"); ax[1].set_ylabel("heading change (deg)")
            ax[1].set_xlabel("model step (stride 4)")
            for A in ax:
                A.grid(alpha=0.25); A.axvspan(s, e, color="tab:orange", alpha=0.15, lw=0)
            fig.suptitle(f"{sp} ep{i:02d} — turned away at step {s}, back by {e} (shaded)\n"
                         f"OPEN-LOOP error before / during / after the turn — only the first "
                         f"{P} frames are real, so the scene at the return must have survived in the "
                         f"rolled state", fontsize=10)
            fig.tight_layout(rect=(0, 0, 1, 0.92))
            fig.savefig(os.path.join(d, f"ep{i:02d}_wm_memory.png"), dpi=100); plt.close(fig)
        g = lambda k: float(np.mean([r[k] for r in rows if r[k] is not None]))
        print(f"  {'MEAN':>3s} {'':>5s} {'':>5s} {g('lpips_before'):>13.4f} {g('lpips_during'):>12.4f} "
              f"{g('lpips_after'):>13.4f}")
        report[sp] = {"per_episode": rows, "mean_before": g("lpips_before"),
                      "mean_during": g("lpips_during"), "mean_after": g("lpips_after")}
    json.dump(report, open(os.path.join(out_root, "wm_anomaly_detection.json"), "w"), indent=1)
    print(f"\n  report -> {out_root}/wm_anomaly_detection.json")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
