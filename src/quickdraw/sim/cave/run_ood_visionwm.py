"""Vision-WM OOD detection: can the JOINT cave_fpv world model (one model, two heads) catch all three anomalies?

This replaces the model-free CHROMA hack with a LEARNED detector. The joint WM predicts both the next proprio state
and the next FPV frame; we score one-step surprise on BOTH heads:
  - proprio-head surprise (normalized L2)  -> dynamics (lighter drone) + glitch (corrupted action)
  - fpv-head surprise (pixel MSE == trainer's val/metric/fpv/mse) -> person (novel red mesh pops into the FPV)
Chroma (per-frame red-pixel count) is kept as the baseline the fpv-head must beat on the person case.

Same split-conformal pipeline as run_ood_proprio: held-out NOMINAL calibration -> (1-alpha) quantile threshold;
classify InD/OOD; report accuracy, AUC, latency, coverage. Cross-detector table = each detector vs all 3 anomalies.

CHECKPOINT QUIRK: cave_fpv best.ckpt predates the `.back` decoder refactor -> key remap + strict=False load
(load_fpv_wm below; verified missing_params=0, one-step fpv mse ~0.005 on settled nominal).

Bridge:  PATH=prospector/.venv/bin:$PATH PYTHONPATH=quickdraw/src CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cpu \
         prospector/.venv/bin/python -m quickdraw.sim.cave.run_ood_visionwm --n_cal 30 --n_test 20 --n_ood 20
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re

import numpy as np

FPV_CKPT = "/home/ubuntu/user_irw/quickdraw/logs/train_world_model_2026_10_20_02_03_20_cave_fpv_v2/checkpoints/best.ckpt"
ROOT = "/home/ubuntu/user_irw/quickdraw/logs/recording_2026_10_19_06_07_09_cave-grapevine"
STEPS = 900
KIND = "corridor"
_UP = ("readout", "ups", "to_ch", "gpool", "mid", "out_conv", "out_norm")


def load_fpv_wm(ckpt, root, dev="cuda"):
    """Load the joint cave_fpv WM. Migrates pre-`.back` decoder keys (refactor 246847d) then strict=False loads;
    NO red-model overrides (this ckpt has no integrate_position/velocity_idx). Returns (core, norm, P)."""
    import torch
    from ...training.setup import build_model, normalizer
    from ..mbari.run_uncertainty import _cfg_for
    cfg = _cfg_for(ckpt)
    cfg.data.root = root
    core = build_model(cfg).to(dev)
    sd = torch.load(ckpt, map_location="cpu"); sd = sd.get("state_dict", sd)
    remap = {}
    for k, v in sd.items():
        k = k[6:] if k.startswith("model.") else k
        k = k.replace("_orig_mod.", "")
        for p in ("act_enc", "enc"):
            k = re.sub(rf"(^|\.)({p})\.(\d+)\.", r"\1\2.net.\3.", k)
        k = re.sub(rf"(decode_head)\.({'|'.join(_UP)})\.", r"\1.back.\2.", k)
        remap[k] = v
    inc = core.load_state_dict(remap, strict=False)
    params = {n for n, _ in core.named_parameters()}
    missing_params = [k for k in inc.missing_keys if k in params]
    print(f"[load] missing_params={len(missing_params)} missing_buffers={len(inc.missing_keys) - len(missing_params)}"
          f" unexpected={len(inc.unexpected_keys)}", flush=True)
    assert not missing_params, f"decoder migration incomplete: {missing_params[:5]}"
    core.eval()
    return core, normalizer(cfg), int(cfg.data.P)


def _gen(scene, mode, seed0, n, rng, tstar=True):
    from .ood import ood_rollout
    out, seed, tried = [], seed0, 0
    while len(out) < n and seed < seed0 + n * 60:
        t_star = int(rng.integers(360, 540)) if tstar else 450
        r = ood_rollout(scene, KIND, seed, mode, t_star, steps=STEPS, want_frames=True)
        seed += 1; tried += 1
        if r is not None:
            out.append(r)
    print(f"[gen]   {mode}: {len(out)}/{n} kept in {tried} tries", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_cal", type=int, default=30)
    ap.add_argument("--n_test", type=int, default=20)
    ap.add_argument("--n_ood", type=int, default=20)
    ap.add_argument("--stride", type=int, default=2, help="window-start subsample for the expensive fpv decode")
    ap.add_argument("--out", default="/home/ubuntu/user_irw/quickdraw/logs/cave/ood")
    a = ap.parse_args()

    from .run_control import _patch_scene_render
    _patch_scene_render()
    from prospector.cave_wm.scene import CaveScene
    from . import ood_detect as D

    print("[setup] scene + joint cave_fpv WM ...", flush=True)
    scene = CaveScene(log=lambda *x: None)
    core, norm, P = load_fpv_wm(FPV_CKPT, ROOT, "cuda")
    core.stochastic_eval = False
    rng = np.random.default_rng(1)

    print("[gen] rendering trajectories (frames) ...", flush=True)
    sets = {"nom_cal": _gen(scene, "nominal", 1000, a.n_cal, rng, tstar=False),
            "nom_test": _gen(scene, "nominal", 5000, a.n_test, rng, tstar=False),
            "dynamics": _gen(scene, "dynamics", 2000, a.n_ood, rng),
            "glitch": _gen(scene, "glitch", 3000, a.n_ood, rng),
            "person": _gen(scene, "person", 4000, a.n_ood, rng)}
    print("[gen] counts: " + ", ".join(f"{k}={len(v)}" for k, v in sets.items()), flush=True)

    # --- per-trajectory per-step streams: joint WM (both heads) + chroma baseline ---
    print(f"[detect] joint cave_fpv surprise (stride={a.stride}) + chroma ...", flush=True)
    pstream, fstream, cstream = {}, {}, {}
    for k, v in sets.items():
        pk, fk, ck = [], [], []
        for i, r in enumerate(v):
            js = D.joint_surprise(core, norm, r["obs"], r["act"], r["frames"]["fpv"], P, stride=a.stride)
            pk.append(js["proprio"]); fk.append(js["fpv"]); ck.append(D.chroma_perframe(r["frames"]["fpv"]))
        pstream[k], fstream[k], cstream[k] = pk, fk, ck
        print(f"[detect]   {k} done ({len(v)})", flush=True)

    # per-trajectory scores: proprio = tail-mean (sustained); fpv/chroma = max settled (person is transient)
    pscore = {k: np.array([D.traj_score(s) for s in pstream[k]]) for k in sets}
    fscore = {k: np.array([D.traj_max(s) for s in fstream[k]]) for k in sets}
    cscore = {k: np.array([D.traj_max(s) for s in cstream[k]]) for k in sets}

    oods = ["dynamics", "glitch", "person"]
    cross = {"fpv_proprio_head": {o: D.roc_auc(pscore["nom_test"], pscore[o]) for o in oods},
             "fpv_image_head": {o: D.roc_auc(fscore["nom_test"], fscore[o]) for o in oods},
             "chroma": {o: D.roc_auc(cscore["nom_test"], cscore[o]) for o in oods}}

    report = {"n": {k: len(v) for k, v in sets.items()}, "WARM_s": D.WARM / 30.0, "stride": a.stride,
              "cross_detector_AUC": cross,
              "detectors": {"fpv_proprio_head": "joint WM proprio one-step surprise, tail-mean",
                            "fpv_image_head": "joint WM fpv one-step MSE (==val/metric/fpv/mse), max",
                            "chroma": "per-frame strongly-red px count, max"}}

    # --- person classifier via the LEARNED fpv-head (the headline) + the proprio head on dyn+glitch ---
    def classify(score, stream, pos_key, pos_sets, higher=True):
        d = {}
        for alpha in (0.05, 0.10):
            tau = D.conformal_threshold(score["nom_cal"], alpha)
            tau_step = D.conformal_threshold(D.perstep_pool(stream["nom_cal"]), alpha)
            fp = int((score["nom_test"] > tau).sum())
            pos = np.concatenate([score[k] for k in pos_key])
            tp = int((pos > tau).sum())
            n_ind, n_ood = len(score["nom_test"]), len(pos)
            lat = [D.detection_latency(s, tau_step, r["t_star"]) for pk in pos_key
                   for s, r in zip(stream[pk], pos_sets[pk])]
            fin = [x for x in lat if np.isfinite(x)]
            d[f"alpha_{alpha}"] = {
                "tau_traj": tau, "tau_step": tau_step,
                "accuracy": (tp + (n_ind - fp)) / (n_ind + n_ood),
                "TPR": tp / n_ood, "FPR_nominal_test": fp / n_ind,
                "confusion": {"TP": tp, "TN": n_ind - fp, "FP": fp, "FN": n_ood - tp},
                "latency_s_median": float(np.median(fin)) if fin else None, "missed": int(np.isinf(lat).sum()),
                "coverage_perstep_FPR": float((D.perstep_pool(stream["nom_test"]) > tau_step).mean())}
        return d

    report["person_fpv_head"] = classify(fscore, fstream, ["person"], sets)
    report["person_chroma"] = classify(cscore, cstream, ["person"], sets)
    report["dynglitch_proprio_head"] = classify(pscore, pstream, ["dynamics", "glitch"], sets)

    outp = pathlib.Path(a.out); outp.mkdir(parents=True, exist_ok=True)
    (outp / "visionwm_report.json").write_text(json.dumps(report, indent=2))
    np.savez(outp / "visionwm_scores.npz",
             **{f"p_{k}": pscore[k] for k in sets}, **{f"f_{k}": fscore[k] for k in sets},
             **{f"c_{k}": cscore[k] for k in sets})
    # also cache per-step streams so the statistic can be re-derived without re-flying/re-decoding
    np.savez(outp / "visionwm_streams.npz",
             **{f"p_{k}_{i}": s for k in sets for i, s in enumerate(pstream[k])},
             **{f"f_{k}_{i}": s for k in sets for i, s in enumerate(fstream[k])},
             **{f"tstar_{k}_{i}": sets[k][i].get("t_star", -1) for k in sets for i in range(len(sets[k]))})
    _plot(outp, sets, fstream, fscore, pscore, cscore, cross, report, D)
    print(json.dumps(report, indent=2), flush=True)
    print(f"[done] wrote {outp}/visionwm_report.json + visionwm_detect.png", flush=True)


def _plot(outp, sets, fstream, fscore, pscore, cscore, cross, report, D):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(STEPS) / 30.0
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    # (0,0) example fpv-surprise traces: person (red) vs nominal (blue)
    a0 = ax[0][0]
    for k in range(min(6, len(sets["person"]))):
        a0.plot(t, fstream["person"][k], lw=0.9, color="#c44e52", alpha=0.7)
        a0.axvline(sets["person"][k]["t_star"] / 30.0, color="r", ls="--", lw=0.6, alpha=0.5)
    for k in range(min(6, len(sets["nom_test"]))):
        a0.plot(t, fstream["nom_test"][k], lw=0.8, color="#4c72b0", alpha=0.6)
    a0.axvspan(0, D.WARM / 30.0, color="gray", alpha=0.15)
    a0.set_title("fpv-head one-step surprise: person (red) vs nominal (blue)", fontsize=10)
    a0.set_xlabel("t (s)"); a0.set_ylabel("next-frame MSE"); a0.grid(alpha=0.3)

    # (0,1) cross-detector AUC heatmap (3 detectors x 3 anomalies)
    a0 = ax[0][1]
    rows = ["fpv_proprio_head", "fpv_image_head", "chroma"]
    M = np.array([[cross[rw][o] for o in ["dynamics", "glitch", "person"]] for rw in rows])
    im = a0.imshow(M, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
    a0.set_xticks(range(3)); a0.set_xticklabels(["dynamics", "glitch", "person"])
    a0.set_yticks(range(3)); a0.set_yticklabels(["WM proprio-head", "WM fpv-head", "chroma (baseline)"], fontsize=9)
    for i in range(3):
        for j in range(3):
            a0.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=12, fontweight="bold")
    a0.set_title("cross-detector AUC (nominal vs OOD)\none joint WM, two heads", fontsize=10)
    fig.colorbar(im, ax=a0, fraction=0.046)

    # (0,2) fpv per-traj score strip
    a0 = ax[0][2]
    for i, (nm, c) in enumerate([("nom_cal", "#999"), ("nom_test", "#4c72b0"), ("dynamics", "#dd8452"),
                                 ("glitch", "#8c8c8c"), ("person", "#c44e52")]):
        sc = fscore[nm]; x = i + 0.08 * np.random.default_rng(i).standard_normal(len(sc))
        a0.scatter(x, sc, s=22, color=c, alpha=0.8)
    a0.axhline(report["person_fpv_head"]["alpha_0.1"]["tau_traj"], color="k", lw=1.2, label="τ_traj(α=.1)")
    a0.set_xticks(range(5)); a0.set_xticklabels(["n-cal", "n-test", "dyn", "glitch", "person"], fontsize=8)
    a0.set_title("fpv-head per-traj score (max MSE)", fontsize=11)
    a0.set_ylabel("score"); a0.grid(alpha=0.3); a0.legend(fontsize=8)

    # (1,0) proprio-head per-traj score strip (log)
    a0 = ax[1][0]
    for i, (nm, c) in enumerate([("nom_cal", "#999"), ("nom_test", "#4c72b0"), ("dynamics", "#dd8452"),
                                 ("glitch", "#8c8c8c"), ("person", "#c44e52")]):
        sc = pscore[nm]; x = i + 0.08 * np.random.default_rng(i).standard_normal(len(sc))
        a0.scatter(x, sc, s=22, color=c, alpha=0.8)
    a0.set_yscale("log"); a0.set_xticks(range(5))
    a0.set_xticklabels(["n-cal", "n-test", "dyn", "glitch", "person"], fontsize=8)
    a0.set_title("WM proprio-head per-traj score (log)", fontsize=11)
    a0.set_ylabel("score (log)"); a0.grid(alpha=0.3)

    # (1,1) person ROC: fpv-head vs chroma
    a0 = ax[1][1]
    for nm, sc_dict, c in [("fpv-head", fscore, "#c44e52"), ("chroma", cscore, "#55a868")]:
        neg, pos = sc_dict["nom_test"], sc_dict["person"]
        ths = np.unique(np.concatenate([neg, pos]))
        tpr = [np.mean(pos > th) for th in ths]; fpr = [np.mean(neg > th) for th in ths]
        order = np.argsort(fpr)
        a0.plot(np.array(fpr)[order], np.array(tpr)[order], "-o", ms=3, color=c,
                label=f"{nm} (AUC={cross['fpv_image_head' if nm == 'fpv-head' else 'chroma']['person']:.3f})")
    a0.plot([0, 1], [0, 1], "k:", lw=0.8)
    a0.set_title("person ROC: learned fpv-head vs chroma", fontsize=11)
    a0.set_xlabel("FPR"); a0.set_ylabel("TPR"); a0.grid(alpha=0.3); a0.legend(fontsize=8)

    # (1,2) person detection latency (fpv-head)
    a0 = ax[1][2]
    tau_step = report["person_fpv_head"]["alpha_0.1"]["tau_step"]
    lat = [D.detection_latency(s, tau_step, r["t_star"]) for s, r in zip(fstream["person"], sets["person"])]
    lat = [x for x in lat if np.isfinite(x)]
    a0.hist(lat, bins=np.linspace(0, 3, 16), color="#c44e52", alpha=0.7)
    a0.set_title(f"person detection latency, fpv-head (n={len(lat)})", fontsize=11)
    a0.set_xlabel("latency (s)"); a0.set_ylabel("count"); a0.grid(alpha=0.3)

    acc = report["person_fpv_head"]["alpha_0.1"]["accuracy"]
    fig.suptitle(f"Vision-WM OOD detection — joint cave_fpv, fpv-head person accuracy {acc:.2f} (α=.1); "
                 f"one model, two heads, all three anomalies", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outp / "visionwm_detect.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
