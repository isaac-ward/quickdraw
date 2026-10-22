"""Red-person (novel-object) OOD detection + the CROSS-DETECTOR table.

Repeats the proprio conformal pipeline for the VISION anomaly (a red person mesh popped into the FPV at t*), using
a model-free CHROMA surprise (per-frame redness). Also builds the cross table that is the point of the study -- run
BOTH detectors (proprio-WM surprise, vision chroma) against ALL THREE OOD types and show each detector catches its
own modality and is BLIND to the other (proprio can't see the person; chroma can't see a mass drop / action glitch).
So you need the right latent-surprise signal per anomaly, and together they "detect them all".

Bridge:  PATH=prospector/.venv/bin:$PATH PYTHONPATH=quickdraw/src CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cpu \
         prospector/.venv/bin/python -m quickdraw.sim.cave.run_ood_vision --n_cal 30 --n_test 20 --n_ood 20
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

CKPT = "/home/ubuntu/user_irw/quickdraw/logs/train_world_model_2026_10_19_11_36_34_cave_proprio_v1/checkpoints/best.ckpt"
ROOT = "/home/ubuntu/user_irw/quickdraw/logs/recording_2026_10_19_06_07_09_cave-grapevine"
STEPS = 900
KIND = "corridor"


def _gen(scene, mode, seed0, n, rng, frames=True, tstar=True):
    from .ood import ood_rollout
    out, seed, tried = [], seed0, 0
    while len(out) < n and seed < seed0 + n * 60:
        t_star = int(rng.integers(360, 540)) if tstar else 450
        r = ood_rollout(scene, KIND, seed, mode, t_star, steps=STEPS, want_frames=frames)
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
    ap.add_argument("--out", default="/home/ubuntu/user_irw/quickdraw/logs/cave/ood")
    a = ap.parse_args()

    from .run_control import _patch_scene_render, load_wm
    _patch_scene_render()
    from prospector.cave_wm.scene import CaveScene
    from . import ood_detect as D

    print("[setup] scene + WM ...", flush=True)
    scene = CaveScene(log=lambda *x: None)
    core, norm, P = load_wm(CKPT, ROOT, "cuda")
    core.stochastic_eval = False
    rng = np.random.default_rng(1)

    print("[gen] rendering trajectories (frames) ...", flush=True)
    sets = {"nom_cal": _gen(scene, "nominal", 1000, a.n_cal, rng, tstar=False),
            "nom_test": _gen(scene, "nominal", 5000, a.n_test, rng, tstar=False),
            "dynamics": _gen(scene, "dynamics", 2000, a.n_ood, rng),
            "glitch": _gen(scene, "glitch", 3000, a.n_ood, rng),
            "person": _gen(scene, "person", 4000, a.n_ood, rng)}
    print("[gen] counts: " + ", ".join(f"{k}={len(v)}" for k, v in sets.items()), flush=True)

    # --- two detector score-streams per trajectory ---
    def proprio_stream(r):
        return D.proprio_surprise(core, norm, r["obs"], r["act"], P)["total"]

    def chroma_stream(r):
        return D.chroma_perframe(r["frames"]["fpv"])

    print("[detect] proprio surprise + chroma ...", flush=True)
    pstream = {k: [proprio_stream(r) for r in v] for k, v in sets.items()}
    cstream = {k: [chroma_stream(r) for r in v] for k, v in sets.items()}
    # per-trajectory scores: proprio = tail-mean (sustained), chroma = max (transient but unambiguous)
    pscore = {k: np.array([D.traj_score(s) for s in pstream[k]]) for k in sets}
    cscore = {k: np.array([D.traj_max(s) for s in cstream[k]]) for k in sets}

    # --- cross-detector AUC table (nominal-test vs each OOD) ---
    oods = ["dynamics", "glitch", "person"]
    cross = {"proprio": {o: D.roc_auc(pscore["nom_test"], pscore[o]) for o in oods},
             "chroma": {o: D.roc_auc(cscore["nom_test"], cscore[o]) for o in oods}}

    report = {"n": {k: len(v) for k, v in sets.items()}, "WARM_s": D.WARM / 30.0,
              "cross_detector_AUC": cross,
              "detectors": {"proprio": "WM one-step surprise, tail-mean score",
                            "chroma": "per-frame redness p99.5, max score"}}

    # --- person classifier (chroma) + its conformal threshold & accuracy, mirroring the proprio run ---
    for alpha in (0.05, 0.10):
        tau = D.conformal_threshold(cscore["nom_cal"], alpha)
        tau_step = D.conformal_threshold(D.perstep_pool(cstream["nom_cal"]), alpha)
        fp = int((cscore["nom_test"] > tau).sum())
        tp = int((cscore["person"] > tau).sum())
        n_ind, n_ood = len(cscore["nom_test"]), len(cscore["person"])
        acc = (tp + (n_ind - fp)) / (n_ind + n_ood)
        lat = [D.detection_latency(s, tau_step, r["t_star"]) for s, r in zip(cstream["person"], sets["person"])]
        fin = [x for x in lat if np.isfinite(x)]
        report[f"person_alpha_{alpha}"] = {
            "tau_traj": tau, "tau_step": tau_step, "accuracy": acc,
            "TPR_person": tp / n_ood, "FPR_nominal_test": fp / n_ind,
            "confusion": {"TP": tp, "TN": n_ind - fp, "FP": fp, "FN": n_ood - tp},
            "latency_s_median": float(np.median(fin)) if fin else None, "missed": int(np.isinf(lat).sum()),
            "coverage_perframe_FPR": float((D.perstep_pool(cstream["nom_test"]) > tau_step).mean())}

    outp = pathlib.Path(a.out); outp.mkdir(parents=True, exist_ok=True)
    (outp / "vision_report.json").write_text(json.dumps(report, indent=2))
    np.savez(outp / "vision_scores.npz", **{f"p_{k}": pscore[k] for k in sets}, **{f"c_{k}": cscore[k] for k in sets})
    _plot(outp, sets, pstream, cstream, pscore, cscore, cross, report, D)
    print(json.dumps(report, indent=2), flush=True)
    print(f"[done] wrote {outp}/vision_report.json + vision_detect.png", flush=True)


def _plot(outp, sets, pstream, cstream, pscore, cscore, cross, report, D):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(STEPS) / 30.0
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    # (0,0) example person chroma traces
    a0 = ax[0][0]
    for k in range(min(5, len(sets["person"]))):
        a0.plot(t, cstream["person"][k], lw=0.9, color="#c44e52", alpha=0.7)
        a0.axvline(sets["person"][k]["t_star"] / 30.0, color="r", ls="--", lw=0.6, alpha=0.5)
    for k in range(min(5, len(sets["nom_test"]))):
        a0.plot(t, cstream["nom_test"][k], lw=0.8, color="#4c72b0", alpha=0.6)
    a0.axhline(report["person_alpha_0.1"]["tau_step"], color="k", lw=1.2, label="per-frame τ (α=.1)")
    a0.axvspan(0, D.WARM / 30.0, color="gray", alpha=0.15)
    a0.set_title("chroma surprise: person (red) vs nominal (blue)", fontsize=11)
    a0.set_xlabel("t (s)"); a0.set_ylabel("redness p99.5"); a0.grid(alpha=0.3); a0.legend(fontsize=7)

    # (0,1) cross-detector AUC heatmap
    a0 = ax[0][1]
    M = np.array([[cross["proprio"][o] for o in ["dynamics", "glitch", "person"]],
                  [cross["chroma"][o] for o in ["dynamics", "glitch", "person"]]])
    im = a0.imshow(M, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
    a0.set_xticks(range(3)); a0.set_xticklabels(["dynamics", "glitch", "person"])
    a0.set_yticks(range(2)); a0.set_yticklabels(["proprio-WM", "chroma"])
    for i in range(2):
        for j in range(3):
            a0.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=12,
                    color="k", fontweight="bold")
    a0.set_title("cross-detector AUC (nominal vs OOD)\neach detector catches its own modality", fontsize=10)
    fig.colorbar(im, ax=a0, fraction=0.046)

    # (0,2) per-trajectory chroma score strip
    a0 = ax[0][2]
    for i, (nm, c) in enumerate([("nom_cal", "#999"), ("nom_test", "#4c72b0"), ("dynamics", "#dd8452"),
                                 ("glitch", "#8c8c8c"), ("person", "#c44e52")]):
        sc = cscore[nm]; x = i + 0.08 * np.random.default_rng(i).standard_normal(len(sc))
        a0.scatter(x, sc, s=22, color=c, alpha=0.8)
    a0.axhline(report["person_alpha_0.1"]["tau_traj"], color="k", lw=1.2, label="τ_traj(α=.1)")
    a0.set_xticks(range(5)); a0.set_xticklabels(["n-cal", "n-test", "dyn", "glitch", "person"], fontsize=8)
    a0.set_title("chroma per-traj score (max redness)", fontsize=11)
    a0.set_ylabel("score"); a0.grid(alpha=0.3); a0.legend(fontsize=8)

    # (1,0) proprio score strip (shows proprio blind to person)
    a0 = ax[1][0]
    for i, (nm, c) in enumerate([("nom_cal", "#999"), ("nom_test", "#4c72b0"), ("dynamics", "#dd8452"),
                                 ("glitch", "#8c8c8c"), ("person", "#c44e52")]):
        sc = pscore[nm]; x = i + 0.08 * np.random.default_rng(i).standard_normal(len(sc))
        a0.scatter(x, sc, s=22, color=c, alpha=0.8)
    a0.set_yscale("log"); a0.set_xticks(range(5))
    a0.set_xticklabels(["n-cal", "n-test", "dyn", "glitch", "person"], fontsize=8)
    a0.set_title("proprio per-traj score (person ~ nominal = blind)", fontsize=11)
    a0.set_ylabel("score (log)"); a0.grid(alpha=0.3)

    # (1,1) person ROC (chroma)
    a0 = ax[1][1]
    neg, pos = cscore["nom_test"], cscore["person"]
    ths = np.unique(np.concatenate([neg, pos]))
    tpr = [np.mean(pos > th) for th in ths]; fpr = [np.mean(neg > th) for th in ths]
    order = np.argsort(fpr)
    a0.plot(np.array(fpr)[order], np.array(tpr)[order], "-o", ms=3, color="#c44e52",
            label=f"person/chroma (AUC={cross['chroma']['person']:.3f})")
    a0.plot([0, 1], [0, 1], "k:", lw=0.8)
    a0.set_title("ROC: nominal vs person (chroma)", fontsize=11)
    a0.set_xlabel("FPR"); a0.set_ylabel("TPR"); a0.grid(alpha=0.3); a0.legend(fontsize=8)

    # (1,2) person detection latency
    a0 = ax[1][2]
    tau_step = report["person_alpha_0.1"]["tau_step"]
    lat = [D.detection_latency(s, tau_step, r["t_star"]) for s, r in zip(cstream["person"], sets["person"])]
    lat = [x for x in lat if np.isfinite(x)]
    a0.hist(lat, bins=np.linspace(0, 3, 16), color="#c44e52", alpha=0.7)
    a0.set_title(f"person detection latency after t* (n={len(lat)})", fontsize=11)
    a0.set_xlabel("latency (s)"); a0.set_ylabel("count"); a0.grid(alpha=0.3)

    acc = report["person_alpha_0.1"]["accuracy"]
    fig.suptitle(f"Red-person (vision) OOD detection — chroma accuracy {acc:.2f} (α=.1); "
                 f"cross-detector AUC shows each signal catches its own modality", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outp / "vision_detect.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
