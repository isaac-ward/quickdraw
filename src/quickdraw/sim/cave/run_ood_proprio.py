"""Proprio-WM OOD detection run: generate nominal + dynamics + glitch onset trajectories, compute one-step
surprise, fit a split-conformal InD/OOD threshold on HELD-OUT nominal, and report classification accuracy
(InD = nominal-test, OOD = dynamics/glitch), AUC, detection latency, and the conformal coverage check.

Bridge:  PATH=prospector/.venv/bin:$PATH PYTHONPATH=quickdraw/src CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cpu \
         prospector/.venv/bin/python -m quickdraw.sim.cave.run_ood_proprio --n_cal 30 --n_test 20 --n_ood 20
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


def _gen_set(scene, mode, seed0, n, rng, want_tstar=True):
    """n surviving trajectories for `mode`; onset t* ~ U[360,540] (12-18s, after WARM). Returns list of dicts."""
    from .ood import ood_rollout
    out = []
    seed = seed0
    while len(out) < n and seed < seed0 + n * 20:
        t_star = int(rng.integers(360, 540)) if want_tstar else 450
        r = ood_rollout(scene, KIND, seed, mode, t_star, steps=STEPS)
        seed += 1
        if r is not None:
            out.append(r)
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
    rng = np.random.default_rng(0)

    print(f"[gen] nominal cal={a.n_cal} test={a.n_test}, dynamics={a.n_ood}, glitch={a.n_ood} ...", flush=True)
    cal = _gen_set(scene, "nominal", 1000, a.n_cal, rng, want_tstar=False)
    test = _gen_set(scene, "nominal", 5000, a.n_test, rng, want_tstar=False)
    dyn = _gen_set(scene, "dynamics", 2000, a.n_ood, rng)
    glt = _gen_set(scene, "glitch", 3000, a.n_ood, rng)
    print(f"[gen] got cal={len(cal)} test={len(test)} dyn={len(dyn)} glt={len(glt)}", flush=True)

    def surp_set(S):
        return [D.proprio_surprise(core, norm, r["obs"], r["act"], P)["total"] for r in S]

    print("[surprise] computing ...", flush=True)
    s_cal, s_test, s_dyn, s_glt = surp_set(cal), surp_set(test), surp_set(dyn), surp_set(glt)
    sc_cal = np.array([D.traj_score(s) for s in s_cal])
    sc_test = np.array([D.traj_score(s) for s in s_test])
    sc_dyn = np.array([D.traj_score(s) for s in s_dyn])
    sc_glt = np.array([D.traj_score(s) for s in s_glt])

    report = {"n": {"cal": len(cal), "test": len(test), "dyn": len(dyn), "glt": len(glt)},
              "WARM_s": D.WARM / 30.0, "steps_s": STEPS / 30.0,
              "score": "mean post-warm one-step surprise (per-traj); per-step surprise (latency)",
              "mass_scale_dynamics": 0.65, "glitch_p": 0.10}
    perstep_cal = D.perstep_pool(s_cal)                              # nominal settled per-step surprise pool
    perstep_test = D.perstep_pool(s_test)
    for alpha in (0.05, 0.10):
        tau = D.conformal_threshold(sc_cal, alpha)                  # per-TRAJECTORY threshold (classification)
        tau_step = D.conformal_threshold(perstep_cal, alpha)        # per-STEP threshold (latency / coverage)
        # per-trajectory InD/OOD classification: InD = nominal-test, OOD = dyn + glt
        fp = int((sc_test > tau).sum())                              # nominal-test wrongly flagged
        tp_dyn = int((sc_dyn > tau).sum()); tp_glt = int((sc_glt > tau).sum())
        n_ind, n_ood = len(sc_test), len(sc_dyn) + len(sc_glt)
        tp = tp_dyn + tp_glt; tn = n_ind - fp
        acc = (tp + tn) / (n_ind + n_ood)
        lat_dyn = [D.detection_latency(s, tau_step, r["t_star"]) for s, r in zip(s_dyn, dyn)]
        lat_glt = [D.detection_latency(s, tau_step, r["t_star"]) for s, r in zip(s_glt, glt)]
        fin = lambda L: [x for x in L if np.isfinite(x)]            # noqa: E731
        report[f"alpha_{alpha}"] = {
            "tau_traj": tau, "tau_step": tau_step,
            "accuracy": acc,
            "confusion": {"TP": tp, "TN": tn, "FP": fp, "FN": n_ood - tp},
            "TPR_dynamics": tp_dyn / len(sc_dyn), "TPR_glitch": tp_glt / len(sc_glt),
            "FPR_nominal_test": fp / n_ind,
            "latency_s_dynamics": {"median": float(np.median(fin(lat_dyn))) if fin(lat_dyn) else None,
                                   "missed": int(np.isinf(lat_dyn).sum())},
            "latency_s_glitch": {"median": float(np.median(fin(lat_glt))) if fin(lat_glt) else None,
                                 "missed": int(np.isinf(lat_glt).sum())},
            "coverage_perstep_FPR_nominal_test": float((perstep_test > tau_step).mean()),
        }
    report["AUC"] = {"dynamics": D.roc_auc(sc_test, sc_dyn), "glitch": D.roc_auc(sc_test, sc_glt),
                     "combined": D.roc_auc(sc_test, np.concatenate([sc_dyn, sc_glt]))}

    outp = pathlib.Path(a.out); outp.mkdir(parents=True, exist_ok=True)
    (outp / "proprio_report.json").write_text(json.dumps(report, indent=2))
    np.savez(outp / "proprio_scores.npz", cal=sc_cal, test=sc_test, dyn=sc_dyn, glt=sc_glt,
             tstar_dyn=[r["t_star"] for r in dyn], tstar_glt=[r["t_star"] for r in glt])
    _plot(outp, s_cal, s_test, s_dyn, s_glt, sc_cal, sc_test, sc_dyn, sc_glt, dyn, glt, report, D)
    print(json.dumps(report, indent=2), flush=True)
    print(f"[done] wrote {outp}/proprio_report.json + proprio_detect.png", flush=True)


def _plot(outp, s_cal, s_test, s_dyn, s_glt, sc_cal, sc_test, sc_dyn, sc_glt, dyn, glt, report, D):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(STEPS) / 30.0
    tau = report["alpha_0.1"]["tau_traj"]
    tau_step = report["alpha_0.1"]["tau_step"]
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    # row 0: example surprise traces with WARM, tau, t*
    for col, (name, S, SS, col_c) in enumerate([("nominal (test)", s_test, None, "#4c72b0"),
                                                ("dynamics", s_dyn, dyn, "#dd8452"),
                                                ("glitch", s_glt, glt, "#55a868")]):
        a0 = ax[0][col]
        for k in range(min(4, len(S))):
            a0.plot(t, S[k], lw=0.9, alpha=0.7, color=col_c)
        a0.axhline(tau_step, color="k", ls="-", lw=1.3, label=f"per-step τ (α=.1)={tau_step:.2f}")
        a0.axvspan(0, D.WARM / 30.0, color="gray", alpha=0.15, label="warm-up (discarded)")
        if SS is not None:
            for k in range(min(4, len(SS))):
                a0.axvline(SS[k]["t_star"] / 30.0, color="r", ls="--", lw=0.8, alpha=0.6)
        a0.set_title(f"{name}: proprio surprise", fontsize=11)
        a0.set_xlabel("t (s)"); a0.set_ylabel("surprise"); a0.grid(alpha=0.3); a0.legend(fontsize=7)
        if col == 2:
            a0.set_yscale("log")

    # (1,0) per-trajectory score strip by class
    a0 = ax[1][0]
    for i, (nm, sc, c) in enumerate([("nom-cal", sc_cal, "#999"), ("nom-test", sc_test, "#4c72b0"),
                                     ("dynamics", sc_dyn, "#dd8452"), ("glitch", sc_glt, "#55a868")]):
        x = i + 0.08 * np.random.default_rng(i).standard_normal(len(sc))
        a0.scatter(x, sc, s=22, color=c, alpha=0.8)
    for alpha, ls in [(0.05, "--"), (0.10, "-")]:
        a0.axhline(report[f"alpha_{alpha}"]["tau_traj"], color="k", ls=ls, lw=1.2, label=f"τ_traj(α={alpha})")
    a0.set_yscale("log"); a0.set_xticks(range(4)); a0.set_xticklabels(["nom-cal", "nom-test", "dyn", "glitch"])
    a0.set_title("per-trajectory score (max settled surprise)", fontsize=11)
    a0.set_ylabel("score (log)"); a0.grid(alpha=0.3); a0.legend(fontsize=8)

    # (1,1) ROC
    a0 = ax[1][1]
    for nm, pos, c in [("dynamics", sc_dyn, "#dd8452"), ("glitch", sc_glt, "#55a868")]:
        ths = np.unique(np.concatenate([sc_test, pos]))
        tpr = [np.mean(pos > th) for th in ths]; fpr = [np.mean(sc_test > th) for th in ths]
        order = np.argsort(fpr)
        a0.plot(np.array(fpr)[order], np.array(tpr)[order], "-o", ms=3, color=c,
                label=f"{nm} (AUC={report['AUC'][nm]:.3f})")
    a0.plot([0, 1], [0, 1], "k:", lw=0.8)
    a0.set_title("ROC: nominal-test vs OOD", fontsize=11); a0.set_xlabel("false-positive rate")
    a0.set_ylabel("true-positive rate"); a0.grid(alpha=0.3); a0.legend(fontsize=8)

    # (1,2) latency hist
    a0 = ax[1][2]
    for nm, S, SS, c in [("dynamics", s_dyn, dyn, "#dd8452"), ("glitch", s_glt, glt, "#55a868")]:
        lat = [D.detection_latency(s, tau_step, r["t_star"]) for s, r in zip(S, SS)]
        lat = [x for x in lat if np.isfinite(x)]
        a0.hist(lat, bins=np.linspace(0, 3, 16), alpha=0.6, color=c, label=f"{nm} (n={len(lat)})")
    a0.set_title("detection latency after t* (α=.1)", fontsize=11); a0.set_xlabel("latency (s)")
    a0.set_ylabel("count"); a0.grid(alpha=0.3); a0.legend(fontsize=8)

    acc = report["alpha_0.1"]["accuracy"]
    fig.suptitle(f"Proprio-WM conformal OOD detection — accuracy {acc:.2f} (α=.1), "
                 f"AUC dyn {report['AUC']['dynamics']:.3f} / glitch {report['AUC']['glitch']:.3f}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outp / "proprio_detect.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
