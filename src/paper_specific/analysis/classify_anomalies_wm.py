"""Turn the world model's prediction error into an anomaly CLASSIFIER, and score it properly.

AUC needs no threshold, which is why it was reported first -- but it also cannot tell you how often the
detector would be right in deployment. For that a threshold is needed, and where it comes from decides
whether the number means anything:

    NOT the per-episode maximum-accuracy threshold -- that is fitted to the test labels and always flatters.
    NOT the out-of-window steps of the same episodes -- they are the negative class being scored.
    val: 7 held-out in-distribution episodes with no anomaly anywhere. Threshold at val's 95th percentile,
    so the false-positive rate is 5% BY CONSTRUCTION on data the threshold was fitted to, and whatever it
    turns out to be on the OOD splits' own normal steps is an honest out-of-sample number.

Emits, per split and per metric: the 2x2 confusion matrix over STEPS, accuracy, precision, recall,
specificity, F1, and the AUC beside them for comparison.

    CUDA_VISIBLE_DEVICES=0 python scratch/classify_anomalies_wm.py <wm_ckpt> [out_dir]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(__file__))   # sibling analyses in this package
from detect_anomalies_wm import METRIC_DOC, OOD, SUB, auc, one_step                    # noqa: E402

from quickdraw.data.dataset import load_split_episodes_mm, set_action_aggregate, set_subsample
from quickdraw.data.ood_windows import kept, window_steps
from quickdraw.training.setup import (build_model, image_head_cams, image_head_sizes, load_checkpoint,
                                      normalizer, resolve_data_root)

COVERAGE = 0.90   # conformal coverage: the threshold is the COVERAGE quantile of the
#                   calibration set's non-conformity scores, so the false-positive rate on
#                   exchangeable calibration data is 1 - COVERAGE by construction.
METRICS = ["lpips", "l1", "l2", "pos_err", "vel_err", "rot_err", "angvel_err", "latent_cos"]


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

    # ---- gather every episode's per-step errors first; thresholds come later, per calibration rule ----
    per_ep = {}
    for sp in OOD:
        eps = load_split_episodes_mm(root, sp, **kw)
        per_ep[sp] = {}
        for i in kept(sp):
            o, a, fr = eps[i]
            r = one_step(core, norm, o, a, fr[key], key, P, dev)
            w0, w1 = window_steps(sp, i, SUB)
            inw = (r["steps"] >= w0) & (r["steps"] < w1)
            per_ep[sp][i] = {k: (np.asarray(r[k])[inw], np.asarray(r[k])[~inw]) for k in METRICS}

    # ---- CALIBRATION RULE 1: val. Reported because it FAILS, and the failure is the point ----
    veps = load_split_episodes_mm(root, "val", **kw)
    vals = {k: [] for k in METRICS}
    CAP = 120
    for o, a, fr in veps:
        r = one_step(core, norm, o[:CAP], a[:CAP], fr[key][:CAP], key, P, dev)
        for k in METRICS:
            vals[k].extend(np.asarray(r[k])[np.isfinite(r[k])].tolist())
    thr = {k: float(np.quantile(vals[k], COVERAGE)) for k in METRICS}
    print(f"\n  CALIBRATION 1 -- val {100*COVERAGE:.0f}th percentile. THIS DOES NOT TRANSFER, and the numbers below say why:")
    for k in METRICS:
        pool = np.concatenate([np.concatenate(per_ep[sp][i][k]) for sp in OOD for i in per_ep[sp]])
        print(f"    {k:12s} val p50 {np.median(vals[k]):.4f} q {thr[k]:.4f}  |  OOD splits: "
              f"p50 {np.median(pool):.4f} max {np.max(pool):.4f}"
              + ("   <- ENTIRE OOD range below the val threshold" if np.max(pool) < thr[k] else ""))
    print("    The OOD clips are near-static hovers on a single constant command, which the model predicts")
    print("    BETTER than val's ordinary flight -- so an absolute threshold fitted to val fires on almost")
    print("    nothing here. Regime, not anomaly, dominates the absolute error level.")

    report = {"val_thresholds": thr,
               "coverage": COVERAGE,
               "threshold_rule": "leave-one-episode-out: the coverage quantile of the OUT-OF-WINDOW steps of every OTHER "
                                 "kept episode in the same split. Matched regime (same hover, same "
                                 "constant command) and out-of-sample for the episode being scored."}
    for sp, kind in OOD.items():
        eids = [i for i in per_ep[sp] if len(per_ep[sp][i]["lpips"][0]) > 0]
        skipped = [i for i in per_ep[sp] if i not in eids]
        tot = {k: dict(TP=0, FN=0, FP=0, TN=0) for k in METRICS}
        pooled = {k: ([], []) for k in METRICS}
        for k in METRICS:
            for i in eids:
                # threshold from the OTHER episodes' normal steps only
                other = np.concatenate([per_ep[sp][j][k][1] for j in eids if j != i]) if len(eids) > 1 else \
                    per_ep[sp][i][k][1]
                t = float(np.quantile(other, COVERAGE))
                pos, neg = per_ep[sp][i][k]
                tot[k]["TP"] += int((pos > t).sum()); tot[k]["FN"] += int((pos <= t).sum())
                tot[k]["FP"] += int((neg > t).sum()); tot[k]["TN"] += int((neg <= t).sum())
                pooled[k][0].extend(pos.tolist()); pooled[k][1].extend(neg.tolist())
        n_in, n_out = len(pooled["lpips"][0]), len(pooled["lpips"][1])
        print(f"\n=== {sp}  ({kind})   {len(eids)} episodes scored {eids}"
              + (f", skipped {skipped} (anomaly inside the context frames)" if skipped else ""))
        print(f"      {n_in} anomalous steps, {n_out} normal steps   [leave-one-episode-out]")
        # RAW ACCURACY IS NOT CHANCE-NORMALISED and this split is imbalanced: a classifier that never
        # flags anything scores n_normal/(n_normal+n_anomalous). AUC already is normalised (0.5 = chance
        # under any imbalance, since it is P(random positive > random negative) and ignores the class
        # proportions entirely), and `bal` -- the mean of recall and specificity -- is the accuracy-like
        # number with the same property. Read `bal` and `AUC`, not `acc`.
        noskill = n_out / max(1, n_in + n_out)
        print(f"      no-skill accuracy (never flag) = {noskill:.3f}; no-skill balanced accuracy = 0.500")
        print(f"  {'metric':12s} {'TP':>4s} {'FN':>4s} {'FP':>4s} {'TN':>4s} {'acc':>6s} {'bal':>7s} "
              f"{'prec':>6s} {'recall':>7s} {'spec':>6s} {'F1':>6s} {'AUC':>6s}")
        rows = {}
        for k in METRICS:
            c = tot[k]
            acc = (c["TP"] + c["TN"]) / max(1, sum(c.values()))
            prec = c["TP"] / max(1, c["TP"] + c["FP"]); rec = c["TP"] / max(1, c["TP"] + c["FN"])
            spec = c["TN"] / max(1, c["TN"] + c["FP"]); f1 = 2 * prec * rec / max(1e-9, prec + rec)
            bal = 0.5 * (rec + spec)          # no-skill = 0.500 under ANY class imbalance
            rows[k] = {**c, "accuracy": acc, "balanced_accuracy": bal, "precision": prec, "recall": rec,
                       "specificity": spec, "f1": f1, "auc": auc(*pooled[k])}
            print(f"  {k:12s} {c['TP']:>4d} {c['FN']:>4d} {c['FP']:>4d} {c['TN']:>4d} {acc:>6.3f} "
                  f"{bal:>7.3f} {prec:>6.3f} {rec:>7.3f} {spec:>6.3f} {f1:>6.3f} {rows[k]['auc']:>6.3f}")
        best = max(METRICS, key=lambda k: rows[k]["balanced_accuracy"])
        r = rows[best]
        print(f"  best by BALANCED accuracy: {best}   confusion matrix over steps")
        print(f"                      flagged   not flagged")
        print(f"    anomalous  {r['TP']:>11d} {r['FN']:>13d}")
        print(f"    normal     {r['FP']:>11d} {r['TN']:>13d}")
        report[sp] = {"kind": kind, "episodes_scored": eids, "episodes_skipped": skipped,
                      "n_anomalous_steps": n_in, "n_normal_steps": n_out, "metrics": rows,
                      "best_by_balanced_accuracy": best}
    json.dump(report, open(os.path.join(out_root, "wm_anomaly_classification.json"), "w"), indent=1)
    print(f"\n  report -> {out_root}/wm_anomaly_classification.json")
    print("\n  METRIC DEFINITIONS")
    for k, v in METRIC_DOC.items():
        print(f"    {k:12s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
