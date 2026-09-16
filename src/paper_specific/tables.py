"""Emit the paper's result tables as standalone .tex files the paper \\input's.

EVERY TABLE CARRIES A BASELINE, which is a rule for this paper rather than a convention: a row is only
interpretable next to something that differs from it in one named way. Where no competing method exists in
the literature-with-our-data sense, the baseline is one of our own ablations (the setting that was tried
and rejected) or a bound (the autoencoder floor, which no dynamics model can beat).

    python -m paper_specific.tables <paper_repo>
"""
from __future__ import annotations

import os
import sys
import numpy as np

from .harvest import config, fmt, metrics, suffix, contains

LOGS = "logs"

# ---- the world-model ablation: same data, same autoencoder, one setting changed per row ----------
# Row labels say what the setting IS, not what the run was called: `vl128 base` named a file, not a
# configuration, and stride 5 with summed actions is the thing a reader needs to know.
WM_ROWS = [
    # PLAIN TEXT, not codes: a reader should not have to decode "4 / concat" to know what a row is.
    # Verified against each run's own config.json -- the stride-1 run uses SUMMED actions, which an
    # earlier version of this table got wrong.
    # THE INHERITED-RECIPE ROW AND THE STRIDE-1 ROW WERE REMOVED at the author's ask. With the reference
    # row gone the remaining labels can no longer be deltas from it, so each one states its own stride,
    # its own action aggregation and any change to the denoiser. The stride-1 measurement (0.0783 at +1,
    # 0.3155 at +128) now lives only in the results prose.
    ("train_world_model_2026_09_07_03_45_54_starling2_heavy",
     "Every fifth frame, summed actions, deeper denoiser", "rejected"),
    ("train_world_model_2026_09_12_08_06_49_s2_sub4_concat_deriv",
     "Every fourth frame, concatenated action increments", "rejected"),
    ("train_world_model_2026_09_11_03_17_22_s2_sub3_concat",
     "Every third frame, concatenated actions", ""),
    ("train_world_model_2026_09_08_22_04_09_s2_sub4",
     "Every fourth frame, summed actions", ""),
    ("train_world_model_2026_09_11_03_17_47_s2_sub4_concat",
     r"Every fourth frame, concatenated actions$^{*}$", "kept"),
]
# ---- the Action Model: the 2x2 of context pooling x target space, plus the chunk and pit_delta arms --
AH_ROWS = [
    # EVERY CONFIGURATION WRITTEN OUT IN FULL, at the author's ask: no indented "longer chunk" rows that
    # only parse by reading upward. Each label states its context, its target transform and its chunk.
    # THE RAW-TARGET ROW WAS REMOVED at the author's ask. It was the only measurement of what the
    # percentile transform buys -- rest AUC 0.500 against 0.995, W_1 0.1023 against 0.0445 -- so those
    # two comparisons now live in the results prose instead of in the table.
    ("train_action_2026_09_13_21_57_41_s2_ah_pit",
     r"Pooled context, raw target, chunk $8$"),
    ("train_action_2026_09_13_23_42_49_s2_ah_pit",
     r"Grouped context, raw target, chunk $8$"),
    ("train_action_2026_09_14_04_41_17_s2_ah_chunk32_full",
     r"Grouped context, raw target, chunk $32^{*}$"),
    ("train_action_model_2026_09_14_22_28_22_s2_ah_chunk32_pitdelta",
     r"Grouped context, delta target, chunk $32$"),
]


def wm_table() -> str:
    hs = [1, 8, 32, 128]
    L = [r"\begin{table}[t]", r"  \centering", r"  \footnotesize",
         r"  \setlength{\tabcolsep}{3pt}",
         r"  \caption{\textbf{World Model.} Open-loop prediction on held-out \dataname flight, by "
         r"horizon: how far the action-conditioned observation prediction holds up. Each row names its "
         r"own frame stride and how the commands skipped between kept frames are aggregated; the best "
         r"per column is bold. The autoencoder floor re-encodes and decodes the true frame, so it "
         r"bounds what any dynamics model on this tokenizer can reach. "
         r"$^{\dagger}$The autoencoder floor is not a model: it encodes and decodes the true frame, so "
         r"it is the same at every horizon and no dynamics model on this tokenizer can beat it. "
         r"$^{*}$The configuration \modelname{} uses.}",
         r"  \label{tab:longhorizon}",
         r"  \begin{tabular}{@{}p{0.40\columnwidth}cccc@{}}", r"    \toprule",
         r"    & \multicolumn{4}{c}{LPIPS $\downarrow$ at open-loop horizon} \\",
         r"    \cmidrule(lr){2-5}",
         r"    Configuration & $+1$ & $+8$ & $+32$ & $+128$ \\", r"    \midrule"]
    # THE FLOOR IS THE DEPLOYED MODEL'S OWN, not a mean over the rows. Averaging it moved the number
    # every time a row was added or removed (0.0461 -> 0.0480 when two rows went), which is wrong for a
    # quantity that is meant to be a property of the tokenizer: the rows run at different frame strides
    # and so encode different frames.
    floor, rows = None, []
    for run, name, verdict in WM_ROWS:
        d = metrics(os.path.join(LOGS, run))
        if not d:
            continue
        vals = [suffix(d, f"open_loop/image/lpips/@+{h}") for h in hs]
        if verdict == "kept":
            floor = suffix(d, "eval_ae_floor/image/lpips_mean")
        rows.append((name, vals))
    assert floor is not None, "no WM_ROWS entry marked 'kept' carries an autoencoder floor"
    # BOLD THE BEST IN EACH COLUMN, not our own row: bolding `ours` at a horizon where an ablation wins
    # (frame stride 1 is far better at +1) would assert something the table itself contradicts.
    best = [min((r[1][j] for r in rows if r[1][j] is not None), default=None) for j in range(len(hs))]
    for name, vals in rows:
        cells = " & ".join(
            (r"\textbf{" + fmt(v, 4) + "}") if (v is not None and best[j] is not None and abs(v - best[j]) < 1e-9)
            else fmt(v, 4) for j, v in enumerate(vals))
        L.append(f"    {name} & {cells} \\\\")
    L += [r"    \midrule",
          # A ROW LIKE ANY OTHER, with the same number in every column: the floor does not depend on
          # horizon, and spanning it across the four columns made it look like a different kind of thing.
          r"    Autoencoder floor$^{\dagger}$ & "
          + " & ".join([fmt(floor, 4)] * 4) + r" \\",
          r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


def ah_table() -> str:
    L = [r"\begin{table}[t]", r"  \centering", r"  \footnotesize",
         r"  \setlength{\tabcolsep}{3pt}",
         r"  \caption{\textbf{Action Model performance.} Here we enumerate the model's ability to predict "
         r"what distribution of actions follows from a given context. Both skill columns are \emph{energy "
         r"skill} against a context-blind null --- the recorded chunks shuffled across contexts, so the "
         r"null has the right marginal and the wrong context --- scored as "
         r"$1-\mathrm{ES}/\mathrm{ES}_{\mathrm{null}}$, so $0$ is a model that ignores its context and "
         r"$1$ is perfect. Skill$_{+1}$ scores the first action of the chunk alone, and "
         r"Skill$_{\mathrm{chunk}}$ scores all $K$ actions jointly, which is the harder question because "
         r"it asks one draw to be right about the whole manoeuvre at once. $W_1$ is the distance to the "
         r"recorded action marginal, i.e.\ whether it flies like the data. Rest AUC asks whether the "
         r"model can place mass on a zero valued action. Best per column in bold. $^{*}$The "
         r"configuration \modelname{} uses.}",
         r"  \label{tab:actionhead}",
         # A WRAPPING CONFIGURATION COLUMN: written out in full the labels are too long for one line, and
         # p{} wraps them rather than overflowing the column.
         r"  \begin{tabular}{@{}p{0.40\columnwidth}cccc@{}}", r"    \toprule",
         r"    & Skill$_{+1}$ & Skill$_{\mathrm{chunk}}$ & $W_1$ & Rest AUC \\",
         r"    Configuration & $\uparrow$ & $\uparrow$ & $\downarrow$ & $\uparrow$ \\", r"    \midrule"]
    rows = []
    for run, lab in AH_ROWS:
        d = metrics(os.path.join(LOGS, run))
        if not d:
            continue
        rows.append((lab, [contains(d, "lead_00/energy_skill"), contains(d, "energy_skill_vs_blind"),
                           contains(d, "w1_mean"), contains(d, "rest_auc")]))
    best = [None] * 4
    for k, lo in enumerate((False, False, True, False)):    # W1 is the only lower-is-better column
        vs = [r[1][k] for r in rows if r[1][k] is not None]
        if vs:
            best[k] = min(vs) if lo else max(vs)
    for lab, vals in rows:
        cells = []
        for k, v in enumerate(vals):
            t = "--" if v is None else (f"{v:.3f}" if k != 2 else f"{v:.4f}")
            if v is not None and best[k] is not None and abs(v - best[k]) < 1e-12:
                t = r"\textbf{" + t + "}"
            cells.append(t)
        L.append(f"    {lab} & " + " & ".join(cells) + r" \\")
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


OOD_ROWS = {
    "eval_ood_noodle": [("latent_cos", r"Latent surprise (\textbf{ours})"), ("lpips", "Image LPIPS"),
                        ("l2", "Image RMSE")],
    # LATENT SURPRISE APPEARS IN BOTH BLOCKS, because the obvious question about a two-anomaly claim is
    # whether each channel is specific or just sensitive. It is not specific: on the dynamical anomaly it
    # still reads 72.4, well clear of no-skill, because a pushed drone eventually sees a different room.
    # The proprioceptive channels ARE specific -- on the visual anomaly they sit at no-skill.
    "eval_ood_leafblower": [("pos_err", "Position error"), ("rot_err", "Orientation error"),
                            ("vel_err", "Velocity error"),
                            ("angvel_err", r"Angular velocity error (\textbf{ours})")],
}
OOD_NAME = {"eval_ood_noodle": "Visual anomaly: a pink pool noodle enters the frame", "eval_ood_leafblower": "Dynamical anomaly: an off-camera leaf blower pushes the drone"}


def ood_table(paper: str) -> str:
    import json
    # analyses write their JSON under the repo's own logs/, so read it repo-relative rather than by
    # walking up from the paper directory -- the paper repo lives outside quickdraw and its depth varies.
    j = json.load(open(os.path.join(LOGS, "paper_icra_2027", "wm_anomaly_classification.json")))
    cov = j.get("coverage", 0.9)
    L = [r"\begin{table}[t]", r"  \centering", r"  \small",
         r"  \setlength{\tabcolsep}{3pt}",
         r"  \caption{\textbf{Out-of-distribution detection}, from one-step prediction error, scored per "
         # ESCAPE THE PERCENT SIGN. `{cov:.0%}` emits a bare % and LaTeX comments out the rest of the
         # caption, which ends as "Runaway argument? ... File ended while scanning use of \caption".
         rf"timestep against the reviewed anomaly windows. The threshold is the {100 * cov:.0f}\% quantile "
         r"of the non-conformity scores of the \emph{other} episodes' in-distribution steps, so nominal "
         r"accuracy is calibrated to $\approx$" + rf"{100 * cov:.0f}\%" + r" by construction and failure "
         r"accuracy is what the detector buys. Weighted accuracy is the mean of the two, whose no-skill "
         r"value is $50\%$ under any class imbalance. Within each block the first row is the channel we "
         r"use and the rest are the same pipeline read through a different error channel, which is what "
         r"makes them controls. $\uparrow$ higher is better.}",
         r"  \label{tab:ood}", r"  \begin{tabular}{lccc}", r"    \toprule",
         r"    & Nominal & Failure & Weighted \\",
         r"    Scoring mechanism & acc.\ (\%) $\uparrow$ & acc.\ (\%) $\uparrow$ & acc.\ (\%) $\uparrow$ \\"]
    for split, rows in OOD_ROWS.items():
        L += [r"    \midrule", r"    \multicolumn{4}{c}{" + OOD_NAME[split] + r"} \\", r"    \midrule"]
        m = j[split]["metrics"]
        for key, label in rows:
            r = m[key]
            cells = [f"{100 * r[k]:.1f}" for k in ("specificity", "recall", "balanced_accuracy")]
            if r"\textbf" in label:
                cells = [r"\textbf{" + c + "}" for c in cells]
            L.append(f"    {label} & " + " & ".join(cells) + r" \\")
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"





# ---- steering: per-request, and then the proposal comparison ---------------------------------------
# The direction block: one run per candidate source, all at the same 16 contexts x 8 requests, same
# Reward Model, same objective, same commit -- the candidate source is the ONLY thing that differs.
# `gauss` is the control: MPPI's historic white noise, which is what the planner drew from before the
# prior was wired into it.
STEER_RUNS = {
    "gauss": "logs/eval_steer_2026_09_15_05_06_35_phys16_gauss",
    "data": "logs/eval_steer_2026_09_14_22_08_48_phys16_data",
    "prior": "logs/eval_steer_2026_09_14_22_08_46_phys16_prior",
    "pitdelta": "logs/eval_steer_2026_09_15_01_12_15_phys16_pitdelta",
}
# ORDER, at the author's ask: the reference first, then the noise floor, then the two learned priors
# with the deployed one last.
# "AM" is dropped from every header -- the caption already says the Action Model is the only thing that
# differs -- and the reference is labelled a CEILING, because it is not a method anyone can deploy: it
# draws chunks that were actually flown.
COLS = [("data", r"\makecell{Data Retrieval\\(ceiling)$^{\ddagger}$}"), ("gauss", "Gaussian"),
        ("pitdelta", r"\makecell{Learned\\$\Delta$}"),
        ("prior", r"\makecell{Learned\\Raw (\textbf{ours})}")]
# THE BASELINE IS NEVER BOLDED AS THE WINNER. Retrieval is a reference, not a competitor: it is bounded
# by what the corpus happens to contain, so calling it "best" asserts a target none of the priors could
# reach by construction. Bolding therefore runs over the generative columns only.
BOLD_COLS = [m for m, _ in COLS if m != "data"]
# The place block needs DECODED VIDEO for the labeller, which is why it used to run at 4 contexts while
# the motion block ran at 16: the video suites were built at 4. These are the 15-context re-runs, 16
# object/region requests each, 224 plans per arm (14 contexts survive the horizon at 15 requested).
VLM_RUNS = {
    "gauss": "logs/eval_steer_2026_09_15_22_20_22_loc15_gauss",
    "prior": "logs/eval_steer_2026_09_15_22_42_19_loc15_prior",
    "pitdelta": "logs/eval_steer_2026_09_15_22_42_20_loc15_pitdelta",
    "data": "logs/eval_steer_2026_09_15_22_20_23_loc15_data",
}
UNIT = {"yaw": r"$^\circ$", "altitude": "m", "forward": "m", "lateral": "m"}
PILOT = {"yaw": 471.75, "altitude": 0.58, "forward": 23.63, "lateral": 19.92}   # analysis/steer_physical
REC_DA = 0.0541          # recorded step-to-step |da| at this rate, same fold (check_action_continuity)
# The continuity arms. Everything is held fixed except how a chunk is made to continue the one before it,
# with the two candidate sources that are not the prior kept as references for the smoothness column.
CONT_ROWS = [
    ("logs/eval_steer_2026_09_15_05_06_35_phys16_gauss", "Gaussian prior"),
    ("logs/eval_steer_2026_09_14_22_08_48_phys16_data", "Data prior"),
    ("logs/eval_steer_2026_09_15_05_33_54_phys16_noguid", r"\quad learned prior, no continuity"),
    ("logs/eval_steer_2026_09_15_05_58_34_phys16_xfade", r"\quad learned prior, crossfade"),
    ("logs/eval_steer_2026_09_14_22_08_46_phys16_prior", r"\quad learned prior, prefix guidance (\textbf{ours})"),
    ("logs/eval_steer_2026_09_15_01_12_15_phys16_pitdelta", r"\quad learned PIT-$\Delta$ prior"),
]


def _phys(run):
    """{request: signed achieved motion along the axis it names} from the plans' saved proprio."""
    import glob
    import json as _j

    import numpy as np
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "analysis"))
    from steer_physical import WANTS, physical
    out = {}
    for f in sorted(glob.glob(os.path.join(run, "logs", "epoch_*", "eval_steer", "plans", "*", "*",
                                           "plan.json"))):
        d = _j.load(open(f))
        if d["request"] not in WANTS:
            continue
        key, sgn = WANTS[d["request"]]
        v = physical(np.load(os.path.join(os.path.dirname(f), "proprio.npy")))[key] * sgn
        out.setdefault(d["request"], []).append(v)
    return {k: (sum(v) / len(v), len(v), v) for k, v in out.items()}   # (mean, n, per-context values)


def _jerk(run):
    """(inside a chunk, at the seam) mean |da|, as a multiple of the recorded step-to-step change.

    The seam spacing is `commit`, not the chunk: a plan re-draws every `commit` steps, so at commit 16 of
    a 32-step chunk the seams are at 16, 32, 48 ... Reading the chunk instead counts every other seam as
    interior and dilutes both columns (see analysis/check_action_continuity.py, which this mirrors)."""
    import glob
    import json as _j

    import numpy as np
    fs = sorted(glob.glob(os.path.join(run, "logs", "epoch_*", "eval_steer", "plans", "*", "*",
                                       "actions.npy")))
    if not fs:
        return None
    d = _j.load(open(os.path.join(os.path.dirname(fs[0]), "plan.json")))
    if d.get("commit"):
        K = int(d["commit"])
    else:                                          # runs from before plan.json recorded it: read the header
        import re as _re
        hdr = open(os.path.join(run, "progress.log")).readline()
        mm = _re.search(r"commit (\d+)", hdr)
        assert mm, f"no commit in plan.json or the header of {run}"
        K = int(mm.group(1))
    seam, inside = [], []
    for f in fs:
        a = np.load(f)
        fold = a.reshape(len(a), -1, 4).mean(axis=1)
        dd = np.abs(np.diff(fold, axis=0))
        m = (np.arange(1, len(fold)) % K) == 0
        seam.append(dd[m]); inside.append(dd[~m])
    return float(np.concatenate(inside).mean()), float(np.concatenate(seam).mean())


def _vlm(run):
    """{request: (hit, base, n)} for the object/region requests, from the VLM labels.

    DOES THE TARGET APPEAR IN THE IMAGINED SEQUENCE, which is the question the author asked for and not
    the one the training labels answer. `vlm_object_check.json` reuses the reward head's own schema: the
    ONE most prominent object, and the one region in front of the drone at the END of the clip. Scored
    that way a plan that flies to the table with the ladder also in frame counts as a miss, and the
    learned prior read 0 of 4 on requests its own video satisfies. `vlm_object_appears.json`
    (analysis/steer_vlm_appears.py) asks the same model for EVERY listed object visible at any point and
    every region faced at any point, and is preferred where it exists.

    The base rate is what makes either version readable, and it is why this is reported next to the hit:
    under the permissive question the labeller returns 5-7 of the 10 objects per clip, so a target turns
    up ~0.6 of the time when something else was asked for."""
    import json as _j
    p = os.path.join(run, "vlm_object_appears.json")
    multi = os.path.exists(p)
    if not multi:
        p = os.path.join(run, "vlm_object_check.json")
        if not os.path.exists(p):
            return {}
    rows = [(r["request"], r["label"]) for r in _j.load(open(p)) if r.get("label")]
    import yaml
    ic = yaml.safe_load(open("conf/interpret/starling.yaml"))
    OBJ, REG = list(ic["factors"]["object_in_view"]["buckets"]), list(ic["factors"]["facing"]["buckets"])
    fields = (("objects_seen", "regions_faced") if multi else ("object_in_view", "facing"))
    hit = (lambda o, b: b in o) if multi else (lambda o, b: o == b)
    out = {}
    for buckets, field in zip((OBJ, REG), fields):
        sub = [(q, l[field]) for q, l in rows if q in buckets]
        for b in buckets:
            mine = [o for q, o in sub if q == b]
            other = [o for q, o in sub if q != b]
            if mine:
                out[b] = (sum(hit(o, b) for o in mine) / len(mine),
                          (sum(hit(o, b) for o in other) / len(other)) if other else float("nan"),
                          len(mine))
    return out


LOC_ROWS = ("center of room over mats", "floor to ceiling glass wall", "ladder", "mannequin", "table")

# WEIGHTED STEERING ACCURACY, built exactly the way tab:ood's weighted accuracy is: the mean of the
# true-positive and true-negative rates, so no-skill is 50% under any imbalance. It exists because a
# hit rate alone rewards an arm that simply moves a lot, and a location hit rate alone rewards a
# labeller that lists everything; folding in the false-positive rate is what removes both.
MOTION_ROWS = ("rotate left", "rotate right", "climb", "descend",
               "strafe left", "strafe right", "fly forward", "fly backward")


def avg_motion(ph):
    """{arm: mean hits out of 15} over MOTION_ROWS -- the plain average of the cells above it."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "analysis"))
    from steer_physical import WANTS
    out = {}
    for m, _ in COLS:
        f = [float(np.mean([v > 0.05 * PILOT[WANTS[q][0]] for v in ph[m][q][2]]))
             for q in MOTION_ROWS if ph[m].get(q)]
        out[m] = float(np.mean(f)) if f else None
    return out


def avg_locations(vl):
    """{arm: mean hits out of 4} over LOC_ROWS."""
    out = {}
    for m, _ in COLS:
        f = [v[0] for v in (vl.get(m, {}).get(q) for q in LOC_ROWS) if v]
        out[m] = float(np.mean(f)) if f else None
    return out


def wacc_motion(ph):
    """{arm: weighted accuracy} over MOTION_ROWS. The negative class is the OPPOSING request on the same
    axis. _phys stores motion*sgn for the request it was asked under, so for the opposing request a
    false positive -- motion in THIS request's direction past the threshold -- is a stored value below
    -threshold."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "analysis"))
    from steer_physical import WANTS
    opp = {q: next(p for p in MOTION_ROWS if p != q and WANTS[p][0] == WANTS[q][0]) for q in MOTION_ROWS}
    out = {}
    for m, _ in COLS:
        w = []
        for q in MOTION_ROWS:
            thr = 0.05 * PILOT[WANTS[q][0]]
            pos, neg = ph[m].get(q), ph[m].get(opp[q])
            if not pos or not neg:
                continue
            tpr = float(np.mean([v > thr for v in pos[2]]))
            fpr = float(np.mean([v < -thr for v in neg[2]]))
            w.append(0.5 * (tpr + 1.0 - fpr))
        out[m] = 100.0 * float(np.mean(w)) if w else None
    return out


def wacc_locations(vl):
    """{arm: weighted accuracy} over LOC_ROWS, from the VLM hit and base rates."""
    out = {}
    for m, _ in COLS:
        w = [0.5 * (v[0] + 1.0 - v[1]) for v in (vl.get(m, {}).get(q) for q in LOC_ROWS)
             if v and np.isfinite(v[1])]
        out[m] = 100.0 * float(np.mean(w)) if w else None
    return out


def _avg_row(wa, what):
    best = max((v for m, v in wa.items() if v is not None and m in BOLD_COLS), default=None)
    cells = ["--" if wa[m] is None else
             ((r"\textbf{" + f"{100 * wa[m]:.0f}" + r"}\%")
              if (m in BOLD_COLS and best and abs(wa[m] - best) < 1e-9)
              else f"{100 * wa[m]:.0f}\%") for m, _ in COLS]
    return r"    \midrule" + "\n" + f"    Mean over {what} " + r"$\uparrow$ & " \
        + " & ".join(cells) + r" \\"


def steer_table(paper: str) -> str:
    """Per-request steering as fractions of contexts, plus the aggregates the continuity table used to
    carry on its own -- the author asked for one table, since the second was mostly the same comparison
    summarised."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "analysis"))
    from steer_physical import WANTS
    ph = {k: _phys(v) for k, v in STEER_RUNS.items()}
    vl = {k: _vlm(v) for k, v in VLM_RUNS.items()}
    nc = len(COLS)
    L = [r"\begin{table*}[t]", r"  \centering", r"  \small",
         r"  \caption{\textbf{Language steering.} The AM is the only thing that differs "
         r"between columns. Every cell is the fraction of starting contexts that met the request: for a "
         r"motion primitive, that the imagined trajectory moved along the axis the words name by more "
         r"than $5\%$ of what a pilot covers in the same $34$\,s, read off the imagined proprioception "
         r"and independent of the reward the planner maximised; for a location, that a VLM asked to list "
         r"every object visible and every region faced \emph{at any point} in the imagined video named "
         r"it, as a percentage of the contexts tried. The last row of each block is the plain mean of the "
         r"cells above it. Bold marks the best \emph{generative} arm, which is why the reference column "
         r"is never bold: it is bounded by what the corpus happens to contain, so calling it best would "
         r"assert a target none of the priors could reach by construction. "
         r"$^{\ddagger}$Data Retrieval is the \textbf{baseline}, and the informative one: its "
         r"candidates are real recorded chunks, so it is the best a fixed planner and a fixed reward can "
         r"do by searching over flight that actually happened. It is blind to the request -- the reward "
         r"alone does the steering -- and a learned prior earns its place only by beating it. What a "
         r"learned prior can offer is not fidelity but reach: it can propose a motion the corpus does "
         r"not contain, and retrieval never can. $^{\S}$The corpus contains no backward flight at all, "
         r"and no arm gets more than one context out of fifteen -- a motion primitive absent from the "
         r"data is not reachable by steering, however the candidates are drawn. Motion primitives are "
         r"scored over $15$ starting contexts and locations over $14$.}",
         r"  \label{tab:planningandcontrol}", r"  \begin{tabular}{l" + "c" * nc + "}", r"    \toprule",
         r"    \diagbox[width=0.19\textwidth, height=2.1\line]{Request}{Action Model} & "
         + " & ".join(lab for _, lab in COLS) + r" \\", r"    \midrule",
         r"    \multicolumn{" + str(1 + nc) + r"}{c}{Motion primitives} \\", r"    \midrule"]
    hits_all = {m: [] for m, _ in COLS}
    frac_all = {m: [] for m, _ in COLS}
    for q in ("rotate left", "rotate right", "climb", "descend", "strafe left", "strafe right",
              "fly forward", "fly backward"):
        key = WANTS[q][0]
        vals = []
        for m, _ in COLS:
            v = ph[m].get(q)
            if v is None:
                vals.append(None); continue
            hits = sum(x > 0.05 * PILOT[key] for x in v[2])
            hits_all[m].append(v[0] > 0.05 * PILOT[key])   # the MEAN, as in the original
            frac_all[m].append(sum(v[2]) / len(v[2]) / PILOT[key])
            vals.append((hits, len(v[2])))
        best = max((v[0] / v[1] for (m, _), v in zip(COLS, vals) if v and m in BOLD_COLS), default=None)
        cells = ["--" if v is None else
                 ((r"\textbf{" + f"{100 * v[0] / v[1]:.0f}" + r"}\%")
                  if (m in BOLD_COLS and best and abs(v[0] / v[1] - best) < 1e-9)
                  else f"{100 * v[0] / v[1]:.0f}\%") for (m, _), v in zip(COLS, vals)]
        nm = q + (r"$^{\S}$" if q == "fly backward" else "")
        L.append(f"    ``{nm}\'\' & " + " & ".join(cells) + r" \\")
    # THE AGGREGATE ROWS ARE GONE, at the author's ask: obeyed, motion against a pilot and the two
    # jerk multiples summarised the per-request cells above them and a continuity comparison this
    # table no longer makes. hits_all/frac_all stay accumulated -- the prose quotes them.
    L += [_avg_row(avg_motion(ph), "motion primitives"),
          r"    \midrule", r"    \multicolumn{" + str(1 + nc) +
          r"}{c}{Locations} \\", r"    \midrule"]
    for q in LOC_ROWS:
        vals = []
        for m, _ in COLS:
            v = vl.get(m, {}).get(q)
            vals.append(None if v is None else (round(v[0] * int(v[2])), int(v[2])))
        best = max((v[0] / v[1] for (m, _), v in zip(COLS, vals) if v and m in BOLD_COLS), default=None)
        cells = ["--" if v is None else
                 ((r"\textbf{" + f"{100 * v[0] / v[1]:.0f}" + r"}\%")
                  if (m in BOLD_COLS and best and abs(v[0] / v[1] - best) < 1e-9)
                  else f"{100 * v[0] / v[1]:.0f}\%") for (m, _), v in zip(COLS, vals)]
        L.append(f"    ``{q}\'\' & " + " & ".join(cells) + r" \\")
    L += [_avg_row(avg_locations(vl), "locations"),
          r"    \bottomrule", r"  \end{tabular}", r"\end{table*}"]
    # THE FALSE-POSITIVE-AWARE VERSION, printed rather than tabulated. The author wants the plain mean in
    # the table; this stays reproducible because the results prose quotes it, and it is the number that
    # shows the gaussian arm is at no-skill once the opposing request is counted as a negative.
    wm_, wl_ = wacc_motion(ph), wacc_locations(vl)
    print("  weighted steering accuracy (%, no-skill 50) -- quoted in the results prose:")
    for m, lab in COLS:
        print(f"    {lab[:30]:32s} motion {wm_[m]:5.1f}   locations {wl_[m]:5.1f}")
    return "\n".join(L) + "\n"


def cont_table(paper: str) -> str:
    """Obeyed / motion / smoothness per arm. The one table where the continuity mechanism varies."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "analysis"))
    from steer_physical import WANTS
    L = [r"\begin{table*}[t]", r"  \centering", r"  \small",
         r"  \caption{The candidate source and the continuity mechanism, over the same $8$ requests "
         r"$\times$ $16$ contexts, same Reward Model, same objective, same commit. `Obeyed' counts requests "
         r"whose imagined trajectory moved along the named axis by more than $5\%$ of a pilot's mean, and "
         r"`motion' is the mean of that fraction. $|\Delta a|$ is the commanded step-to-step change as a "
         r"multiple of the recorded one, inside a chunk and at the seam where a fresh chunk begins; the "
         r"last column is the ratio, so $1.0$ means the join is indistinguishable from an ordinary step.}",
         r"  \label{tab:continuity}", r"  \begin{tabular}{lccccc}", r"    \toprule",
         r"    & Obeyed & Motion & \multicolumn{2}{c}{$|\Delta a| \times$ recorded $\downarrow$} & Seam / \\",
         r"    \cmidrule(lr){4-5}",
         r"    Candidates & of $8$ $\uparrow$ & of pilot & in chunk & at seam & inside \\", r"    \midrule"]
    for run, lab in CONT_ROWS:
        ph, jk = _phys(run), _jerk(run)
        if not ph or jk is None:
            L.append(f"    {lab} & -- & -- & -- & -- & -- \\\\")
            continue
        hits = sum(ph[q][0] > 0.05 * PILOT[WANTS[q][0]] for q in ph)
        frac = sum(ph[q][0] / PILOT[WANTS[q][0]] for q in ph) / len(ph)
        L.append(f"    {lab} & {hits}/8 & {100 * frac:+.0f}\\% & {jk[0] / REC_DA:.2f} & "
                 f"{jk[1] / REC_DA:.2f} & {jk[1] / jk[0]:.2f} \\\\")
    L += [r"    \midrule",
          r"    Recorded flight & -- & $100\%$ & $1.00$ & $1.00$ & $1.00$ \\",
          r"    \bottomrule", r"  \end{tabular}", r"\end{table*}"]
    return "\n".join(L) + "\n"


def main(paper: str) -> int:
    out = os.path.join(paper, "tables")
    os.makedirs(out, exist_ok=True)
    # cont_table is retired: its aggregates now live in the steering table, which is where the
    # comparison they summarise already was.
    for name, fn in (("longhorizon", wm_table), ("actionhead", ah_table), ("ood", ood_table),
                     ("steering", steer_table)):
        t = fn(paper) if fn in (ood_table, steer_table) else fn()
        open(os.path.join(out, f"{name}.tex"), "w").write(t)
        print(f"  tables/{name}.tex  {len(t.splitlines())} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/user_irw/icra2027-seamstress"))
