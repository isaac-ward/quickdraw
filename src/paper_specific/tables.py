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

from .harvest import config, fmt, metrics, suffix, contains

LOGS = "logs"

# ---- the world-model ablation: same data, same autoencoder, one setting changed per row ----------
# Row labels say what the setting IS, not what the run was called: `vl128 base` named a file, not a
# configuration, and stride 5 with summed actions is the thing a reader needs to know.
WM_ROWS = [
    # PLAIN TEXT, not codes: a reader should not have to decode "4 / concat" to know what a row is.
    # Verified against each run's own config.json -- the stride-1 run uses SUMMED actions, which an
    # earlier version of this table got wrong.
    ("train_world_model_2026_09_05_20_21_02_starling2_vl128", "Inherited recipe", ""),
    ("train_world_model_2026_09_07_03_45_54_starling2_heavy", r"\quad deeper denoiser", "rejected"),
    ("train_world_model_2026_09_12_08_06_49_s2_sub4_concat_deriv", r"\quad action increments", "rejected"),
    ("train_world_model_2026_09_08_21_46_53_s2_sub1", r"\quad every frame, summed", ""),
    ("train_world_model_2026_09_11_03_17_22_s2_sub3_concat", r"\quad every third frame", ""),
    ("train_world_model_2026_09_08_22_04_09_s2_sub4", r"\quad every fourth frame, summed", ""),
    ("train_world_model_2026_09_11_03_17_47_s2_sub4_concat",
     r"\quad every fourth frame, concatenated$^{*}$", "kept"),
]
# ---- the Action Model: the 2x2 of context pooling x target space, plus the chunk and pit_delta arms --
AH_ROWS = [
    # ONE PLAIN-TEXT CONFIGURATION COLUMN, like the World Model table. The 2x2 of context x target is
    # incomplete by design: pooled context with a raw target is the cell nothing recommends and it was
    # never trained, which the caption says.
    ("train_action_2026_09_13_21_57_41_s2_ah_pit", "Pooled context, percentile target"),
    ("train_action_2026_09_14_01_15_48_s2_ah_grouped", "Grouped context, raw target"),
    ("train_action_2026_09_13_23_42_49_s2_ah_pit",
     r"Grouped context, percentile target$^{*}$"),
    ("train_action_2026_09_14_04_41_17_s2_ah_chunk32_full", r"\quad longer chunk ($32$)"),
    ("train_action_model_2026_09_14_22_28_22_s2_ah_chunk32_pitdelta",
     r"\quad longer chunk, increment target"),
]


def wm_table() -> str:
    hs = [1, 8, 32, 128]
    L = [r"\begin{table}[t]", r"  \centering", r"  \footnotesize",
         r"  \setlength{\tabcolsep}{3pt}",
         r"  \caption{\textbf{World Model.} Open-loop prediction on held-out \texttt{starling-2} flight, by "
         r"horizon: how far the action-conditioned observation prediction holds up. The first row is the "
         r"recipe as inherited from a manipulation dataset; every row below it changes one setting, and the "
         r"best per column is bold. The autoencoder floor re-encodes and decodes the true frame, so it "
         r"bounds what any dynamics model on this tokenizer can reach. The inherited recipe kept every "
         r"fifth frame and summed the commands it skipped; each row below changes one thing. "
         r"$^{\dagger}$The autoencoder floor is not a model: it encodes and decodes the TRUE frame, so "
         r"it is the same at every horizon and no dynamics model on this tokenizer can beat it. "
         r"$^{*}$The configuration \modelname{} uses.}",
         r"  \label{tab:longhorizon}", r"  \begin{tabular}{lcccc}", r"    \toprule",
         r"    & \multicolumn{4}{c}{LPIPS $\downarrow$ at open-loop horizon} \\",
         r"    \cmidrule(lr){2-5}",
         r"    Configuration & $+1$ & $+8$ & $+32$ & $+128$ \\", r"    \midrule"]
    floors, rows = [], []
    for run, name, verdict in WM_ROWS:
        d = metrics(os.path.join(LOGS, run))
        if not d:
            continue
        vals = [suffix(d, f"open_loop/image/lpips/@+{h}") for h in hs]
        ae = suffix(d, "eval_ae_floor/image/lpips_mean")
        if ae is not None:
            floors.append(ae)
        rows.append((name, vals))
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
          + " & ".join([fmt(sum(floors) / len(floors), 4)] * 4) + r" \\",
          r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


def ah_table() -> str:
    L = [r"\begin{table}[t]", r"  \centering", r"  \footnotesize",
         r"  \setlength{\tabcolsep}{3pt}",
         r"  \caption{\textbf{Action Model.} The counterpart of Table~\ref{tab:longhorizon} for the other "
         r"half of what \modelname{} predicts: not what follows from an action, but what action follows "
         r"from a context. One setting changes per row, best per column in bold. Energy skill is the only "
         r"column that measures \emph{conditioning}, against a context-blind null, so $0$ is a model that "
         r"ignores its context; it is given at the first lead time and at the worst. $W_1$ is the distance "
         r"to the recorded action marginal, i.e.\ whether it flies like the data. Rest AUC asks whether "
         r"the model can place mass on a stick being HELD still, which is what the percentile transform "
         r"buys and what a flow cannot do without it. Pooled context with a raw target is the one cell of "
         r"the $2\times2$ that was never trained: nothing recommends it. No row wins every column, because "
         r"the trade is real --- Fig.~\ref{fig:marginals} is the same question answered by eye. "
         r"$^{*}$The configuration \modelname{} uses, at the chunk length the planner wants.}",
         r"  \label{tab:actionhead}", r"  \begin{tabular}{lcccc}", r"    \toprule",
         r"    & Skill$_{+1}$ & Skill$_{\max}$ & $W_1$ & Rest AUC \\",
         r"    Configuration & $\uparrow$ & $\uparrow$ & $\downarrow$ & $\uparrow$ \\", r"    \midrule"]
    rows = []
    for run, lab in AH_ROWS:
        d = metrics(os.path.join(LOGS, run))
        if not d:
            continue
        rows.append((lab, [contains(d, "lead_00/energy_skill"), contains(d, "energy_skill"),
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
                        ("l2", "Image RMSE"), ("angvel_err", "Angular velocity error"),
                        ("pos_err", "Position error")],
    "eval_ood_leafblower": [("angvel_err", r"Angular velocity error (\textbf{ours})"),
                            ("vel_err", "Velocity error"), ("lpips", "Image LPIPS"),
                            ("rot_err", "Orientation error"), ("pos_err", "Position error")],
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
COLS = [("gauss", "Gaussian AM"), ("data", r"Data Retrieval AM$^{\ddagger}$"),
        ("prior", r"Learned AM (\textbf{ours})"), ("pitdelta", r"Learned $\Delta$ AM")]
# The place block needs DECODED VIDEO for the labeller, so these are the video-on 26-request suites
# rather than the 16-context physical runs above.
VLM_RUNS = {
    "gauss": "logs/eval_steer_2026_09_15_06_29_37_suite_gauss",
    "prior": "logs/eval_steer_2026_09_14_21_57_53_best_prior_guided",
    "pitdelta": "logs/eval_steer_2026_09_15_01_29_12_suite_pitdelta",
    "data": "logs/eval_steer_2026_09_15_01_28_55_suite_data_retrieved",
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
    """{request: (hit, base)} for the object/region requests, from the VLM labels."""
    import json as _j
    p = os.path.join(run, "vlm_object_check.json")
    if not os.path.exists(p):
        return {}
    rows = [(r["request"], r["label"]) for r in _j.load(open(p)) if r.get("label")]
    import yaml
    ic = yaml.safe_load(open("conf/interpret/starling.yaml"))
    OBJ, REG = list(ic["factors"]["object_in_view"]["buckets"]), list(ic["factors"]["facing"]["buckets"])
    out = {}
    for buckets, field in ((OBJ, "object_in_view"), (REG, "facing")):
        sub = [(q, l[field]) for q, l in rows if q in buckets]
        for b in buckets:
            mine = [o for q, o in sub if q == b]
            other = [o for q, o in sub if q != b]
            if mine:
                out[b] = (sum(o == b for o in mine) / len(mine),
                          (sum(o == b for o in other) / len(other)) if other else float("nan"), len(mine))
    return out


def steer_table(paper: str) -> str:
    """Per-request steering as fractions of contexts, plus the aggregates the continuity table used to
    carry on its own -- the author asked for one table, since the second was mostly the same comparison
    summarised."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "analysis"))
    from steer_physical import WANTS
    ph = {k: _phys(v) for k, v in STEER_RUNS.items()}
    jk = {k: _jerk(v) for k, v in STEER_RUNS.items()}
    vl = {k: _vlm(v) for k, v in VLM_RUNS.items()}
    nc = len(COLS)
    L = [r"\begin{table*}[t]", r"  \centering", r"  \small",
         r"  \caption{\textbf{Language steering.} The Action Model (AM) is the only thing that differs "
         r"between columns. Every cell is the fraction of starting contexts that met the request: for a "
         r"motion primitive, that the imagined trajectory moved along the axis the words name by more "
         r"than $5\%$ of what a pilot covers in the same $34$\,s, read off the imagined proprioception "
         r"and independent of the reward the planner maximised; for a location, that a VLM labelling the "
         r"imagined video named it. Best per row in bold. $^{\ddagger}$Data Retrieval is the BASELINE: it "
         r"draws real recorded chunks, so it is perfectly flyable and completely blind to the request, "
         r"and beating it is the bar a learned prior has to clear. $^{\S}$The corpus contains no "
         r"backward flight at all, and no arm gets more than one context out of fifteen -- a motion "
         r"primitive absent from the data is not reachable by steering, however the candidates are "
         r"drawn. The location block does "
         r"not separate the columns and is scored strictly: the VLM names ONE object and ONE region per "
         r"clip, so a plan that reaches the table while the ladder is also in view scores nothing here.}",
         r"  \label{tab:planningandcontrol}", r"  \begin{tabular}{l" + "c" * nc + "}", r"    \toprule",
         r"    Request & " + " & ".join(lab for _, lab in COLS) + r" \\", r"    \midrule",
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
        best = max((h for h, _ in (x for x in vals if x)), default=None)
        cells = ["--" if v is None else
                 ((r"\textbf{" + f"{v[0]}" + "}/" + f"{v[1]}") if v[0] == best and best else
                  f"{v[0]}/{v[1]}") for v in vals]
        nm = q + (r"$^{\S}$" if q == "fly backward" else "")
        L.append(f"    ``{nm}\'\' & " + " & ".join(cells) + r" \\")
    # THE AGGREGATES the continuity table used to hold: obeyed, motion against a pilot, and the two jerk
    # columns, which is what makes the comparison between candidate sources legible in one place.
    L += [r"    \midrule",
          r"    Primitives obeyed (of $8$) & " + " & ".join(
              (r"\textbf{" + f"{sum(hits_all[m])}" + "}/8")
              if sum(hits_all[m]) == max(sum(h) for h in hits_all.values()) else f"{sum(hits_all[m])}/8"
              for m, _ in COLS) + r" \\",
          r"    Motion, fraction of a pilot & " + " & ".join(
              f"{sum(frac_all[m]) / max(1, len(frac_all[m])):+.2f}" for m, _ in COLS) + r" \\",
          r"    $|\Delta a|$ in a chunk ($\times$ recorded) & " + " & ".join(
              "--" if jk[m] is None else f"{jk[m][0] / REC_DA:.2f}" for m, _ in COLS) + r" \\",
          r"    $|\Delta a|$ at the seam ($\times$ recorded) & " + " & ".join(
              "--" if jk[m] is None else f"{jk[m][1] / REC_DA:.2f}" for m, _ in COLS) + r" \\"]
    L += [r"    \midrule", r"    \multicolumn{" + str(1 + nc) +
          r"}{c}{Locations} \\", r"    \midrule"]
    for q in ("wall with black panels", "center of room over mats", "floor to ceiling glass wall",
              "white wall with table", "ladder", "mannequin", "colored floor mat", "table"):
        vals = []
        for m, _ in COLS:
            v = vl.get(m, {}).get(q)
            vals.append(None if v is None else (round(v[0] * int(v[2])), int(v[2])))
        best = max((h for h, _ in (x for x in vals if x)), default=None)
        cells = ["--" if v is None else
                 ((r"\textbf{" + f"{v[0]}" + "}/" + f"{v[1]}") if best and v[0] == best else
                  f"{v[0]}/{v[1]}") for v in vals]
        L.append(f"    ``{q}\'\' & " + " & ".join(cells) + r" \\")
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table*}"]
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
