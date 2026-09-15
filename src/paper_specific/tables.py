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

# ---- the world-model ablation: same data, same codec, one setting changed per row -----------------
WM_ROWS = [
    ("train_world_model_2026_09_05_20_21_02_starling2_vl128", r"\texttt{vl128} base", ""),
    ("train_world_model_2026_09_07_03_45_54_starling2_heavy", r"\quad + denoiser depth 4", "rejected"),
    ("train_world_model_2026_09_12_08_06_49_s2_sub4_concat_deriv", r"\quad + derivative actions", "rejected"),
    ("train_world_model_2026_09_11_03_17_22_s2_sub3_concat", r"\quad frame stride 3", ""),
    ("train_world_model_2026_09_08_21_46_53_s2_sub1", r"\quad frame stride 1", ""),
    ("train_world_model_2026_09_08_22_04_09_s2_sub4", r"\quad frame stride 4, summed actions", ""),
    ("train_world_model_2026_09_11_03_17_47_s2_sub4_concat", r"\quad frame stride 4, concatenated (\textbf{ours})", "kept"),
]
# ---- the action prior: the 2x2 of context pooling x target space, plus the chunk and pit_delta arms --
AH_ROWS = [
    ("train_action_2026_09_13_21_57_41_s2_ah_pit", "8", "pooled", "PIT", ""),
    ("train_action_2026_09_14_01_15_48_s2_ah_grouped", "8", "grouped", "none", ""),
    ("train_action_2026_09_13_23_42_49_s2_ah_pit", "8", "grouped", "PIT", r"\textbf{ours}"),
    ("train_action_2026_09_14_04_41_17_s2_ah_chunk32_full", "32", "grouped", "PIT", ""),
    ("train_action_model_2026_09_14_22_28_22_s2_ah_chunk32_pitdelta", "32", "grouped", "PIT-$\\Delta$", "rejected"),
]


def wm_table() -> str:
    hs = [1, 8, 32, 128]
    L = [r"\begin{table*}[t]", r"  \centering", r"  \small",
         r"  \caption{Open-loop prediction on held-out \texttt{starling-2} flight, by horizon. Rows change "
         r"one setting at a time; best per column in bold. The autoencoder floor re-encodes and decodes the "
         r"true frame, so it bounds what any dynamics model can reach.}",
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
          r"    Autoencoder floor & \multicolumn{4}{c}{" + fmt(sum(floors) / len(floors), 4) + r"} \\",
          r"    \bottomrule", r"  \end{tabular}", r"\end{table*}"]
    return "\n".join(L) + "\n"


def ah_table() -> str:
    L = [r"\begin{table*}[t]", r"  \centering", r"  \small",
         r"  \caption{The play action prior. Energy skill is measured against a context-blind null, so $0$ is "
         r"a model that ignores its context. Rest AUC asks whether the prior identifies the stick being held "
         r"at rest, which a rectified flow cannot place mass on without the percentile transform. $W_1$ is "
         r"against the recorded action marginal. Best per column in bold.}",
         r"  \label{tab:actionhead}", r"  \begin{tabular}{lllcccc}", r"    \toprule",
         r"    Chunk & Context & Target & Skill$_{+1}$ $\uparrow$ & Skill$_{\max}$ $\uparrow$ "
         r"& $W_1$ $\downarrow$ & Rest AUC $\uparrow$ \\", r"    \midrule"]
    rows = []
    for run, chunk, ctx, tgt, note in AH_ROWS:
        d = metrics(os.path.join(LOGS, run))
        if not d:
            continue
        rows.append((f"{chunk} & {ctx} & {tgt}", note,
                     [contains(d, "lead_00/energy_skill"), contains(d, "energy_skill"),
                      contains(d, "w1_mean"), contains(d, "rest_auc")]))
    hi = [True, True, False, True]        # higher-is-better per column; W1 is lower-is-better
    best = []
    for j, up in enumerate(hi):
        vs = [r[2][j] for r in rows if r[2][j] is not None]
        best.append((max if up else min)(vs) if vs else None)
    for nm, note, vals in rows:
        cells = []
        for j, v in enumerate(vals):
            t = fmt(v, 4 if j == 2 else 3)
            if v is not None and best[j] is not None and abs(v - best[j]) < 1e-9:
                t = r"\textbf{" + t + "}"
            cells.append(t)
        tail = f"  {note}" if note and "rejected" in note else ""
        L.append(f"    {nm} & " + " & ".join(cells) + (r" \\" if not tail else r" \\"))
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table*}"]
    return "\n".join(L) + "\n"




# ---- OOD detection: the columns the paper asks for, with metric variants as the baselines ----------
OOD_ROWS = {
    "eval_ood_noodle": [("latent_cos", r"latent surprise (\textbf{ours})"), ("lpips", "image LPIPS"),
                        ("l2", "image RMSE"), ("angvel_err", "angular velocity error"),
                        ("pos_err", "position error")],
    "eval_ood_leafblower": [("angvel_err", r"angular velocity error (\textbf{ours})"),
                            ("vel_err", "velocity error"), ("lpips", "image LPIPS"),
                            ("rot_err", "orientation error"), ("pos_err", "position error")],
}
OOD_NAME = {"eval_ood_noodle": "Visual (pool noodle)", "eval_ood_leafblower": "Dynamical (leaf blower)"}


def ood_table(paper: str) -> str:
    import json
    # analyses write their JSON under the repo's own logs/, so read it repo-relative rather than by
    # walking up from the paper directory -- the paper repo lives outside quickdraw and its depth varies.
    j = json.load(open(os.path.join(LOGS, "paper_icra_2027", "wm_anomaly_classification.json")))
    cov = j.get("coverage", 0.9)
    L = [r"\begin{table*}[t]", r"  \centering", r"  \small",
         r"  \caption{Out-of-distribution detection from one-step prediction error, scored per timestep "
         # ESCAPE THE PERCENT SIGN. `{cov:.0%}` emits a bare % and LaTeX comments out the rest of the
         # caption, which ends as "Runaway argument? ... File ended while scanning use of \caption".
         rf"against the reviewed anomaly windows. The threshold is the {100 * cov:.0f}\% quantile of the "
         r"non-conformity scores of the \emph{other} episodes' in-distribution steps, so nominal accuracy "
         r"is calibrated to $\approx$" + rf"{100 * cov:.0f}\%" + r" by construction and failure accuracy is what the "
         r"detector buys. Weighted accuracy is the mean of the two, whose no-skill value is $0.500$ under "
         r"any class imbalance. Rows below the first in each block are the same pipeline read through a "
         r"different error channel.}",
         r"  \label{tab:ood}", r"  \begin{tabular}{llccc}", r"    \toprule",
         r"    & & Nominal & Failure & Weighted \\",
         r"    Anomaly & Score & acc.\ $\uparrow$ & acc.\ $\uparrow$ & acc.\ $\uparrow$ \\"]
    for split, rows in OOD_ROWS.items():
        L += [r"    \midrule", r"    \multicolumn{5}{c}{" + OOD_NAME[split] + r"} \\", r"    \midrule"]
        m = j[split]["metrics"]
        for key, label in rows:
            r = m[key]
            cells = [fmt(r["specificity"]), fmt(r["recall"]), fmt(r["balanced_accuracy"])]
            if r"\textbf" in label:
                cells = [r"\textbf{" + c + "}" for c in cells]
            L.append(f"    & {label} & " + " & ".join(cells) + r" \\")
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table*}"]
    return "\n".join(L) + "\n"




# ---- steering: per-request, and then the proposal comparison ---------------------------------------
# The direction block: one run per candidate source, all at the same 16 contexts x 8 requests, same
# reward head, same objective, same commit -- the candidate source is the ONLY thing that differs.
# `gauss` is the control: MPPI's historic white noise, which is what the planner drew from before the
# prior was wired into it.
STEER_RUNS = {
    "gauss": "logs/eval_steer_2026_09_15_05_06_35_phys16_gauss",
    "data": "logs/eval_steer_2026_09_14_22_08_48_phys16_data",
    "prior": "logs/eval_steer_2026_09_14_22_08_46_phys16_prior",
    "pitdelta": "logs/eval_steer_2026_09_15_01_12_15_phys16_pitdelta",
}
COLS = [("gauss", "Gaussian"), ("data", "Data chunks"), ("prior", r"Prior (\textbf{ours})"),
        ("pitdelta", r"Prior, PIT-$\Delta$")]
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
    ("logs/eval_steer_2026_09_15_05_06_35_phys16_gauss", "Gaussian noise"),
    ("logs/eval_steer_2026_09_14_22_08_48_phys16_data", "Real data chunks"),
    ("logs/eval_steer_2026_09_15_05_33_54_phys16_noguid", r"\quad prior, no continuity"),
    ("logs/eval_steer_2026_09_15_05_58_34_phys16_xfade", r"\quad prior, crossfade"),
    ("logs/eval_steer_2026_09_14_22_08_46_phys16_prior", r"\quad prior, prefix guidance (\textbf{ours})"),
    ("logs/eval_steer_2026_09_15_01_12_15_phys16_pitdelta", r"\quad prior, PIT-$\Delta$"),
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
    return {k: (sum(v) / len(v), len(v)) for k, v in out.items()}


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
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "analysis"))
    from steer_physical import WANTS
    ph = {k: _phys(v) for k, v in STEER_RUNS.items()}
    jk = {k: _jerk(v) for k, v in STEER_RUNS.items()}
    vl = {k: _vlm(v) for k, v in VLM_RUNS.items()}
    # what a pilot covers on each axis in the same 34 s, so "achieved" has a scale (analysis/steer_physical)
    PILOT = {"yaw": 471.75, "altitude": 0.58, "forward": 23.63, "lateral": 19.92}
    REC = 0.0541                                        # recorded step-to-step |da|, same fold
    nc = len(COLS)
    L = [r"\begin{table*}[t]", r"  \centering", r"  \small",
         r"  \caption{Language steering, per request, with the candidate source as the only difference "
         r"between columns: gaussian noise (MPPI's historic candidates, the control), real recorded action "
         r"chunks (state-blind but perfectly flyable), the trained prior, and the same prior trained on "
         r"increments. For a request naming a \emph{direction} the readout is the imagined trajectory's net "
         r"motion along the axis the words name, signed so positive means obeyed, in physical units and "
         r"independent of the reward the planner maximised. For a request naming a \emph{place} it is a "
         r"VLM's label of the imagined video, with `null' --- how often that place is reached when "
         r"something \emph{else} was asked for --- as the context-blind baseline beside it. The place block "
         r"is inconclusive and we report it as such: every arm beats its own null, but so does the gaussian "
         r"control, and at four contexts per request the differences between columns are noise.}",
         r"  \label{tab:planningandcontrol}", r"  \begin{tabular}{ll" + "c" * nc + "}", r"    \toprule",
         r"    Request & Asked for & " + " & ".join(lab for _, lab in COLS) + r" \\", r"    \midrule",
         r"    \multicolumn{" + str(2 + nc) + r"}{c}{Directions --- net motion achieved, and $\%$ of what a "
         r"pilot covers in the same $34$\,s} \\", r"    \midrule"]
    hits = {m: [] for m, _ in COLS}
    frac = {m: [] for m, _ in COLS}
    for q in ("rotate left", "rotate right", "climb", "descend", "strafe left", "strafe right",
              "fly forward", "fly backward"):
        key, sgn = WANTS[q]
        cells = []
        for m, _ in COLS:
            v = ph[m].get(q)
            if v is None:
                cells.append("--")
                continue
            r = v[0] / PILOT[key]
            hits[m].append(v[0] > 0.05 * PILOT[key]); frac[m].append(r)
            cells.append(f"{v[0]:+.1f}{UNIT[key]} ({100 * r:+.0f}\\%)")
        L.append(f"    {q} & {key} {'+' if sgn > 0 else '$-$'} & " + " & ".join(cells) + r" \\")
    L += [r"    \midrule", r"    \multicolumn{" + str(2 + nc) +
          r"}{c}{Places --- fraction of plans a VLM confirms reached it, (null)} \\", r"    \midrule"]
    for q in ("wall with black panels", "center of room over mats", "floor to ceiling glass wall",
              "white wall with table", "ladder", "mannequin", "colored floor mat", "table"):
        cells = []
        for m, _ in COLS:
            v = vl.get(m, {}).get(q)
            cells.append("--" if v is None else f"{v[0]:.2f}~({v[1]:.2f})")
        L.append(f"    {q} & place & " + " & ".join(cells) + r" \\")
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table*}"]
    return "\n".join(L) + "\n"


def cont_table(paper: str) -> str:
    """Obeyed / motion / smoothness per arm. The one table where the continuity mechanism varies."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "analysis"))
    from steer_physical import WANTS
    L = [r"\begin{table*}[t]", r"  \centering", r"  \small",
         r"  \caption{The candidate source and the continuity mechanism, over the same $8$ requests "
         r"$\times$ $16$ contexts, same reward head, same objective, same commit. `Obeyed' counts requests "
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
    for name, fn in (("longhorizon", wm_table), ("actionhead", ah_table), ("ood", ood_table),
                     ("steering", steer_table), ("continuity", cont_table)):
        t = fn(paper) if fn in (ood_table, steer_table, cont_table) else fn()
        open(os.path.join(out, f"{name}.tex"), "w").write(t)
        print(f"  tables/{name}.tex  {len(t.splitlines())} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/user_irw/icra2027-seamstress"))
