"""`evaluation/conditional.py` -- do the conditional scores actually say what they are claimed to say?

A scoring rule is only worth logging if it RANKS models correctly, so this checks the ranking on cases whose
answer is known by construction rather than checking arithmetic against itself:

    a model that knows the answer          -> energy skill near 1
    a model that knows only the marginal   -> energy skill near 0        <- the whole point
    a model that hedges (too wide)         -> worse than the sharp one, and "overdispersed"
    a model that is too sharp              -> worse too, and "overconfident"
    a model that knows WHEN the atom is    -> rest AUC near 1
    a model that knows only HOW OFTEN      -> rest AUC near 0.5          <- the other whole point

The second and last lines are the ones that matter: they are exactly what pooled W1 cannot distinguish, and
the reason this module exists.

    python -m quickdraw.smoke.conditional
"""
from __future__ import annotations

import sys

import numpy as np

from ..evaluation.conditional import blind_null, energy_skill, rank_calibration, rest_skill

ok = bad = 0


def check(n, cond, extra=""):
    global ok, bad
    ok, bad = ok + bool(cond), bad + (not cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {n}{('  ' + extra) if extra else ''}", flush=True)


def main() -> int:
    rng = np.random.default_rng(0)
    N, M, D = 4000, 32, 4
    # A world where the observation IS predictable from something the model may or may not use.
    mu = rng.normal(0, 1.0, (N, D))                      # the "context" the truth depends on
    y = mu + rng.normal(0, 0.25, (N, D))                 # what actually happened
    null = blind_null(y, M)

    print("\nENERGY SKILL ranks models the way it must")
    knows = mu[:, None, :] + rng.normal(0, 0.25, (N, M, D))          # right mean, right spread
    blind = null                                                      # right marginal, no context
    hedge = mu[:, None, :] + rng.normal(0, 1.50, (N, M, D))           # right mean, far too wide
    tight = mu[:, None, :] + rng.normal(0, 0.02, (N, M, D))           # right mean, far too sharp
    s_k, s_b = energy_skill(knows, y, null), energy_skill(blind, y, null)
    s_h, s_t = energy_skill(hedge, y, null), energy_skill(tight, y, null)
    check(f"knows the answer   skill {s_k:+.3f} (near 1)", s_k > 0.6)
    check(f"context-BLIND      skill {s_b:+.3f} (near 0)", abs(s_b) < 0.05)
    check(f"hedging is punished  {s_h:+.3f} < {s_k:+.3f}", s_h < s_k)
    check(f"over-sharp is punished {s_t:+.3f} < {s_k:+.3f}", s_t < s_k)
    check("a model that ignores its context scores ZERO, whatever its marginal", abs(s_b) < 0.05)

    print("\nRANK CALIBRATION says WHICH WAY the head is wrong")
    for name, X, want in (("well calibrated", knows, "roughly calibrated"),
                          ("too wide", hedge, "overdispersed (draws too wide)"),
                          ("too sharp", tight, "overconfident (truth lands outside the draws)")):
        hist, dev, verdict = rank_calibration(X, y)
        check(f"{name:<16} -> {verdict}   (deviation {dev:.3f})", verdict == want)

    print("\nREST SKILL separates knowing WHEN from knowing HOW OFTEN")
    # an atom at 0.0 that occurs on 40% of contexts, decided by the context itself
    at_rest = mu[:, 0] > np.quantile(mu[:, 0], 0.6)
    ya = y.copy(); ya[at_rest, 0] = 0.0
    knows_when = knows.copy()
    knows_when[at_rest, :, 0] = np.where(rng.random((at_rest.sum(), M)) < 0.9, 0.0,
                                         knows_when[at_rest, :, 0])
    only_often = knows.copy()                                         # right RATE, wrong contexts
    pick = rng.random((N, M)) < float(at_rest.mean())
    only_often[:, :, 0] = np.where(pick, 0.0, only_often[:, :, 0])
    for name, X, lo, hi in (("knows WHEN", knows_when, 0.8, 1.01), ("knows only HOW OFTEN", only_often,
                                                                   0.45, 0.55)):
        r = [d for d in rest_skill(X, ya) if d["dim"] == 0]
        check(f"{name:<22} AUC {r[0]['auc']:.3f} (want {lo:.2f}-{hi:.2f}), "
              f"predicted rate {r[0]['pred_rate'] * 100:.0f}% vs actual {r[0]['rate'] * 100:.0f}%",
              lo <= r[0]["auc"] <= hi)
    r_of = [d for d in rest_skill(only_often, ya) if d["dim"] == 0][0]
    check("...and the HOW OFTEN model gets the RATE right while its AUC says it knows nothing",
          abs(r_of["pred_rate"] - r_of["rate"]) < 0.06 and abs(r_of["auc"] - 0.5) < 0.05)

    print(f"\n{ok} passed, {bad} failed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
