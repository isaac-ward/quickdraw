"""Does the reward actually track the MOTION it names? The domain check ruled out a train/test gap: the head
discriminates requests on planned latents at ~78% of its fitted spread, flat out to step 128, and a plan can
move R by ~0.44 within one request. So there IS signal and the planner CAN climb it -- yet the chosen actions
do not obey the words. That leaves one question: is high R("climb") the same thing as climbing?

Every plan folder already holds both halves, so this needs no model: R per step (plan.json reward_curve) and
the command per step (actions.npy, raw stick units). Correlate them WITHIN each plan, along the axis the
request names. Alignment matters: actions[i] is applied at step i and produces step i+1, so the reward to
correlate with actions[i] is reward[i+1].

    python scratch/check_reward_vs_motion.py <steer_run>
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import yaml

AXES = yaml.safe_load(open("conf/interpret/starling.yaml"))["action_axes"]
# request -> (axis index, sign the request asks for) from the axis labels themselves
WANT = {}
for j, a in enumerate(AXES):
    WANT[a["positive"]] = (j, +1.0)
    WANT[a["negative"]] = (j, -1.0)
WANT["fly forward toward the ladder"] = WANT["fly forward"]


def main(run: str) -> int:
    rows = {}
    for f in sorted(glob.glob(os.path.join(run, "logs", "epoch_*", "eval_steer", "plans", "*", "*",
                                           "plan.json"))):
        d = json.load(open(f))
        if d["request"] not in WANT:
            continue
        j, want = WANT[d["request"]]
        a = np.load(os.path.join(os.path.dirname(f), "actions.npy"))
        stick = a.reshape(len(a), -1, len(AXES)).mean(axis=1)[:, j]      # (T,) raw, sign kept
        R = np.asarray(d["reward_curve"], dtype=np.float64)
        n = min(len(stick) - 1, len(R) - 1)
        x, y = stick[:n], R[1:n + 1]                                     # action[i] -> reward[i+1]
        if x.std() < 1e-6:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        rows.setdefault(d["request"], {"r": [], "want": want, "mean": [], "hi": [], "lo": []}).append if 0 else None
        e = rows.setdefault(d["request"], {"r": [], "want": want, "mean": [], "split": []})
        e["r"].append(r)
        e["mean"].append(float(x.mean()))
        # the planner's own choice, put bluntly: of the steps where R was in its TOP quartile, which way did
        # the stick go? That is what maximising R commits to, with no correlation assumption.
        e["split"].append(float(x[y >= np.quantile(y, 0.75)].mean()))
    print(f"\n  {'request':32s} {'wants':>6s} {'corr(stick, R)':>15s} {'stick@topR':>11s} {'stick(all)':>11s}  verdict")
    print("  " + "-" * 100)
    for q, e in rows.items():
        want = e["want"]
        r, top, allm = np.mean(e["r"]), np.mean(e["split"]), np.mean(e["mean"])
        # a reward that tracks the request should correlate with the stick IN THE REQUESTED DIRECTION
        aligned = r * want
        v = ("TRACKS" if aligned > 0.2 else "weak" if aligned > 0.05 else
             "NONE" if aligned > -0.05 else "ANTI-TRACKS")
        print(f"  {q:32s} {('+' if want > 0 else '-'):>6s} {r:>+15.3f} {top:>+11.3f} {allm:>+11.3f}  {v}")
    print("\n  `wants` is the sign of the stick the request asks for (from conf/interpret/starling.yaml).")
    print("  TRACKS means R rises as the stick goes the requested way -- the only thing that would let a")
    print("  reward-maximising planner obey the words. stick@topR is where the stick sat on the steps the")
    print("  reward liked best: it should carry the `wants` sign.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
