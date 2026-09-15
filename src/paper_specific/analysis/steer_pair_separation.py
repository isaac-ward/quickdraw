"""The bias-free readout: OPPOSING PAIRS on the same axis, from the same starting context.

`obey` (does the plan's mean stick carry the requested sign) is contaminated by a standing bias. The
recorded fore/aft stick averages -0.46, so EVERY plan flies forward whatever was asked -- and `fly forward`
scores "obeyed" for free while `fly backward` cannot score at all. Differencing the two opposing requests
on one axis cancels that offset exactly, leaving only what the words changed:

    yaw       rotate left  - rotate right     vertical   descend - climb
    lateral   strafe right - strafe left      fore/aft   fly backward - fly forward

Positive separation = the pair moved apart the way the words ask. Paired per context (both requests were
planned from the same start with the same seed), so the sign test over contexts is the significance.

    python scratch/steer_pair_separation.py <run> [<run> ...]
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import yaml

AXES = yaml.safe_load(open("conf/interpret/starling.yaml"))["action_axes"]
PAIRS = [(j, a["positive"], a["negative"], a["name"]) for j, a in enumerate(AXES)]


def main(*runs: str) -> int:
    for run in runs:
        per, obj = {}, None
        for f in sorted(glob.glob(os.path.join(run, "logs", "epoch_*", "eval_steer", "plans", "*", "*",
                                               "plan.json"))):
            d = json.load(open(f))
            obj = d.get("objective", "?")
            a = np.load(os.path.join(os.path.dirname(f), "actions.npy"))
            per.setdefault((d["episode"], d["start"]), {})[d["request"]] = \
                a.reshape(len(a), -1, len(AXES)).mean(axis=1).mean(axis=0)
        print(f"\n=== {os.path.basename(run)}   objective={obj}   {len(per)} contexts")
        print(f"  {'axis':9s} {'+ request':14s} {'- request':14s} {'mean(+)':>8s} {'mean(-)':>8s} "
              f"{'separation':>11s} {'sem':>7s}  contexts correct")
        print("  " + "-" * 104)
        for j, pos, neg, name in PAIRS:
            d = [(v[pos][j], v[neg][j]) for v in per.values() if pos in v and neg in v]
            if not d:
                continue
            p, n = np.array([x[0] for x in d]), np.array([x[1] for x in d])
            sep = p - n                      # should be POSITIVE: + request pushes the axis more positive
            print(f"  {name:9s} {pos:14s} {neg:14s} {p.mean():>+8.3f} {n.mean():>+8.3f} "
                  f"{sep.mean():>+11.3f} {sep.std() / np.sqrt(len(sep)):>7.3f}  "
                  f"{int((sep > 0).sum())}/{len(sep)}")
        print("  separation > 0 means the two words moved the SAME axis apart the right way; the standing")
        print("  bias on that axis cancels in the difference, so this is what the language actually bought.")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
