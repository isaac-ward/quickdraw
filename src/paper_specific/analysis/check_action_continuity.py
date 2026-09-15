"""Are the planned commands FLYABLE, and is anything enforcing that?

Nothing is: the imagination planner has no control cost and no smoothness term. MPPI has `beta_ctrl` (a
penalty on action magnitude) but eval_steer's `plan` scores reward alone, so the only continuity in a plan
is whatever the prior puts inside one chunk. Across a chunk BOUNDARY -- where one 32-step draw is committed
and the next is drawn fresh -- nothing connects them at all.

So measure the seam. |a_t - a_{t-1}| at the stitch indices (multiples of the committed chunk) against the
same quantity inside a chunk, and both against what the recorded sticks actually do. If the seams are much
larger than the recorded step-to-step change, the plan is asking for jerks no pilot commanded, and a
continuity penalty is worth adding; if they are comparable, the prior's chunks already stitch smoothly.

    python scratch/check_action_continuity.py <steer_run> [<steer_run> ...]
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import yaml
from omegaconf import OmegaConf

AXES = yaml.safe_load(open("conf/interpret/starling.yaml"))["action_axes"]
NA = len(AXES)


def recorded_baseline() -> tuple:
    """What step-to-step change the real sticks make, at the SAME rate the planner commands at."""
    from quickdraw.data.dataset import load_split_episodes_mm, set_action_aggregate, set_subsample
    from quickdraw.training.setup import image_head_cams, image_head_sizes, resolve_data_root
    cfg = OmegaConf.create(json.load(open(
        "logs/train_action_2026_09_14_04_41_17_s2_ah_chunk32_full/logs/config.json")))
    set_subsample(int(cfg.data.get("subsample", 1) or 1)); set_action_aggregate("concat")
    eps = load_split_episodes_mm(resolve_data_root(cfg), "val", img_size=image_head_sizes(cfg),
                                 cam=image_head_cams(cfg), repo_id="starling-2")
    d = []
    for _, a, _ in eps:
        f = a.reshape(len(a), -1, NA).mean(axis=1)          # same fold the readouts use
        d.append(np.abs(np.diff(f, axis=0)))
    d = np.concatenate(d, 0)
    return float(d.mean()), np.percentile(d, 95, axis=0)


def main(*runs: str) -> int:
    rec, rec95 = recorded_baseline()
    runs = [r for r in runs if not r.startswith("stride=")]
    print(f"\n  RECORDED sticks, step-to-step |da|: mean {rec:.4f}   p95 per axis "
          f"{[round(float(x), 3) for x in rec95]}")
    for run in runs:
        fs = sorted(glob.glob(os.path.join(run, "logs", "epoch_*", "eval_steer", "plans", "*", "*",
                                           "actions.npy")))
        if not fs:
            continue
        obj = json.load(open(os.path.join(os.path.dirname(fs[0]), "plan.json")))
        # The chunk comes from the proposal's OWN name, not a guess: PriorProposal writes "chunk=32" and
        # DataProposal writes "K=32", and reading the wrong one silently counts interior steps as seams.
        # THE SEAM SPACING IS `commit`, NOT THE CHUNK. A plan re-draws every `commit` steps, so with
        # commit 16 out of a 32-chunk the seams are at 16, 32, 48 ... -- reading the chunk instead counted
        # every other seam as interior and diluted BOTH columns. plan.json now records commit; older runs
        # fall back to the chunk, or to the `stride` argument.
        import re as _re
        if len(sys.argv) > 1 and sys.argv[-1].startswith("stride="):
            K = int(sys.argv[-1].split("=")[1])
        elif obj.get("commit"):
            K = int(obj["commit"])
        else:
            mm = _re.search(r"(?:chunk|K)=(\d+)", obj.get("proposal", ""))
            assert mm, f"cannot read the seam spacing from proposal {obj.get('proposal')!r}"
            K = int(mm.group(1))
        seam, inside = [], []
        for f in fs:
            a = np.load(f).reshape(-1, 1, NA * 0 + NA) if False else np.load(f)
            fold = a.reshape(len(a), -1, NA).mean(axis=1)
            dd = np.abs(np.diff(fold, axis=0))              # dd[i] is the change INTO step i+1
            idx = np.arange(1, len(fold))
            m = (idx % K) == 0                              # the first step of a freshly drawn chunk
            seam.append(dd[m]); inside.append(dd[~m])
        seam, inside = np.concatenate(seam, 0), np.concatenate(inside, 0)
        print(f"\n  {os.path.basename(run)}   objective={obj.get('objective')}  commit/chunk={K}")
        print(f"    inside a chunk   |da| mean {inside.mean():.4f}   ({inside.mean() / rec:5.2f}x recorded)")
        print(f"    at the seam      |da| mean {seam.mean():.4f}   ({seam.mean() / rec:5.2f}x recorded)"
              f"   seam/inside {seam.mean() / max(1e-9, inside.mean()):.2f}x")
        print(f"    per axis, seam:  " + "  ".join(f"{ax['name']} {v:.3f}"
                                                   for ax, v in zip(AXES, seam.mean(axis=0))))
    print("\n  A seam much larger than `inside` means the stitch is where the plan jerks, and a continuity")
    print("  penalty across the boundary would be the thing to add. Comparable numbers mean the prior's")
    print("  chunks already join smoothly and a penalty would only cost reward for nothing.")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
