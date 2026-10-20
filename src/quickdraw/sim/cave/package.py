"""Pack the generated cave npz episodes into the house dataset format, then push to HF.

Goes through `data.processors.build_recorded_dataset` -- the same path starling / robocasa / block-stack /
mbari use -- so the result needs no loader change: `data.cam=fpv` picks the one training view.

TWO THINGS SET DELIBERATELY (same reasoning as mbari):
  per-episode `task` = behaviour class (nominal_corridor ...) or OOD condition, the only free-text label
  that survives packaging and lets a consumer slice by condition.
  `obs_identity` on the quaternion (dims 3:7): components of a UNIT quaternion, already O(1); z-scoring
  them per-component would distort the rotation manifold.

Obs (13): x y z | qw qx qy qz | vx vy vz (world) | p q r (body rates). Action (4): vx vy vz | yaw_rate.
20% val -- val is both the checkpoint monitor AND the OOD detector's nominal reference (held-out).
"""
from __future__ import annotations

import ast
import glob
import pathlib

import numpy as np

STATE_NAMES = ["x", "y", "z", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "p", "q", "r"]
ACTION_NAMES = ["vx_cmd", "vy_cmd", "vz_cmd", "yaw_rate_cmd"]
QUAT_DIMS = [3, 4, 5, 6]
CAMS = ("fpv",)
FPS = 30
VAL_FRAC = 0.2
SPLIT_TASK = {"ood_thruster": "ood_rotor_glitch", "ood_person": "ood_object_person",
              "ood_appearance": "ood_appearance_shift"}


def episodes_from(root, split, cams=CAMS, limit=None, log=print):
    from ...data.processors import Episode

    files = sorted(glob.glob(str(pathlib.Path(root) / split / "ep_*.npz")))[:limit]
    eps = []
    for f in files:
        z = np.load(f)
        meta = ast.literal_eval(str(z["meta"]))
        task = SPLIT_TASK.get(split, f"nominal_{meta['kind']}")
        eps.append(Episode(states=z["obs"].astype(np.float32),
                           actions=z["act"].astype(np.float32),
                           frames={c: z[f"frames_{c}"] for c in cams},
                           task=task))
    log(f"  {split:14s} {len(eps):4d} episodes, {sum(len(e.states) for e in eps):7d} steps")
    return eps


def build(root="/home/ubuntu/user_irw/quickdraw/logs/cave/data/v1", name="cave-grapevine",
          cams=CAMS, limit=None, log=print):
    from ...data.processors import build_recorded_dataset

    train = episodes_from(root, "train", cams, limit=limit, log=log)
    extra = {s: episodes_from(root, s, cams, log=log) for s in SPLIT_TASK
             if (pathlib.Path(root) / s).exists()}
    extra = {k: v for k, v in extra.items() if v}
    if not train:
        raise RuntimeError(f"no train episodes under {root}")
    return build_recorded_dataset(name, train, FPS, list(cams), log=log, extra_splits=extra,
                                  obs_identity=QUAT_DIMS, obs_names=STATE_NAMES,
                                  act_names=ACTION_NAMES, val_frac=VAL_FRAC)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/ubuntu/user_irw/quickdraw/logs/cave/data/v1")
    ap.add_argument("--name", default="cave-grapevine")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    print("run_dir:", build(a.root, a.name, limit=a.limit), flush=True)
