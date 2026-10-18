"""Pack the generated npz episodes into the house dataset format, then optionally push to HF.

Goes through `data.processors.build_recorded_dataset`, the same path starling / robocasa / block-stack
use, so the result needs no loader change: `data.cam=fpv` picks the training view and the other three
cameras ride along as extra `observation.images.*` keys.

TWO THINGS THIS SETS THAT THE DEFAULTS WOULD GET WRONG:

  per-episode `task`. lerobot's task field is the only free-text label that survives packaging, and it
  is how a consumer slices the dataset by condition. Every episode is labelled with its behaviour
  class (`nominal_circle`) or its OOD condition (`ood_thruster_glitch`). The record in
  data/processors.py spells out what happens otherwise: starling-2's eval split shipped 49 episodes
  from four separate campaigns all labelled `starling-2`, and which was which is now unrecoverable.

  `obs_identity` on the quaternion. Dims 3:7 are the components of a UNIT quaternion -- already O(1),
  already zero-mean-ish, and z-scoring them per-component would distort the rotation manifold and let
  the model produce "quaternions" that are not unit. Same argument block-stack's rot6 dims get.
"""
from __future__ import annotations

import ast
import glob
import pathlib

import numpy as np

from .mola import ACTION_NAMES, STATE_NAMES
from .record import CAMS, FPS

QUAT_DIMS = [3, 4, 5, 6]
SPLIT_TASK = {"ood_thruster": "ood_thruster_glitch", "ood_fish": "ood_object_shark",
              "ood_buoy": "ood_appearance_buoy"}


def _load(path):
    z = np.load(path)
    meta = ast.literal_eval(str(z["meta"]))
    return z, meta


def episodes_from(root, split, cams=CAMS, limit=None, log=print):
    """npz shards -> list[processors.Episode], in episode-index order."""
    from ...data.processors import Episode

    files = sorted(glob.glob(str(pathlib.Path(root) / split / "ep_*.npz")))[:limit]
    eps = []
    for f in files:
        z, meta = _load(f)
        task = SPLIT_TASK.get(split, f"nominal_{meta['kind']}")
        eps.append(Episode(states=z["obs"].astype(np.float32),
                           actions=z["act"].astype(np.float32),
                           frames={c: z[f"frames_{c}"] for c in cams},
                           task=task))
    log(f"  {split:14s} {len(eps):4d} episodes, {sum(len(e.states) for e in eps):7d} steps")
    return eps


def build(root="/app/logs/mbari/data/v6", name="mbari-tank", cams=CAMS, log=print):
    """MEMORY. build_recorded_dataset holds every Episode in RAM before encoding, and four 144x192
    streams over 390 episodes is ~116 GB of uint8. That fits on this box but loading it took longer
    than training would. Pass `cams=("fpv",)` for the training dataset (18 GB) and run the full
    four-camera pack separately -- the npz shards on disk are the source of truth for both."""
    from ...data.processors import build_recorded_dataset

    train = episodes_from(root, "train", cams, log=log)
    extra = {s: episodes_from(root, s, cams, log=log) for s in SPLIT_TASK if
             (pathlib.Path(root) / s).exists()}
    extra = {k: v for k, v in extra.items() if v}
    if not train:
        raise RuntimeError(f"no train episodes under {root}")
    return build_recorded_dataset(name, train, FPS, list(cams), log=log, extra_splits=extra,
                                  obs_identity=QUAT_DIMS, obs_names=STATE_NAMES,
                                  act_names=ACTION_NAMES)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/app/logs/mbari/data/v6")
    ap.add_argument("--name", default="mbari-tank")
    ap.add_argument("--cams", default=",".join(CAMS))
    a = ap.parse_args()
    print("run_dir:", build(a.root, a.name, cams=tuple(a.cams.split(","))), flush=True)
