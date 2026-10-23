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
# 20% val, not the house 10%. Val does double duty here: it is the checkpoint monitor AND the
# nominal reference the OOD detector calibrates against, and that second job is the binding one --
# thresholds set on data the model trained on are optimistically biased, so the reference has to be
# held out, and 72 episodes is a thin sample to set a 90th-percentile threshold from. 144 is not.
VAL_FRAC = 0.2
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


def build(root="/app/logs/mbari/data/v6", name="mbari-tank", cams=CAMS, limit=None, log=print):
    """MEMORY. build_recorded_dataset holds every Episode in RAM before encoding, and four 144x192
    streams over 390 episodes is ~116 GB of uint8. That fits on this box but loading it took longer
    than training would. Pass `cams=("fpv",)` for the training dataset (18 GB) and run the full
    four-camera pack separately -- the npz shards on disk are the source of truth for both."""
    from ...data.processors import build_recorded_dataset

    # `limit` caps the NOMINAL episodes only, taking the first N. Job i always carries seed 1000+i
    # and class CLASSES[i%3], so the first 240 of 720 are exactly the 2.00 h dataset and any prefix
    # stays class-balanced -- which is what makes a data-volume ablation a strict subset rather than
    # a different draw.
    train = episodes_from(root, "train", cams, limit=limit, log=log)
    extra = {s: episodes_from(root, s, cams, log=log) for s in SPLIT_TASK if
             (pathlib.Path(root) / s).exists()}
    extra = {k: v for k, v in extra.items() if v}
    if not train:
        raise RuntimeError(f"no train episodes under {root}")
    return build_recorded_dataset(name, train, FPS, list(cams), log=log, extra_splits=extra,
                                  obs_identity=QUAT_DIMS, obs_names=STATE_NAMES,
                                  act_names=ACTION_NAMES, val_frac=VAL_FRAC)


CONTROLLED_TASK = {"nominal_test": "nominal_circle_inward",
                   "ood_thruster": "ood_thruster_glitch_onset",
                   "ood_fish": "ood_object_shark_onset",
                   "ood_buoy": "ood_appearance_buoy_onset"}


def build_controlled(root="/app/logs/mbari/data/ood_controlled", name="mbari-tank-ood-controlled",
                     cams=("fpv",), log=print):
    """Pack the CONTROLLED onset set so its calibration/test boundary survives packing.

    The normal packer draws train/val randomly out of the primary split. That is exactly wrong here:
    the 60 nominal episodes are deliberately 30 CALIBRATION + 30 TEST so a conformal threshold can be
    fitted on one half and the false-positive rate measured on the other half the fitting never saw.
    A re-draw would mix them and destroy the one property the set was generated for.

    So the two halves are handed over TOGETHER as the primary, with `val_ids` naming the test indices
    exactly -- the packer's "random" split is thereby pinned to the boundary we want:

        split `train` = the 30 nominal CALIBRATION episodes
        split `val`   = the 30 nominal TEST episodes
        splits ood_*  = 30 each, as generated

    NOTHING IS TRAINED ON ANY OF IT. `train` and `val` are the packer's names, not a claim about use;
    this dataset is for scoring a model trained elsewhere. (An earlier attempt passed `val_ids=set()`
    to suppress the val split altogether and tripped an assertion that exists to stop a caller
    silently shipping an empty val.)
    """
    from ...data.processors import build_recorded_dataset

    calib = episodes_from(root, "nominal_calib", cams, log=log)
    test = episodes_from(root, "nominal_test", cams, log=log)
    if not calib or not test:
        raise RuntimeError(f"need both nominal_calib and nominal_test under {root}")
    nominal = calib + test
    val_ids = set(range(len(calib), len(nominal)))        # the test half, pinned
    extra = {s: episodes_from(root, s, cams, log=log) for s in CONTROLLED_TASK
             if s != "nominal_test" and (pathlib.Path(root) / s).exists()}
    extra = {k: v for k, v in extra.items() if v}
    log(f"  train<-calib {len(calib)} | val<-test {len(test)} | extras " +
        ", ".join(f"{k} {len(v)}" for k, v in extra.items()))
    return build_recorded_dataset(name, nominal, FPS, list(cams), log=log, extra_splits=extra,
                                  obs_identity=QUAT_DIMS, obs_names=STATE_NAMES,
                                  act_names=ACTION_NAMES, val_ids=val_ids)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/app/logs/mbari/data/v6")
    ap.add_argument("--name", default="mbari-tank")
    ap.add_argument("--cams", default=",".join(CAMS))
    ap.add_argument("--limit", type=int, default=None, help="cap nominal episodes (data ablation)")
    ap.add_argument("--controlled", action="store_true",
                    help="pack the controlled onset set, preserving its calibration/test split")
    a = ap.parse_args()
    if a.controlled:
        print("run_dir:", build_controlled(a.root, a.name, cams=tuple(a.cams.split(","))), flush=True)
    else:
        print("run_dir:", build(a.root, a.name, cams=tuple(a.cams.split(",")), limit=a.limit), flush=True)
