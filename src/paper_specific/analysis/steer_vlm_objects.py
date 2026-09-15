"""Does asking for an object actually PUT IT IN VIEW? The measurement the object requests never had.

The 8 direction requests can be checked from the committed actions -- exact, and independent of the reward
head. The 16 object and region requests cannot: nothing in the imagination measures "is the ladder in
view", so the only number that moved for them was the reward the planner maximises, which is circular.

So ask the VLM, with the SAME prompt, schema and model that produced the labels the reward head was
trained on (evaluation/interpret.label_clip + conf/interpret/starling.yaml), applied now to the PLANNED
imaginations instead of to unconditioned ones.

THE NULL IS INSIDE THE MATRIX, which is what makes this honest. For each object X, compare
P(VLM sees X | the request was X) against P(VLM sees X | the request was something else). A planner whose
plans always drift to the black panels scores a high hit rate on "black panel" and gains nothing on the
diagonal -- the lift is what language bought.

    OPENAI_API_KEY=... python scratch/steer_vlm_objects.py <steer_run> [<steer_run> ...]
"""
from __future__ import annotations

import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import yaml

from quickdraw.evaluation.interpret import (build_action_text, build_label_schema, label_clip,
                                            openai_api_key)

IC = yaml.safe_load(open("conf/interpret/starling.yaml"))
AXES = IC["action_axes"]
OBJ = list(IC["factors"]["object_in_view"]["buckets"])
REG = list(IC["factors"]["facing"]["buckets"])
MODEL = IC.get("model") or "gpt-4o"      # conf/interpret/starling.yaml:84 -- the SAME model the
#                                          training labels came from, which is the point of reusing it
NFR = int(IC.get("vlm_frames", 15))


def read_mp4(path, n):
    cap = cv2.VideoCapture(path)
    fr = []
    while True:
        ok, x = cap.read()
        if not ok:
            break
        fr.append(x[..., ::-1])
    cap.release()
    fr = np.stack(fr)
    # the plan mp4 carries a caption bar under the frame; crop it off so the VLM sees only the imagination
    h = fr.shape[1]
    for cut in range(h - 1, h // 2, -1):
        if fr[:, cut - 1:cut].max() > 8:            # first non-black row from the bottom
            fr = fr[:, :cut]
            break
    idx = np.unique(np.linspace(0, len(fr) - 1, min(n, len(fr))).round().astype(int))
    return fr[idx]


def matrix(rows, wanted, buckets, title):
    """rows: list of (requested_bucket, observed_bucket). Diagonal vs the per-object base rate."""
    req = [r for r in rows if r[0] in buckets]
    if not req:
        return
    print(f"\n  {title}")
    print(f"  {'request':34s} {'n':>3s} {'hit':>6s} {'base':>6s} {'lift':>6s}   most common observation")
    print("  " + "-" * 92)
    hits, bases = [], []
    for b in buckets:
        mine = [o for q, o in req if q == b]
        others = [o for q, o in req if q != b]
        if not mine:
            continue
        hit = float(np.mean([o == b for o in mine]))
        base = float(np.mean([o == b for o in others])) if others else float("nan")
        top = max(set(mine), key=mine.count)
        lift = hit / base if base and base > 0 else float("inf") if hit > 0 else float("nan")
        hits.append(hit); bases.append(base)
        print(f"  {b:34s} {len(mine):>3d} {hit:>6.2f} {base:>6.2f} "
              f"{(f'{lift:>6.1f}' if np.isfinite(lift) else '   inf' if hit > 0 else '     -')}"
              f"   {top} ({mine.count(top)}/{len(mine)})")
    print(f"  {'OVERALL':34s} {len(req):>3d} {np.mean(hits):>6.2f} {np.nanmean(bases):>6.2f}"
          f" {np.mean(hits) / max(1e-9, np.nanmean(bases)):>6.1f}")


def main(*runs: str) -> int:
    key = openai_api_key()
    schema = build_label_schema({k: IC["factors"][k] for k in ("facing", "object_in_view")}, n_captions=0)
    for run in runs:
        jobs = []
        for f in sorted(glob.glob(os.path.join(run, "logs", "epoch_*", "eval_steer", "plans", "*", "*",
                                               "plan.json"))):
            d = json.load(open(f))
            if d["request"] not in OBJ + REG:
                continue                                   # directions are checked from the actions
            jobs.append((d["request"], os.path.dirname(f)))
        if not jobs:
            print(f"\n=== {os.path.basename(run)}: no object/region plans found"); continue
        print(f"\n=== {os.path.basename(run)}   {len(jobs)} clips -> VLM ({MODEL}, {NFR} frames each)")

        def one(job):
            req, dd = job
            fr = read_mp4(os.path.join(dd, "image.mp4"), NFR)
            at = build_action_text(np.load(os.path.join(dd, "actions.npy")), AXES)
            lab = label_clip(api_key=key, model=MODEL, prompt=IC["prompt"], schema=schema,
                             frames_uint8=fr, action_text=at)
            return (req, lab)

        with ThreadPoolExecutor(max_workers=int(IC.get("max_workers", 8))) as ex:
            out = list(ex.map(one, jobs))
        bad = sum(1 for _, l in out if l is None)
        rows_o = [(q, l["object_in_view"]) for q, l in out if l]
        rows_r = [(q, l["facing"]) for q, l in out if l]
        print(f"  {len(out) - bad}/{len(out)} labelled ({bad} failed)")
        matrix(rows_o, OBJ, OBJ, "OBJECT requests -- did the VLM see the object that was asked for?")
        matrix(rows_r, REG, REG, "REGION requests -- was the drone facing the region that was asked for?")
        json.dump([{"request": q, "label": l} for q, l in out],
                  open(os.path.join(run, "vlm_object_check.json"), "w"), indent=1)
        print(f"  raw labels -> {run}/vlm_object_check.json")
    print("\n  lift = hit / base. 1.0 means asking for the object changed nothing; the base rate is how")
    print("  often that object turns up when something ELSE was requested, so the null is inside the run.")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
