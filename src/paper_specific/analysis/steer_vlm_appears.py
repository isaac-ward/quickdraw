"""Does the requested target APPEAR IN THE IMAGINED SEQUENCE? The location score, re-measured.

steer_vlm_objects.py reuses the labelling schema the reward head was trained on, which asks for the ONE
most prominent object and the ONE region in front of the drone at the END of the clip. That is the right
question for a training label and the wrong question for steering: a plan that flies to the table while
the ladder also stands in frame scores nothing, and a plan that reaches the requested wall and then turns
away in the last second scores nothing either. Measured that way the learned prior read 0/4 on requests
its own video plainly satisfies.

So ask a different question, of the same model, on the same clips: which of these objects is clearly
visible AT ANY POINT, and which of these regions does the drone face AT ANY POINT. Multi-label, not
argmax. The null is still inside the matrix -- P(target listed | target requested) against
P(target listed | something else requested) -- so a planner that always drifts to the black panels still
gains nothing on the diagonal, and a labeller that says "everything" drives hit and base together and the
lift to 1.0. Being more permissive cannot manufacture a result; it can only stop discarding one.

    OPENAI_API_KEY=... python -m paper_specific.analysis.steer_vlm_appears <steer_run> [<steer_run> ...]
"""
from __future__ import annotations

import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import yaml

from quickdraw.evaluation.interpret import build_action_text, label_clip, openai_api_key

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from steer_vlm_objects import IC, AXES, OBJ, REG, MODEL, NFR, read_mp4    # noqa: E402

OUT_NAME = "vlm_object_appears.json"

# The scene paragraph is worth keeping verbatim -- it is what lets the model read 112x192 frames at all --
# so the shared prompt is cut at the point where it starts asking for single labels and the question is
# replaced. Everything about direction and blur stays.
_SCENE = IC["prompt"].split("Report three things.")[0].rstrip()
PROMPT = _SCENE + """

Report two things, and for BOTH of them consider the WHOLE clip, not just its final frame. Something
counts if it is clearly visible at any point, even briefly, and even if it is off to the side.

  objects_seen — every object from this list that is clearly visible at some point in the clip:
    black panel, ladder, clock, mannequin, black floor mat, colored floor mat, door, window, whiteboard,
    table.
    List all that apply. Do not list something you cannot actually make out; the frames are
    LOW-RESOLUTION and a guess is worse than an omission. An empty list is a valid answer.

  regions_faced — every part of the room the drone is pointed at, at some point in the clip:
    floor to ceiling glass wall — the long side of the room walled in full-height glass panels/windows.
    wall with black panels — the end of the room where the large black backdrop panels stand.
    white wall with flat window and door — a plain white wall carrying a small flat window and a door.
    white wall with table — a plain white wall with workbenches or tables against it.
    center of room over mats — NOT facing a wall: looking out across the open middle, coloured floor
      mats below or ahead.
    center of room near ladder — NOT facing a wall: the open middle, with the step ladder nearby.
    The drone turns and flies during the clip, so more than one region is normal. List all it faces.

In "reasoning", FIRST describe in a few words what the clip shows and how the view changes, THEN commit
to the two lists. Also give a self_reported_confidence in [0,1], lowering it when the frames are blurry
"""


SCHEMA = {"type": "object", "properties": {
    "reasoning": {"type": "string"},
    "objects_seen": {"type": "array", "items": {"type": "string", "enum": OBJ}},
    "regions_faced": {"type": "array", "items": {"type": "string", "enum": REG}},
    "self_reported_confidence": {"type": "number"}},
    "required": ["reasoning", "objects_seen", "regions_faced", "self_reported_confidence"],
    "additionalProperties": False}


def matrix(rows, buckets, title):
    """rows: (requested_bucket, [observed buckets]). Diagonal against the per-target base rate."""
    req = [r for r in rows if r[0] in buckets]
    if not req:
        return
    print(f"\n  {title}")
    print(f"  {'request':38s} {'n':>3s} {'hit':>6s} {'base':>6s} {'lift':>6s} {'listed':>7s}")
    print("  " + "-" * 78)
    hits, bases = [], []
    for b in buckets:
        mine = [o for q, o in req if q == b]
        others = [o for q, o in req if q != b]
        if not mine:
            continue
        hit = float(np.mean([b in o for o in mine]))
        base = float(np.mean([b in o for o in others])) if others else float("nan")
        lift = hit / base if base and base > 0 else float("inf") if hit > 0 else float("nan")
        hits.append(hit); bases.append(base)
        print(f"  {b:38s} {len(mine):>3d} {hit:>6.2f} {base:>6.2f} "
              f"{(f'{lift:>6.1f}' if np.isfinite(lift) else '   inf' if hit > 0 else '     -')}"
              f" {np.mean([len(o) for o in mine]):>7.1f}")
    print(f"  {'overall':38s} {len(req):>3d} {np.mean(hits):>6.2f} {np.nanmean(bases):>6.2f}"
          f" {np.mean(hits) / max(1e-9, np.nanmean(bases)):>6.1f}")


def main(*runs: str) -> int:
    key = openai_api_key()
    for run in runs:
        jobs = []
        for f in sorted(glob.glob(os.path.join(run, "logs", "epoch_*", "eval_steer", "plans", "*", "*",
                                               "plan.json"))):
            d = json.load(open(f))
            if d["request"] in OBJ + REG:                  # directions are checked from the actions
                jobs.append((d["request"], os.path.dirname(f)))
        if not jobs:
            print(f"\n=== {os.path.basename(run)}: no object/region plans found"); continue
        print(f"\n=== {os.path.basename(run)}   {len(jobs)} clips -> VLM ({MODEL}, {NFR} frames each)")

        def one(job):
            req, dd = job
            fr = read_mp4(os.path.join(dd, "image.mp4"), NFR)
            at = build_action_text(np.load(os.path.join(dd, "actions.npy")), AXES)
            return (req, label_clip(api_key=key, model=MODEL, prompt=PROMPT, schema=SCHEMA,
                                    frames_uint8=fr, action_text=at))

        with ThreadPoolExecutor(max_workers=int(IC.get("max_workers", 8))) as ex:
            out = list(ex.map(one, jobs))
        bad = sum(1 for _, l in out if l is None)
        print(f"  {len(out) - bad}/{len(out)} labelled ({bad} failed)")
        matrix([(q, l["objects_seen"]) for q, l in out if l], OBJ,
               "OBJECT requests -- did the object appear anywhere in the imagined clip?")
        matrix([(q, l["regions_faced"]) for q, l in out if l], REG,
               "REGION requests -- was the region faced at any point?")
        json.dump([{"request": q, "label": l} for q, l in out],
                  open(os.path.join(run, OUT_NAME), "w"), indent=1)
        print(f"  raw labels -> {run}/{OUT_NAME}")
    print("\n  `listed` is the mean number of items the labeller returned per clip. If it climbs toward")
    print("  the list length, hit and base climb together and the lift falls to 1.0 -- permissiveness")
    print("  cannot fake a diagonal.")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
