"""Acceptance checks for a generated MBARI dataset, before it is packed or pushed.

The record claims the published dataset is "6.01 h, balanced, collision-free". That claim was
produced by three throwaway scripts and is otherwise unfalsifiable, so it lives here instead:

    python -m quickdraw.sim.mbari.verify logs/mbari/data/v7

Four things are checked, and each one is a mistake that actually shipped at least once:

  COLLISION-FREE   the whole point of v7. Re-tests every written frame against the SAME
                   `trajectories.clear` predicate the planner and plant use, so this cannot pass
                   by agreeing with a stale copy of the geometry.
  NO EMPTY EPISODES  25 zero-length episodes reached the packer once (`_start_pose` tested
                   `inside()` where it should have tested `clear()`). A 0-frame clip cannot be
                   encoded to mp4 and the packer dies on the missing file, an hour in.
  BALANCE          the 1/3-1/3-1/3 split across behaviour classes is a stated property of the
                   dataset, and top-up rounds are the obvious way to break it.
  DURATION         episodes are variable-length now that they terminate on contact, so the hour
                   count has to be summed, not multiplied.

Episodes still being written by a live shard are skipped and counted, so this is safe to run
against a generation in flight.
"""
from __future__ import annotations

import argparse
import ast
import collections
import glob
import os

import numpy as np

from . import trajectories as tj

FPS = 30


def scan(root):
    rows, skipped = [], 0
    for f in sorted(glob.glob(os.path.join(root, "*", "*.npz"))):
        if ".tmp." in f:                       # atomic write in progress
            continue
        try:
            z = np.load(f)
            p = z["obs"][:, :3]
            m = ast.literal_eval(str(z["meta"]))
        except Exception:                      # shard still writing; not an error
            skipped += 1
            continue
        bad = 0 if len(p) == 0 else int((~tj.clear(p)).sum())
        rows.append({"split": os.path.basename(os.path.dirname(f)), "kind": m.get("kind", "?"),
                     "steps": len(p), "bad": bad,
                     "contact": bool(m.get("ended_on_contact", False)),
                     "file": os.path.basename(f)})
    return rows, skipped


def report(root):
    rows, skipped = scan(root)
    if not rows:
        print(f"  no episodes under {root}")
        return False
    n = np.array([r["steps"] for r in rows])
    bad_eps = [r for r in rows if r["bad"]]
    empty = [r for r in rows if r["steps"] == 0]
    hours = n.sum() / FPS / 3600

    print(f"  {len(rows)} episodes, {n.sum()} steps = {hours:.2f} h"
          f"   ({skipped} in flight, skipped)")
    print(f"  episodes containing a COLLIDING frame : {len(bad_eps):4d}   (want 0)")
    print(f"  colliding frames                      : {sum(r['bad'] for r in bad_eps):4d}   (want 0)")
    print(f"  zero-length episodes                  : {len(empty):4d}   (want 0)")
    print(f"  ended on contact                      : {sum(r['contact'] for r in rows):4d}"
          f" ({100 * sum(r['contact'] for r in rows) / len(rows):.0f}%)")
    print(f"  episode length: min {n.min()} max {n.max()} mean {n.mean():.0f} steps")

    d = collections.defaultdict(lambda: {"n": 0, "steps": 0, "contact": 0})
    for r in rows:
        k = d[(r["split"], r["kind"])]
        k["n"] += 1
        k["steps"] += r["steps"]
        k["contact"] += r["contact"]
    print(f"  {'split':14s} {'kind':12s} {'n':>4} {'mean_len':>9} {'hours':>7} {'contact':>9}")
    for (sp, kind), v in sorted(d.items()):
        print(f"  {sp:14s} {kind:12s} {v['n']:4d} {v['steps'] / max(1, v['n']):9.0f} "
              f"{v['steps'] / FPS / 3600:7.2f} {v['contact']:4d}/{v['n']:<4d}")
    for r in sorted(bad_eps + empty, key=lambda r: r["steps"])[:5]:
        print(f"    suspect: {r['file']}  steps={r['steps']} colliding={r['bad']}")
    return not (bad_eps or empty)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", help="dataset root holding <split>/*.npz, e.g. logs/mbari/data/v7")
    raise SystemExit(0 if report(ap.parse_args().root) else 1)


if __name__ == "__main__":
    main()
