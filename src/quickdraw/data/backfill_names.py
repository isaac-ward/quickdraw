"""Write the observation/action DIM NAMES into an already-built dataset's lerobot metadata.

`write_lerobot_split` now takes `obs_names`/`act_names` and the rosbag processor passes the flight layout,
so datasets built from 2026-09-14 ship their names. starling-2 predates that and went out as 16 anonymous
floats -- which is why data/rosbag.py had to write STATE_COLUMNS down in a comment, why the robocasa layout
cost a day to reverse-engineer, and why analytic interpretability factors (which must know which dim is
altitude) could not be written for it. Rebuilding a dataset to add a list of strings is not a trade worth
making, so this patches `<split>/meta/info.json` in place.

Additive and safe: it only fills `features.<key>.names` where it is currently null, refuses to overwrite
names that disagree, and checks the count against the declared shape before writing anything.

    python -m quickdraw.data.backfill_names <dataset_root> [--layout starling]

`--layout` selects a known column set (default `starling`, from data/rosbag.py). A dataset whose layout is
not known here should have its processor updated instead -- the names belong at the source.
"""
from __future__ import annotations

import glob
import json
import os
import sys

LAYOUTS = {}


def _layouts():
    """Imported lazily so this module does not drag in the rosbag reader on every dataset import."""
    if not LAYOUTS:
        from .rosbag import ACTION_COLUMNS, STATE_COLUMNS
        LAYOUTS["starling"] = {"observation_vector": list(STATE_COLUMNS), "action": list(ACTION_COLUMNS)}
    return LAYOUTS


def patch(info_path: str, names: dict) -> list[str]:
    """Fill in `features.<key>.names` for one meta/info.json. Returns the changes made, [] if already named."""
    info = json.load(open(info_path))
    feats = info.get("features", {})
    changed = []
    for key, cols in names.items():
        f = feats.get(key)
        if f is None:
            continue                                    # this split does not carry that feature
        n = int(f.get("shape", [len(cols)])[0])
        assert n == len(cols), f"{info_path}: {key} has {n} dims but the layout names {len(cols)}"
        cur = f.get("names")
        if cur == cols:
            continue
        assert cur is None, f"{info_path}: {key} already has DIFFERENT names {cur} -- refusing to overwrite"
        f["names"] = list(cols)
        changed.append(f"{key} ({n} dims)")
    if not changed:
        return []
    # A HuggingFace cache snapshot stores each file as a SYMLINK into ../../blobs/<sha>. Writing through it
    # would edit a content-addressed blob under a hash that no longer describes it, and that blob may be
    # shared with another snapshot -- so replace the link with a real file and leave the blob untouched.
    if os.path.islink(info_path):
        os.unlink(info_path)
    with open(info_path, "w") as f_:
        json.dump(info, f_, indent=2)
    return changed


def main(root: str, layout: str = "starling") -> int:
    names = _layouts()[layout]
    metas = sorted(glob.glob(os.path.join(root, "*", "meta", "info.json")))
    assert metas, f"no <split>/meta/info.json under {root}"
    print(f"  {root}\n  layout {layout!r}: " + ", ".join(f"{k} x{len(v)}" for k, v in names.items()))
    total = 0
    for m in metas:
        ch = patch(m, names)
        split = os.path.basename(os.path.dirname(os.path.dirname(m)))
        print(f"    {split:<24} " + (", ".join(ch) if ch else "already named, unchanged"))
        total += len(ch)
    print(f"  {total} feature(s) named across {len(metas)} split(s)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m quickdraw.data.backfill_names <dataset_root> [layout]")
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "starling"))
