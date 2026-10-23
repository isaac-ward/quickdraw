"""Four-camera preview videos, one per split, so a trajectory can be WATCHED rather than inferred.

Every visual fault in this environment so far -- the destroyed AUV mesh, the window columns on the
end walls, the planked concrete, the upside-down diver -- was invisible in every metric and obvious
in the first frame someone looked at. These previews exist so that looking is cheap.

MP4 ONLY BY DEFAULT. A 30 s four-panel clip is 900 frames, and as a gif that is 80 MB -- over
`save_gif`'s 45 MB cap, which raises rather than silently shipping something unloadable. The mp4 is
the deliverable; `--gif` writes a 10 s slice at 12 fps for anywhere a video will not play. The panel
layout is the demo frame: FPV, Chase, Top-down, Side-on, each labelled, FPV first because that is
the only stream the world model is trained on.
"""
from __future__ import annotations

import glob
import pathlib

import numpy as np

CAMS = ("fpv", "chase", "topdown", "sideon")
TITLES = ("FPV", "Chase", "Top-down", "Side-on")
# DISCOVERED, NOT LISTED. This was a fixed tuple and the controlled onset set -- whose nominal
# halves are `nominal_calib` and `nominal_test` -- silently got previews for its three ood_* splits
# and none for its 60 nominal episodes. A hardcoded split list quietly skips any dataset that does
# not use the names it was written for, and says nothing.
SPLITS = ("train", "ood_thruster", "ood_fish", "ood_buoy")


def _splits(root):
    """Every subdirectory holding npz episodes, with the known names first for stable ordering."""
    import os
    found = sorted(d for d in os.listdir(root)
                   if (pathlib.Path(root) / d).is_dir() and glob.glob(f"{root}/{d}/*.npz"))
    return [s for s in SPLITS if s in found] + [s for s in found if s not in SPLITS]
CAP = 900                         # 30 s at 30 Hz; the whole nominal episode


def one(npz_path, out_stem, cap=CAP, fps=30, gif=False, log=print):
    """Write <out_stem>.mp4 (and .gif if asked) from one episode npz. Returns a one-line summary."""
    from ...logging import viz

    z = np.load(npz_path)
    panels = [z[f"frames_{c}"][:cap].astype(np.float32) / 255 for c in CAMS]
    canvas = viz.label_panels(panels, list(TITLES), panel_wh=(230, 172), header_px=40, font_px=25)
    viz.save_mp4(f"{out_stem}.mp4", canvas, fps, quality=7)   # 7 not 9: ~870 clips ship with the
    #                                                            dataset, and 9 doubles the upload
    if gif:
        viz.save_gif(f"{out_stem}.gif", canvas[:300], fps=12, max_mb=45, log=lambda *_: None)
    o = z["obs"]
    return (f"{len(o)} steps | speed {np.linalg.norm(o[:, 7:10], axis=1).mean():.2f} m/s | "
            f"pos span {np.round(np.ptp(o[:, :3], 0), 1)} | {str(z['meta'])[:70]}")


def main(root, out, n_per=None, gif=False, shard=0, shards=1, log=print):
    """A preview for EVERY episode (or the first `n_per` per split), as <out>/<split>/ep_xxxxx.mp4.

    Every trajectory gets one, because a dataset you cannot watch is a dataset nobody checks. At
    ~40 s per 30 s episode that is 3.3 hours serial for 390, so the work is sharded the same way
    generation is: `--shard k --shards n` takes every n-th job and the caller runs n processes.
    Episodes already on disk are skipped, so a re-run resumes."""
    out = pathlib.Path(out)
    jobs = []
    for split in _splits(root):
        for i, f in enumerate(sorted(glob.glob(f"{root}/{split}/*.npz"))):
            if n_per is not None and i >= n_per:
                break
            jobs.append((split, pathlib.Path(f).stem, f))
    mine = [j for n, j in enumerate(jobs) if n % shards == shard]
    log(f"[prev] {len(jobs)} episodes, {len(mine)} in shard {shard}/{shards}")
    for n, (split, stem, f) in enumerate(mine, 1):
        d = out / split
        d.mkdir(parents=True, exist_ok=True)
        if (d / f"{stem}.mp4").exists():
            continue
        line = one(f, str(d / stem), gif=gif, log=log)
        if n % 10 == 0 or n == len(mine):
            log(f"[prev] {n}/{len(mine)}  {split}/{stem}  {line}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Four-camera preview videos, one per split.")
    ap.add_argument("--root", default="/app/logs/mbari/data/v6")
    ap.add_argument("--out", default="/app/logs/mbari/data/previews")
    ap.add_argument("--n-per", type=int, default=None, help="cap per split; default ALL episodes")
    ap.add_argument("--gif", action="store_true", help="also write a 10 s gif slice")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    a = ap.parse_args()
    main(a.root, a.out, a.n_per, gif=a.gif, shard=a.shard, shards=a.shards,
         log=lambda m: print(m, flush=True))
