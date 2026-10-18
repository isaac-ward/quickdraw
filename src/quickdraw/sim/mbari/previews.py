"""Four-camera preview videos, one per split, so a trajectory can be WATCHED rather than inferred.

Every visual fault in this environment so far -- the destroyed AUV mesh, the window columns on the
end walls, the planked concrete, the upside-down diver -- was invisible in every metric and obvious
in the first frame someone looked at. These previews exist so that looking is cheap.

MP4 AND GIF. The mp4 is the one to watch (h264, full rate); the gif is for anywhere a video will not
play, capped by `save_gif` to a size that will actually load. The panel layout is the demo frame:
FPV, Chase, Top-down, Side-on, each labelled, FPV first because that is the only stream the world
model is trained on.
"""
from __future__ import annotations

import glob
import pathlib

import numpy as np

CAMS = ("fpv", "chase", "topdown", "sideon")
TITLES = ("FPV", "Chase", "Top-down", "Side-on")
SPLITS = ("train", "ood_thruster", "ood_fish", "ood_buoy")
CAP = 900                         # 30 s at 30 Hz; the whole nominal episode


def one(npz_path, out_stem, cap=CAP, fps=30, log=print):
    """Write <out_stem>.mp4 and .gif from one episode npz. Returns a one-line summary."""
    from ...logging import viz

    z = np.load(npz_path)
    panels = [z[f"frames_{c}"][:cap].astype(np.float32) / 255 for c in CAMS]
    canvas = viz.label_panels(panels, list(TITLES), panel_wh=(230, 172), header_px=40, font_px=25)
    viz.save_mp4(f"{out_stem}.mp4", canvas, fps)
    viz.save_gif(f"{out_stem}.gif", canvas, fps=15, max_mb=45, log=lambda *_: None)
    o = z["obs"]
    return (f"{len(o)} steps | speed {np.linalg.norm(o[:, 7:10], axis=1).mean():.2f} m/s | "
            f"pos span {np.round(np.ptp(o[:, :3], 0), 1)} | {str(z['meta'])[:70]}")


def main(root, out, n_per=1, log=print):
    """One preview per split (or `n_per`), written to `out` as mbari_<split>_<i>.{mp4,gif}."""
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        for i, f in enumerate(sorted(glob.glob(f"{root}/{split}/*.npz"))[:n_per]):
            stem = out / f"mbari_{split}_{i}"
            log(f"  {split:13s} {one(f, str(stem), log=log)}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Four-camera preview videos, one per split.")
    ap.add_argument("--root", default="/app/logs/mbari/data/v4")
    ap.add_argument("--out", default="/app/logs/mbari/data/previews")
    ap.add_argument("--n-per", type=int, default=1)
    a = ap.parse_args()
    main(a.root, a.out, a.n_per, log=lambda m: print(m, flush=True))
