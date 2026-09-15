"""Pre-decode a recorded dataset's video streams into the .npy frame caches quickdraw trains from.

WHY THIS EXISTS. `data.dataset.load_fpv_frames` already does the work -- read
`<split>/videos/observation.images.<cam>/*/*.mp4`, AREA-downsample once, cache to
`<split>/<cam>_<tag>.npy` -- but it does it LAZILY on first use, inside the training run. On these
datasets that is ~475k frames of H.264 to decode before step 1, twice (train + val), while a GPU sits
idle. The robocasa/torus repos ship their caches; the starling repos do not, so they are built here.

No new decoding logic: this calls the same tested loader, so the cache is bit-identical to what training
would have produced on its own. Idempotent -- an existing cache is returned untouched.

    python -m quickdraw._build_frame_cache                 # every repo in REPOS
    python -m quickdraw._build_frame_cache block-stack     # just one
"""
from __future__ import annotations

import os
import sys
import time

from .data.dataset import load_fpv_frames

# repo -> (cameras to decode, sizes). `None` = native resolution.
#
# BLOCK-STACK DECODES ONLY THE SCENE CAMERAS, deliberately. It ships four (scene_left, scene_right,
# gripper_right_bottom, gripper_right_top) and the gripper pair moves WITH the gripper -- so on those the
# frame difference is dominated by camera egomotion, which is the regime that makes a first-order loss a
# poor fit (design/derivative_loss.md §9, and the starling measurements it cites). A fixed scene camera is
# the case where the frame difference IS the object motion. Decoding the gripper pair too would cost
# ~37 GB for frames the experiment should not use.
REPOS = {
    "isaac-ronald-ward/starling":   (("ego",), (None, (56, 96))),
    "isaac-ronald-ward/starling-2": (("ego",), (None, (56, 96))),
    "isaac-ronald-ward/block-stack": (("scene_left", "scene_right"), (None,)),
}


def main() -> int:
    from huggingface_hub import snapshot_download
    only = set(sys.argv[1:])                                   # optional: restrict to named repos
    for repo, (cams, sizes) in REPOS.items():
        if only and not any(o in repo for o in only):
            continue
        root = snapshot_download(repo, repo_type="dataset", ignore_patterns=["media/*"])
        splits = [d for d in sorted(os.listdir(root))
                  if os.path.isdir(os.path.join(root, d, "meta"))]
        for split in splits:
            for cam in cams:
                for size in sizes:
                    t = time.time()
                    a = load_fpv_frames(root, split, size=size, cam=cam)
                    print(f"[cache] {repo.split('/')[-1]:12s} {split:18s} {cam:22s} "
                          f"{str(size or 'native'):9s} -> {a.shape} {a.dtype} "
                          f"{a.nbytes / 1e9:.2f} GB in {time.time() - t:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
