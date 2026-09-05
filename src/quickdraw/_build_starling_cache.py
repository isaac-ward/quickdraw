"""Pre-decode the starling video streams into the .npy frame caches quickdraw trains from.

WHY THIS EXISTS. `data.dataset.load_fpv_frames` already does the work -- read
`<split>/videos/observation.images.<cam>/*/*.mp4`, AREA-downsample once, cache to
`<split>/<cam>_<tag>.npy` -- but it does it LAZILY on first use, inside the training run. On these
datasets that is ~475k frames of H.264 to decode before step 1, twice (train + val), while a GPU sits
idle. The robocasa/torus repos ship their caches; the starling repos do not, so they are built here.

No new decoding logic: this calls the same tested loader, so the cache is bit-identical to what training
would have produced on its own. Idempotent -- an existing cache is returned untouched.

    python -m quickdraw._build_starling_cache            # both repos, native + (56,96)
"""
from __future__ import annotations

import os
import sys
import time

from .data.dataset import load_fpv_frames

REPOS = ("isaac-ronald-ward/starling", "isaac-ronald-ward/starling-2")
# NATIVE (112,192) is the source resolution -- the honest full-detail cache. (56,96) is the halved
# aspect-preserving one: 96 is this project's working width, and a square resize would distort a 12:7
# egocentric frame, which is exactly the kind of silent geometry error the record keeps warning about.
SIZES = (None, (56, 96))
CAM = "ego"


def main() -> int:
    from huggingface_hub import snapshot_download
    for repo in REPOS:
        root = snapshot_download(repo, repo_type="dataset", ignore_patterns=["media/*"])
        for split in ("train", "val", "eval"):
            if not os.path.isdir(os.path.join(root, split)):
                continue
            for size in SIZES:
                t = time.time()
                a = load_fpv_frames(root, split, size=size, cam=CAM)
                print(f"[starling] {repo.split('/')[-1]:11s} {split:5s} "
                      f"{str(size or 'native'):9s} -> {a.shape} {a.dtype} "
                      f"{a.nbytes / 1e9:.2f} GB in {time.time() - t:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
