"""Add the `action_pit` knots to an ALREADY-BUILT dataset's normalization_stats.json.

`compute_norm_stats` now fits them at build time, but starling-2 was recorded before that and rebuilding a
dataset to gain one additive key is not a trade worth making. This fits the knots from the dataset's own
TRAIN split -- the same arrays the build would have used -- and writes them into the existing file beside
mean/std, after backing it up. Every existing consumer ignores unknown keys, so this is additive.

    python -m quickdraw.data.backfill_pit <dataset_root> [n_knots]   # default 1024, matching the build
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import sys

import numpy as np
import torch

from quickdraw.data.transforms import PIT


def train_actions(root: str) -> np.ndarray:
    fs = sorted(glob.glob(os.path.join(root, "train", "data", "**", "*.parquet"), recursive=True))
    assert fs, f"no train parquet under {root}"
    import pyarrow.parquet as pq
    return np.concatenate([np.stack([np.asarray(x, np.float32) for x in pq.read_table(f).to_pydict()["action"]])
                           for f in fs])


def main(root: str, n_knots: int = 1024) -> int:
    a = train_actions(root)
    p = PIT.fit(a, n_knots=n_knots)
    x = torch.from_numpy(a)
    back = p.invert(p.apply(x, generator=torch.Generator().manual_seed(0)))
    rng = float(x.max() - x.min())
    print(f"  {root}\n  train actions {a.shape}  ->  {n_knots} knots x {a.shape[1]} dims")
    print(f"  round trip: max {float((back - x).abs().max()):.2e} ({float((back - x).abs().max()) / rng * 100:.3f}%"
          f" of range), mean {float((back - x).abs().mean()):.2e}")
    for d in range(a.shape[1]):
        at = x[:, d] == 0.0
        if float(at.float().mean()) < 0.01:
            continue
        exact = bool(torch.equal(back[at, d], x[at, d]))
        print(f"    dim {d}: atom {float(at.float().mean()) * 100:5.1f}% of mass, recovered exactly: {exact}")
        assert exact, f"dim {d}: the atom did not survive -- do NOT write these knots"

    path = os.path.join(root, "normalization_stats.json")
    stats = json.load(open(path))
    if "action_pit" in stats:
        print(f"  {path} already has action_pit ({len(stats['action_pit']['knots'][0])} knots) -- overwriting")
    stats["action_pit"] = p.state_dict()
    # A HuggingFace cache snapshot stores every file as a SYMLINK into ../../blobs/<sha>, shared with any
    # other snapshot of the same content. Writing through the link would edit the blob in place, i.e. edit
    # a content-addressed store under a hash that no longer describes it. Replace the link with a real file
    # instead; the blob stays untouched and is itself the backup.
    if os.path.islink(path):
        print(f"  {path}\n    is a symlink -> {os.readlink(path)}; replacing it with a real file "
              f"(the blob is left as the backup)")
        os.unlink(path)
    else:
        shutil.copy2(path, path + ".pre_pit")
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  wrote {path}  ({os.path.getsize(path) / 1e3:.0f} KB)")

    from quickdraw.data.dataset import Normalizer
    n = Normalizer.from_file(root)
    assert n.act_pit is not None and torch.equal(n.act_pit.knots, p.knots), "reload mismatch"
    print(f"  Normalizer.from_file reloads it: knots {tuple(n.act_pit.knots.shape)}, "
          f"tile_act(4) -> {tuple(Normalizer.from_file(root).tile_act(4).act_pit.knots.shape)}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m quickdraw.data.backfill_pit <dataset_root> [n_knots]")
    raise SystemExit(main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1024))
