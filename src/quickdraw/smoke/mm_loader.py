"""Greenlight smoke for the multimodal data loader (data/dataset.py). Loads the VAL split (obs + act +
aligned FPV frames), builds an MMWindowLoader, and checks a batch has aligned shapes, image range [0,1],
and that the gathered image window actually matches the source frames (alignment). Run:
  uv run python -m quickdraw.smoke.mm_loader <data_root>
"""
import sys

import torch

from quickdraw.data.dataset import MMWindowLoader, Normalizer, load_split_episodes_mm

DEV = "cuda" if torch.cuda.is_available() else "cpu"
R = []


def check(name, cond, extra=""):
    R.append(bool(cond))
    print(f"[{'OK' if cond else 'FAIL'}] {name}{(' — ' + extra) if extra else ''}")


def main():
    root = sys.argv[1]
    P, Fh, B = 8, 6, 4
    L = P + Fh
    print(f"[mm_loader] loading VAL episodes (obs+act+FPV128) from {root} ...")
    eps = load_split_episodes_mm(root, "val", img_size=128)
    norm = Normalizer.from_file(root)
    check("episodes loaded with frames", len(eps) > 0 and eps[0][2].ndim == 4,
          f"{len(eps)} eps, frame shape {tuple(eps[0][2].shape)}")
    check("per-episode obs/frame counts aligned", all(len(o) == len(img) for o, _, img in eps))

    loader = MMWindowLoader(eps, P, Fh, norm, batch=B, shuffle=False, device=DEV)
    batch = next(iter(loader))
    check("obs_seq (B,L,6)", batch["obs_seq"].shape == (B, L, 6), str(tuple(batch["obs_seq"].shape)))
    check("act_seq (B,L,2)", batch["act_seq"].shape == (B, L, 2))
    check("image (B,L,128,128,3)", batch["image"].shape == (B, L, 128, 128, 3), str(tuple(batch["image"].shape)))
    img = batch["image"]
    check("image in [0,1]", float(img.min()) >= 0.0 and float(img.max()) <= 1.0,
          f"[{float(img.min()):.3f},{float(img.max()):.3f}]")

    # alignment: window 0 (shuffle=False) is episode 0, start 0 -> frames[0][0:L]
    ref = eps[0][2][0:L].astype("float32") / 255.0
    got = img[0].cpu().numpy()
    import numpy as np
    check("gathered image window == source frames (alignment)", np.allclose(got, ref, atol=1e-6))

    print(f"\n{'ALL OK' if all(R) else 'SOME FAILED'} ({sum(R)}/{len(R)})")
    sys.exit(0 if all(R) else 1)


if __name__ == "__main__":
    main()
