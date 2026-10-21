"""Pin down the rot6 layout and the yaw sign, then say what each stick LOOKS like in scene_left.

The world-frame signs are already measured (_bs_axes.py: r ~ +0.94 on the diagonal). What a text prompt
for a VIDEO model needs is the apparent motion in the camera, so the second half cross-correlates frames
either side of a strong single-axis stick input and reports which way the image actually moves.
"""
import sys
import numpy as np
sys.path.insert(0, "/app/src")
from huggingface_hub import snapshot_download
from quickdraw.data.dataset import load_split_episodes_mm, set_subsample

set_subsample(1)
root = snapshot_download(repo_id="isaac-ronald-ward/block-stack", repo_type="dataset")

# ---- 1. rot6 layout: which flattening makes the two 3-vectors orthonormal? -------------------------
from quickdraw.data.dataset import load_split_episodes
from quickdraw.data.dataset import DataConfig
eps = load_split_episodes(root, "train", repo_id="block-stack", dcfg=DataConfig.legacy())
R6 = np.concatenate([o[:, 3:9] for o, _ in eps], 0)
for label, (u, v) in {"col-major [c0|c1]": (R6[:, 0:3], R6[:, 3:6]),
                      "row-major [r0|r1]": (R6[:, 0::2], R6[:, 1::2])}.items():
    nu, nv = np.linalg.norm(u, axis=1), np.linalg.norm(v, axis=1)
    dot = (u * v).sum(1)
    print(f"  {label:20s} |u|={nu.mean():.4f}+-{nu.std():.4f}  |v|={nv.mean():.4f}+-{nv.std():.4f}  "
          f"u.v={np.abs(dot).mean():.4f}")

# ---- 2. yaw sign, under the layout that wins above --------------------------------------------------
A = np.concatenate([a[:-1] for _, a in eps], 0)
for label, sl in {"col-major": (0, 1), "row-major": (0, 2)}.items():
    ang = np.concatenate([np.unwrap(np.arctan2(o[:, 3 + sl[1]], o[:, 3 + sl[0]])) for o, _ in eps])
    d = np.concatenate([np.diff(np.unwrap(np.arctan2(o[:, 3 + sl[1]], o[:, 3 + sl[0]]))) for o, _ in eps])
    r = float(np.corrcoef(A[:, 3], d)[0, 1])
    print(f"  yaw stick vs d(atan2) under {label}: r={r:+.3f}   "
          f"(+ stick -> {'CCW seen from +z' if r > 0 else 'CW seen from +z'})")

# ---- 3. apparent motion in scene_left, for the prompt wording ---------------------------------------
mm = load_split_episodes_mm(root, "val", img_size=(72, 96), cam="scene_left", repo_id="block-stack", dcfg=DataConfig.legacy())
o, a, fr = mm[0]
img = fr["scene_left"].astype(np.float32).mean(-1) / 255.0        # (T,72,96) grey
K = 24                                                            # frames apart: ~0.8 s at 30 Hz

def shift(a0, a1, axis):
    """Integer shift of a1 vs a0 along `axis`, by brute-force SAD over +-6 px on the central crop."""
    best, bd = 0, 1e9
    for s in range(-6, 7):
        b = np.roll(a1, s, axis=axis)[8:-8, 8:-8]
        d = np.abs(a0[8:-8, 8:-8] - b).mean()
        if d < bd:
            bd, best = d, s
    return best

print("\n  apparent shift in scene_left when ONE stick dominates (72x96 grey, 24 frames apart):")
for j, nm in enumerate(["move_x", "move_y", "height"]):
    for sgn in (+1, -1):
        m = np.ones(len(a) - K, bool)
        for k in range(3):
            m &= (np.abs(a[:len(m), k]) < 0.15) if k != j else (sgn * a[:len(m), k] > 0.5)
        idx = np.where(m)[0][:60]
        if len(idx) < 5:
            print(f"    {nm:>7s} {sgn:+d}: only {len(idx)} clean samples")
            continue
        dy = np.median([shift(img[i], img[i + K], 0) for i in idx])
        dx = np.median([shift(img[i], img[i + K], 1) for i in idx])
        print(f"    {nm:>7s} {sgn:+d}: n={len(idx):3d}  image content moves dx={dx:+.1f} dy={dy:+.1f} px "
              f"(+dx = content moves RIGHT, +dy = content moves DOWN)")
