"""How do the world axes appear in scene_left? The camera is FIXED, so nothing moves but the arm --
locate the arm as the centroid of frame-to-frame change, then regress its pixel position on ee_xyz.
The signs of that 2x3 map are what the action_axes wording needs."""
import sys
import numpy as np
sys.path.insert(0, "/app/src")
from huggingface_hub import snapshot_download
from quickdraw.data.dataset import load_split_episodes_mm, set_subsample

set_subsample(1)
root = snapshot_download(repo_id="isaac-ronald-ward/block-stack", repo_type="dataset")
o, a, fr = load_split_episodes_mm(root, "val", img_size=(144, 192), cam="scene_left",
                                  repo_id="block-stack")[0]
img = fr["scene_left"].astype(np.float32).mean(-1) / 255.0
T, Hh, Ww = img.shape
print(f"episode: {T} frames {Hh}x{Ww}")

K = 6
d = np.abs(img[K:] - img[:-K])                       # (T-K,H,W) where the arm is moving
d = np.clip(d - np.percentile(d, 70, axis=(1, 2), keepdims=True), 0, None)
w = d.reshape(len(d), -1)
ok = w.sum(1) > 1e-3
ys, xs = np.mgrid[0:Hh, 0:Ww]
u = (w @ xs.ravel()) / np.maximum(w.sum(1), 1e-9)    # centroid column (px, + = right)
v = (w @ ys.ravel()) / np.maximum(w.sum(1), 1e-9)    # centroid row    (px, + = down)

X = o[:len(u), 0:3]                                  # ee_xyz in world mm
X = np.c_[X - X.mean(0), np.ones(len(X))][ok]
for name, y in (("u (px right)", u[ok]), ("v (px down)", v[ok])):
    b, *_ = np.linalg.lstsq(X, y - y.mean(), rcond=None)
    pred = X @ b
    r2 = 1 - ((y - y.mean() - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"  {name}:  d/d ee_x {b[0]*100:+7.2f}  d/d ee_y {b[1]*100:+7.2f}  d/d ee_z {b[2]*100:+7.2f}"
          f"   px per 100 mm   (R2 {r2:.2f}, n={ok.sum()})")
