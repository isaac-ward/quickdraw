"""Which cube ACTUALLY moved? Colour-segment the three cubes and report centroids, so the VLM's claims
are checked against a measurement rather than my eyeballing a thumbnail."""
import sys
import numpy as np
sys.path.insert(0, "/app/src")
from huggingface_hub import snapshot_download
from quickdraw.data.dataset import load_split_episodes_mm, set_action_aggregate, set_subsample
from quickdraw.data.dataset import DataConfig

set_subsample(2); set_action_aggregate("concat")
root = snapshot_download(repo_id="isaac-ronald-ward/block-stack", repo_type="dataset")
o, a, fr = load_split_episodes_mm(root, "val", img_size=(288, 384), cam="scene_left",
                                  repo_id="block-stack", dcfg=DataConfig.legacy())[0]
im = fr["scene_left"].astype(np.float32) / 255.0
P, NEW = 8, 88
ts = [P - 1, P - 1 + NEW, P - 1 + 2 * NEW, P - 1 + 3 * NEW]

def centroid(f, which):
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    mx = f.max(-1) + 1e-6
    sat = (f.max(-1) - f.min(-1)) / mx
    m = {"green": (g > r + 0.06) & (g > b + 0.04),
         "blue":  (b > r + 0.10) & (b > g + 0.04),
         "red":   (r > g + 0.15) & (r > b + 0.15)}[which] & (sat > 0.22) & (mx > 0.18)
    if m.sum() < 30:
        return None
    ys, xs = np.nonzero(m)
    return float(xs.mean()), float(ys.mean()), int(m.sum())

print("  centroids in the 288x384 frame (x = right, y = down); 'moved' = shift from the previous mark")
for c in ("green", "blue", "red"):
    prev = None
    row = []
    for t in ts:
        p = centroid(im[min(t, len(im) - 1)], c)
        if p is None:
            row.append("      --      "); prev = None; continue
        d = "" if prev is None else f" ({np.hypot(p[0]-prev[0], p[1]-prev[1]):4.0f}px)"
        row.append(f"({p[0]:5.0f},{p[1]:4.0f}){d}")
        prev = p
    print(f"   {c:6s} " + "  ".join(row))
print(f"   marks:  " + "  ".join(f"{('+%d'%(t-P+1)):>14s}" for t in ts))
