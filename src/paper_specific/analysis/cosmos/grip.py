"""The gripper channel chatters. How much, and what debounce cleans it without eating real grasps?"""
import sys
import numpy as np
sys.path.insert(0, "/app/src")
from huggingface_hub import snapshot_download
from quickdraw.data.dataset import load_split_episodes, set_action_aggregate, set_subsample
from scipy.ndimage import median_filter
from quickdraw.data.dataset import DataConfig

set_subsample(2); set_action_aggregate("concat")
a = load_split_episodes(snapshot_download(repo_id="isaac-ronald-ward/block-stack",
                                          repo_type="dataset"), "val", repo_id="block-stack", dcfg=DataConfig.legacy())[0][1]
g = a[:, 4::5].mean(1)                                   # gripper, concat sub-steps folded
raw = g > 0.5
print(f"  {len(g)} steps at 15 Hz. closed {100*raw.mean():.0f}% of the time")
print(f"  raw transitions: {int(np.abs(np.diff(raw)).sum())}  "
      f"-> one 'grasp' every {len(g)/max(1,np.abs(np.diff(raw)).sum()/2):.0f} steps")
print("\n  median filter   transitions   grasps   mean hold (steps / s)")
for w in (1, 3, 5, 7, 11, 15, 21, 31):
    f = median_filter(raw.astype(np.uint8), size=w, mode="nearest").astype(bool)
    tr = int(np.abs(np.diff(f)).sum())
    runs = np.split(np.arange(len(f)), np.where(np.diff(f))[0] + 1)
    holds = [len(r) for r in runs if f[r[0]]]
    print(f"    w={w:3d} ({w*2/30:4.2f} s)  {tr:6d}      {tr//2:5d}   "
          f"{np.mean(holds) if holds else 0:6.1f} / {np.mean(holds)*2/30 if holds else 0:5.2f} s")
