"""Which way does each block-stack stick push? MEASURED, not assumed.

Correlate each action dim against the resulting end-effector motion in the recorded train split, so the
action_axes wording for the Cosmos text bridge states what the stick actually does rather than what its
name suggests. obs layout (PROCESSING.md 5): 0:3 ee_xyz_mm world, 3:9 ee_rot6, 9 gripper (1 = OPEN).
action (4): [move_x, move_y, height, yaw, gripper], gripper 1 = SQUEEZE.
"""
import sys
import numpy as np
sys.path.insert(0, "/app/src")
from huggingface_hub import snapshot_download
from quickdraw.data.dataset import load_split_episodes, set_subsample

set_subsample(1)
root = snapshot_download(repo_id="isaac-ronald-ward/block-stack", repo_type="dataset")
eps = load_split_episodes(root, "train", repo_id="block-stack")
print(f"{len(eps)} episodes, obs {eps[0][0].shape}, act {eps[0][1].shape}")

A = np.concatenate([a[:-1] for _, a in eps], 0)
dO = np.concatenate([o[1:] - o[:-1] for o, _ in eps], 0)
names = ["move_x", "move_y", "height", "yaw", "gripper"]
obsn = ["d ee_x", "d ee_y", "d ee_z", "d rot6_0", "d rot6_1", "d rot6_2",
        "d rot6_3", "d rot6_4", "d rot6_5", "d grip_open"]

print("\ncorrelation of action[t] with observation delta[t->t+1]  (|r| > 0.15 marked *)")
print("            " + "".join(f"{n:>11s}" for n in obsn))
for j, nm in enumerate(names):
    row = []
    for k in range(10):
        a, b = A[:, j], dO[:, k]
        r = 0.0 if a.std() < 1e-9 or b.std() < 1e-9 else float(np.corrcoef(a, b)[0, 1])
        row.append(f"{r:+8.3f}{'*' if abs(r) > 0.15 else ' '}  ")
    print(f"{nm:>11s} " + "".join(row))
print("\nstick ranges:", {n: (round(float(A[:, j].min()), 2), round(float(A[:, j].max()), 2))
                          for j, n in enumerate(names)})
