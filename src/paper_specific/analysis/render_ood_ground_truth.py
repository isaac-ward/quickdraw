"""Re-render the OOD artifacts against the REVIEWED windows (data/ood_windows.py), not the detector.

One mp4 and one png per KEPT episode, named `_gt` so the detector's versions stay beside them for
comparison. Excluded episodes get nothing, and are listed at the end so a missing file is never a mystery.

The mp4 flashes a fully white frame at the window's first and last frame and carries a white border
throughout it. The png keeps the detector signal on the top panel with the reviewed window shaded, so the
two can be compared by eye -- that is the point of keeping both.

    python scratch/render_ood_ground_truth.py [out_dir]
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(__file__))   # sibling analyses in this package
from analyse_ood_anomaly import AX, dynamic_novelty, visual_novelty                    # noqa: E402

from quickdraw.data.dataset import load_split_episodes_mm, set_action_aggregate, set_subsample
from quickdraw.data.ood_windows import FPS, WINDOWS, kept
from quickdraw.logging import viz
from quickdraw.training.setup import image_head_cams, image_head_sizes, resolve_data_root

KIND = {"eval_ood_noodle": "visual", "eval_ood_leafblower": "dynamic"}


def main(out_root: str = "logs/paper_icra_2027") -> int:
    cfg = OmegaConf.create(json.load(open(
        "logs/paper_icra_2027/model_backups/train_action_2026_09_14_04_41_17_s2_ah_chunk32_full/logs/config.json")))
    set_subsample(1); set_action_aggregate("concat")
    root = resolve_data_root(cfg)
    for sp, kind in KIND.items():
        eps = load_split_episodes_mm(root, sp, img_size=image_head_sizes(cfg),
                                     cam=image_head_cams(cfg), repo_id="starling-2")
        key = list(eps[0][2])[0]
        d = os.path.join(out_root, "eval_ood", sp)
        keep = kept(sp)
        print(f"\n  {sp}  ({kind})  keeping {len(keep)}/{len(eps)}")
        for i in keep:
            o, a, fr = eps[i]
            s0, s1 = WINDOWS[sp][i]
            s1 = min(s1, len(o))
            sig = visual_novelty(fr[key]) if kind == "visual" else dynamic_novelty(o, a)
            v = fr[key].copy()
            v[s0:s1, :3] = 255; v[s0:s1, -3:] = 255
            v[s0:s1, :, :3] = 255; v[s0:s1, :, -3:] = 255
            out = []
            for t in range(len(v)):
                if t in (s0, s1 - 1):
                    out.append(np.full_like(v[t], 255))
                out.append(v[t])
            viz.save_mp4(os.path.join(d, f"ep{i:02d}_gt.mp4"), np.stack(out), FPS)
            t = np.arange(len(o)) / FPS
            fig, ax = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
            ax[0].plot(t, sig, color="crimson")
            ax[0].set_ylabel("pink pixel\nfraction" if kind == "visual" else "unexplained\nspeed (m/s)")
            ax[1].plot(t, np.linalg.norm(o[:, 3:6], axis=1), label="speed (m/s)")
            ax[1].plot(t, o[:, 2], label="altitude (m)")
            ax[1].legend(fontsize=7, ncol=2); ax[1].set_ylabel("proprio")
            st = a.reshape(len(a), -1, 4).mean(axis=1)
            for k in range(4):
                ax[2].plot(t, st[:, k], lw=1.0, label=AX[k])
            ax[2].set_ylabel("stick"); ax[2].set_xlabel("seconds"); ax[2].legend(fontsize=7, ncol=4)
            for A in ax:
                A.grid(alpha=0.25)
                A.axvspan(s0 / FPS, s1 / FPS, color="tab:green", alpha=0.15, lw=0)
            fig.suptitle(f"{sp} ep{i:02d} — REVIEWED window {s0 / FPS:.1f}-{s1 / FPS:.1f}s "
                         f"(frames {s0}-{s1}, model steps {s0 // 4}-{-(-s1 // 4)})\n"
                         f"green = ground truth; the red trace is the detector, kept for comparison",
                         fontsize=10)
            fig.tight_layout(rect=(0, 0, 1, 0.93))
            fig.savefig(os.path.join(d, f"ep{i:02d}_gt.png"), dpi=100); plt.close(fig)
            print(f"    ep{i:02d}  {s0 / FPS:>4.1f}-{s1 / FPS:<4.1f}s -> ep{i:02d}_gt.{{mp4,png}}")
        dropped = [i for i in range(len(eps)) if i not in keep]
        print(f"    excluded (no _gt files written): {dropped}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
