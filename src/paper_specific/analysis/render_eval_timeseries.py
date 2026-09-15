"""A proprio+action timeseries PNG beside every eval-split mp4, to watch alongside the video.

Laid out so the two OOD splits can be told apart by eye, because they are different kinds of OOD:

    noodle       VISUAL   -- a novel object enters the frame. Shows up in the IMAGE, and the proprio
                             traces should look like ordinary flight.
    leafblower   DYNAMIC  -- airflow pushes the drone. Shows up as acceleration the COMMANDED STICKS DO
                             NOT EXPLAIN, which is why |a_imu| is plotted against the stick traces rather
                             than on its own: the tell is acceleration with no command behind it.

    memory       the drone turns away from a scene and back. The yaw panel carries the away-and-back, and
                 the shaded span marks it (detected from the unwrapped heading), because whether the
                 return lands inside the model's 32-step window decides whether the experiment tests
                 retained state or just attention.

    python scratch/render_eval_timeseries.py [out_dir]
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from quickdraw.data.dataset import load_split_episodes_mm, set_action_aggregate, set_subsample
from quickdraw.training.setup import image_head_cams, image_head_sizes, resolve_data_root

SPLITS = {"eval_ood": ["eval_ood_noodle", "eval_ood_leafblower"],
          "eval_memory": ["eval_memory_backwall1", "eval_memory_backwall2"]}
AX = ["yaw (+=rot left)", "vertical (+=descend)", "lateral (+=right)", "fore/aft (-=forward)"]
FPS = 15.0


def yaw_deg(q):
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.degrees(np.unwrap(np.arctan2(2 * (w * z + x * y), 1.0 - 2 * (y * y + z * z))))


def turn_span(yaw):
    """The away-and-back: from the first sustained departure of the heading to its return. Returns
    (start, end) step indices, or None. Heuristic and reported as such -- it exists to make the timing
    VISIBLE on the plot, not to define a metric."""
    d = yaw - yaw[0]
    away = np.where(np.abs(d) > 45.0)[0]
    if len(away) == 0:
        return None
    s = int(away[0])
    back = np.where(np.abs(d[s:]) < 25.0)[0]
    return (s, s + int(back[0])) if len(back) else (s, len(yaw) - 1)


def main(out_root: str = "logs/paper_icra_2027") -> int:
    cfg = OmegaConf.create(json.load(open(
        "logs/paper_icra_2027/model_backups/train_action_2026_09_14_04_41_17_s2_ah_chunk32_full/logs/config.json")))
    set_subsample(1); set_action_aggregate("concat")
    root = resolve_data_root(cfg)
    for group, splits in SPLITS.items():
        for sp in splits:
            eps = load_split_episodes_mm(root, sp, img_size=image_head_sizes(cfg),
                                         cam=image_head_cams(cfg), repo_id="starling-2")
            d = os.path.join(out_root, group, sp)
            os.makedirs(d, exist_ok=True)
            spans = []
            for i, (o, a, _) in enumerate(eps):
                t = np.arange(len(o)) / FPS
                p, v, q, acc = o[:, 0:3], o[:, 3:6], o[:, 6:10], o[:, 13:16]
                yw = yaw_deg(q)
                sp_ = turn_span(yw)
                spans.append(sp_ if sp_ else (-1, -1))
                fig, ax = plt.subplots(5, 1, figsize=(9, 10), sharex=True)
                for k, lab in zip(range(3), "xyz"):
                    ax[0].plot(t, p[:, k], label=f"position {lab}")
                ax[0].set_ylabel("position (m)"); ax[0].legend(fontsize=7, ncol=3)
                ax[1].plot(t, yw, color="k"); ax[1].set_ylabel("heading (deg, unwrapped)")
                ax[2].plot(t, np.linalg.norm(v[:, :2], axis=1), label="horizontal speed")
                ax[2].plot(t, v[:, 2], label="vertical speed")
                ax[2].set_ylabel("speed (m/s)"); ax[2].legend(fontsize=7, ncol=2)
                ax[3].plot(t, np.linalg.norm(acc, axis=1), color="crimson")
                ax[3].set_ylabel("|IMU accel| (m/s2)")
                for k in range(4):
                    ax[4].plot(t, a[:, k], label=AX[k], lw=1.1)
                ax[4].set_ylabel("commanded stick"); ax[4].set_xlabel("seconds (15 Hz)")
                ax[4].legend(fontsize=7, ncol=2); ax[4].set_ylim(-1.05, 1.05)
                for A in ax:
                    A.grid(alpha=0.25)
                    if sp_:
                        A.axvspan(sp_[0] / FPS, sp_[1] / FPS, color="tab:orange", alpha=0.12, lw=0)
                ttl = f"{sp}  ep{i:02d}   {len(o)} frames = {len(o) / FPS:.1f}s   ({len(o) // 4} model steps at stride 4)"
                if sp_:
                    ttl += f"\nheading leaves >45deg at {sp_[0] / FPS:.1f}s and returns <25deg by {sp_[1] / FPS:.1f}s (shaded)"
                fig.suptitle(ttl, fontsize=10)
                fig.tight_layout(rect=(0, 0, 1, 0.96))
                fig.savefig(os.path.join(d, f"ep{i:02d}.png"), dpi=100)
                plt.close(fig)
            sv = np.array(spans)
            ok = sv[:, 0] >= 0
            print(f"  {sp:24s} {len(eps):2d} eps -> ep*.png"
                  + (f" | away-and-back detected in {int(ok.sum())}/{len(eps)}: "
                     f"span {np.mean((sv[ok, 1] - sv[ok, 0]) / 4):.0f} model steps "
                     f"(min {np.min((sv[ok, 1] - sv[ok, 0]) // 4)}, max {np.max((sv[ok, 1] - sv[ok, 0]) // 4)})"
                     if ok.any() else " | no away-and-back detected"))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
