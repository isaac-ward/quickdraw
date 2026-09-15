"""Is a plan that moves MORE than a pilot doing something impossible, or just something sustained?

">489% of pilot" needs unpacking, because the pilot baseline is the mean |NET| motion over a 34 s segment
and net motion cancels: a pilot who turns 300 deg left then 250 deg right nets 50. A planner asked to
rotate left picks one direction and holds it, so it can exceed the mean net without ever exceeding
anything physical. Two different questions get conflated:

  SUSTAINED?    net motion vs the recorded net distribution (mean, p95, max). Beating the mean is
                expected for a directed request; beating the recorded MAX means the plan is more
                single-minded than any real flight, which is a claim about the planner, not a fault.
  PHYSICAL?     per-step RATE vs the recorded rate distribution. A plan that exceeds the recorded max
                rate is asking the world model for motion it has never seen, and whatever it renders
                there is extrapolation. THIS is the number that says whether to trust the trajectory.

    python scratch/check_rates.py <steer_run> [<steer_run> ...]
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))   # sibling analyses in this package
from steer_physical import WANTS, physical, yaw_of                          # noqa: E402

DT = 4.0 / 15.0                     # subsample 4 on a 15 Hz dataset


def rates(pro: np.ndarray) -> dict:
    """Per-step magnitudes -> the fastest the trajectory ever moves, in physical units per second."""
    p, q = pro[:, 0:3], pro[:, 6:10]
    yaw = np.unwrap(yaw_of(q))
    d = np.diff(p, axis=0)
    return {"yaw": np.abs(np.degrees(np.diff(yaw))) / DT,
            "altitude": np.abs(d[:, 2]) / DT,
            "horizontal": np.linalg.norm(d[:, :2], axis=1) / DT}


def recorded(span: int = 128):
    from omegaconf import OmegaConf
    from quickdraw.data.dataset import load_split_episodes_mm, set_action_aggregate, set_subsample
    from quickdraw.training.setup import image_head_cams, image_head_sizes, resolve_data_root
    cfg = OmegaConf.create(json.load(open(
        "logs/paper_icra_2027/model_backups/train_action_2026_09_14_04_41_17_s2_ah_chunk32_full/logs/config.json")))
    set_subsample(4); set_action_aggregate("concat")
    eps = load_split_episodes_mm(resolve_data_root(cfg), "val", img_size=image_head_sizes(cfg),
                                 cam=image_head_cams(cfg), repo_id="starling-2")
    nets, rr = [], {k: [] for k in ("yaw", "altitude", "horizontal")}
    for o, _, _ in eps:
        for i in range(0, len(o) - span, span // 2):
            nets.append(physical(o[i:i + span]))
        for k, v in rates(o).items():
            rr[k].append(v)
    return nets, {k: np.concatenate(v) for k, v in rr.items()}


def main(*runs: str) -> int:
    nets, rr = recorded()
    print(f"\n  RECORDED, {len(nets)} segments of 34 s")
    print(f"  {'quantity':12s} {'|net| mean':>11s} {'|net| p95':>10s} {'|net| max':>10s}   "
          f"{'rate p50':>9s} {'rate p95':>9s} {'rate max':>9s}")
    for k, unit in (("yaw", "deg"), ("altitude", "m"), ("horizontal", "m")):
        n = np.abs([x[k] for x in nets]) if k != "horizontal" else \
            np.abs([np.hypot(x["forward"], x["lateral"]) for x in nets])
        print(f"  {k + ' (' + unit + ')':12s} {n.mean():>11.1f} {np.percentile(n, 95):>10.1f} "
              f"{n.max():>10.1f}   {np.percentile(rr[k], 50):>9.2f} {np.percentile(rr[k], 95):>9.2f} "
              f"{rr[k].max():>9.2f}   per second")

    for run in runs:
        got = {}
        for f in sorted(glob.glob(os.path.join(run, "logs", "epoch_*", "eval_steer", "plans", "*", "*",
                                               "plan.json"))):
            d = json.load(open(f))
            if d["request"] not in WANTS:
                continue
            pro = np.load(os.path.join(os.path.dirname(f), "proprio.npy"))
            got.setdefault(d["request"], []).append(rates(pro))
        if not got:
            continue
        print(f"\n=== {os.path.basename(run)}")
        print(f"  {'request':16s} " + "  ".join(f"{k + ' p95/max':>18s}" for k in
                                                ("yaw", "altitude", "horizontal")))
        print("  " + "-" * 74)
        for q in sorted(got):
            cells = []
            for k in ("yaw", "altitude", "horizontal"):
                a = np.concatenate([r[k] for r in got[q]])
                over = 100.0 * float((a > rr[k].max()).mean())
                cells.append(f"{np.percentile(a, 95):>7.2f}/{a.max():<6.2f}{('*' if over > 1 else ' ')}")
            print(f"  {q:16s} " + "  ".join(f"{c:>18s}" for c in cells))
        print("  * = more than 1% of this plan's steps move FASTER than anything in the recorded data")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
