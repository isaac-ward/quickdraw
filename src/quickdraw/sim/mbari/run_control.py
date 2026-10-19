"""The control benchmark: reach a pose with MPPI, swapping only the dynamics model.

One task: reach a RANDOM free POSE -- position and attitude. Run with every available dynamics
model over the same start/goal pairs and the same planner seed, so the rows differ only in what
predicts the next state.

    true        MuJoCo itself -- the ceiling
    analytic    the same equations at the control rate, coarse integrator -- shows how much of any
                gap is model CLASS rather than learning
    learned     the trained world model's proprio head (added when a checkpoint exists)

Reported per task: final distance, final ATTITUDE error, best distance, and steps to within 0.5 m.
Distance rather than a success rate because with 8 starts a rate is 8 numbers wide and hides how
close the misses were -- and attitude beside it because a vehicle at the right point facing the
wrong way has not reached the pose.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np


def goals(model, n=8, seed=0):
    """-> [(position (3), quaternion (4))] -- n random target POSES in free water.

    THE DOCK IS GONE. The benchmark used to reach a dock entry; the dock's mesh was never square to
    its own body frame, so the structure never agreed with the attitude target it was supposed to
    define. Random poses are the cleaner task anyway: they probe the whole workspace instead of one
    corner, every trial is a different attitude rather than the same one, and the target is defined
    by numbers rather than by a model's internal frame. The translucent ghost marks each one.

    POSES, NOT POSITIONS. A vehicle at the right point facing the wrong way has not reached the
    pose, and `run_episode` scores both -- see `control._rollout_fn`."""
    from . import trajectories as tj

    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        p = np.array([rng.uniform(-1, 1) * (tj.TANK_L / 2 - 2.0),
                      rng.uniform(-1, 1) * (tj.TANK_W / 2 - 2.0),
                      -rng.uniform(2.0, tj.TANK_D - 2.0)], np.float32)
        # BOTH tests. `inside` keeps it off the walls with a planning margin; `clear` is the
        # physical one that also rejects a pose intersecting a buoy, the ladder or the crane hook
        # -- a target the vehicle could only reach by colliding is not a target.
        if not (tj.inside(p[None], margin=1.5)[0] and tj.clear(p[None])[0]):
            continue
        ax = rng.normal(size=3)
        q = tj.mat_to_quat(tj.rot_from_axis_angle(ax / np.linalg.norm(ax), rng.uniform(0, 2 * np.pi)))
        out.append((p, np.asarray(q, np.float32)))
    return out


def starts(model, n, seed=1):
    import mujoco

    from . import trajectories as tj

    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        p = np.array([rng.uniform(-1, 1) * (tj.TANK_L / 2 - 2.0),
                      rng.uniform(-1, 1) * (tj.TANK_W / 2 - 2.0),
                      -rng.uniform(2.0, tj.TANK_D - 2.0)], np.float32)
        if not tj.inside(p[None], margin=1.5)[0]:
            continue
        if not tj.clear(p[None])[0]:
            continue
        s = np.zeros(13, np.float32)
        s[:3] = p
        ax = rng.normal(size=3)
        s[3:7] = tj.mat_to_quat(tj.rot_from_axis_angle(ax / np.linalg.norm(ax), rng.uniform(0, 2 * np.pi)))
        out.append(s)
    return out


def run(dyns, out="/app/logs/mbari/control", n_trials=6, steps=600, log=print):
    from . import control as C, record

    model = record.load_model()
    tgt = goals(model, n=max(n_trials, 8))
    st = starts(model, n_trials)
    tasks = [("random_pose", [tgt[i % len(tgt)] for i in range(n_trials)])]
    rows = []
    for dyn in dyns:
        for task, gs in tasks:
            res = []
            for i, (s0, (gp, gq)) in enumerate(zip(st, gs)):
                r = C.run_episode(model, dyn, s0, gp, goal_q=gq, steps=steps, seed=100 + i)
                r["start_dist"] = float(np.linalg.norm(s0[:3] - gp))
                qf = r["traj"][-1, 3:7]
                qf = qf / max(1e-9, np.linalg.norm(qf))
                r["final_att_deg"] = float(np.degrees(2 * np.arccos(min(1.0, abs(float(qf @ gq))))))
                r["crashed"] = bool(r.get("crashed", False))
                res.append(r)
                log(f"  {dyn.name:9s} {task:12s} trial {i}: {r['start_dist']:.2f} -> "
                    f"{r['final_dist']:.3f} m, {r['final_att_deg']:.1f} deg "
                    f"(best {r['best_dist']:.3f})" + ("  CRASHED" if r["crashed"] else ""))
            rows.append({"model": dyn.name, "task": task,
                         "start_dist_mean": float(np.mean([r["start_dist"] for r in res])),
                         "final_dist_mean": float(np.mean([r["final_dist"] for r in res])),
                         "final_dist_median": float(np.median([r["final_dist"] for r in res])),
                         "best_dist_mean": float(np.mean([r["best_dist"] for r in res])),
                         "final_att_deg_mean": float(np.mean([r["final_att_deg"] for r in res])),
                         "reached_0p5m": float(np.mean([r["steps_to_0p5m"] >= 0 for r in res])),
                         "crashed_frac": float(np.mean([r["crashed"] for r in res])),
                         "n": len(res)})
            log(f"  == {dyn.name:9s} {task:12s} final {rows[-1]['final_dist_mean']:.3f} m, "
                f"{rows[-1]['final_att_deg_mean']:.1f} deg mean, "
                f"reached<0.5m {rows[-1]['reached_0p5m']*100:.0f}%")
    p = pathlib.Path(out)
    p.mkdir(parents=True, exist_ok=True)
    (p / "benchmark.json").write_text(json.dumps(rows, indent=1))
    return rows


def learned_from(ckpt, plant_model, dev="cuda"):
    """Build the learned arm from a training run directory or a .ckpt path."""
    from ...evaluation.standalone import load_checkpoint
    from ...training.setup import build_model, normalizer
    from . import control as C
    from .run_uncertainty import _cfg_for

    cfg = _cfg_for(ckpt)
    core = build_model(cfg).to(dev)
    load_checkpoint(core, ckpt)
    core.eval()
    return C.LearnedDynamics(core, normalizer(cfg), plant_model, key=str(cfg.data.cam),
                             P=int(cfg.data.P), dev=dev)


if __name__ == "__main__":
    import argparse
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    ap = argparse.ArgumentParser(description="Control benchmark: true vs analytic vs learned.")
    ap.add_argument("--ckpt", default=None, help="training run dir or .ckpt; adds the learned arm")
    ap.add_argument("--out", default="/app/logs/mbari/control")
    ap.add_argument("--trials", type=int, default=6)
    a = ap.parse_args()

    from . import control as C, record

    m = record.load_model()
    dyns = [C.TrueDynamics(m), C.AnalyticDynamics()]
    if a.ckpt:
        dyns.append(learned_from(a.ckpt, m))
    run(dyns, out=a.out, n_trials=a.trials, log=lambda s: print(s, flush=True))
