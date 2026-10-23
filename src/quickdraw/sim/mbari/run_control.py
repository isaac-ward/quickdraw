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


# SUSTAINED HOLD, not a terminal snapshot. `final_dist` called a controller successful while it sat
# on the goal firing a third of full thrust, so success here requires the vehicle to be inside POS_TOL
# AND slower than VEL_TOL for HOLD_STEPS consecutive steps, with no collision -- arrive, settle, stay.
# The tolerances are OURS, not the cave project's: their 0.40 m sits just above a 0.37 m controller
# floor, and ours is 0.009 m, so 0.40 would score every arm 100% and discriminate nothing.
POS_TOL, VEL_TOL, HOLD_STEPS = 0.10, 0.05, 20


def trial_metrics(r, goal_q, n=60):
    """Every number the control study reports, on one episode. Settling quantities are measured over
    the last `n` steps (2 s): a single terminal frame cannot tell 'stopped' from 'passing through'."""
    tr, ac, d = r["traj"], r["act"], r["dist"]
    qf = tr[-1, 3:7] / max(1e-9, np.linalg.norm(tr[-1, 3:7]))
    spd = np.linalg.norm(tr[:, 7:10], axis=1)
    ok = (d < POS_TOL) & (spd < VEL_TOL)                      # per-step "held"
    run_len = held = 0
    for v in ok:                                              # longest consecutive hold
        run_len = run_len + 1 if v else 0
        held = max(held, run_len)
    return {
        "final_m": float(r["final_dist"]),
        "best_m": float(r["best_dist"]),
        "att_deg": float(np.degrees(2 * np.arccos(min(1.0, abs(float(qf @ np.asarray(goal_q))))))),
        "wander": float(d[-n:].max() - d[-n:].min()),
        "v_end": float(spd[-n:].mean()),
        "a_end": float(np.abs(ac[-n:]).mean()),
        "da_end": float(np.abs(np.diff(ac[-n:], axis=0)).mean()),
        "steps_to_0p5m": int(r["steps_to_0p5m"]),
        "hold_steps": int(held),
        "success": bool(held >= HOLD_STEPS and not r.get("crashed", False)),
        "crashed": bool(r.get("crashed", False)),
    }


def run(dyns, out="/app/logs/mbari/control", n_trials=6, steps=600, trial0=0, log=print):
    from . import control as C, record

    model = record.load_model()
    # THE TRIAL SET IS GLOBAL, the shard is a window into it. goals/starts are drawn for the FULL
    # count so trial i is the same task in every shard and every arm -- sharding must not change
    # what is being measured, only who measures it.
    total = trial0 + n_trials
    tgt = goals(model, n=max(total, 8))
    st = starts(model, total)
    idx = list(range(trial0, total))
    tasks = [("random_pose", [tgt[i % len(tgt)] for i in idx])]
    st = [st[i] for i in idx]
    rows, trials = [], []
    for dyn in dyns:
        for task, gs in tasks:
            res = []
            for n_i, (s0, (gp, gq)) in enumerate(zip(st, gs)):
                i = idx[n_i]
                r = C.run_episode(model, dyn, s0, gp, goal_q=gq, steps=steps, seed=100 + i)
                m = trial_metrics(r, gq)
                m.update(trial=i, model=dyn.name, task=task,
                         start_dist=float(np.linalg.norm(s0[:3] - gp)))
                trials.append(m)
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
    # PER-TRIAL rows are what a sharded run merges on; the aggregate above is only valid within a shard.
    (p / "trials.json").write_text(json.dumps(trials, indent=1))
    return rows


METRICS = ("final_m", "best_m", "att_deg", "wander", "v_end", "a_end", "da_end", "hold_steps")


def aggregate(root, out=None, log=print):
    """Merge sharded per-trial rows into one table per arm.

    ON RAW TRIALS, NOT ON SHARD AVERAGES. Each shard writes its own `benchmark.json` summary, and
    averaging those would weight an 8-trial shard the same as a 2-trial one and make the median
    meaningless. `trials.json` is the unit that merges.

    Reports median beside mean because control error is long-tailed -- one trial that never arrives
    moves the mean by metres and says nothing about the typical case, and hiding it would be worse.
    """
    import collections

    rows = collections.defaultdict(list)
    for f in sorted(pathlib.Path(root).glob("*/trials.json")):
        for t in json.loads(f.read_text()):
            rows[t["model"] + "|" + pathlib.Path(f).parent.name.split("_s")[0]].append(t)
    merged = {}
    for key, ts in rows.items():
        name = key.split("|")[1]
        seen = {t["trial"] for t in ts}
        d = {"arm": name, "n": len(ts), "trials_covered": sorted(seen),
             "duplicate_trials": len(ts) - len(seen),
             "success_rate": float(np.mean([t["success"] for t in ts])),
             "crashed_frac": float(np.mean([t["crashed"] for t in ts])),
             "reached_0p5m": float(np.mean([t["steps_to_0p5m"] >= 0 for t in ts]))}
        for m in METRICS:
            v = np.array([t[m] for t in ts], float)
            d[f"{m}_mean"] = float(v.mean())
            d[f"{m}_median"] = float(np.median(v))
            d[f"{m}_p90"] = float(np.percentile(v, 90))
        merged[name] = d
    log(f"  {'arm':10s} {'n':>3} {'success':>8} {'final_m':>18} {'att_deg':>14} {'wander':>8} {'|a|':>7} {'crash':>6}")
    log(f"  {'':10s} {'':>3} {'(hold)':>8} {'mean / med / p90':>18} {'mean / med':>14}")
    for name, d in sorted(merged.items()):
        log(f"  {name:10s} {d['n']:3d} {d['success_rate']*100:7.0f}% "
            f"{d['final_m_mean']:6.3f} /{d['final_m_median']:6.3f} /{d['final_m_p90']:6.3f} "
            f"{d['att_deg_mean']:6.1f} /{d['att_deg_median']:5.1f} "
            f"{d['wander_mean']:8.3f} {d['a_end_mean']:7.3f} {d['crashed_frac']*100:5.0f}%")
        if d["duplicate_trials"]:
            log(f"    !! {name}: {d['duplicate_trials']} duplicate trial indices -- shards overlap")
    if out:
        pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(out).write_text(json.dumps(merged, indent=1))
        log(f"  wrote {out}")
    return merged


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
    ap.add_argument("--aggregate", default=None,
                    help="merge */trials.json under this directory and print the table")
    ap.add_argument("--trial0", type=int, default=0,
                    help="first global trial index -- shards cover disjoint windows of ONE trial set")
    # One arm at a time, because the arms cost wildly different amounts and an A/B of the LEARNED one
    # should not re-pay for true+analytic every time -- those two are a fixed reference, not a variable.
    ap.add_argument("--arms", default="true,analytic,learned",
                    help="comma-separated subset of true,analytic,learned")
    ap.add_argument("--no-integrate", action="store_true",
                    help="learned arm: use the model's RAW absolute pose instead of integrating its "
                         "predicted velocities from the true state (see control._integrate_from_true)")
    a = ap.parse_args()

    from . import control as C, record

    if a.aggregate:
        aggregate(a.aggregate, out=str(pathlib.Path(a.aggregate) / "aggregate.json"))
        raise SystemExit(0)
    if a.no_integrate:
        C.LEARNED_INTEGRATE = False
    want = [x.strip() for x in a.arms.split(",") if x.strip()]
    m = record.load_model()
    dyns = []
    if "true" in want:
        dyns.append(C.TrueDynamics(m))
    if "analytic" in want:
        dyns.append(C.AnalyticDynamics())
    if "learned" in want and a.ckpt:
        dyns.append(learned_from(a.ckpt, m))
    run(dyns, out=a.out, n_trials=a.trials, trial0=a.trial0, log=lambda s: print(s, flush=True))
