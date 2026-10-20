"""Run the cave point-to-point pose control benchmark: LEARNED (red WM) vs TRUE dynamics (+ ANALYTIC).

Same 20 start->goal pose pairs (sampled from recorded val trajectories, so both endpoints are collision-free
and reachable) and the same MPPI config/seed for every arm -- the only variable is the dynamics model used
inside the planner. Execution is always the true Dynamics3D plant.

Metrics (per the spec): success rate (collision = immediate failure), time to completion, settle time,
path length as a multiple of the geodesic shortest path (plan_fast), and closest-obstacle clearance.

RUN IN THE BRIDGE:
  PYTHONPATH=/home/ubuntu/user_irw/quickdraw/src CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cpu \
    /home/ubuntu/user_irw/prospector/.venv/bin/python -m quickdraw.sim.cave.run_control \
      --ckpt <red_best.ckpt> --root <dataset_dir> --out <logs/cave/control> --n_pairs 20
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


def _patch_scene_render():
    """Build CaveScene collision-only: stub the gsplat render palette/bake (control needs clearance, not color)."""
    import prospector.cave_wm.scene as S
    S.CaveScene._palette = lambda self, sp: np.zeros((100, 3), np.float32)
    S.CaveScene._bake = lambda self, mv, mn, mesh_full, lights, pal, ambient: np.zeros((len(mv), 3), np.float32)


def load_wm(ckpt, root, dev="cuda"):
    import torch  # noqa: F401
    from ...evaluation.standalone import load_checkpoint
    from ...training.setup import build_model, normalizer
    from ..mbari.run_uncertainty import _cfg_for
    cfg = _cfg_for(ckpt)
    cfg.data.root = root
    cfg.model.integrate_position_from_velocity = True        # the red model
    cfg.environments.velocity_idx = [7, 8, 9]                # saved config lacks it; the override needs it
    core = build_model(cfg).to(dev)
    load_checkpoint(core, ckpt)
    core.eval()
    return core, normalizer(cfg), int(cfg.data.P)


def sample_pose_pairs(scene, points_npz, n_pairs, seed=0, min_clear=0.85, min_sep=2.5, max_sep=8.0,
                      max_detour=1.6):
    """n (start, goal) pose pairs from recorded val trajectory points: both collision-free and REACHABLE.
    Each pose = (position, yaw). start12 is from rest (pitch/roll/vel/rates = 0). `points_npz` holds pts (N,7) =
    pos(3)+quat(4), pre-extracted in the container (the bridge venv has no lerobot to read the LeRobot dataset).
    Pairs are kept near-line-of-sight (geodesic <= max_detour*euclid) so the benchmark measures the DYNAMICS
    model under MPPI, not global path planning around blind corners (a separate problem)."""
    from .control import quat_to_yaw
    raw = np.load(points_npz)["pts"]    # (N,7) pos+quat
    pts = []    # (pos(3), yaw) with safe clearance
    for row in raw:
        p = row[:3]
        if scene.clearance(p) >= min_clear:
            pts.append((p.astype(np.float32), float(quat_to_yaw(row[3:7]))))
    rng = np.random.default_rng(seed)
    pairs = []
    tries = 0
    while len(pairs) < n_pairs and tries < n_pairs * 400:
        tries += 1
        i, j = rng.integers(0, len(pts), 2)
        (ps, ys), (pg, yg) = pts[i], pts[j]
        sep = np.linalg.norm(pg - ps)
        if not (min_sep <= sep <= max_sep):
            continue
        try:
            path = scene.plan_fast(ps, pg)
        except Exception:
            continue
        geo = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        if geo > max_detour * sep:        # skip blind-corner goals; keep reachable point-to-point hops
            continue
        s0 = np.zeros(12, np.float32); s0[:3] = ps; s0[3] = ys
        pairs.append({"s0": s0, "goal_p": pg, "goal_yaw": yg, "geodesic": geo, "euclid": float(sep)})
    return pairs


def run(ckpt, root, out, points_npz, n_pairs=20, steps=300, dev="cuda", n_trials=3,
        arms=("true", "learned", "analytic"), log=print):
    import jax.numpy as jnp  # noqa: F401  (ensure jax under CPU is importable before heavy work)
    from prospector.cave_wm.record import build_dynamics
    from .control import TrueDyn, LearnedDyn, AnalyticDyn, run_episode

    _patch_scene_render()
    from prospector.cave_wm.scene import CaveScene
    log("[setup] building CaveScene (collision-only) ...")
    scene = CaveScene(log=lambda *a: None)
    dyn = build_dynamics(dt=1.0 / 30.0)
    log("[setup] loading red WM ...")
    core, norm, P = load_wm(ckpt, root, dev)

    log(f"[setup] sampling {n_pairs} pose pairs from val ...")
    pairs = sample_pose_pairs(scene, points_npz, n_pairs)
    log(f"[setup] got {len(pairs)} pairs (geodesic {np.mean([p['geodesic'] for p in pairs]):.1f} m avg)")

    def make(arm):
        if arm == "true":
            return TrueDyn(dyn)
        if arm == "analytic":
            return AnalyticDyn()
        return LearnedDyn(core, norm, P=P, dev=dev)

    # n_trials per pair with a DISTINCT seed each (submarine discipline: identical numbers across nominally
    # independent trials are a seed or a budget, never a result -- vary the seed, report the spread). results_all
    # is the flat list of every (pair x trial) episode (summary stats + spread); results keeps the first trial per
    # pair for the trajectory plot. Seeds are shared across arms per (pair,trial) so the only variable is the model.
    results = {a: [] for a in arms}
    results_all = {a: [] for a in arms}
    for a in arms:
        log(f"[run] arm={a}")
        for i, pr in enumerate(pairs):
            for t in range(n_trials):
                model = make(a)
                r = run_episode(TrueDyn(dyn), model, pr["s0"], pr["goal_p"], pr["goal_yaw"], scene,
                                pr["euclid"], steps=steps, seed=2000 + i * 17 + t * 101)
                results_all[a].append(r)
                if t == 0:
                    results[a].append(r)
            r0 = results[a][i]
            log(f"    pair {i:2d}: success={int(r0['success'])} coll={int(r0['collided'])} "
                f"final={r0['final_dist']:.2f}m att={r0['att_deg']:.0f}deg wander={r0['wander']:.3f} "
                f"a={r0['a_goal']:.3f} clr_min={r0['closest_obstacle']:.2f}")

    outp = pathlib.Path(out); outp.mkdir(parents=True, exist_ok=True)
    summary = _summarize(results_all, arms)
    summary["_meta"] = {"n_pairs": len(pairs), "n_trials": n_trials}
    (outp / "metrics.json").write_text(json.dumps(summary, indent=2))
    _print_table(summary, arms, log)
    _plot_summary(summary, arms, outp)
    _plot_trajectories(results, pairs, arms, scene, outp)
    log(f"[done] wrote metrics.json + plots to {outp}")
    return summary


def _summarize(results, arms):
    """Submarine-spec summary: mean + spread (std across trials) for every settling metric, so the report always
    shows wander and |a| next to final error. Window stats (wander/v/a/da) come from each episode's last 2 s."""
    def ms(arr):
        arr = np.asarray(arr, float); arr = arr[np.isfinite(arr)]
        return [float(arr.mean()), float(arr.std())] if len(arr) else [float("nan"), 0.0]
    out = {}
    for a in arms:
        R = results[a]
        succ = np.array([r["success"] for r in R], bool)
        s2p = np.array([r["steps_to_0p5m"] for r in R], float)
        out[a] = {
            "n": len(R),
            "success_rate": float(succ.mean()),
            "collision_rate": float(np.mean([r["collided"] for r in R])),
            "final_m": ms([r["final_dist"] for r in R]),            # terminal position error (report, don't tune on)
            "best_m": ms([r["best_dist"] for r in R]),              # best along path
            "att_deg": ms([r["att_deg"] for r in R]),              # geodesic attitude error to the goal heading
            "wander": ms([r["wander"] for r in R]),                # peak-to-peak dist over last 2 s (THE jitter metric)
            "v": ms([r["v_goal"] for r in R]),                     # terminal speed (m/s)
            "a": ms([r["a_goal"] for r in R]),                     # mean |a| at rest (fraction of full command)
            "da": ms([r["da_goal"] for r in R]),                   # command change at rest (fraction/step)
            "path_mult": ms([r["path_mult"] for r in R if r["success"]]),
            "steps_to_0p5m": ms(s2p[s2p >= 0]),
            "closest_obstacle_m": ms([r["closest_obstacle"] for r in R]),
            "closest_obstacle_min_m": float(np.min([r["closest_obstacle"] for r in R])),
        }
    return out


def _print_table(summary, arms, log):
    # wander and |a| sit right next to final error, always (the submarine design rule).
    cols = [("final_m", "final(m)"), ("best_m", "best(m)"), ("att_deg", "att(deg)"), ("wander", "wander(m)"),
            ("v", "v(m/s)"), ("a", "|a|"), ("da", "da"), ("path_mult", "path_x"), ("closest_obstacle_m", "clr(m)")]
    meta = summary.get("_meta", {})
    log(f"\n=== cave point-to-point control benchmark  ({meta.get('n_pairs','?')} pairs x "
        f"{meta.get('n_trials','?')} trials; mean +/- std) ===")
    log(f"{'arm':>9} | {'success':>8} {'collide':>8} | " + " ".join(f"{h:>16}" for _, h in cols))
    for a in arms:
        s = summary[a]
        cells = [f"{s[k][0]:.3f}+-{s[k][1]:.3f}" for k, _ in cols]
        log(f"{a:>9} | {s['success_rate']:>8.2f} {s['collision_rate']:>8.2f} | "
            + " ".join(f"{c:>16}" for c in cells))


def _plot_summary(summary, arms, outp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # scalar metrics (success/collision) and [mean,std] metrics (the rest, shown with error bars = the spread).
    metrics = [("success_rate", "Success rate", "higher better", False),
               ("final_m", "Final error (m)", "lower better", True),
               ("att_deg", "Attitude error (deg)", "lower better", True),
               ("wander", "Wander (m, last 2 s)", "lower better", True),
               ("a", "|a| at rest (frac)", "lower better", True),
               ("closest_obstacle_m", "Closest obstacle (m)", "higher better", True)]
    colors = {"true": "#4c72b0", "learned": "#c44e52", "analytic": "#55a868"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (k, title, note, has_std) in zip(axes.ravel(), metrics):
        vals = [summary[a][k][0] if has_std else summary[a][k] for a in arms]
        errs = [summary[a][k][1] if has_std else 0.0 for a in arms]
        ax.bar(range(len(arms)), vals, yerr=errs, capsize=4, color=[colors.get(a, "#888") for a in arms])
        ax.set_xticks(range(len(arms))); ax.set_xticklabels(arms)
        ax.set_title(f"{title}\n({note})", fontsize=11); ax.grid(alpha=0.3, axis="y")
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Cave point-to-point control: learned WM vs true dynamics vs analytic "
                 f"({summary[arms[0]]['n']} pose pairs)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outp / "control_metrics.png", dpi=120); plt.close(fig)


def _plot_trajectories(results, pairs, arms, scene, outp, n=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"true": "#4c72b0", "learned": "#c44e52", "analytic": "#55a868"}
    n = min(n, len(pairs)); cols = 3; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.3 * rows), squeeze=False)
    for k in range(rows * cols):
        ax = axes[k // cols][k % cols]
        if k >= n:
            ax.axis("off"); continue
        gp = pairs[k]["goal_p"]; s0 = pairs[k]["s0"]
        for a in arms:
            tr = results[a][k]["traj"]
            ls = "-" if results[a][k]["success"] else "--"
            ax.plot(tr[:, 0], tr[:, 1], color=colors.get(a, "#888"), lw=1.8, ls=ls, label=a)
            if results[a][k]["collided"]:
                ax.scatter([tr[-1, 0]], [tr[-1, 1]], marker="x", c=colors.get(a), s=60, zorder=6)
        ax.scatter([s0[0]], [s0[1]], c="green", s=50, zorder=5, label="start")
        ax.scatter([gp[0]], [gp[1]], c="black", marker="*", s=130, zorder=5, label="goal")
        ax.set_title(f"pair {k}  (geo {pairs[k]['geodesic']:.1f} m)", fontsize=10)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.axis("equal"); ax.grid(alpha=0.3)
        if k == 0:
            ax.legend(fontsize=8)
    fig.suptitle("Executed trajectories (top-down): solid=success, dashed=failed, x=collision", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outp / "control_trajectories.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Cave point-to-point pose control: learned vs true dynamics.")
    ap.add_argument("--ckpt", required=True, help="red WM best.ckpt (prior-free + velocity-integral)")
    ap.add_argument("--root", required=True, help="packaged dataset dir (has val/)")
    ap.add_argument("--out", default="/home/ubuntu/user_irw/quickdraw/logs/cave/control")
    ap.add_argument("--points_npz", default="/home/ubuntu/user_irw/quickdraw/logs/cave/control/val_points.npz",
                    help="pre-extracted val pose points (pos+quat); made in the container (lerobot)")
    ap.add_argument("--n_pairs", type=int, default=20)
    ap.add_argument("--n_trials", type=int, default=3, help="trials per pair, distinct seeds (report the spread)")
    ap.add_argument("--steps", type=int, default=360)
    ap.add_argument("--arms", default="true,learned,analytic")
    a = ap.parse_args()
    run(a.ckpt, a.root, a.out, a.points_npz, n_pairs=a.n_pairs, steps=a.steps, n_trials=a.n_trials,
        arms=tuple(a.arms.split(",")), log=lambda s: print(s, flush=True))
