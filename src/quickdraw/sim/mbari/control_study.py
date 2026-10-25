"""Every controller measurement behind the tuned MPPI schedule, as one reproducible entry point.

WHY THIS IS IN THE CODEBASE AND NOT IN scratch/. The numbers in `wizard/records/mbari.md` --
"mean_decay 0.7 measured best", "w_att above 0.3 is pure cost", "exec_n 2 is the knee" -- were each
produced by a separate throwaway script that differed from its neighbour only in the grid it looped
over. That is four copies of the same twenty lines, none of them committed, and a claim nobody can
re-check six months from now. The loop is the same every time:

    build the tank -> take the same N goals and starts -> override the schedule and the mppi kwargs
    -> run a closed-loop episode on TRUE dynamics -> report the same settling metrics

so it is written once here, the grids are data, and every table in the record names the command that
regenerates it.

    python -m quickdraw.sim.mbari.control_study sweep decay      # warm-start retirement
    python -m quickdraw.sim.mbari.control_study sweep effort     # control cost vs velocity brake
    python -m quickdraw.sim.mbari.control_study sweep settle     # sigma / horizon / brake
    python -m quickdraw.sim.mbari.control_study sweep attitude   # w_att and exec_n
    python -m quickdraw.sim.mbari.control_study probe            # per-step dump at the goal
    python -m quickdraw.sim.mbari.control_study clips --schedule best --trials 3

MEASURE SETTLING, NOT ARRIVAL. `final_dist` alone called a controller successful while it sat on the
goal firing a third of full thrust on all six axes. Every row below therefore carries `wander` (the
peak-to-peak spread of distance over the last 2 s), `|v|`, `|a|` and `|da|`; the failure that mattered
was only ever visible in those.

ALL SWEEPS RUN ON TrueDynamics. The point is to tune the PLANNER, so the dynamics model is held
perfect -- any settling failure seen here is the search, not the model. The learned arm is
`run_control.py`, which takes whatever schedule this module settles on.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from . import control as C, record, run_control

# Named schedules. A "schedule" is what `run_episode` reads out of C.COARSE / C.FINE each step.
#   repeat  how many control steps one planned action is HELD (action coarseness)
#   exec_n  how many steps of the plan are SPENT before replanning (defaults to `repeat`)
# These were one number until the attitude sweep separated them; see control.py's note.
SCHEDULES = {
    # The two-mode default that ships in control.py, flattened to one mode for sweeping.
    "coarse": dict(repeat=6, horizon=12, samples=24, sigma=0.50),
    # What produced logs/mbari/control/clips_tuned: more samples, a third of the action noise.
    "tuned": dict(repeat=6, horizon=12, samples=96, sigma=0.20),
    # Tuned plus the attitude sweep's two winners. `w_att` rides in MPPI_KW, not here.
    "best": dict(repeat=6, horizon=12, samples=96, sigma=0.20, exec_n=2),
}
MPPI_KW = {"best": dict(w_att=0.3)}

TUNED = SCHEDULES["tuned"]
GRIDS = {
    # Does retiring the warm start stop the thrashing? 1.0 = never decays, which was the bug.
    "decay": [(f"mean_decay {m:.1f}", TUNED, dict(mean_decay=m)) for m in (1.0, 0.7, 0.4, 0.0)],
    # Penalise EFFORT or penalise SPEED? beta_ctrl is a flat cost on command magnitude; beta_vel is
    # the near-goal velocity gate that controller/mppi.py ships switched off.
    "effort": [("bctrl 0.02 bvel 0.00", TUNED, dict(beta_ctrl=0.02, w_vel=0.0)),
               ("bctrl 0.20 bvel 0.00", TUNED, dict(beta_ctrl=0.20, w_vel=0.0)),
               ("bctrl 1.00 bvel 0.00", TUNED, dict(beta_ctrl=1.00, w_vel=0.0)),
               ("bctrl 0.02 bvel 0.50", TUNED, dict(beta_ctrl=0.02, w_vel=0.5))],
    # Terminal speed was the complaint: it parked by passing THROUGH the goal. Three candidate
    # cures -- brake harder, sample finer, or see further past the overshoot.
    "settle": [("tuned baseline", TUNED, {}),
               ("+ brake bvel 1.0", TUNED, dict(w_vel=1.0)),
               ("+ brake bvel 3.0", TUNED, dict(w_vel=3.0)),
               ("sigma 0.08", dict(TUNED, sigma=0.08), {}),
               ("sigma 0.08 + brake 3.0", dict(TUNED, sigma=0.08), dict(w_vel=3.0)),
               ("horizon 24 + brake 3.0", dict(TUNED, horizon=24), dict(w_vel=3.0))],
    # w_att: position and attitude are FUNGIBLE in the cost at w_att rad^-1. If position error falls
    # as w_att falls while attitude does NOT rise, the weight was buying nothing.
    # exec_n: actions are held 6 steps (0.2 s), so even a perfect correction runs open-loop through
    # 0.2 s of staleness. Shortening the SPEND without changing the hold isolates that.
    "attitude": [("w_att 0.3", TUNED, dict(w_att=0.3)),
                 ("w_att 1.2 (shipped)", TUNED, dict(w_att=1.2)),
                 ("w_att 4.0", TUNED, dict(w_att=4.0)),
                 ("exec_n 2", dict(TUNED, exec_n=2), dict(w_att=1.2)),
                 ("exec_n 1", dict(TUNED, exec_n=1), dict(w_att=1.2))],
    # The corner the one-at-a-time sweep never ran. They fix different failures, so they ought to
    # compose -- "ought to" being why it is measured.
    "combined": [("w_att 0.3 + exec_n 2", dict(TUNED, exec_n=2), dict(w_att=0.3)),
                 ("w_att 1.2 exec_n 6 (shipped)", TUNED, dict(w_att=1.2))],
}


def metrics(r, goal_q, n=60):
    """The settling numbers, all measured over the last `n` steps (2 s at 30 Hz) rather than at the
    final instant -- a single frame cannot distinguish 'stopped' from 'passing through'."""
    tr, ac, d = r["traj"], r["act"], r["dist"]
    qf = tr[-1, 3:7] / max(1e-9, np.linalg.norm(tr[-1, 3:7]))
    return {
        "final_m": r["final_dist"],
        "att_deg": float(np.degrees(2 * np.arccos(min(1.0, abs(float(qf @ np.asarray(goal_q))))))),
        "wander": float(d[-n:].max() - d[-n:].min()),   # peak-to-peak: "it jitters", quantified
        "v": float(np.linalg.norm(tr[-n:, 7:10], axis=1).mean()),
        "a": float(np.abs(ac[-n:]).mean()),             # command magnitude: 0.33 was the bug
        "da": float(np.abs(np.diff(ac[-n:], axis=0)).mean()),
        "crashed": r["crashed"],
    }


def _task(model, trials, seed=0):
    """The same goals and starts for every row, so a sweep is an A/B of the schedule, not of luck."""
    return run_control.goals(model, n=trials, seed=seed), run_control.starts(model, trials)


def _run(model, sched, kw, goals, starts, steps, seed=100):
    """One row: `trials` closed-loop episodes under one (schedule, mppi kwargs) pair, averaged.

    The schedule is installed by assignment because `run_episode` reads the module globals every
    step; NEAR=99 collapses the two-mode switch so a sweep measures ONE schedule rather than a
    blend of two. Both are restored by the caller."""
    C.FINE = C.COARSE = sched
    orig = C.mppi
    C.mppi = lambda *a, **k: orig(*a, **{**k, **kw})
    try:
        rows = []
        for i, (gp, gq) in enumerate(goals):
            r = C.run_episode(model, C.TrueDynamics(model), starts[i], gp, goal_q=gq,
                              steps=steps, seed=seed + i)
            rows.append(metrics(r, gq))
        return {k: (np.mean([row[k] for row in rows]) if k != "crashed"
                    else any(row[k] for row in rows)) for k in rows[0]}
    finally:
        C.mppi = orig


def sweep(name, trials=3, steps=600, out=None):
    model = record.load_model()
    goals, starts = _task(model, trials)
    near, coarse, fine = C.NEAR, C.COARSE, C.FINE
    C.NEAR = 99.0
    lines = [f"  {'config':28s} {'final_m':>8} {'att':>6} {'wander':>8} {'|v|':>7} {'|a|':>7} {'|da|':>7}"]
    print(lines[0], flush=True)
    try:
        for label, sched, kw in GRIDS[name]:
            t0 = time.time()
            m = _run(model, sched, kw, goals, starts, steps)
            line = (f"  {label:28s} {m['final_m']:8.3f} {m['att_deg']:6.1f} {m['wander']:8.3f} "
                    f"{m['v']:7.3f} {m['a']:7.3f} {m['da']:7.3f}   [{(time.time()-t0)/60:.1f} min]")
            lines.append(line)
            print(line, flush=True)
    finally:
        C.NEAR, C.COARSE, C.FINE = near, coarse, fine
    if out:
        with open(out, "w") as fh:
            fh.write("\n".join(lines) + "\n")
    return lines


def probe(schedule="tuned", steps=600, seed=100, n=120):
    """Per-step dump of the last seconds at the goal. WHICH quantity fails to reach zero names the
    mechanism: |a| high with |da| low is a stuck warm start, both high is search noise, |v| high
    with |a| low is coasting. This is the diagnostic that found the un-retired warm start."""
    model = record.load_model()
    (goal,), (start,) = _task(model, 1)
    near, coarse, fine = C.NEAR, C.COARSE, C.FINE
    C.NEAR, C.COARSE, C.FINE = 99.0, SCHEDULES[schedule], SCHEDULES[schedule]
    orig = C.mppi
    C.mppi = lambda *a, **k: orig(*a, **{**k, **MPPI_KW.get(schedule, {})})
    try:
        r = C.run_episode(model, C.TrueDynamics(model), start, goal[0], goal_q=goal[1],
                          steps=steps, seed=seed)
    finally:
        C.NEAR, C.COARSE, C.FINE, C.mppi = near, coarse, fine, orig
    tr, ac, d = r["traj"], r["act"], r["dist"]
    v = np.linalg.norm(tr[:, 7:10], axis=1)
    w = np.linalg.norm(tr[:, 10:13], axis=1)
    am = np.abs(ac).mean(1)
    da = np.r_[0, np.abs(np.diff(ac, axis=0)).mean(1)]
    print(f"  schedule {schedule}   final {d[-1]:.3f} m   crashed={r['crashed']}")
    print(f"  {'step':>5} {'dist':>7} {'|v|':>7} {'|w|':>7} {'|a|':>7} {'|da|':>7}")
    for k in range(max(0, len(d) - n), len(d), 10):
        print(f"  {k:5d} {d[k]:7.3f} {v[k]:7.3f} {w[k]:7.3f} {am[k]:7.3f} {da[k]:7.3f}")
    print(f"  --- last 60 steps (2 s) means ---")
    print(f"  dist {d[-60:].mean():.3f}  |v| {v[-60:].mean():.3f}  |w| {w[-60:].mean():.3f}  "
          f"|a| {am[-60:].mean():.3f}  |da| {da[-60:].mean():.3f}")
    print(f"  per-axis |a| last 2 s: {np.round(np.abs(ac[-60:]).mean(0), 3)}")
    print(f"       (surge sway heave | roll pitch yaw)")
    return r


def clips(schedule="best", trials=3, steps=840, out=None, trial=None, egl_device=None):
    """Render closed-loop episodes in the tank from every camera, with the goal-pose ghost placed.

    ONE TRIAL PER PROCESS when `trial` is given. Two reasons, and the second is a bug, not taste:
    a schedule that replans 3x as often takes ~3x as long, so three sequential renders is half a
    day on a box whose 80 cores sit at load 2 -- and a SECOND UnderwaterRenderer in one process
    returns an uninitialised buffer, so the frames come back as pure noise. One renderer per
    process is both faster and the only arrangement that has ever been correct.

    REAL TIME: every frame at the 30 Hz sim rate, written at 30 fps, so a second of video is a
    second of tank. An earlier version kept every 4th frame and wrote 15 fps, running a 20 s
    approach in 10.
    """
    import os

    import mujoco

    from ...logging import viz
    from . import tank as _tank
    from .water import UnderwaterRenderer

    if egl_device is not None:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl_device)
    cams = ("fpv", "chase", "topdown", "sideon")
    out = out or f"logs/mbari/control/clips_{schedule}"
    os.makedirs(out, exist_ok=True)
    # A per-trial scene file: concurrent processes must not race on one path.
    suffix = "" if trial is None else str(trial)
    path, _ = _tank.build(textures=False, name=f"tank_clip{suffix}.xml", ghost=True)
    model = mujoco.MjModel.from_xml_path(str(path))
    goals, starts = _task(model, trials)
    near, coarse, fine = C.NEAR, C.COARSE, C.FINE
    C.NEAR, C.COARSE, C.FINE = 99.0, SCHEDULES[schedule], SCHEDULES[schedule]
    orig = C.mppi
    C.mppi = lambda *a, **k: orig(*a, **{**k, **MPPI_KW.get(schedule, {})})
    gid = int(model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ghost")])
    rend = UnderwaterRenderer(model, 288, 384, cams=cams + ("dockcam",))
    try:
        for i in ([trial] if trial is not None else range(trials)):
            gp, gq = goals[i]
            # A FIXED camera anchored to the goal. The chase camera is 4.2 m directly behind the
            # vehicle, so at the moment of arrival -- the moment worth watching -- it is buried in
            # a wall. This one sits on the tank-centre side of the goal and does not orbit, because
            # a circling camera sweeps through the glass.
            rend.look_at(gp, dist=4.6, elev=-18.0)
            t0 = time.time()
            r = C.run_episode(model, C.TrueDynamics(model), starts[i], gp, goal_q=gq,
                              steps=steps, seed=100, render=rend, ghost=(gid, gp, gq))
            dockcam = r["frames"].pop("dockcam")
            m = metrics(r, gq)
            panels = [r["frames"][c].astype(np.float32) / 255 for c in cams] + [
                np.asarray(dockcam, np.float32) / 255]
            canvas = viz.label_panels(panels, ["FPV", "Chase", "Top-down", "Side-on", "Goal cam"],
                                      panel_wh=(384, 288), header_px=58, font_px=36)
            viz.save_mp4(f"{out}/pose_trial{i}.mp4", canvas, 30, quality=8)
            print(f"  pose_trial{i}  {r['dist'][0]:.2f} m -> {m['final_m']:.3f} m, "
                  f"{m['att_deg']:.1f} deg, wander {m['wander']:.3f}, |v| {m['v']:.3f}, "
                  f"|a| {m['a']:.3f}, crashed={m['crashed']}  [{(time.time()-t0)/60:.1f} min]",
                  flush=True)
    finally:
        C.NEAR, C.COARSE, C.FINE, C.mppi = near, coarse, fine, orig
    print("wrote", out)


def compare_clips(ckpt, trial, out=None, steps=840, egl_device=None, log=print):
    """ONE trial, both arms, tiled -- true dynamics above, the learned world model below.

    The point is that the numbers alone ("0.009 m vs 0.022 m") do not show whether the learned arm
    is doing the same THING or getting there by luck. Same trial, same start, same goal, same
    planner, same seed; the only difference between the rows is which dynamics model MPPI planned
    against. If the two rows move together the model has learned the dynamics; if they diverge and
    reconverge, it has not.

    TWO cameras per arm: FPV and the goal camera. Chase was dropped -- it shows the vehicle from
    behind at a fixed offset, which duplicates what the goal camera already says about where it is
    and adds nothing about whether it arrived.

    THE WORLD MODEL IS THE TOP ROW. It is the thing being presented; true dynamics underneath is
    the reference it is read against, and a reader scans down from the claim to the baseline.

    NO STATS BAKED INTO THE FRAME. The per-trial numbers go to a text file beside the clips, so the
    deck can quote them at whatever size it likes and a re-measure does not require a re-render.
    """
    import os

    import mujoco

    from ...logging import viz
    from . import run_control, tank as _tank
    from .water import UnderwaterRenderer

    if egl_device is not None:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl_device)
    cams = ("fpv",)
    out = out or "logs/mbari/control/compare"
    os.makedirs(out, exist_ok=True)
    path, _ = _tank.build(textures=False, name=f"tank_cmp{trial}.xml", ghost=True)
    model = mujoco.MjModel.from_xml_path(str(path))
    goals, starts = _task(model, max(trial + 1, 8))
    gp, gq = goals[trial]
    gid = int(model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ghost")])

    near, coarse, fine = C.NEAR, C.COARSE, C.FINE
    C.NEAR, C.COARSE, C.FINE = 99.0, SCHEDULES["best"], SCHEDULES["best"]
    orig = C.mppi
    C.mppi = lambda *a, **k: orig(*a, **{**k, **MPPI_KW["best"]})
    # Two panels per row now, so render them at the size they are shown at rather
    # than upscaling 288x384 into a 512-wide slot.
    rend = UnderwaterRenderer(model, 384, 512, cams=cams + ("dockcam",))
    rend.look_at(gp, dist=4.6, elev=-18.0)
    rows, labels, stats = [], [], []
    try:
        for name, dyn in (("WORLD MODEL", run_control.learned_from(ckpt, model)),
                          ("TRUE DYNAMICS", C.TrueDynamics(model))):
            t0 = time.time()
            r = C.run_episode(model, dyn, starts[trial], gp, goal_q=gq, steps=steps, seed=100 + trial,
                              render=rend, ghost=(gid, gp, gq))
            m = metrics(r, gq)
            dock = r["frames"].pop("dockcam")
            panels = [r["frames"][c].astype(np.float32) / 255 for c in cams] + [
                np.asarray(dock, np.float32) / 255]
            rows.append(panels)
            # SHORT LABELS, AND THE ARM NAMED ONCE. The first version repeated the arm name on all
            # three captions at font 26 against a 384 px panel; they overlapped each other and ran
            # off the right edge, which makes the one figure whose entire job is side-by-side
            # comparison unreadable. The arm belongs on its row, not on every panel.
            labels += [f"{name} (FPV)", f"{name} (GOAL)"]
            stats.append((name, m))
            log(f"  trial {trial} {name:22s} {m['final_m']:.3f} m, {m['att_deg']:.1f} deg, "
                f"wander {m['wander']:.3f}  [{(time.time()-t0)/60:.1f} min]")
    finally:
        C.NEAR, C.COARSE, C.FINE, C.mppi = near, coarse, fine, orig
    # ONE STRIP PER ARM, STACKED. label_panels lays panels out in a single row and has no grid
    # option, so the two arms are rendered as two strips and concatenated vertically -- which is
    # better anyway: each arm keeps its own label band directly above its own footage, instead of
    # six captions in one line that the eye has to re-associate.
    n = min(len(p) for row in rows for p in row)              # arms can differ in length on a crash
    strips = [viz.label_panels([p[:n] for p in row], labels[2 * i:2 * i + 2],
                               panel_wh=(512, 384), header_px=46, font_px=22)
              for i, row in enumerate(rows)]
    canvas = np.concatenate(strips, axis=1)                   # (T, 2*H, W, 3): stack rows vertically
    viz.save_mp4(f"{out}/compare_trial{trial}.mp4", canvas, 30, quality=8)
    with open(f"{out}/compare_trial{trial}.txt", "w") as fh:
        fh.write(f"Trial {trial}\n")
        for nm, m in stats:
            fh.write(f"  {nm:14s} final {m['final_m']:.3f} m   attitude {m['att_deg']:.1f} deg   "
                     f"wander {m['wander']:.3f} m   crashed {m['crashed']}\n")
    log(f"  wrote {out}/compare_trial{trial}.mp4 (+ .txt)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sweep");  s.add_argument("grid", choices=sorted(GRIDS))
    s.add_argument("--trials", type=int, default=3); s.add_argument("--steps", type=int, default=600)
    s.add_argument("--out")
    p = sub.add_parser("probe"); p.add_argument("--schedule", default="tuned", choices=sorted(SCHEDULES))
    p.add_argument("--steps", type=int, default=600)
    k = sub.add_parser("compare", help="one trial, true vs learned, tiled into one video")
    k.add_argument("--ckpt", required=True); k.add_argument("--trial", type=int, required=True)
    k.add_argument("--out"); k.add_argument("--steps", type=int, default=840)
    k.add_argument("--egl-device", type=int)
    c = sub.add_parser("clips"); c.add_argument("--schedule", default="best", choices=sorted(SCHEDULES))
    c.add_argument("--trials", type=int, default=3); c.add_argument("--steps", type=int, default=840)
    c.add_argument("--trial", type=int, help="render only this index, for one-per-process parallelism")
    c.add_argument("--out"); c.add_argument("--egl-device", type=int)
    a = ap.parse_args()
    if a.cmd == "compare":
        compare_clips(a.ckpt, a.trial, out=a.out, steps=a.steps, egl_device=a.egl_device)
    elif a.cmd == "sweep":
        sweep(a.grid, trials=a.trials, steps=a.steps, out=a.out)
    elif a.cmd == "probe":
        probe(schedule=a.schedule, steps=a.steps)
    else:
        clips(schedule=a.schedule, trials=a.trials, steps=a.steps, out=a.out,
              trial=a.trial, egl_device=a.egl_device)


if __name__ == "__main__":
    main()
