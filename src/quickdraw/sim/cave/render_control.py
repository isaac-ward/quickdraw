"""Render control episodes LEARNED vs TRUE, real-time, as an ALL-MESH scene (no Gaussian splats).

Two rows (top = learned WM, bottom = true Dynamics3D), same pose pair + identical MPPI -- a direct comparison.
The cave is the grapevine surface MESH in rock browns/greys, the drone is the real SISL quadrotor mesh (4 props
spinning) and the goal is a translucent green GHOST of the same mesh; all composited by depth in one metric frame
(drone_render.render_view). Top-down slices the cave ceiling off above the drone so the overhead camera sees in.
Each episode holds station ~8 s after arrival so you see the drone parked on the ghost. The house video layout
(viz.label_panels + viz.save_mp4) does the tiling / titles. 30 fps, one sim step = one frame = REAL TIME.

Run in the BRIDGE with the venv bin on PATH and EGL for pyrender:
  PATH=<prospector>/.venv/bin:$PATH PYTHONPATH=<quickdraw>/src CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cpu \
  PYOPENGL_PLATFORM=egl <prospector>/.venv/bin/python -m quickdraw.sim.cave.render_control --ckpt .. --root .. --pairs 0,1
"""
from __future__ import annotations

import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import argparse
import pathlib

import numpy as np

UP = np.array([0.0, 0.0, -1.0], np.float32)   # grapevine true-up (gravity is +z)
HFOV = 82.0                                    # render field of view (deg), all views
SPIN = 1.3                                     # blade angle increment per frame (rad) -> props visibly spinning
GHOST_A = 0.25                                 # ghost opacity (external views only; FPV gets no ghost)
TOP_H = 2.4                                    # top-down camera height above the drone (m)
CEIL_SLICE = 0.35                             # keep cave from this far above the drone downward (ceiling cut, m)
PANEL_WH = (768, 576)                          # per-view output size (4x the 192x144 source); 3x2 grid for big screen


def _safe_eye(scene, eye, target, min_clear=0.3):
    for t in np.linspace(0.0, 1.0, 10):
        e = eye * (1 - t) + np.asarray(target) * t
        if scene.clearance(e) > min_clear:
            return e.astype(np.float32)
    return np.asarray(target, np.float32)


def _cams(scene, p, R):
    """(eye, fwd, up, ceil_cut) for fpv / chase / top. ceil_cut = cave near-plane for the overhead ceiling slice.
    The SISL logo on the drone top reads upright from the drone's LEFT side, so the chase camera sits off the left
    (slightly behind + above) with up = body-up -> the logo is readable left-to-right."""
    body_fwd = R @ np.array([1, 0, 0.0]); body_right = R @ np.array([0, 1, 0.0]); body_up = R @ np.array([0, 0, -1.0])
    fwd_h = body_fwd - (body_fwd @ UP) * UP; fwd_h /= np.linalg.norm(fwd_h) + 1e-9
    fpv = (p, body_fwd, UP, None)
    off = -0.18 * body_fwd - 0.72 * body_right + 0.38 * body_up         # behind, LEFT, above (logo side); close
    chase_eye = _safe_eye(scene, p + off, p)
    chase = (chase_eye, p - chase_eye, body_up, None)
    top_up = np.array([-fwd_h[1], fwd_h[0], 0.0], np.float32)            # fwd_h rotated 90 deg CCW -> SISL reads L-to-R
    top = (p + TOP_H * UP, -UP + 0.12 * fwd_h, top_up, TOP_H - CEIL_SLICE)
    return fpv, chase, top


def _views(scene, s, goal_p, goal_yaw, fi):
    """The 3 rendered views (fpv, chase, top) for one state s -- each (H,W,3) uint8."""
    from . import drone_render as DR
    from prospector.cave_wm.record import euler_zyx_R
    p = s[:3].astype(np.float32); R = euler_zyx_R(s[3], s[4], s[5])
    Rg = euler_zyx_R(goal_yaw, 0.0, 0.0)
    ang = fi * SPIN
    W, H = PANEL_WH
    fpv, chase, top = _cams(scene, p, R)
    out = []
    for (eye, fwd, up, cut), show in ((fpv, False), (chase, True), (top, True)):   # FPV shows no drone (it IS the drone)
        out.append(DR.render_view(scene, eye, fwd, up, W, H, HFOV, p, R, ang, goal_p, Rg,
                                  show_drone=show, show_ghost=show, ghost_a=GHOST_A, ceil_cut=cut))
    return out


def render_compare(scene, dyn, learned_fn, true_fn, pr, out_mp4, steps=360, seed=0, log=print):
    from .control import run_episode, TrueDyn
    from quickdraw.logging import viz
    rows = {}
    for tag, fn in (("learned", learned_fn), ("true", true_fn)):
        r = run_episode(TrueDyn(dyn), fn(), pr["s0"], pr["goal_p"], pr["goal_yaw"], scene, pr["euclid"],
                        steps=steps, hold_extra=240, seed=seed)
        rows[tag] = r
        log(f"    {tag}: {len(r['traj'])} steps, final {r['final_dist']:.2f}m, success={int(r['success'])}")
    L = max(len(rows["learned"]["traj"]), len(rows["true"]["traj"]))

    streams = {tag: [[], [], []] for tag in rows}                       # tag -> [fpv[], chase[], top[]]
    for i in range(L):
        for tag in rows:
            tr = rows[tag]["traj"]
            s = tr[min(i, len(tr) - 1)]
            for v, img in enumerate(_views(scene, s, pr["goal_p"], pr["goal_yaw"], i)):
                streams[tag][v].append(img)
    titles = ["FPV", "Chase", "Top-down"]
    canvas_rows = []
    for tag in ("learned", "true"):
        panels = [np.asarray(streams[tag][v]) for v in range(3)]
        name = "LEARNED WM" if tag == "learned" else "TRUE DYNAMICS"
        canvas_rows.append(viz.label_panels(panels, [f"{name}  ·  {t}" for t in titles],
                                            panel_wh=PANEL_WH, header_px=58, font_px=34))
    canvas = np.concatenate(canvas_rows, axis=1)                        # learned row over true row
    viz.save_mp4(out_mp4, canvas, 30, quality=8)
    log(f"    -> {out_mp4}  ({L} frames, {L/30:.1f}s)")
    return rows


def run(ckpt, root, out, points_npz, pairs, steps=360, dev="cuda", log=print):
    import jax.numpy as jnp  # noqa
    from prospector.cave_wm.record import build_dynamics
    from .control import TrueDyn, LearnedDyn
    from .run_control import load_wm, sample_pose_pairs

    from prospector.cave_wm.scene import CaveScene
    log("[setup] CaveScene (FULL: splat + rock mesh fallback, rendered for the video) ...")
    scene = CaveScene(log=lambda *a: None)
    dyn = build_dynamics(dt=1.0 / 30.0)
    core, norm, P = load_wm(ckpt, root, dev)
    core.stochastic_eval = False
    all_pairs = sample_pose_pairs(scene, points_npz, max(pairs) + 1)
    log(f"[setup] hfov={HFOV} deg; panels {PANEL_WH}; rendering pairs {pairs} (learned vs true)")

    def learned_fn():
        return LearnedDyn(core, norm, P=P, dev=dev)

    def true_fn():
        return TrueDyn(dyn)

    outp = pathlib.Path(out); outp.mkdir(parents=True, exist_ok=True)
    for i in pairs:
        log(f"[render] pair {i}")
        render_compare(scene, dyn, learned_fn, true_fn, all_pairs[i], str(outp / f"pair{i:02d}_compare.mp4"),
                       steps=steps, seed=1000 + i, log=log)
    log(f"[done] videos in {outp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Render cave control: learned-vs-true, all-mesh (cave+drone+ghost).")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="/home/ubuntu/user_irw/quickdraw/logs/cave/control_bench/videos")
    ap.add_argument("--points_npz", default="/home/ubuntu/user_irw/quickdraw/logs/cave/control/val_points.npz")
    ap.add_argument("--pairs", default="0,1,2,3")
    ap.add_argument("--steps", type=int, default=360)
    a = ap.parse_args()
    run(a.ckpt, a.root, a.out, a.points_npz, [int(x) for x in a.pairs.split(",")], steps=a.steps,
        log=lambda s: print(s, flush=True))
