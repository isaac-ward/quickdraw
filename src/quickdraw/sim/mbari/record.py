"""Roll episodes and record them: state, action, and four camera streams.

ONE EPISODE = one trajectory of `steps` control steps at 30 Hz, with
    obs    (T, 13)  x y z | qw qx qy qz | vx vy vz world | wx wy wz BODY
    act    (T, 6)   surge sway heave | roll pitch yaw, each in [-1,1]
    frames {cam: (T, H, W, 3) uint8} for fpv / chase / topdown / sideon

The action recorded is what was COMMANDED, not what the plant received. For nominal episodes those are
the same thing; for the glitch OOD they are deliberately not, and that gap is the whole point -- a
world model conditioned on the commanded action cannot predict the glitched outcome, which is what the
uncertainty is meant to expose.

fpv is the training view. The other three exist so a human can see what happened, and so the control
benchmark can be shown convincingly.
"""
from __future__ import annotations

import numpy as np

from . import mola, trajectories as tj
from .tank import ASSETS
from .water import UnderwaterRenderer

FPS = 30
CAMS = ("fpv", "chase", "topdown", "sideon")
TRAIN_CAM = "fpv"


def load_model():
    import mujoco

    from .tank import build

    p, _ = build(textures=False, name="tank.xml")
    return mujoco.MjModel.from_xml_path(str(p))


def _start_pose(rng, spec=None):
    """Either the circle's own start, or a random free pose well inside the tank."""
    if spec is not None:
        p, v, _ = tj.circle_at(spec, np.array(0.0))
        return p, tj.mat_to_quat(tj.frame_on_plane(v, spec["R"][:, 2]))
    for _ in range(200):
        p = np.array([rng.uniform(-1, 1) * (tj.TANK_L / 2 - tj.MARGIN),
                      rng.uniform(-1, 1) * (tj.TANK_W / 2 - tj.MARGIN),
                      -rng.uniform(tj.MARGIN, tj.TANK_D - tj.MARGIN)])
        if tj.inside(p[None])[0]:
            break
    ax = rng.normal(size=3)
    return p, tj.mat_to_quat(tj.rot_from_axis_angle(ax / np.linalg.norm(ax), rng.uniform(0, 2 * np.pi)))


def rollout(model, kind, seed, steps=900, size=(144, 192), cams=CAMS, renderer=None,
            glitch_p=0.0, extra=None):
    """One episode. `kind` in trajectories.CLASSES. Returns dict(obs, act, frames, meta).

    `extra(model, data, k)` runs once per control step, for the OOD intruders that need to be moved.
    """
    import mujoco

    rng = np.random.default_rng(seed)
    dt = 1.0 / FPS
    sub = int(round(dt / model.opt.timestep))
    data = mujoco.MjData(model)

    spec = None
    if kind in ("circle", "circle_yaw"):
        spec = tj.fit_circle(rng, steps * dt, dt)
        if spec is None:
            return None
        policy = tj.policy_circle(spec, yaw_sweep=(kind == "circle_yaw"))
    elif kind == "random":
        policy = tj.policy_random(rng, dt)
    else:
        raise ValueError(f"unknown kind {kind!r}")

    p0, q0 = _start_pose(rng, spec)
    data.qpos[:3], data.qpos[3:7] = p0, q0
    mujoco.mj_forward(model, data)

    own = renderer is None
    r = renderer or UnderwaterRenderer(model, size[0], size[1])
    obs = np.zeros((steps, 13), np.float32)
    act = np.zeros((steps, 6), np.float32)
    frames = {c: np.zeros((steps, size[0], size[1], 3), np.uint8) for c in cams}
    try:
        for k in range(steps):
            s = mola.state(data)
            if not np.isfinite(s).all():
                return None                                   # drop rather than ship a NaN episode
            a_cmd = np.asarray(policy(k * dt, s), np.float32)
            obs[k], act[k] = s, a_cmd
            for c in cams:
                frames[c][k] = r.render(data, c)
            a_plant = tj.glitch(a_cmd, rng, glitch_p) if glitch_p > 0 else a_cmd
            if extra is not None:
                extra(model, data, k)
            for _ in range(sub):
                mola.apply(model, data, a_plant)
                mujoco.mj_step(model, data)
    finally:
        if own:
            r.close()

    meta = {"kind": kind, "seed": int(seed), "steps": int(steps), "glitch_p": float(glitch_p)}
    if spec is not None:
        meta.update(radius=float(spec["rad"]), speed=float(spec["speed"]),
                    centre=[float(v) for v in spec["centre"]])
    return {"obs": obs, "act": act, "frames": frames, "meta": meta}
