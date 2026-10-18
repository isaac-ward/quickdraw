"""The object/appearance OOD conditions: three great white sharks, or a purple buoy.

MOCAP BODIES, NOT FREE BODIES. Both are scripted: they follow analytic paths across the tank and
nothing in the scene reacts to them. A mocap body is kinematic -- driven by `data.mocap_pos` -- with
no dynamics and no contacts. Freejoints would let them drift into the vehicle and turn an
*appearance* OOD into a *dynamics* OOD, confounding the three classes the detector has to separate.

THREE FISH, NOT ONE. A single intruder crossing a 15 m tank during a 12 s episode is often never in
the FPV frustum at all, and an OOD the camera never sees is not one the detector can be asked about.
Three independent transits at different depths raise the odds of at least one being seen.

THE DIVER OOD BECAME A PURPLE BUOY. The source diver is a T-pose; sweeping its arms back by rotating
the vertices beyond 22% of the arm-span did not read as a dive pose when rendered, and the user's
standing instruction for that case was to swap the condition for the hanging buoy turning purple
instead of neon green. That condition is static -- there is nothing to move -- so `mover` returns a
no-op for it.
"""
from __future__ import annotations

import numpy as np

from .tank import ASSETS, TANK_D, TANK_L, TANK_W, build

N_FISH = 3


def load_model(intruder: str):
    """Tank MJCF with the intruder bodies present. Textures are reused, not rebuilt."""
    import mujoco

    name = f"tank_{intruder}.xml"
    p, _ = build(textures=False, name=name,
                 n_fish=N_FISH if intruder == "fish" else 0,
                 buoy_purple=(intruder == "buoy"))
    return mujoco.MjModel.from_xml_path(str(p))


def _transit(rng):
    """A straight crossing of the tank at roughly constant depth, with a little vertical life."""
    z = -rng.uniform(2.0, TANK_D - 2.0)
    y0 = (1 if rng.integers(0, 2) else -1) * (TANK_W / 2 - 1.0)
    a = np.array([-TANK_L / 2 + 1.0, y0, z])
    b = np.array([TANK_L / 2 - 1.0, -y0 + rng.uniform(-1.5, 1.5), z + rng.uniform(-1.0, 1.0)])
    if rng.random() < 0.5:
        a, b = b, a
    d = b - a
    L = float(np.linalg.norm(d))
    yaw = float(np.arctan2(d[1], d[0]))
    return dict(a=a, d=d, L=L, speed=rng.uniform(0.45, 0.9),
                quat=np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]),
                wiggle=rng.uniform(0.1, 0.3), phase=rng.uniform(0, 6.28))


def mover(model, intruder: str, seed: int):
    """Returns extra(model, data, k) that drives every intruder body for this condition."""
    import mujoco

    rng = np.random.default_rng(seed)
    names = [f"fish{i}" for i in range(N_FISH)] if intruder == "fish" else []
    plan = []
    for nm in names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
        if bid < 0:
            continue
        plan.append((int(model.body_mocapid[bid]), _transit(rng)))

    def extra(_model, data, k):
        t = k / 30.0
        for mid, s in plan:
            u = min(1.0, s["speed"] * t / max(1e-6, s["L"]))
            p = s["a"] + s["d"] * u
            p[2] += s["wiggle"] * np.sin(2 * np.pi * 0.6 * t + s["phase"])
            data.mocap_pos[mid] = p
            data.mocap_quat[mid] = s["quat"]
    return extra
