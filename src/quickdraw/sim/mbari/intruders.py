"""The object/appearance OOD conditions: three great white sharks, or a purple buoy.

MOCAP BODIES, NOT FREE BODIES. Both are scripted: they follow analytic paths across the tank and
nothing in the scene reacts to them. A mocap body is kinematic -- driven by `data.mocap_pos` -- with
no dynamics and no contacts. Freejoints would let them drift into the vehicle and turn an
*appearance* OOD into a *dynamics* OOD, confounding the three classes the detector has to separate.

THREE SHARKS, NOT ONE. A single intruder crossing a 15 m tank during a 12 s episode is often never
in the FPV frustum at all, and an OOD the camera never sees is not one the detector can be asked
about. Three patrolling at different depths raise the odds of at least one being seen.

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


SHARK_L = 4.0                 # the mesh is 2.5 m scaled 1.6x
DEPTHS = (-2.3, -5.2, -8.1)   # one shark per depth band, 2.9 m apart


def _patrol(rng, i):
    """A circuit for shark `i`: its own depth band, radius, speed and direction.

    NO INTERSECTIONS, BY CONSTRUCTION. The three sharks are given depth bands 2.6 m apart and a
    band jitter of at most 0.2 m and a vertical bob of at most 0.28 m, so the closest two can ever
    come is 2.9 - 2*0.2 - 2*0.28 = 1.94 m against a body 1.33 m tall: 0.6 m of clearance in the
    WORST case, not merely the sampled one. Separating in depth rather than rejecting sampled paths
    means it holds for every seed and every episode length, with nothing to check at runtime.

    FORWARD, ON A CIRCLE. The heading is the tangent to the circle, so the animal always swims nose
    first. The first version drove straight transits with a heading taken from the chord, and with
    the geom yawed the wrong way (the mesh swims along +Y, not -Y) they crossed the tank tail
    first."""
    z = DEPTHS[i % len(DEPTHS)] + rng.uniform(-0.2, 0.2)
    rad = rng.uniform(2.2, 4.0)
    cx = rng.uniform(-1, 1) * max(0.0, TANK_L / 2 - rad - 1.2)
    cy = rng.uniform(-1, 1) * max(0.0, TANK_W / 2 - rad - 1.2)
    spd = rng.uniform(0.55, 1.05)
    return dict(c=np.array([cx, cy, z]), rad=rad, w=(spd / rad) * (1 if rng.random() < 0.5 else -1),
                phase=rng.uniform(0, 2 * np.pi), bob=rng.uniform(0.10, 0.28),
                bob_f=rng.uniform(0.10, 0.22), sway=rng.uniform(0.05, 0.11))


def mover(model, intruder: str, seed: int):
    """Returns extra(model, data, k) that drives every intruder body for this condition.

    The purple-buoy condition has nothing to move, so this returns a no-op for it."""
    import mujoco

    rng = np.random.default_rng(seed)
    names = [f"fish{i}" for i in range(N_FISH)] if intruder == "fish" else []
    plan = []
    for i, nm in enumerate(names):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
        if bid < 0:
            continue
        plan.append((int(model.body_mocapid[bid]), _patrol(rng, i)))

    def extra(_model, data, k):
        t = k / 30.0
        for mid, s in plan:
            a = s["w"] * t + s["phase"]
            p = s["c"] + np.array([s["rad"] * np.cos(a), s["rad"] * np.sin(a), 0.0])
            p[2] += s["bob"] * np.sin(2 * np.pi * s["bob_f"] * t)
            # heading = the tangent, plus a small tail sway so it does not look rail-mounted
            yaw = a + (np.pi / 2 if s["w"] > 0 else -np.pi / 2)
            yaw += s["sway"] * np.sin(2 * np.pi * 0.7 * t + s["phase"])
            data.mocap_pos[mid] = p
            data.mocap_quat[mid] = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    return extra
