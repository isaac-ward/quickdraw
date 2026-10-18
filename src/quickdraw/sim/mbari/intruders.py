"""The two object-OOD intruders: an orange fish and a diver.

MOCAP BODIES, NOT FREE BODIES. Both are scripted: they follow an analytic path across the tank and
nothing in the scene reacts to them. A mocap body is exactly that -- kinematic, driven by
`data.mocap_pos` / `mocap_quat`, no dynamics, no contacts, no effect on the AUV. Giving them freejoints
would add two more state vectors to integrate and let them drift into the vehicle, which would turn an
*appearance* OOD into a *dynamics* OOD and confound the three classes the detector has to separate.

THE FISH IS PROCEDURAL. The requested great white was CGTrader, which has no public download API, so
the fish is built from primitives: a flattened ellipsoid body, a tail and a dorsal fin, in orange. For
an appearance-OOD that is the right trade anyway -- what matters is that it is a compact moving object
of a colour and shape absent from every nominal frame, not that it is a particular artist's shark.
"""
from __future__ import annotations

import numpy as np

from .tank import ASSETS, TANK_D, TANK_L, TANK_W, mjcf

FISH_XML = """
    <body name="intruder" mocap="true" pos="0 0 -5">
      <geom name="fish_body" type="ellipsoid" size="0.42 0.15 0.22" material="m_fish"
            contype="0" conaffinity="0"/>
      <geom name="fish_tail" type="box" size="0.17 0.02 0.19" pos="-0.52 0 0.04" material="m_fish"
            contype="0" conaffinity="0"/>
      <geom name="fish_dors" type="box" size="0.12 0.02 0.14" pos="0.02 0 0.3" material="m_fish"
            contype="0" conaffinity="0"/>
      <geom name="fish_eye" type="sphere" size="0.045" pos="0.3 0.1 0.08" material="m_eye"
            contype="0" conaffinity="0"/>
    </body>
"""

DIVER_XML = """
    <body name="intruder" mocap="true" pos="0 0 -5">
      <geom name="diver_g" type="mesh" mesh="diver" material="m_diver" contype="0" conaffinity="0"/>
    </body>
"""

EXTRA_MATS = """
    <material name="m_fish" rgba="0.96 0.42 0.07 1" specular="0.5" shininess="0.5"/>
    <material name="m_eye" rgba="0.05 0.05 0.05 1"/>
    <material name="m_diver" rgba="0.17 0.21 0.27 1" specular="0.4"/>
"""


def xml_with(intruder: str) -> str:
    """The tank MJCF plus one mocap intruder body."""
    body = {"fish": FISH_XML, "diver": DIVER_XML}[intruder]
    s = mjcf()
    s = s.replace('    <mesh name="mola"', EXTRA_MATS + '    <mesh name="mola"')
    return s.replace("  </worldbody>", body + "  </worldbody>")


def load_model(intruder: str):
    import mujoco

    p = ASSETS / f"tank_{intruder}.xml"
    p.write_text(xml_with(intruder))
    return mujoco.MjModel.from_xml_path(str(p))


def mover(model, intruder: str, seed: int):
    """Returns extra(model, data, k): a straight transit across the tank at constant depth.

    A transit rather than a loiter so the intruder enters and leaves frame during a 12 s episode --
    an object parked in one corner may never appear in the FPV view at all, and an OOD the camera
    never sees is not an OOD the detector can be asked about."""
    import mujoco

    rng = np.random.default_rng(seed)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "intruder")
    mid = int(model.body_mocapid[bid])
    z = -rng.uniform(2.0, TANK_D - 2.0)
    side = rng.integers(0, 2)
    y0 = (1 if side else -1) * (TANK_W / 2 - 1.0)
    a = np.array([-TANK_L / 2 + 1.0, y0, z])
    b = np.array([TANK_L / 2 - 1.0, -y0 + rng.uniform(-1.5, 1.5), z + rng.uniform(-1.0, 1.0)])
    if rng.random() < 0.5:
        a, b = b, a
    speed = rng.uniform(0.45, 0.9)
    d = b - a
    L = float(np.linalg.norm(d))
    heading = d / max(1e-9, L)
    yaw = float(np.arctan2(heading[1], heading[0]))
    quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    wiggle = rng.uniform(0.1, 0.3)

    def extra(_model, data, k):
        t = k / 30.0
        u = min(1.0, speed * t / max(1e-6, L))
        p = a + d * u
        p[2] += wiggle * np.sin(2 * np.pi * 0.6 * t)         # a little life, and a moving silhouette
        data.mocap_pos[mid] = p
        data.mocap_quat[mid] = quat
    return extra
