"""Rooms behind the tank windows: conference rooms and libraries, rendered once and baked in.

WHY BAKED RATHER THAN GEOMETRY. The AUV can never enter these rooms and only ever sees them through a
1.5 m window from several metres away, under water that attenuates hard. Modelling them as real geoms
would mean cutting 32 openings out of four walls -- replacing each textured plane with ~13 box strips
-- and the payoff at 144x192 is parallax through a small bright rectangle. Baking costs one render per
room at build time and nothing at all per frame. The honest cost is that the view through a window does
not shift as the AUV moves past it; at this resolution and that attenuation it is not visible, and it is
written down here so nobody wonders later.

PARAMETERISED, so every window row on every wall opens onto a different room: `room_spec` draws a type
(conference or library) and then its own table wood, chair colour, shelf count, light temperature and
camera yaw. Eight rows (4 walls x 2 rows) get eight distinct interiors.
"""
from __future__ import annotations

import numpy as np

WOODS = {"oak": "0.72 0.52 0.30", "walnut": "0.40 0.26 0.16", "cherry": "0.58 0.30 0.21",
         "ash": "0.80 0.70 0.52"}
CHAIRS = {"red": "0.72 0.10 0.10", "crimson": "0.55 0.06 0.12", "brick": "0.64 0.22 0.16"}
BOOKS = ["0.65 0.18 0.15", "0.20 0.35 0.60", "0.25 0.50 0.28", "0.70 0.60 0.20",
         "0.45 0.25 0.55", "0.60 0.35 0.15"]


def room_spec(rng):
    """One room's parameters. `kind` is conference or library."""
    kind = "conference" if rng.random() < 0.55 else "library"
    return {"kind": kind,
            "wood": str(rng.choice(list(WOODS))),
            "chair": str(rng.choice(list(CHAIRS))),
            "seats": int(rng.integers(4, 9)),
            "shelves": int(rng.integers(3, 6)),
            "bays": int(rng.integers(2, 5)),
            "warm": float(rng.uniform(0.75, 1.0)),
            "yaw": float(rng.uniform(-0.5, 0.5)),
            "wall": float(rng.uniform(0.55, 0.85))}


def _conference(s):
    g, w, c = [], WOODS[s["wood"]], CHAIRS[s["chair"]]
    g.append(f'<geom type="box" size="1.5 0.75 0.06" pos="0 0 0.72" material="m_wood"/>')
    for i in range(2):
        for j in range(2):
            g.append(f'<geom type="box" size="0.06 0.06 0.36" pos="{-1.3+2.6*i} {-0.55+1.1*j} 0.36" '
                     f'material="m_wood"/>')
    n = s["seats"]
    for k in range(n):
        side = -1 if k % 2 else 1
        x = -1.2 + 2.4 * (k // 2) / max(1, (n - 1) // 2)
        y = side * 1.15
        g.append(f'<geom type="box" size="0.22 0.22 0.04" pos="{x:.2f} {y:.2f} 0.45" material="m_chair"/>')
        g.append(f'<geom type="box" size="0.22 0.04 0.28" pos="{x:.2f} {y+side*0.18:.2f} 0.75" '
                 f'material="m_chair"/>')
        g.append(f'<geom type="cylinder" size="0.04 0.22" pos="{x:.2f} {y:.2f} 0.22" material="m_metal"/>')
    g.append('<geom type="box" size="0.9 0.02 0.55" pos="0 2.3 1.6" material="m_screen"/>')
    return "\n".join(g)


def _library(s):
    g = []
    for b in range(s["bays"]):
        x = -2.0 + 4.0 * b / max(1, s["bays"] - 1)
        g.append(f'<geom type="box" size="0.55 0.14 1.05" pos="{x:.2f} 1.6 1.05" material="m_wood"/>')
        for sh in range(s["shelves"]):
            z = 0.28 + sh * (1.9 / s["shelves"])
            for k in range(5):
                col = BOOKS[(b * 7 + sh * 3 + k) % len(BOOKS)]
                g.append(f'<geom type="box" size="0.045 0.11 0.14" pos="{x-0.4+0.2*k:.2f} 1.55 {z:.2f}" '
                         f'rgba="{col} 1"/>')
    g.append(f'<geom type="box" size="0.8 0.45 0.05" pos="0 -0.6 0.7" material="m_wood"/>')
    g.append(f'<geom type="box" size="0.2 0.2 0.04" pos="0 0.1 0.45" material="m_chair"/>')
    return "\n".join(g)


def room_xml(s):
    wood, chair, wall = WOODS[s["wood"]], CHAIRS[s["chair"]], s["wall"]
    warm = s["warm"]
    body = _conference(s) if s["kind"] == "conference" else _library(s)
    return f"""<mujoco>
  <visual><headlight ambient="{0.55*warm:.2f} {0.52*warm:.2f} {0.46*warm:.2f}"
     diffuse="{0.5*warm:.2f} {0.48*warm:.2f} {0.42*warm:.2f}"/><quality shadowsize="0" offsamples="0"/></visual>
  <asset>
    <material name="m_wood" rgba="{wood} 1" specular="0.3" shininess="0.4"/>
    <material name="m_chair" rgba="{chair} 1" specular="0.2"/>
    <material name="m_metal" rgba="0.55 0.56 0.58 1" specular="0.7" shininess="0.7"/>
    <material name="m_screen" rgba="0.08 0.09 0.11 1" specular="0.5"/>
    <material name="m_wall" rgba="{wall:.2f} {wall*0.97:.2f} {wall*0.92:.2f} 1"/>
    <material name="m_floor" rgba="{0.42*warm:.2f} {0.36*warm:.2f} {0.30*warm:.2f} 1"/>
  </asset>
  <worldbody>
    <light pos="0 0 2.6" dir="0 0 -1" diffuse="{0.7*warm:.2f} {0.66*warm:.2f} {0.58*warm:.2f}"/>
    <geom type="plane" material="m_floor" pos="0 0 0" size="5 4 0.1"/>
    <geom type="plane" material="m_wall" pos="0 3.0 1.6" size="5 1.6 0.1" euler="1.5708 0 0"/>
    <geom type="plane" material="m_wall" pos="-3.4 0 1.6" size="1.6 4 0.1" euler="0 1.5708 0"/>
    <geom type="plane" material="m_wall" pos="3.4 0 1.6" size="1.6 4 0.1" euler="0 -1.5708 0"/>
    <geom type="plane" material="m_wall" pos="0 0 3.2" size="5 4 0.1" euler="3.14159 0 0"/>
    {body}
    <camera name="win" pos="0 -2.9 1.45" xyaxes="{np.cos(s['yaw']):.3f} {np.sin(s['yaw']):.3f} 0  0 0 1" fovy="68"/>
  </worldbody>
</mujoco>"""


def render_room(s, h=96, w=128):
    import mujoco

    m = mujoco.MjModel.from_xml_string(room_xml(s))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, h, w)
    r.update_scene(d, camera="win")
    img = r.render()
    r.close()
    return img
