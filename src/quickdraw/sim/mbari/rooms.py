"""Real rooms behind the tank windows — geometry, not a painted image.

A baked picture pasted into the window rectangle does not move when the camera does, so the window
reads as a poster rather than an opening. These are actual MuJoCo bodies sitting behind actual holes
in the wall, so the view through a window parallaxes exactly as it should.

ONE ROOM PER (WALL, RING). All three windows in a row look into the SAME room, from three different
positions along its front face, which is what a row of windows into a conference room looks like.
Eight rooms in all: four walls x two rings.

The AUV cannot reach them: the rooms sit outside the tank shell, and the vehicle has `contype=0`
anyway, so the wall is a visual boundary rather than a physical one. Nothing in a room moves.
"""
from __future__ import annotations

import numpy as np

WOODS = {"oak": (0.72, 0.52, 0.30), "walnut": (0.40, 0.26, 0.16),
         "cherry": (0.58, 0.30, 0.21), "ash": (0.80, 0.70, 0.52)}
CHAIRS = {"red": (0.72, 0.10, 0.10), "crimson": (0.55, 0.06, 0.12), "brick": (0.64, 0.22, 0.16)}
BOOKS = [(0.65, 0.18, 0.15), (0.20, 0.35, 0.60), (0.25, 0.50, 0.28),
         (0.70, 0.60, 0.20), (0.45, 0.25, 0.55), (0.60, 0.35, 0.15)]
DEPTH = 5.0            # how far the room extends behind the wall
HEIGHT = 3.2


def room_spec(rng):
    kind = "conference" if rng.random() < 0.55 else "library"
    return {"kind": kind, "wood": str(rng.choice(list(WOODS))), "chair": str(rng.choice(list(CHAIRS))),
            "seats": int(rng.integers(4, 9)), "shelves": int(rng.integers(3, 5)),
            "bays": int(rng.integers(2, 5)), "warm": float(rng.uniform(0.55, 0.78)),
            "spin": float(rng.uniform(-0.35, 0.35))}


def _rgba(c, a=1.0):
    return f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {a}"


def room_body(name, spec, pos, yaw, span):
    """MJCF for one room, placed behind a wall.

    `pos` is the centre of its front face (the wall), `yaw` rotates it so the front face is parallel
    to that wall, `span` is how wide the room is along the wall.
    """
    w, wood, chair = span / 2, WOODS[spec["wood"]], CHAIRS[spec["chair"]]
    warm = spec["warm"]
    g = [f'<geom type="box" size="{w} 0.05 {HEIGHT/2}" pos="0 {DEPTH} {HEIGHT/2}" '
         f'material="rm_wall_{name}"/>',                                        # back wall
         f'<geom type="box" size="0.05 {DEPTH/2} {HEIGHT/2}" pos="{-w} {DEPTH/2} {HEIGHT/2}" '
         f'material="rm_wall_{name}"/>',
         f'<geom type="box" size="0.05 {DEPTH/2} {HEIGHT/2}" pos="{w} {DEPTH/2} {HEIGHT/2}" '
         f'material="rm_wall_{name}"/>',
         f'<geom type="box" size="{w} {DEPTH/2} 0.05" pos="0 {DEPTH/2} 0" material="rm_floor_{name}"/>',
         f'<geom type="box" size="{w} {DEPTH/2} 0.05" pos="0 {DEPTH/2} {HEIGHT}" '
         f'material="rm_ceil_{name}"/>']
    for lx in (-w * 0.45, w * 0.45):
        g.append(f'<geom type="box" size="0.55 0.18 0.03" pos="{lx} {DEPTH*0.5} {HEIGHT-0.08}" '
                 f'material="rm_lamp_{name}"/>')
    s = spec["spin"]
    if spec["kind"] == "conference":
        g.append(f'<geom type="box" size="{min(2.2, w*0.6):.2f} 0.85 0.06" pos="0 {DEPTH*0.45} 0.74" '
                 f'euler="0 0 {s:.3f}" material="rm_wood_{name}"/>')
        n = spec["seats"]
        for k in range(n):
            side = -1 if k % 2 else 1
            x = -w * 0.45 + (2 * w * 0.45) * (k // 2) / max(1, (n - 1) // 2)
            y = DEPTH * 0.45 + side * 1.25
            g.append(f'<geom type="box" size="0.24 0.24 0.05" pos="{x:.2f} {y:.2f} 0.46" '
                     f'material="rm_chair_{name}"/>')
            g.append(f'<geom type="box" size="0.24 0.05 0.30" pos="{x:.2f} {y+side*0.19:.2f} 0.78" '
                     f'material="rm_chair_{name}"/>')
            g.append(f'<geom type="cylinder" size="0.05 0.23" pos="{x:.2f} {y:.2f} 0.23" '
                     f'material="rm_metal_{name}"/>')
        g.append(f'<geom type="box" size="1.0 0.04 0.58" pos="0 {DEPTH-0.1} 1.75" '
                 f'material="rm_screen_{name}"/>')
    else:
        for b in range(spec["bays"]):
            x = -w * 0.7 + (1.4 * w) * b / max(1, spec["bays"] - 1)
            g.append(f'<geom type="box" size="0.6 0.16 1.1" pos="{x:.2f} {DEPTH-0.9} 1.1" '
                     f'material="rm_wood_{name}"/>')
            for sh in range(spec["shelves"]):
                z = 0.3 + sh * (1.95 / spec["shelves"])
                for k in range(6):
                    col = BOOKS[(b * 5 + sh * 3 + k) % len(BOOKS)]
                    g.append(f'<geom type="box" size="0.045 0.12 0.15" '
                             f'pos="{x-0.44+0.175*k:.2f} {DEPTH-1.03} {z:.2f}" rgba="{_rgba(col)}"/>')
        g.append(f'<geom type="box" size="1.0 0.5 0.05" pos="0 {DEPTH*0.42} 0.72" '
                 f'euler="0 0 {s:.3f}" material="rm_wood_{name}"/>')
        g.append(f'<geom type="box" size="0.24 0.24 0.05" pos="0 {DEPTH*0.42-0.9} 0.46" '
                 f'material="rm_chair_{name}"/>')
    inner = "\n      ".join(g)
    mats = (
        f'<material name="rm_wall_{name}" rgba="{0.62*warm:.2f} {0.60*warm:.2f} {0.56*warm:.2f} 1"/>'
        f'<material name="rm_floor_{name}" rgba="{0.40*warm:.2f} {0.33*warm:.2f} {0.27*warm:.2f} 1"/>'
        f'<material name="rm_ceil_{name}" rgba="{0.70*warm:.2f} {0.69*warm:.2f} {0.66*warm:.2f} 1"/>'
        f'<material name="rm_lamp_{name}" rgba="1 0.95 0.72 1" emission="0.9"/>'
        f'<material name="rm_wood_{name}" rgba="{_rgba(wood)}" specular="0.3" shininess="0.4"/>'
        f'<material name="rm_chair_{name}" rgba="{_rgba(chair)}" specular="0.2"/>'
        f'<material name="rm_metal_{name}" rgba="0.55 0.56 0.58 1" specular="0.7" shininess="0.6"/>'
        f'<material name="rm_screen_{name}" rgba="0.07 0.08 0.10 1" specular="0.5"/>')
    body = f"""
    <body name="room_{name}" pos="{pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}" euler="0 0 {yaw:.4f}">
      {inner}
      <light pos="0 {DEPTH*0.5} {HEIGHT-0.3}" dir="0 0 -1" diffuse="{0.55*warm:.2f} {0.52*warm:.2f} {0.45*warm:.2f}" castshadow="false"/>
    </body>"""
    return mats, body
