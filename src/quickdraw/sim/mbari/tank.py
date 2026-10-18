"""The MBARI test tank as MuJoCo MJCF: 15 x 10 x 10 m, concrete, windows, ArUco, crane hook, dock.

WALLS ARE BAKED TEXTURES, NOT GEOMETRY. Concrete, the two window rows and the ArUco grid are drawn into
one RGB image per surface and applied to a single plane geom. A grid of tag geoms would be ~100 extra
bodies to no benefit: nothing interacts with a tag, the AUV cannot collide with a window, and at
144x192 the camera resolves a painted tag and a modelled tag identically. It also keeps the model small
enough that 240 episodes render in minutes.

WATER IS FOG, NOT FLUID. MuJoCo has no free surface, and we do not need one: the AUV is submerged for
every frame of every episode. What the camera must show is depth-dependent attenuation, which is
exactly what fog gives, and what the dynamics must show is damping, which `mola.py` applies directly.
Gravity is zero because a trimmed AUV is neutrally buoyant -- modelling weight and buoyancy as two
large opposing forces would add stiffness for no behaviour.
"""
from __future__ import annotations

import pathlib

import numpy as np

from .textures import aruco_tile, concrete

TANK_L, TANK_W, TANK_D = 15.0, 10.0, 10.0          # metres: length (x), width (y), depth (z)
WATER_RGB = (0.08, 0.26, 0.30)
ASSETS = pathlib.Path(__file__).parent / "assets"
BUILD = ASSETS / "build"


def _windows(img, rows, cols, pad=0.10, h_frac=0.13, seed=0):
    """Two rows of viewing ports, as the real tank has. Dark glass with a pale frame."""
    H, W = img.shape[:2]
    for r in range(rows):
        y0 = int(H * (0.18 + r * 0.42))
        y1 = y0 + int(H * h_frac)
        for c in range(cols):
            x0 = int(W * (pad + c * (1 - 2 * pad) / max(1, cols - 1))) - int(W * 0.035)
            x1 = x0 + int(W * 0.07)
            x0, x1 = max(0, x0), min(W, x1)
            if x1 <= x0:
                continue
            img[y0 - 3:y1 + 3, x0 - 3:x1 + 3] = (205, 205, 200)        # frame
            img[y0:y1, x0:x1] = (18, 34, 44)                            # glass
            img[y0:y0 + max(1, (y1 - y0) // 6), x0:x1] = (70, 110, 125)  # specular band
    return img


def _tags(img, nx, ny, px=72, margin=0.06, first_id=0):
    """Evenly spaced ArUco markers. Returns the ids used, in row-major order, so a detector result can
    be checked against what was actually painted."""
    H, W = img.shape[:2]
    ids = []
    k = first_id
    for j in range(ny):
        for i in range(nx):
            cx = int(W * (margin + (1 - 2 * margin) * (i + 0.5) / nx))
            cy = int(H * (margin + (1 - 2 * margin) * (j + 0.5) / ny))
            t = aruco_tile(k, px)
            y0, x0 = cy - px // 2, cx - px // 2
            if y0 < 0 or x0 < 0 or y0 + px > H or x0 + px > W:
                k += 1
                continue
            img[y0:y0 + px, x0:x0 + px] = t
            ids.append(k)
            k += 1
    return ids


def build_textures(out=BUILD, res=1024, seed=0):
    """Bake the four walls and the floor. Returns {name: [aruco ids painted on it]}."""
    import imageio.v2 as imageio

    out.mkdir(parents=True, exist_ok=True)
    painted = {}
    # long walls are 15 x 10 m, end walls 10 x 10 m -> keep ~square pixels
    specs = {"wall_long_a": (int(res * TANK_D / TANK_L), res, 7, 2, 0),
             "wall_long_b": (int(res * TANK_D / TANK_L), res, 7, 2, 14),
             "wall_end_a": (res, int(res * TANK_W / TANK_D), 4, 2, 28),
             "wall_end_b": (res, int(res * TANK_W / TANK_D), 4, 2, 36)}
    for name, (h, w, nx, ny, fid) in specs.items():
        img = concrete(h, w, seed=seed + fid)
        img = _windows(img, rows=2, cols=nx)
        painted[name] = _tags(img, nx=nx, ny=ny, px=max(40, w // 18), first_id=fid)
        imageio.imwrite(out / f"{name}.png", img)
    floor = concrete(res, res, seed=seed + 99)
    painted["floor"] = _tags(floor, nx=4, ny=6, px=res // 16, first_id=44)
    imageio.imwrite(out / "floor.png", floor)
    return painted


def mjcf(mola_scale=2.48, dock_corner=(-1, -1), fog=True) -> str:
    """The tank MJCF. `mola_scale` multiplies the raw mesh so the AUV is 2.0 m long (0.806 * 2.48)."""
    hl, hw = TANK_L / 2, TANK_W / 2
    r, g, b = WATER_RGB
    dx, dy = dock_corner
    dock_pos = (dx * (hl - 1.6), dy * (hw - 1.6), -TANK_D + 1.0)
    fogtag = (f'<map fogstart="2.5" fogend="16"/><rgba fog="{r} {g} {b} 1" haze="{r} {g} {b} 1"/>'
              if fog else "")
    return f"""<mujoco model="mbari_tank">
  <compiler angle="radian" meshdir="." texturedir="build"/>
  <!-- gravity 0: a trimmed AUV is neutrally buoyant; weight/buoyancy as opposing forces adds
       stiffness and no behaviour. Damping lives in mola.py, applied with the thrust. -->
  <option gravity="0 0 0" timestep="0.002" integrator="RK4" density="0" viscosity="0"/>
  <visual>
    <headlight ambient="0.62 0.66 0.68" diffuse="0.6 0.63 0.64" specular="0.1 0.1 0.1"/>
    <quality shadowsize="2048" offsamples="4"/>
    {fogtag}
  </visual>
  <asset>
    <texture name="t_la" type="2d" file="wall_long_a.png"/>
    <texture name="t_lb" type="2d" file="wall_long_b.png"/>
    <texture name="t_ea" type="2d" file="wall_end_a.png"/>
    <texture name="t_eb" type="2d" file="wall_end_b.png"/>
    <texture name="t_fl" type="2d" file="floor.png"/>
    <material name="m_la" texture="t_la" texuniform="false" texrepeat="1 1" specular="0.1" shininess="0.1"/>
    <material name="m_lb" texture="t_lb" texuniform="false" texrepeat="1 1" specular="0.1" shininess="0.1"/>
    <material name="m_ea" texture="t_ea" texuniform="false" texrepeat="1 1" specular="0.1" shininess="0.1"/>
    <material name="m_eb" texture="t_eb" texuniform="false" texrepeat="1 1" specular="0.1" shininess="0.1"/>
    <material name="m_fl" texture="t_fl" texuniform="false" texrepeat="1 1" specular="0.05"/>
    <texture name="t_surf" type="2d" builtin="flat" width="8" height="8" rgb1="0.55 0.78 0.82"/>
    <material name="m_surf" texture="t_surf" rgba="0.55 0.78 0.82 1" specular="0.9" shininess="0.9" reflectance="0.25"/>
    <material name="m_crane" rgba="0.93 0.74 0.05 1" specular="0.5" shininess="0.5"/>
    <material name="m_mola" rgba="0.95 0.78 0.08 1" specular="0.4" shininess="0.4"/>
    <material name="m_dock" rgba="0.72 0.74 0.78 1" specular="0.3"/>
    <mesh name="mola" file="mola/mola.obj" scale="{mola_scale} {mola_scale} {mola_scale}"/>
    <mesh name="dock" file="dock/dock.obj" scale="1 1 1"/>
    <mesh name="diver" file="diver/diver.obj" scale="1 1 1"/>
  </asset>

  <worldbody>
    <light pos="0 0 2" dir="0 0 -1" diffuse="0.5 0.55 0.55" specular="0.05 0.05 0.05" castshadow="true"/>
    <light pos="{-hl*0.6} 0 -1.5" dir="0.3 0 -1" diffuse="0.25 0.3 0.3" castshadow="false"/>

    <!-- floor at -D, walls facing inward. Planes are one-sided; euler turns each normal into the tank. -->
    <geom name="floor"  type="plane" material="m_fl" pos="0 0 {-TANK_D}" size="{hl} {hw} 0.1"/>
    <geom name="wall_x+" type="plane" material="m_ea" pos="{hl} 0 {-TANK_D/2}" size="{TANK_D/2} {hw} 0.1" euler="0 -1.5708 0"/>
    <geom name="wall_x-" type="plane" material="m_eb" pos="{-hl} 0 {-TANK_D/2}" size="{TANK_D/2} {hw} 0.1" euler="0 1.5708 0"/>
    <geom name="wall_y+" type="plane" material="m_la" pos="0 {hw} {-TANK_D/2}" size="{hl} {TANK_D/2} 0.1" euler="1.5708 0 0"/>
    <geom name="wall_y-" type="plane" material="m_lb" pos="0 {-hw} {-TANK_D/2}" size="{hl} {TANK_D/2} 0.1" euler="-1.5708 0 0"/>

    <!-- yellow crane hook, hanging to 1/4 depth in the middle. NEVER a target: it is a fixed obstacle
         that must appear in training footage so the model learns it is static. -->
    <body name="crane" pos="0 0 0">
      <geom name="crane_arm"  type="capsule" material="m_crane" fromto="0 0 0.4 0 0 {-TANK_D/4}" size="0.09"/>
      <geom name="crane_hook" type="capsule" material="m_crane"
            fromto="0 0 {-TANK_D/4} 0.35 0 {-TANK_D/4 - 0.35}" size="0.07"/>
      <geom name="crane_tip"  type="sphere"  material="m_crane" pos="0.35 0 {-TANK_D/4 - 0.45}" size="0.1"/>
    </body>

    <!-- docking station in a corner, sitting on the floor. Its AprilTag faces are in the mesh texture. -->
    <body name="dock" pos="{dock_pos[0]} {dock_pos[1]} {dock_pos[2]}" euler="0 0 {np.arctan2(-dy, -dx):.4f}">
      <geom name="dock_g" type="mesh" mesh="dock" material="m_dock" contype="0" conaffinity="0"/>
      <site name="dock_entry" pos="1.6 0 0.2" size="0.08" rgba="0 1 0 0.0"/>
    </body>

    <body name="mola" pos="0 0 -5">
      <freejoint name="mola_free"/>
      <geom name="mola_g" type="mesh" mesh="mola" material="m_mola" mass="60" contype="0" conaffinity="0"/>
      <camera name="fpv" pos="0.95 0 0.08" xyaxes="0 -1 0 0 0 1" fovy="70"/>
      <site name="mola_c" pos="0 0 0" size="0.05" rgba="1 0 0 0.0"/>
    </body>

    <!-- the surface, seen from below. A plane's normal is +z in its own frame, so euler x=pi turns it
         down into the tank; without it the camera looking up sees the void and the scene reads as dry. -->
    <geom name="surface" type="plane" material="m_surf" pos="0 0 0.02" size="{hl} {hw} 0.1" euler="3.14159 0 0"/>

    <camera name="topdown" pos="0 0 3.5" xyaxes="1 0 0 0 1 0" fovy="78" mode="fixed"/>
    <camera name="sideon" pos="0 {-hw - 7.5} {-TANK_D/2}" xyaxes="1 0 0 0 0 1" fovy="62" mode="fixed"/>
  </worldbody>
</mujoco>
"""


def build(out=BUILD, **kw):
    """Write textures + tank.xml under assets/build/ and return (xml_path, painted_ids)."""
    painted = build_textures(out)
    ASSETS.mkdir(parents=True, exist_ok=True)
    p = ASSETS / "tank.xml"
    p.write_text(mjcf(**kw))
    return p, painted


if __name__ == "__main__":
    p, painted = build()
    print(f"  wrote {p}")
    for k, v in painted.items():
        print(f"    {k:12s} {len(v):3d} aruco ids {v[:6]}{'...' if len(v) > 6 else ''}")
