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


# the two viewing-port depths, as a fraction of tank depth. Shared by every wall so the ports form
# two RINGS at the same two depths rather than four unrelated rows.
RING_Z = (0.22, 0.62)


def _windows(img, cols, pad=0.12, h_frac=0.22, w_frac=0.16, seed=0):
    """Two rings of viewing ports. Deliberately large: these are a visual check that the camera and
    the wall indexing are working, and a small port at 144x192 is three pixels of nothing."""
    H, W = img.shape[:2]
    for r, zf in enumerate(RING_Z):
        y0 = int(H * zf)
        y1 = y0 + int(H * h_frac)
        for c in range(cols):
            cx = W * (pad + c * (1 - 2 * pad) / max(1, cols - 1))
            x0 = int(cx - W * w_frac / 2)
            x1 = int(cx + W * w_frac / 2)
            x0, x1 = max(0, x0), min(W, x1)
            if x1 <= x0:
                continue
            img[max(0, y0 - 5):y1 + 5, max(0, x0 - 5):x1 + 5] = (210, 210, 205)   # frame
            img[y0:y1, x0:x1] = (14, 30, 40)                                       # glass
            img[y0:y0 + max(2, (y1 - y0) // 5), x0:x1] = (90, 140, 155)            # specular band
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


# EACH SURFACE A DIFFERENT SHADE, on purpose. With five identically-textured grey planes there is no
# way to tell from a frame whether the camera is pointing at +x or -y, which makes a wrong pose or a
# mirrored texture invisible in review. The spread is large enough to read at a glance.
SHADE = {"wall_x+": 1.30, "wall_x-": 0.70, "wall_y+": 1.08, "wall_y-": 0.86, "floor": 0.55}


def build_textures(out=BUILD, res=1024, seed=0):
    """Bake the four walls and the floor. Returns {name: [aruco ids painted on it]}."""
    import imageio.v2 as imageio

    out.mkdir(parents=True, exist_ok=True)
    painted = {}
    # long walls span 15 x 10 m, end walls 10 x 10 m -> keep pixels roughly square
    # id blocks spaced 40 apart: _tags advances its counter past markers it cannot fit, so adjacent
    # blocks have to be wider than the tag count or the walls start sharing ids again.
    specs = {"wall_y+": (int(res * TANK_D / TANK_L), res, 7, 3, 0),
             "wall_y-": (int(res * TANK_D / TANK_L), res, 7, 3, 40),
             "wall_x+": (res, int(res * TANK_W / TANK_D), 5, 3, 80),
             "wall_x-": (res, int(res * TANK_W / TANK_D), 5, 3, 120)}
    for name, (h, w, nx, ny, fid) in specs.items():
        img = concrete(h, w, seed=seed + fid).astype(np.float32) * SHADE[name]
        img = _windows(np.clip(img, 0, 255).astype(np.uint8), cols=nx)
        painted[name] = _tags(img, nx=nx, ny=ny, px=max(48, w // 15), first_id=fid)
        imageio.imwrite(out / f"{name}.png", img)
    floor = np.clip(concrete(res, res, seed=seed + 99).astype(np.float32) * SHADE["floor"],
                    0, 255).astype(np.uint8)
    painted["floor"] = _tags(floor, nx=4, ny=6, px=res // 16, first_id=160)
    imageio.imwrite(out / "floor.png", floor)
    return painted


def mjcf(mola_scale=2.48, dock_corner=(-1, -1), fog=True) -> str:
    """The tank MJCF. `mola_scale` multiplies the raw mesh so the AUV is 2.0 m long (0.806 * 2.48)."""
    hl, hw = TANK_L / 2, TANK_W / 2
    r, g, b = WATER_RGB
    dx, dy = dock_corner
    # the dock sits AT THE SURFACE in a corner, as the real one does -- the AUV docks near the top,
    # not on the tank floor.
    dock_pos = (dx * (hl - 2.0), dy * (hw - 2.0), -1.4)
    return f"""<mujoco model="mbari_tank">
  <compiler angle="radian" meshdir="." texturedir="build"/>
  <!-- gravity 0: a trimmed AUV is neutrally buoyant; weight/buoyancy as opposing forces adds
       stiffness and no behaviour. Damping lives in mola.py, applied with the thrust. -->
  <option gravity="0 0 0" timestep="0.002" integrator="RK4" density="0" viscosity="0"/>
  <visual>
    <headlight ambient="0.62 0.66 0.68" diffuse="0.6 0.63 0.64" specular="0.1 0.1 0.1"/>
    <quality shadowsize="2048" offsamples="4"/>
  </visual>
  <asset>
    <texture name="t_yp" type="2d" file="wall_y+.png"/>
    <texture name="t_ym" type="2d" file="wall_y-.png"/>
    <texture name="t_xp" type="2d" file="wall_x+.png"/>
    <texture name="t_xm" type="2d" file="wall_x-.png"/>
    <texture name="t_fl" type="2d" file="floor.png"/>
    <material name="m_yp" texture="t_yp" texuniform="false" texrepeat="1 1" specular="0.1" shininess="0.1"/>
    <material name="m_ym" texture="t_ym" texuniform="false" texrepeat="1 1" specular="0.1" shininess="0.1"/>
    <material name="m_xp" texture="t_xp" texuniform="false" texrepeat="1 1" specular="0.1" shininess="0.1"/>
    <material name="m_xm" texture="t_xm" texuniform="false" texrepeat="1 1" specular="0.1" shininess="0.1"/>
    <material name="m_fl" texture="t_fl" texuniform="false" texrepeat="1 1" specular="0.05"/>
    <!-- open water overhead: bright blue, seen from below -->
    <material name="m_surf" rgba="0.24 0.60 0.82 1" specular="0.9" shininess="0.9" reflectance="0.35" emission="0.25"/>
    <material name="m_crane" rgba="0.95 0.76 0.03 1" specular="0.5" shininess="0.5"/>
    <material name="m_mola" rgba="0.95 0.78 0.08 1" specular="0.4" shininess="0.4"/>
    <material name="m_dock" rgba="0.72 0.74 0.78 1" specular="0.3"/>
    <material name="m_buoy_o" rgba="0.98 0.42 0.03 1" specular="0.6" shininess="0.6" emission="0.12"/>
    <material name="m_buoy_g" rgba="0.08 0.80 0.25 1" specular="0.6" shininess="0.6" emission="0.12"/>
    <material name="m_rope" rgba="0.85 0.85 0.80 1"/>
    <mesh name="mola" file="mola/mola.obj" scale="{mola_scale} {mola_scale} {mola_scale}"/>
    <mesh name="dock" file="dock/dock.obj" scale="1 1 1"/>
    <mesh name="diver" file="diver/diver.obj" scale="1 1 1"/>
  </asset>

  <worldbody>
    <light pos="0 0 2" dir="0 0 -1" diffuse="0.5 0.55 0.55" specular="0.05 0.05 0.05" castshadow="true"/>
    <light pos="{-hl*0.6} 0 -1.5" dir="0.3 0 -1" diffuse="0.25 0.3 0.3" castshadow="false"/>

    <!-- floor at -D, walls facing inward. Planes are one-sided; euler turns each normal into the tank.
         Each surface has a different concrete shade (see SHADE) so a frame identifies which wall it is. -->
    <geom name="floor"  type="plane" material="m_fl" pos="0 0 {-TANK_D}" size="{hl} {hw} 0.1"/>
    <geom name="wall_x+" type="plane" material="m_xp" pos="{hl} 0 {-TANK_D/2}" size="{TANK_D/2} {hw} 0.1" euler="0 -1.5708 0"/>
    <geom name="wall_x-" type="plane" material="m_xm" pos="{-hl} 0 {-TANK_D/2}" size="{TANK_D/2} {hw} 0.1" euler="0 1.5708 0"/>
    <geom name="wall_y+" type="plane" material="m_yp" pos="0 {hw} {-TANK_D/2}" size="{hl} {TANK_D/2} 0.1" euler="1.5708 0 0"/>
    <geom name="wall_y-" type="plane" material="m_ym" pos="0 {-hw} {-TANK_D/2}" size="{hl} {TANK_D/2} 0.1" euler="-1.5708 0 0"/>
    <!-- the surface, seen from below: open water, bright blue. A plane's normal is +z in its own
         frame, so euler x=pi turns it down into the tank. -->
    <geom name="surface" type="plane" material="m_surf" pos="0 0 0.02" size="{hl} {hw} 0.1" euler="3.14159 0 0"/>

    <!-- yellow crane hook, hanging to 1/4 depth in the middle. NEVER a target: a fixed obstacle that
         must appear in training footage so the model learns it is static. -->
    <body name="crane" pos="0 0 0">
      <geom name="crane_arm"  type="capsule" material="m_crane" fromto="0 0 0.4 0 0 {-TANK_D/4}" size="0.11"/>
      <geom name="crane_hook" type="capsule" material="m_crane"
            fromto="0 0 {-TANK_D/4} 0.4 0 {-TANK_D/4 - 0.4}" size="0.08"/>
      <geom name="crane_tip"  type="sphere"  material="m_crane" pos="0.4 0 {-TANK_D/4 - 0.52}" size="0.12"/>
    </body>

    <!-- two moored buoys at mid depth, orange and green. Static landmarks: distinct colours that give
         the world model something unambiguous to localise against away from the walls. -->
    <body name="buoy_orange" pos="{hl*0.45} {hw*0.5} -5.0">
      <geom name="rope_o" type="capsule" material="m_rope" fromto="0 0 0 0 0 5.0" size="0.025"/>
      <geom name="buoy_o" type="sphere" material="m_buoy_o" size="0.38"/>
    </body>
    <body name="buoy_green" pos="{-hl*0.40} {-hw*0.45} -5.6">
      <geom name="rope_g" type="capsule" material="m_rope" fromto="0 0 0 0 0 5.6" size="0.025"/>
      <geom name="buoy_g" type="sphere" material="m_buoy_g" size="0.34"/>
    </body>

    <!-- docking station in a corner AT THE SURFACE. AprilTag faces come from the mesh texture. -->
    <body name="dock" pos="{dock_pos[0]} {dock_pos[1]} {dock_pos[2]}" euler="0 0 {np.arctan2(-dy, -dx):.4f}">
      <geom name="dock_g" type="mesh" mesh="dock" material="m_dock" contype="0" conaffinity="0"/>
      <site name="dock_entry" pos="1.6 0 0.2" size="0.08" rgba="0 1 0 0"/>
    </body>

    <body name="mola" pos="0 0 -5">
      <freejoint name="mola_free"/>
      <geom name="mola_g" type="mesh" mesh="mola" material="m_mola" mass="60" contype="0" conaffinity="0"/>
      <camera name="fpv" pos="0.95 0 0.08" xyaxes="0 -1 0 0 0 1" fovy="70"/>
      <!-- over-the-shoulder chase view: 5.4 m behind, 1.9 m up, pitched ~19 deg down. Pulled back
           from a first attempt at 2.6 m where the hull filled the frame and the tank was invisible.
           A SECOND view of the same episode, NOT a training input -- the world model trains on fpv. -->
      <camera name="chase" pos="-5.4 0 1.9" xyaxes="0 -1 0 0.33 0 0.944" fovy="58"/>
      <site name="mola_c" pos="0 0 0" size="0.05" rgba="1 0 0 0"/>
    </body>

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
