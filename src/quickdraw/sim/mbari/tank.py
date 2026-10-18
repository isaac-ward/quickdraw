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

from .rooms import render_room, room_spec
from .textures import aruco_tile, concrete, window_rows

TANK_L, TANK_W, TANK_D = 15.0, 10.0, 10.0          # metres: length (x), width (y), depth (z)
RING_Z = (0.24, 0.62)                              # window-ring depths, as a fraction of TANK_D
GANTRY_H, GANTRY_W = 1.0, 2.0                      # gantry height above water, and its width
ROOM_H = 6.0                                       # warehouse headroom above the water line
WATER_RGB = (0.08, 0.26, 0.30)
ASSETS = pathlib.Path(__file__).parent / "assets"
BUILD = ASSETS / "build"


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
    # WINDOW AND TAG DENSITY BOTH HALVED. The first build put 7 ports and 21 markers on a long wall
    # and 24 markers on the floor, which read as clutter at 144x192 rather than as landmarks.
    # (h, w, window_cols, tag_nx, tag_ny, first_tag_id)
    specs = {"wall_y+": (int(res * TANK_D / TANK_L), res, 4, 4, 2, 0),
             "wall_y-": (int(res * TANK_D / TANK_L), res, 4, 4, 2, 40),
             "wall_x+": (res, int(res * TANK_W / TANK_D), 3, 3, 2, 80),
             "wall_x-": (res, int(res * TANK_W / TANK_D), 3, 3, 2, 120)}
    rng = np.random.default_rng(seed + 4242)
    rooms_used = {}
    for name, (h, w, ncols, nx, ny, fid) in specs.items():
        img = concrete(h, w, seed=seed + fid).astype(np.float32) * SHADE[name]
        img = np.clip(img, 0, 255).astype(np.uint8)
        # one DISTINCT room per ring per wall -> eight interiors around the tank
        specs_r = [room_spec(rng) for _ in RING_Z]
        rooms_used[name] = [r["kind"] for r in specs_r]
        img = window_rows(img, cols=ncols, rooms=[render_room(r, 140, 190) for r in specs_r],
                          rings=RING_Z)
        painted[name] = _tags(img, nx=nx, ny=ny, px=max(48, w // 15), first_id=fid)
        imageio.imwrite(out / f"{name}.png", img)
    floor = np.clip(concrete(res, res, seed=seed + 99).astype(np.float32) * SHADE["floor"],
                    0, 255).astype(np.uint8)
    painted["floor"] = _tags(floor, nx=3, ny=4, px=res // 16, first_id=160)
    imageio.imwrite(out / "floor.png", floor)
    painted["_rooms"] = rooms_used
    return painted


def _mola_geoms(scale):
    """One mesh + material + geom per colour group, so the AUV renders in its real colours.

    The model is NOT textured: it is 14 parts carrying six baseColorFactor materials -- a black
    frame, a light-grey hull, a cyan nose dome and yellow thruster pods. Painting the whole thing one
    yellow (the first version) threw that away, and decimating it to 5,586 faces destroyed the hull
    outright."""
    import json

    meta = json.loads((ASSETS / "mola_groups.json").read_text())
    a = "".join(
        f'<mesh name="mola{k}" file="mola/{row[0]}" scale="{scale} {scale} {scale}"/>'
        f'<material name="mola_c{k}" rgba="{row[1][0]} {row[1][1]} {row[1][2]} 1" '
        f'specular="0.25" shininess="0.3"/>' for k, row in enumerate(meta))
    # The whole 60 kg sits on the first group and the rest are near-massless: MuJoCo sums geom
    # masses, and six geoms at 60 kg each would give a 360 kg vehicle whose inertia no longer matches
    # the constants in mola.py.
    parts = []
    for k in range(len(meta)):
        mass = "60" if k == 0 else "0.0001"
        parts.append(f'<geom name="mola_g{k}" type="mesh" mesh="mola{k}" material="mola_c{k}" '
                     f'mass="{mass}" contype="0" conaffinity="0"/>')
    return a, "".join(parts)


def mjcf(mola_scale=2.48, dock_corner=(-1, -1), n_fish=0, diver=False) -> str:
    """The tank MJCF. `mola_scale` makes the AUV 2.0 m long (0.806 x 2.48)."""
    hl, hw = TANK_L / 2, TANK_W / 2
    dx, dy = dock_corner
    dock_pos = (dx * (hl - 2.0), dy * (hw - 2.0), -1.4)
    mola_assets, mola_geoms = _mola_geoms(mola_scale)

    lights = "".join(
        f'<geom name="ceil_{i}" type="cylinder" material="m_lamp" size="0.45 0.04" '
        f'pos="{x} {y} {ROOM_H - 0.06}" euler="0 0 0"/>'
        f'<light pos="{x} {y} {ROOM_H - 0.2}" dir="0 0 -1" diffuse="0.34 0.32 0.24" '
        f'castshadow="false"/>'
        for i, (x, y) in enumerate([(ax, ay) for ax in (-5.2, -1.7, 1.7, 5.2) for ay in (-2.6, 2.6)]))

    fish = "".join(f"""
    <body name="fish{i}" mocap="true" pos="0 0 -5">
      <geom type="ellipsoid" size="0.40 0.14 0.21" material="m_fish" contype="0" conaffinity="0"/>
      <geom type="box" size="0.16 0.02 0.18" pos="-0.50 0 0.04" material="m_fish" contype="0" conaffinity="0"/>
      <geom type="box" size="0.11 0.02 0.13" pos="0.02 0 0.28" material="m_fish" contype="0" conaffinity="0"/>
      <geom type="sphere" size="0.042" pos="0.29 0.095 0.075" material="m_eye" contype="0" conaffinity="0"/>
    </body>""" for i in range(n_fish))

    # DIVER NOSE-DOWN. euler pitches the model so it is head-first into the tank rather than standing
    # upright; the source mesh is a T-pose and `pose_diver` in convert_assets sweeps the arms back.
    diver_xml = """
    <body name="diver" mocap="true" pos="0 0 -5" euler="0 1.3963 0">
      <geom name="diver_g" type="mesh" mesh="diver" material="m_diver" contype="0" conaffinity="0"/>
    </body>""" if diver else ""

    extra_assets = ""
    if n_fish:
        extra_assets += ('<material name="m_fish" rgba="0.96 0.42 0.07 1" specular="0.5" shininess="0.5"/>'
                         '<material name="m_eye" rgba="0.05 0.05 0.05 1"/>')
    if diver:
        extra_assets += '<material name="m_diver" rgba="0.14 0.17 0.22 1" specular="0.4"/>'

    return f"""<mujoco model="mbari_tank">
  <compiler angle="radian" meshdir="." texturedir="build"/>
  <!-- gravity 0: a trimmed AUV is neutrally buoyant. Damping lives in mola.py with the thrust. -->
  <option gravity="0 0 0" timestep="0.002" integrator="RK4" density="0" viscosity="0"/>
  <visual>
    <headlight ambient="0.58 0.60 0.62" diffuse="0.46 0.47 0.48" specular="0.06 0.06 0.06"/>
    <!-- shadows off, no multisampling: 141 -> 65 ms per 4-camera control step, and at 144x192
         under attenuating water a shadow is a handful of dim pixels. -->
    <quality shadowsize="512" offsamples="0"/>
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
    <material name="m_surf" rgba="0.26 0.52 0.62 0.72" specular="0.9" shininess="0.9" reflectance="0.3"/>
    <material name="m_room" rgba="0.085 0.085 0.095 1" specular="0.02"/>
    <material name="m_lamp" rgba="1.0 0.92 0.45 1" emission="0.95" specular="0.1"/>
    <material name="m_gantry" rgba="0.93 0.74 0.05 1" specular="0.5" shininess="0.5"/>
    <material name="m_dock" rgba="0.72 0.74 0.78 1" specular="0.3"/>
    <material name="m_buoy_o" rgba="0.98 0.42 0.03 1" specular="0.6" shininess="0.6" emission="0.1"/>
    <material name="m_buoy_g" rgba="0.08 0.80 0.25 1" specular="0.6" shininess="0.6" emission="0.1"/>
    <material name="m_rope" rgba="0.85 0.85 0.80 1"/>
    {mola_assets}{extra_assets}
    <mesh name="dock" file="dock/dock_g0.obj" scale="1 1 1"/>
    <mesh name="diver" file="diver/diver_g0.obj" scale="1 1 1"/>
  </asset>

  <worldbody>
    <light pos="0 0 {ROOM_H - 0.5}" dir="0 0 -1" diffuse="0.42 0.43 0.42" castshadow="false"/>
    <light pos="{-hl*0.6} 0 -1.5" dir="0.3 0 -1" diffuse="0.26 0.28 0.28" castshadow="false"/>
    <light pos="{hl*0.6} 0 -4.0" dir="-0.3 0 -0.6" diffuse="0.22 0.24 0.24" castshadow="false"/>

    <!-- THE TANK. Walls span the water column only: floor at -D up to z = 0, nothing above. -->
    <geom name="floor"  type="plane" material="m_fl" pos="0 0 {-TANK_D}" size="{hl} {hw} 0.1"/>
    <geom name="wall_x+" type="plane" material="m_xp" pos="{hl} 0 {-TANK_D/2}" size="{TANK_D/2} {hw} 0.1" euler="0 -1.5708 0"/>
    <geom name="wall_x-" type="plane" material="m_xm" pos="{-hl} 0 {-TANK_D/2}" size="{TANK_D/2} {hw} 0.1" euler="0 1.5708 0"/>
    <geom name="wall_y+" type="plane" material="m_yp" pos="0 {hw} {-TANK_D/2}" size="{hl} {TANK_D/2} 0.1" euler="1.5708 0 0"/>
    <geom name="wall_y-" type="plane" material="m_ym" pos="0 {-hw} {-TANK_D/2}" size="{hl} {TANK_D/2} 0.1" euler="-1.5708 0 0"/>
    <geom name="surface" type="plane" material="m_surf" pos="0 0 -0.02" size="{hl} {hw} 0.1" euler="3.14159 0 0"/>

    <!-- THE WAREHOUSE. A dark enclosed room around the tank, lit by ceiling lamps. The tank is a
         hole in its floor, so from underwater the surface reads as a lid with a lit room above. -->
    <geom name="room_ceil" type="plane" material="m_room" pos="0 0 {ROOM_H}" size="{hl+6} {hw+6} 0.1" euler="3.14159 0 0"/>
    <geom name="room_x+" type="plane" material="m_room" pos="{hl+6} 0 {ROOM_H/2}" size="{ROOM_H/2+1} {hw+6} 0.1" euler="0 -1.5708 0"/>
    <geom name="room_x-" type="plane" material="m_room" pos="{-hl-6} 0 {ROOM_H/2}" size="{ROOM_H/2+1} {hw+6} 0.1" euler="0 1.5708 0"/>
    <geom name="room_y+" type="plane" material="m_room" pos="0 {hw+6} {ROOM_H/2}" size="{hl+6} {ROOM_H/2+1} 0.1" euler="1.5708 0 0"/>
    <geom name="room_y-" type="plane" material="m_room" pos="0 {-hw-6} {ROOM_H/2}" size="{hl+6} {ROOM_H/2+1} 0.1" euler="-1.5708 0 0"/>
    <geom name="deck_x+" type="box" material="m_room" pos="{hl+3} 0 -0.05" size="3 {hw+6} 0.05"/>
    <geom name="deck_x-" type="box" material="m_room" pos="{-hl-3} 0 -0.05" size="3 {hw+6} 0.05"/>
    <geom name="deck_y+" type="box" material="m_room" pos="0 {hw+3} -0.05" size="{hl} 3 0.05"/>
    <geom name="deck_y-" type="box" material="m_room" pos="0 {-hw-3} -0.05" size="{hl} 3 0.05"/>
    {lights}

    <!-- GANTRY: spans -X to +X at 1 m above the water, 2 m wide. The hook hangs from it to 1/4
         depth and is NEVER a target -- a fixed obstacle the model must learn is static. -->
    <body name="gantry" pos="0 0 {GANTRY_H}">
      <geom name="g_beam_a" type="box" material="m_gantry" pos="0 {GANTRY_W/2} 0" size="{hl} 0.10 0.12"/>
      <geom name="g_beam_b" type="box" material="m_gantry" pos="0 {-GANTRY_W/2} 0" size="{hl} 0.10 0.12"/>
      <geom name="g_cross_a" type="box" material="m_gantry" pos="{-hl+0.3} 0 0" size="0.10 {GANTRY_W/2} 0.10"/>
      <geom name="g_cross_b" type="box" material="m_gantry" pos="0 0 0" size="0.10 {GANTRY_W/2} 0.10"/>
      <geom name="g_cross_c" type="box" material="m_gantry" pos="{hl-0.3} 0 0" size="0.10 {GANTRY_W/2} 0.10"/>
      <geom name="g_leg_a" type="box" material="m_gantry" pos="{-hl} {GANTRY_W/2} -0.55" size="0.12 0.12 0.55"/>
      <geom name="g_leg_b" type="box" material="m_gantry" pos="{hl} {GANTRY_W/2} -0.55" size="0.12 0.12 0.55"/>
      <geom name="g_leg_c" type="box" material="m_gantry" pos="{-hl} {-GANTRY_W/2} -0.55" size="0.12 0.12 0.55"/>
      <geom name="g_leg_d" type="box" material="m_gantry" pos="{hl} {-GANTRY_W/2} -0.55" size="0.12 0.12 0.55"/>
    </body>
    <body name="hook" pos="1.2 0 0">
      <geom name="hook_line" type="capsule" material="m_gantry" fromto="0 0 {GANTRY_H} 0 0 {-TANK_D/4}" size="0.055"/>
      <geom name="hook_j" type="capsule" material="m_gantry" fromto="0 0 {-TANK_D/4} 0.42 0 {-TANK_D/4 - 0.42}" size="0.085"/>
      <geom name="hook_tip" type="sphere" material="m_gantry" pos="0.42 0 {-TANK_D/4 - 0.55}" size="0.12"/>
    </body>

    <body name="buoy_orange" pos="{hl*0.45} {hw*0.5} -5.0">
      <geom name="rope_o" type="capsule" material="m_rope" fromto="0 0 0 0 0 5.0" size="0.025"/>
      <geom name="buoy_o" type="sphere" material="m_buoy_o" size="0.38"/>
    </body>
    <body name="buoy_green" pos="{-hl*0.40} {-hw*0.45} -5.6">
      <geom name="rope_g" type="capsule" material="m_rope" fromto="0 0 0 0 0 5.6" size="0.025"/>
      <geom name="buoy_g" type="sphere" material="m_buoy_g" size="0.34"/>
    </body>

    <body name="dock" pos="{dock_pos[0]} {dock_pos[1]} {dock_pos[2]}" euler="0 0 {np.arctan2(-dy, -dx):.4f}">
      <geom name="dock_g" type="mesh" mesh="dock" material="m_dock" contype="0" conaffinity="0"/>
      <site name="dock_entry" pos="1.6 0 0.2" size="0.08" rgba="0 1 0 0"/>
    </body>

    <body name="mola" pos="0 0 -5">
      <freejoint name="mola_free"/>
      {mola_geoms}
      <camera name="fpv" pos="0.95 0 0.08" xyaxes="0 -1 0 0 0 1" fovy="70"/>
      <camera name="chase" pos="-5.4 0 1.9" xyaxes="0 -1 0 0.33 0 0.944" fovy="58"/>
      <site name="mola_c" pos="0 0 0" size="0.05" rgba="1 0 0 0"/>
    </body>
    {fish}{diver_xml}

    <camera name="topdown" pos="0 0 3.2" xyaxes="1 0 0 0 1 0" fovy="84" mode="fixed"/>
    <!-- SIDE-ON IS INSIDE THE TANK, against the -Y wall looking across. Outside the glass it was
         shooting through a textured wall plane and seeing mostly concrete. -->
    <camera name="sideon" pos="0 {-hw + 0.6} {-TANK_D/2 + 1.0}" xyaxes="1 0 0 0 0.35 0.937" fovy="75" mode="fixed"/>
  </worldbody>
</mujoco>
"""


def build(out=BUILD, textures=True, name="tank.xml", **kw):
    """Write textures + <name> under assets/ and return (xml_path, painted_ids)."""
    import json

    from .convert_assets import convert

    ASSETS.mkdir(parents=True, exist_ok=True)
    if not (ASSETS / "mola_groups.json").exists():
        rows = convert(log=lambda *_: None)
        (ASSETS / "mola_groups.json").write_text(json.dumps([[r[0], list(r[1])] for r in rows["mola"]]))
    painted = build_textures(out) if textures else {}
    p = ASSETS / name
    p.write_text(mjcf(**kw))
    return p, painted


if __name__ == "__main__":
    p, painted = build()
    print(f"  wrote {p}")
    for k, v in painted.items():
        print(f"    {k:12s} {len(v):3d} aruco ids {v[:6]}{'...' if len(v) > 6 else ''}")
