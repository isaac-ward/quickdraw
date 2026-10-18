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
    """Bake the four wall surfaces and the floor.

    WALL TEXTURES ARE SQUARE AND CUBE-MAPPED. The walls are box geoms, and a `type="2d"` texture on
    a box is stretched to each face independently -- on a 7.5 x 0.18 x 1.5 m strip that turned the
    concrete into vertical planks. A `type="cube"` texture with `texuniform="true"` maps in world
    units instead, so the grain is the same size on a long strip and a narrow pillar. Square because
    cube faces are square. No ArUco on the walls (removed at the user's request); the floor keeps
    its markers at the halved density.
    """
    import imageio.v2 as imageio

    out.mkdir(parents=True, exist_ok=True)
    painted = {}
    for name, fid in (("wall_y+", 0), ("wall_y-", 40), ("wall_x+", 80), ("wall_x-", 120)):
        img = np.clip(concrete(res, res, seed=seed + fid).astype(np.float32) * SHADE[name],
                      0, 255).astype(np.uint8)
        imageio.imwrite(out / f"{name}.png", img)
        painted[name] = []
    floor = np.clip(concrete(res, res, seed=seed + 99).astype(np.float32) * SHADE["floor"],
                    0, 255).astype(np.uint8)
    painted["floor"] = _tags(floor, nx=3, ny=4, px=res // 16, first_id=160)
    imageio.imwrite(out / "floor.png", floor)
    return painted


MIN_RGB, SPEC, SHIN = 0.09, 0.55, 0.65
# The frame group's baseColorFactor is literally (0,0,0), so it rendered as a hole in the water
# whatever the lighting -- the chase and top-down views showed a silhouette, not a vehicle. A floor
# of 0.09 plus real specular is what dark anodised aluminium actually looks like underwater.


def _mola_geoms(scale):
    """One mesh + material + geom per colour group, so the AUV renders in its real colours.

    The model is NOT textured: 14 parts carrying six baseColorFactor materials -- a black frame, a
    light-grey hull, a cyan nose dome and yellow thruster pods. Painting it one yellow discarded that,
    and decimating 84,619 -> 5,586 faces destroyed the hull outright."""
    import json

    meta = json.loads((ASSETS / "mola_groups.json").read_text())
    a = "".join(
        f'<mesh name="mola{k}" file="mola/{row[0]}" scale="{scale} {scale} {scale}"/>'
        f'<material name="mola_c{k}" rgba="{max(MIN_RGB, row[1][0]):.3f} {max(MIN_RGB, row[1][1]):.3f} '
        f'{max(MIN_RGB, row[1][2]):.3f} 1" specular="{SPEC}" shininess="{SHIN}"/>'
        for k, row in enumerate(meta))
    # the whole 60 kg sits on the first group: MuJoCo SUMS geom masses, and six geoms at 60 kg each
    # would give a 360 kg vehicle whose inertia no longer matches the constants in mola.py
    parts = []
    for k in range(len(meta)):
        mass = "60" if k == 0 else "0.0001"
        parts.append(f'<geom name="mola_g{k}" type="mesh" mesh="mola{k}" material="mola_c{k}" '
                     f'mass="{mass}" contype="0" conaffinity="0"/>')
    return a, "".join(parts)


def _walls_and_rooms(hl, hw, seed=4242):
    """Four boxed walls with openings, plus one room behind each (wall, ring) pair.

    The room is placed at the wall surface, rotated so its depth axis points AWAY from the tank, and
    dropped so the window sits about 1.3 m above the room floor -- which is where a window is."""
    from .rooms import room_body, room_spec
    from .walls import build_wall

    rng = np.random.default_rng(seed)
    walls, mats, bodies = [], [], []
    #            name      axis sign  at   half-length   material   yaw of "away from tank"
    for name, axis, sign, at, half, mat, yaw in (
            ("wall_yp", "y", 1, hw, hl, "m_yp", 0.0),
            ("wall_ym", "y", -1, hw, hl, "m_ym", np.pi),
            ("wall_xp", "x", 1, hl, hw, "m_xp", -np.pi / 2),
            ("wall_xm", "x", -1, hl, hw, "m_xm", np.pi / 2)):
        g, centres, ring_z = build_wall(name, axis, sign, at, half, TANK_D, RING_Z, mat)
        walls += g
        for ri, zc in enumerate(ring_z):
            spec = room_spec(rng)
            if axis == "y":
                pos = (0.0, sign * (at + 0.1), zc - 1.3)
            else:
                pos = (sign * (at + 0.1), 0.0, zc - 1.3)
            m, b = room_body(f"{name}_r{ri}", spec, pos, yaw, span=2 * half * 0.95)
            mats.append(m)
            bodies.append(b)
    return "\n    ".join(walls), "".join(mats), "".join(bodies)


def mjcf(mola_scale=2.48, dock_corner=(-1, -1), n_fish=0, buoy_purple=False) -> str:
    """The tank MJCF. `mola_scale` makes the AUV 2.0 m long (0.806 x 2.48)."""
    hl, hw = TANK_L / 2, TANK_W / 2
    dx, dy = dock_corner
    HOOK_S = 0.28                       # the mesh is 5.53 m tall raw -> 1.55 m, a crane hook's size
    HOOK_H = 5.53 * HOOK_S
    # NEON GREEN normally; PURPLE is the appearance OOD. The diver OOD was dropped for this at the
    # user's instruction -- the source diver is a T-pose and the vertex-sweep posing attempt did not
    # read as a dive pose when rendered.
    BUOY_RGB = "0.72 0.18 0.92" if buoy_purple else "0.18 1.00 0.25"
    dock_pos = (dx * (hl - 2.0), dy * (hw - 2.0), -1.4)
    mola_assets, mola_geoms = _mola_geoms(mola_scale)
    wall_xml, room_mats, room_xml = _walls_and_rooms(hl, hw)

    lights = "".join(
        f'<geom name="ceil_{i}" type="cylinder" material="m_lamp" size="0.45 0.04" '
        f'pos="{x} {y} {ROOM_H - 0.06}" euler="0 0 0"/>'
        f'<light pos="{x} {y} {ROOM_H - 0.2}" dir="0 0 -1" diffuse="0.34 0.32 0.24" '
        f'castshadow="false"/>'
        for i, (x, y) in enumerate([(ax, ay) for ax in (-5.2, -1.7, 1.7, 5.2) for ay in (-2.6, 2.6)]))

    # GREAT WHITES, NOT APPROXIMATED FISH. The source mesh swims along -Y, so the geom yaws +90 deg
    # to put the nose on +X, which is the heading `intruders.mover` writes into mocap_quat.
    fish = "".join(f"""
    <body name="fish{i}" mocap="true" pos="0 0 -5">
      <geom type="mesh" mesh="shark" material="m_shark" euler="0 0 1.5708" contype="0" conaffinity="0"/>
    </body>""" for i in range(n_fish))

    extra_assets = ""
    if n_fish:
        extra_assets += ('<mesh name="shark" file="fish/fish_g0.obj" scale="1.6 1.6 1.6"/>'
                         '<material name="m_shark" rgba="0.40 0.44 0.48 1" specular="0.35" shininess="0.4"/>')

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
    <!-- CUBE TEXTURES ON THE WALLS, texuniform OFF. Rendered side by side on the real wall
         geometry: `type="2d"` smears the image into vertical planks at every texrepeat and at both
         texuniform settings, and cube + texuniform="true" projects it radially into a starburst.
         Cube + texuniform="false" gives each box face the whole image, which for scale-free
         concrete noise reads correctly at every box size. One baked image per surface, each a
         different shade, so a frame tells you which wall you are looking at. -->
    <texture name="t_fl" type="2d" file="floor.png"/>
    <texture name="t_yp" type="cube" file="wall_y+.png"/>
    <texture name="t_ym" type="cube" file="wall_y-.png"/>
    <texture name="t_xp" type="cube" file="wall_x+.png"/>
    <texture name="t_xm" type="cube" file="wall_x-.png"/>
    <material name="m_yp" texture="t_yp" texuniform="false" texrepeat="1 1" specular="0.06" shininess="0.06"/>
    <material name="m_ym" texture="t_ym" texuniform="false" texrepeat="1 1" specular="0.06" shininess="0.06"/>
    <material name="m_xp" texture="t_xp" texuniform="false" texrepeat="1 1" specular="0.06" shininess="0.06"/>
    <material name="m_xm" texture="t_xm" texuniform="false" texrepeat="1 1" specular="0.06" shininess="0.06"/>
    <material name="m_fl" texture="t_fl" texuniform="false" texrepeat="1 1" specular="0.05"/>
    <material name="m_surf" rgba="0.26 0.52 0.62 0.72" specular="0.9" shininess="0.9" reflectance="0.3"/>
    <material name="m_room" rgba="0.085 0.085 0.095 1" specular="0.02"/>
    <material name="m_lamp" rgba="1.0 0.92 0.45 1" emission="0.95" specular="0.1"/>
    <material name="m_gantry" rgba="0.93 0.74 0.05 1" specular="0.5" shininess="0.5"/>
    <material name="m_dock" rgba="0.72 0.74 0.78 1" specular="0.3"/>
    <material name="m_buoy_o" rgba="0.98 0.42 0.03 1" specular="0.6" shininess="0.6" emission="0.1"/>
    <material name="m_buoy_g" rgba="{BUOY_RGB} 1" specular="0.6" shininess="0.6" emission="0.18"/>
    <material name="m_rope" rgba="0.85 0.85 0.80 1"/>
    {mola_assets}{extra_assets}{room_mats}
    <mesh name="dock" file="dock/dock_g0.obj" scale="1 1 1"/>
    <mesh name="hook" file="hook/hook_g0.obj" scale="{HOOK_S} {HOOK_S} {HOOK_S}"/>
  </asset>

  <worldbody>
    <light pos="0 0 {ROOM_H - 0.5}" dir="0 0 -1" diffuse="0.42 0.43 0.42" castshadow="false"/>
    <light pos="{-hl*0.6} 0 -1.5" dir="0.3 0 -1" diffuse="0.26 0.28 0.28" castshadow="false"/>
    <light pos="{hl*0.6} 0 -4.0" dir="-0.3 0 -0.6" diffuse="0.22 0.24 0.24" castshadow="false"/>

    <!-- THE TANK. Walls span the water column only: floor at -D up to z = 0, nothing above.
         Each wall is eleven boxes leaving three real openings per ring (see walls.py). -->
    <geom name="floor"  type="plane" material="m_fl" pos="0 0 {-TANK_D}" size="{hl} {hw} 0.1"/>
    {wall_xml}
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
      <!-- a real crane hook (block + hook, 10.8k faces). The mesh hangs along -Y with the hook at
           +Y, so euler x=-90 deg stands it up: sheave block at the top, hook pointing down. -->
      <geom name="hook_g" type="mesh" mesh="hook" material="m_gantry" euler="-1.5708 0 0"
            pos="0 0 {-TANK_D/4 - HOOK_H/2}" contype="0" conaffinity="0"/>
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
      <camera name="chase" pos="-4.2 0 1.5" xyaxes="0 -1 0 0.33 0 0.944" fovy="58"/>
      <site name="mola_c" pos="0 0 0" size="0.05" rgba="1 0 0 0"/>
    </body>
    {fish}{room_xml}

    <!-- TOP-DOWN SITS UNDER THE GANTRY. From 3.2 m it was shooting straight through the beams,
         which cut the tank in half in every frame. Just below the water line it sees the whole pool
         with nothing in the way; the surface plane is one-sided so it does not occlude from above. -->
    <camera name="topdown" pos="0 0 -0.45" xyaxes="1 0 0 0 1 0" fovy="104" mode="fixed"/>
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
