"""The MBARI test tank as MuJoCo MJCF: 15 x 10 x 10 m, concrete, windows, ArUco, crane hook.

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

import os
import pathlib

import numpy as np


from .textures import aruco_tile, concrete

TANK_L, TANK_W, TANK_D = 15.0, 10.0, 10.0          # metres: length (x), width (y), depth (z)
# Window-ring depths as a fraction of TANK_D. Both rings moved DOWN by 0.95 m -- half a window
# height at the size they were when the move was asked for (1.9 m), before the shrink to 80%.
RING_Z = (0.335, 0.715)
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
# KEEP-OUT VOLUMES, as upright cylinders (x, y, radius, z_lo, z_hi) covering every solid object a
# vehicle could fly into. Defined HERE, beside the geometry that draws them, so the two cannot
# drift: change a buoy's position and the keep-out moves with it.
KEEPOUT = (
    # Each tether runs from its buoy UP to the surface and no further, so the z span stops at the
    # buoy -- an earlier version ran them to the floor and excluded two full-depth columns of open
    # water, which would have truncated 69% of episodes instead of 7%.
    (3.375, 2.5, 0.38, -5.38, -0.52),      # orange buoy + tether above it
    (-3.0, -2.25, 0.34, -5.94, -0.52),     # neon-green buoy + tether above it
    (7.28, -4.25, 0.32, -3.60, -0.52),     # entry ladder on the +X wall
    (1.2, 0.0, 0.34, -2.95, -0.52),        # crane cable and the hook block (halved)
)
VEHICLE_R = 0.55                            # bounding radius of the 1.0 m AUV, generous on purpose

TEX_N = 512                   # cube faces must be square; see build_textures
WATER_Z = -0.52               # the water line, 50 cm below the tank rim, so the rim shows above it
PX_PER_M = 70
XWALL_MODE = 'none'           # 'none' | 'rot' | 'square'; chosen by the A/B in _rotab.py
WALL_TAG_ID0 = 200            # ArUco ids for the wall fiducials; the floor grid uses 160+                 # concrete resolution; every wall box is CROPPED from one wall-wide
#                               field at this density, which is what stops the texture stretching


def _square(crop, n=TEX_N):
    """Resample a crop to n x n, PER AXIS. Undone exactly by MuJoCo's stretch onto the box face.

    The axis being shrunk is area-averaged; the axis being grown is nearest-neighbour. Bilinear on
    the growing axis was the bug behind the streaks: a 15 x 1.2 m strip crops to 1050 x 84, and
    smoothly interpolating 84 rows up to 512 turns honest speckle into soft horizontal bands. Taking
    them nearest keeps each source pixel a sharp block, so the stretched result is square speckle at
    PX_PER_M in both directions."""
    import cv2

    h, w = crop.shape[:2]
    out = cv2.resize(crop, (n, h), interpolation=cv2.INTER_AREA if w > n else cv2.INTER_NEAREST)
    return cv2.resize(out, (n, n), interpolation=cv2.INTER_AREA if h > n else cv2.INTER_NEAREST)


def build_textures(out=BUILD, res=1024, seed=0):
    """Bake ONE concrete field per wall at a fixed px/metre, then a crop per wall box.

    THE STRETCH, AND WHY CROPPING IS THE FIX. The walls are box geoms. A `type="2d"` texture on a box
    is smeared per face into vertical planks; a `type="cube"` texture ignores `texrepeat` entirely
    (measured: rep=1 and rep=4 render identically) and `texuniform="true"` projects it radially into
    a starburst. So MuJoCo offers no way to TILE on a box -- whatever image a face gets is stretched
    to fill it, and a 15 x 0.4 m strip and a 1.5 x 1.9 m pillar given the same square image show
    wildly different grain.

    CROP, THEN SQUASH TO SQUARE, AND LET MUJOCO STRETCH IT BACK. Each box gets the region of one
    wall-spanning concrete field that matches its own footprint -- so neighbouring boxes are
    neighbouring crops and the concrete runs continuously across the wall -- and that crop is then
    resampled to TEX_N x TEX_N. Cube textures must be square, and the square-ing is undone exactly
    when MuJoCo stretches the image onto the face, so the grain lands back at PX_PER_M in both
    directions on every box regardless of its aspect. (A 2d texture cannot be used instead: on a box
    MuJoCo smears it along one axis only, which is the vertical-plank artifact, and feeding it a
    correctly-shaped crop does not help because the mapping never reads the second axis.)
    """
    import imageio.v2 as imageio

    from .walls import build_wall

    out.mkdir(parents=True, exist_ok=True)
    hl, hw = TANK_L / 2, TANK_W / 2
    painted = {}
    for name, axis, sign, at, half, fid in (("wall_y+", "y", 1, hw, hl, 0),
                                            ("wall_y-", "y", -1, hw, hl, 40),
                                            ("wall_x+", "x", 1, hl, hw, 80),
                                            ("wall_x-", "x", -1, hl, hw, 120)):
        W = int(2 * half * PX_PER_M)
        H = int(TANK_D * PX_PER_M)
        field = np.clip(concrete(H, W, seed=seed + fid).astype(np.float32) * SHADE[name],
                        0, 255).astype(np.uint8)
        crops = {}

        def mat_fn(tag, u0, u1, z0, z1, _n=name, _f=field, _h=half, _W=W, _H=H, _c=crops,
                   _rot=(axis == "x")):
            x0 = int(round((u0 + _h) / (2 * _h) * _W))
            x1 = max(x0 + 2, int(round((u1 + _h) / (2 * _h) * _W)))
            y0 = int(round((-z1) / TANK_D * _H))
            y1 = max(y0 + 2, int(round((-z0) / TANK_D * _H)))
            crop = _f[y0:y1, x0:x1]
            if _rot and XWALL_MODE == "rot":
                crop = np.rot90(crop)
            elif _rot and XWALL_MODE == "square":
                k = max(crop.shape[0], crop.shape[1])
                yc, xc = (y0 + y1) // 2, (x0 + x1) // 2
                ya, xa = max(0, yc - k // 2), max(0, xc - k // 2)
                crop = _f[ya:ya + k, xa:xa + k]
            # MuJoCo's cube mapping lays the image onto a +-X face rotated 90 degrees relative to a
            # +-Y face. Without this the aspect correction is applied across the wrong axis there,
            # and the end walls came out streaked while the long walls looked right.
            _c[tag] = _square(crop)
            return f"m_{_n}_{tag}"

        _, _, _, tags = build_wall(name, axis, sign, at, half, TANK_D, RING_Z, mat_fn)
        for tag, img in crops.items():
            imageio.imwrite(out / f"{name}_{tag}.png", img)
        ids = []
        for n, (tname, _pos, _size) in enumerate(tags):
            tid = WALL_TAG_ID0 + fid // 40 * 4 + n
            imageio.imwrite(out / f"{tname}.png", aruco_tile(tid, px=256, border=1))
            ids.append(tid)
        painted[name] = ids
    floor = np.clip(concrete(res, res, seed=seed + 99).astype(np.float32) * SHADE["floor"],
                    0, 255).astype(np.uint8)
    painted["floor"] = _tags(floor, nx=3, ny=4, px=res // 16, first_id=160)
    imageio.imwrite(out / "floor.png", floor)
    return painted


MIN_RGB, SPEC, SHIN = 0.09, 0.55, 0.65
# THE HULL IS YELLOW, NOT SILVER. The source model's large pressure-hull group ships as a flat grey
# (0.4, 0.4, 0.4); recoloured here, at the single place every MOLA geom is emitted, so the change
# reaches the tank, the intruder scenes and the control videos alike rather than being patched per
# caller. Matched by colour, not by group index, because the index moves if the mesh is reconverted.
HULL_GREY = (0.4, 0.4, 0.4)
HULL_YELLOW = (0.88, 0.72, 0.06)
GHOST_RGBA = "0.10 0.95 0.25 0.125"     # the translucent goal-pose marker; see `ghost` in mjcf()


def _recolour(rgb):
    return HULL_YELLOW if tuple(round(float(c), 2) for c in rgb[:3]) == HULL_GREY else rgb
# The frame group's baseColorFactor is literally (0,0,0), so it rendered as a hole in the water
# whatever the lighting. A floor of 0.09 plus real specular is what dark anodised aluminium looks
# like underwater.


def _mola_geoms(scale):
    """One mesh + material + geom per colour group, so the AUV renders in its real colours.

    The model is NOT textured: 14 parts carrying six baseColorFactor materials -- a black frame, a
    light-grey hull, a cyan nose dome and yellow thruster pods. Painting it one yellow discarded that,
    and decimating 84,619 -> 5,586 faces destroyed the hull outright."""
    import json

    meta = json.loads((ASSETS / "mola_groups.json").read_text())
    cols = [_recolour(row[1]) for row in meta]
    a = "".join(
        f'<mesh name="mola{k}" file="mola/{row[0]}" scale="{scale} {scale} {scale}"/>'
        f'<material name="mola_c{k}" rgba="{max(MIN_RGB, c[0]):.3f} {max(MIN_RGB, c[1]):.3f} '
        f'{max(MIN_RGB, c[2]):.3f} 1" specular="{SPEC}" shininess="{SHIN}"/>'
        for k, (row, c) in enumerate(zip(meta, cols)))
    # the whole 60 kg sits on the first group: MuJoCo SUMS geom masses, and six geoms at 60 kg each
    # would give a 360 kg vehicle whose inertia no longer matches the constants in mola.py
    # ROLLED +90 DEG ABOUT X. The source mesh's cross-section is 0.477 m across by 0.561 m tall --
    # taller than wide, which is a vehicle lying on its side. Rolling it upright makes it wider than
    # tall, spreads the four thrusters to the corners, and puts the two small fore/aft fittings on
    # top (measured at +49% of the way to the frame's top edge; the opposite roll puts them under).
    parts = [f'<geom name="mola_g{k}" type="mesh" mesh="mola{k}" material="mola_c{k}" '
             f'euler="1.5708 0 0" mass="0.0001" contype="0" conaffinity="0"/>'
             for k in range(len(meta))]
    return a, "".join(parts)


def _walls_and_rooms(hl, hw, seed=4242):
    """Four boxed walls with openings and frames, one room behind each (wall, ring) pair.

    Returns (wall geoms, wall texture+material assets, room materials, room bodies). Each wall box
    declares its OWN texture -- the crop `build_textures` wrote for it -- which is what keeps the
    concrete isotropic on boxes of wildly different aspect."""
    from .rooms import room_body, room_spec
    from .walls import build_wall

    rng = np.random.default_rng(seed)
    walls, wall_assets, mats, bodies = [], [], [], []
    for name, axis, sign, at, half, yaw in (("wall_y+", "y", 1, hw, hl, 0.0),
                                            ("wall_y-", "y", -1, hw, hl, np.pi),
                                            ("wall_x+", "x", 1, hl, hw, -np.pi / 2),
                                            ("wall_x-", "x", -1, hl, hw, np.pi / 2)):
        seen = []

        def mat_fn(tag, u0, u1, z0, z1, _n=name, _s=seen):
            _s.append(tag)
            return f"m_{_n}_{tag}"

        g, centres, ring_z, tags = build_wall(name, axis, sign, at, half, TANK_D, RING_Z, mat_fn)
        walls += g
        for tname, tpos, tsize in tags:
            # CUBE, not 2d: a 2d texture on a box geom is smeared along one axis, and the markers
            # rendered as plain white squares. The tile is square so cube accepts it.
            wall_assets.append(f'<texture name="t_{tname}" type="cube" file="{tname}.png"/>'
                               f'<material name="m_{tname}" texture="t_{tname}" texuniform="false" '
                               f'specular="0.03"/>')
            walls.append(f'<geom name="{tname}" type="box" material="m_{tname}" pos="{tpos}" '
                         f'size="{tsize}" contype="0" conaffinity="0"/>')
        for tag in seen:
            wall_assets.append(
                f'<texture name="t_{name}_{tag}" type="cube" file="{name}_{tag}.png"/>'
                f'<material name="m_{name}_{tag}" texture="t_{name}_{tag}" texuniform="false" '
                f'specular="0.06" shininess="0.06"/>')
        for ri, zc in enumerate(ring_z):
            spec = room_spec(rng)
            pos = ((0.0, sign * (at + 0.1), zc - 1.3) if axis == "y"
                   else (sign * (at + 0.1), 0.0, zc - 1.3))
            m, b = room_body(f"{name}_r{ri}", spec, pos, yaw, span=2 * half * 0.95)
            mats.append(m)
            bodies.append(b)
    return ("\n    ".join(walls), "".join(wall_assets), "".join(mats), "".join(bodies))


def mjcf(mola_scale=1.24, n_fish=0, buoy_purple=False, ghost=False, buoy_ood=False) -> str:
    """The tank MJCF. `mola_scale` 1.24 makes the AUV 1.0 m long (0.807 x 1.24), half its first size."""
    hl, hw = TANK_L / 2, TANK_W / 2
    HOOK_S = 0.14                       # the mesh is 5.53 m tall raw -> 0.77 m (halved: 1.55 m
    #                                     read as oversized against a 1.0 m vehicle)
    HOOK_H = 5.53 * HOOK_S
    # NEON GREEN normally; PURPLE is the appearance OOD. The diver OOD was dropped for this at the
    # user's instruction -- the source diver is a T-pose and the vertex-sweep posing attempt did not
    # read as a dive pose when rendered.
    BUOY_RGB = "0.72 0.18 0.92" if buoy_purple else "0.18 1.00 0.25"
    mola_assets, mola_geoms = _mola_geoms(mola_scale)
    # A PURPLE BUOY AS A MOCAP BODY, for the controlled OOD episodes (gen_ood_controlled.py).
    # The `buoy_purple` condition above RECOLOURS an existing fixture, which cannot "come into
    # frame" -- it is in the scene from step 0 and only the camera moves. An onset experiment needs
    # an object that is genuinely absent and then present, so this is a separate body the caller
    # parks below the floor and raises. It is off by default and never built for nominal data.
    buoy_ood_body = ("\n    <body name=\"buoy_ood\" mocap=\"true\" pos=\"0 0 -40\">"
                     "\n      <geom name=\"buoy_ood_g\" type=\"sphere\" material=\"m_buoy_p\" size=\"0.34\" "
                     "contype=\"0\" conaffinity=\"0\"/>"
                     "\n    </body>") if buoy_ood else ""
    # GHOST: a translucent green copy of the vehicle marking the GOAL POSE, as a mocap body so a
    # caller can place it per episode. OFF by default and never built for data generation -- a
    # target marker in the training frames would be a feature the model could read instead of
    # learning the dynamics.
    ghost_assets, ghost_body = ("", "")
    if ghost:
        import json as _json
        n_g = len(_json.loads((ASSETS / "mola_groups.json").read_text()))
        ghost_assets = f'<material name="m_ghost" rgba="{GHOST_RGBA}" specular="0.1"/>'
        # rgba ON THE GEOM, not only on the material: a geom's own rgba defaults to
        # (0.5, 0.5, 0.5, 1) and that default WINS over the material's alpha, so the ghost
        # rendered fully opaque while the material said 0.25.
        ghost_body = ("\n    <body name=\"ghost\" mocap=\"true\" pos=\"0 0 -40\">"
                      + "".join(f'<geom type="mesh" mesh="mola{k}" material="m_ghost" '
                                f'rgba="{GHOST_RGBA}" euler="1.5708 0 0" '
                                f'contype="0" conaffinity="0" group="2"/>'
                                for k in range(n_g)) + "</body>")
    wall_xml, wall_assets, room_mats, room_xml = _walls_and_rooms(hl, hw)

    # ENTRY LADDER on the +X wall at y=-4.25, clear of both buoys and clearing the nearest window
    # rim by 0.24 m. (It was originally sited relative to the dock, which no longer exists.)
    # Starts 0.55 m above the water and runs down to the centre of the UPPER window ring.
    lad_y, lad_bot = -4.25, -TANK_D * RING_Z[0]
    lad_rungs = "".join(
        f'<geom name="lad_r{i}" type="capsule" material="m_gantry" size="0.028" '
        f'fromto="{hl-0.22} {lad_y-0.21} {z:.3f} {hl-0.22} {lad_y+0.21} {z:.3f}"/>'
        for i, z in enumerate(np.arange(0.40, lad_bot, -0.30)))
    ladder = f"""
    <body name="ladder" pos="0 0 0">
      <geom name="lad_a" type="capsule" material="m_gantry" size="0.038"
            fromto="{hl-0.22} {lad_y-0.21} 0.55 {hl-0.22} {lad_y-0.21} {lad_bot:.3f}"/>
      <geom name="lad_b" type="capsule" material="m_gantry" size="0.038"
            fromto="{hl-0.22} {lad_y+0.21} 0.55 {hl-0.22} {lad_y+0.21} {lad_bot:.3f}"/>
      {lad_rungs}
    </body>"""

    lights = "".join(
        f'<geom name="ceil_{i}" type="cylinder" material="m_lamp" size="0.45 0.04" '
        f'pos="{x} {y} {ROOM_H - 0.06}" euler="0 0 0"/>'
        f'<light pos="{x} {y} {ROOM_H - 0.2}" dir="0 0 -1" diffuse="0.34 0.32 0.24" '
        f'castshadow="false"/>'
        for i, (x, y) in enumerate([(ax, ay) for ax in (-5.2, -1.7, 1.7, 5.2) for ay in (-2.6, 2.6)]))

    # GREAT WHITES, NOT APPROXIMATED FISH. Rendered against labelled axes the source mesh swims
    # along +Y with its DARK BACK ON -Z -- it is both rotated and upside down. One quaternion fixes
    # both: a 180 deg turn about (1,1,0)/sqrt(2) maps mesh +Y -> world +X (nose forward, the heading
    # `intruders.mover` writes) and mesh -Z -> world +Z (back up). Assuming the nose was on -Y and
    # the back on +Z is what shipped sharks swimming backwards and inverted.
    fish = "".join(f"""
    <body name="fish{i}" mocap="true" pos="0 0 -5">
      <geom type="mesh" mesh="shark" material="m_shark" quat="0 0.7071068 0.7071068 0" contype="0" conaffinity="0"/>
    </body>""" for i in range(n_fish))

    extra_assets = ""
    if n_fish:
        extra_assets += ('<mesh name="shark" file="fish/fish_g0.obj" scale="1.6 1.6 1.6"/>'
                         '<texture name="t_shark" type="2d" file="shark.png"/>'
                         '<material name="m_shark" texture="t_shark" texuniform="false" '
                         'specular="0.3" shininess="0.35"/>')

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
    <!-- WALL CONCRETE IS ONE CROP PER BOX (see build_textures). MuJoCo cannot tile a texture on a
         box geom -- 2d smears into planks, cube ignores texrepeat, cube+texuniform starbursts --
         so instead every box carries an image already cut to its own aspect out of one
         wall-spanning concrete field. Isotropic grain, continuous across the wall, no stretch. -->
    <texture name="t_fl" type="2d" file="floor.png"/>
    {wall_assets}
    <material name="m_frame" rgba="0.115 0.115 0.125 1" specular="0.35" shininess="0.5"/>
    <material name="m_fl" texture="t_fl" texuniform="false" texrepeat="1 1" specular="0.05"/>
    <material name="m_surf" rgba="0.26 0.52 0.62 0.72" specular="0.9" shininess="0.9" reflectance="0.3"/>
    <material name="m_room" rgba="0.085 0.085 0.095 1" specular="0.02"/>
    <material name="m_lamp" rgba="1.0 0.92 0.45 1" emission="0.95" specular="0.1"/>
    <material name="m_gantry" rgba="0.93 0.74 0.05 1" specular="0.5" shininess="0.5"/>
    <material name="m_buoy_o" rgba="0.98 0.42 0.03 1" specular="0.6" shininess="0.6" emission="0.1"/>
    <material name="m_buoy_g" rgba="{BUOY_RGB} 1" specular="0.6" shininess="0.6" emission="0.18"/>
    <material name="m_buoy_p" rgba="0.72 0.18 0.92 1" specular="0.6" shininess="0.6" emission="0.18"/>
    <material name="m_rope" rgba="0.85 0.85 0.80 1"/>
    <material name="m_cable" rgba="0.26 0.26 0.28 1" specular="0.5" shininess="0.6"/>
    {mola_assets}{ghost_assets}{extra_assets}{room_mats}
    <mesh name="hook" file="hook/hook_g0.obj" scale="{HOOK_S} {HOOK_S} {HOOK_S}"/>
    <texture name="t_hook" type="2d" file="hook.png"/>
    <material name="m_hook" texture="t_hook" texuniform="false" specular="0.4" shininess="0.5"/>
  </asset>

  <worldbody>
    <light pos="0 0 {ROOM_H - 0.5}" dir="0 0 -1" diffuse="0.42 0.43 0.42" castshadow="false"/>
    <light pos="{-hl*0.6} 0 -1.5" dir="0.3 0 -1" diffuse="0.26 0.28 0.28" castshadow="false"/>
    <light pos="{hl*0.6} 0 -4.0" dir="-0.3 0 -0.6" diffuse="0.22 0.24 0.24" castshadow="false"/>

    <!-- THE TANK. Walls span the water column only: floor at -D up to z = 0, nothing above.
         Each wall is eleven boxes leaving three real openings per ring (see walls.py). -->
    <geom name="floor"  type="plane" material="m_fl" pos="0 0 {-TANK_D}" size="{hl} {hw} 0.1"/>
    {wall_xml}
    <geom name="surface" type="plane" material="m_surf" pos="0 0 {WATER_Z}" size="{hl} {hw} 0.1" euler="3.14159 0 0"/>

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
      <!-- CYLINDER, NOT CAPSULE, for the cable: a capsule's hemispherical caps swell at the ends
           and the line read as varying width. A cylinder is the same width the whole way down. -->
      <geom name="hook_line" type="cylinder" material="m_cable" size="0.045 {(GANTRY_H + TANK_D/4)/2:.4f}"
            pos="0 0 {(GANTRY_H - TANK_D/4)/2:.4f}" contype="0" conaffinity="0"/>
      <!-- a real crane hook (block + hook, 12.9k faces, its own baseColor texture). The mesh lies
           along -Y with the hook at +Y, so euler x=+90 deg stands it up HOOK-END DOWN. At -90 it
           was upside down, with the hook opening at the top. -->
      <geom name="hook_g" type="mesh" mesh="hook" material="m_hook" euler="1.5708 0 0"
            pos="0 0 {-TANK_D/4 - HOOK_H/2}" contype="0" conaffinity="0"/>
    </body>

    {ladder}

    <body name="buoy_orange" pos="{hl*0.45} {hw*0.5} -5.0">
      <geom name="rope_o" type="capsule" material="m_rope" fromto="0 0 0 0 0 {5.0 + WATER_Z}" size="0.025"/>
      <geom name="buoy_o" type="sphere" material="m_buoy_o" size="0.38"/>
    </body>
    <body name="buoy_green" pos="{-hl*0.40} {-hw*0.45} -5.6">
      <geom name="rope_g" type="capsule" material="m_rope" fromto="0 0 0 0 0 {5.6 + WATER_Z}" size="0.025"/>
      <geom name="buoy_g" type="sphere" material="m_buoy_g" size="0.34"/>
    </body>
    {buoy_ood_body}


    <body name="mola" pos="0 0 -5">
      <freejoint name="mola_free"/>
      <!-- INERTIA PINNED, NOT DERIVED FROM THE GEOMS. Halving the mesh would quarter a
           geom-derived inertia and make the vehicle four times more angularly agile for the same
           torque, silently invalidating every tuned gain and every measured number (cruise speed,
           coast time, 0.040 m tracking error). The size change asked for was visual, so the
           inertial properties are stated explicitly and the dynamics are untouched. The roll
           swaps the y and z principal values: (8.86, 22.45, 20.16) -> (8.86, 20.16, 22.45). -->
      <inertial pos="0 0 0" mass="60" diaginertia="8.86 20.16 22.45"/>
      {mola_geoms}
      <camera name="fpv" pos="0.50 0 0.04" xyaxes="0 -1 0 0 0 1" fovy="70"/>
      <camera name="chase" pos="-4.2 0 1.5" xyaxes="0 -1 0 0.33 0 0.944" fovy="58"/>
      <site name="mola_c" pos="0 0 0" size="0.05" rgba="1 0 0 0"/>
    </body>
    {fish}{ghost_body}{room_xml}

    <!-- TOP-DOWN SITS UNDER THE GANTRY. From 3.2 m it was shooting straight through the beams,
         which cut the tank in half in every frame. Just below the water line it sees the whole pool
         with nothing in the way; the surface plane is one-sided so it does not occlude from above. -->
    <camera name="topdown" pos="0 0 {WATER_Z - 0.45}" xyaxes="1 0 0 0 1 0" fovy="104" mode="fixed"/>
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
    # the shark and hook carry their own baseColor images; MuJoCo looks for textures in `texturedir`
    for src_name, dst in (("fish/fish_tex.png", "shark.png"), ("hook/hook_tex.png", "hook.png")):
        src = ASSETS / src_name
        if src.exists() and not (out / dst).exists():
            out.mkdir(parents=True, exist_ok=True)
            (out / dst).write_bytes(src.read_bytes())
    painted = build_textures(out) if textures else {}
    p = ASSETS / name
    # ATOMIC. Every generation shard calls build(), so 24 processes write this same path while
    # others are reading it -- and two episodes died on `ParseXML: empty file` because a reader
    # caught a truncated write. Writing to a unique temp file and os.replace()-ing it is atomic
    # within a filesystem, so a reader sees either the old file or the new one, never a partial.
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(mjcf(**kw))
    os.replace(tmp, p)
    return p, painted


if __name__ == "__main__":
    p, painted = build()
    print(f"  wrote {p}")
    for k, v in painted.items():
        print(f"    {k:12s} {len(v):3d} aruco ids {v[:6]}{'...' if len(v) > 6 else ''}")
