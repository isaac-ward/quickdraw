"""Cave control rendering: HYBRID splat cave + crisp drone + goal ghost, depth-composited.

The CAVE is a crisp grapevine surface MESH (pyrender, rock browns/greys) with the real Gaussian SPLAT composited
OVER it by the splat's alpha: where the splat has coverage the real reconstruction shows, where it has none the
mesh shows (NOT a low-res splat -- the mesh is real triangles). The DRONE and GHOST are the real SISL quadrotor
MESH (body + 4 separable props at the arm tips, spinning; translucent solid-green ghost at the goal), rendered
crisply in pyrender and composited OVER the cave by DEPTH (grapevine metres, from the mesh), so walls occlude the
drone correctly. `CaveScene.render_splat_rgba` supplies the splat layer; `_load_cave` the mesh fallback.

Drone meshes are extracted from cave-maps.blend to .npz (extract_drone_mesh.py). The 4 blade objects in the blend
are identical geometry stacked at the origin (identity transforms), so we load ONE blade and place 4 copies at
the body's detected motor arm-tips -- the blend's own instancing is not in the vertex data.

Local mesh frame (from the blend): rotor plane is X-Z, up is +Y. `ROT_L2B` maps that to the vehicle BODY frame
(record.euler_zyx_R: body +x forward, +z DOWN), so applying the body->world R places the drone correctly.
"""
from __future__ import annotations

import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

ASSETS = "/home/ubuntu/user_irw/prospector/src/prospector/cave_wm/assets"
CAVE_PLY = "/home/ubuntu/user_irw/prospector/src/assets/point_clouds/grapevine-exped-3.ply"
LOGO_TEX = "/home/ubuntu/user_irw/prospector/src/assets/textures/droneBody_diffuse_SISL_Stanford.png"
MM = 0.001                                         # mesh units (mm) -> meters
PROP_LIFT = 0.012                                  # m: raise props slightly above the motor tops
# mesh-local (x,z rotor plane, +y up) -> body (x forward, y right, z DOWN): swap y<->z and flip for z-down.
ROT_L2B = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], np.float32)

_CACHE = {}


def _load_parts():
    """body (paint colors), logo (UV-TEXTURED SISL shell), ONE blade (origin-centered), and 4 arm-tip offsets.
    body/logo/blade share one center (the whole-drone centroid) so world placement puts the body centre at p."""
    if "body" in _CACHE:
        return _CACHE
    import trimesh
    from PIL import Image

    def load_colored(npz):
        d = np.load(os.path.join(ASSETS, npz))
        return trimesh.Trimesh(vertices=d["verts"].astype(np.float64) * MM, faces=d["faces"],
                               vertex_colors=(d["colors"] * 255).astype(np.uint8), process=False)
    body = load_colored("drone_body.npz")
    # drop the untextured pure-white gimbal CAMERA block (the one part that rendered as a white box)
    db = np.load(os.path.join(ASSETS, "drone_body.npz"))
    keep_face = (~(db["colors"] > 0.9).all(1))[db["faces"]].all(1)
    body.update_faces(keep_face); body.remove_unreferenced_vertices()
    blade = load_colored("drone_blade0.npz")        # all 4 are identical; place copies at the arm tips
    dl = np.load(os.path.join(ASSETS, "drone_logo.npz"))
    tex = Image.open(LOGO_TEX).convert("RGB")
    logo = trimesh.Trimesh(vertices=dl["verts"].astype(np.float64) * MM, faces=dl["faces"],
                           visual=trimesh.visual.TextureVisuals(uv=dl["uv"], image=tex), process=False)
    allv = np.concatenate([body.vertices, logo.vertices], 0)
    center = allv.mean(0)
    body.vertices -= center; logo.vertices -= center
    # 4 motor arm-tips: per x-z quadrant, the outer ring (max radius) of the whole drone, lifted up (+y local)
    v = np.concatenate([body.vertices, logo.vertices], 0)
    tips = []
    for sx in (+1, -1):
        for sz in (+1, -1):
            sub = v[(np.sign(v[:, 0]) == sx) & (np.sign(v[:, 2]) == sz)]
            rr = np.linalg.norm(sub[:, [0, 2]], axis=1)
            c = sub[rr > np.quantile(rr, 0.995)].mean(0); c[1] += PROP_LIFT
            tips.append(c)
    _CACHE.update(body=body, logo=logo, blade=blade, tips=np.array(tips))
    return _CACHE


def _value_noise(p, L, seed):
    """Smooth trilinear VALUE noise over 3D points `p` (N,3), feature size ~L -> (N,) in [0,1]. Random values on an
    L-lattice, smoothstep-interpolated inside each cell -> organic flow (no visible cell squares, unlike per-cell
    constant noise)."""
    x = p / L
    c = np.floor(x).astype(np.int64)
    f = x - c
    f = f * f * (3 - 2 * f)                                              # smoothstep per axis

    def h(ix, iy, iz):
        hh = (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791) ^ (seed * 2654435761)
        u = np.abs(np.sin(hh.astype(np.float64) * 12.9898) * 43758.5453)
        return u - np.floor(u)
    c0, c1, c2 = c[:, 0], c[:, 1], c[:, 2]
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
    x00 = h(c0, c1, c2) * (1 - fx) + h(c0 + 1, c1, c2) * fx
    x10 = h(c0, c1 + 1, c2) * (1 - fx) + h(c0 + 1, c1 + 1, c2) * fx
    x01 = h(c0, c1, c2 + 1) * (1 - fx) + h(c0 + 1, c1, c2 + 1) * fx
    x11 = h(c0, c1 + 1, c2 + 1) * (1 - fx) + h(c0 + 1, c1 + 1, c2 + 1) * fx
    y0 = x00 * (1 - fy) + x10 * fy
    y1 = x01 * (1 - fy) + x11 * fy
    return y0 * (1 - fz) + y1 * fz


def _load_cave():
    """grapevine cave surface MESH (crisp triangles), flat rock browns/greys (height-tinted + mottled). This is the
    FALLBACK everywhere the splat has no coverage; the real splat is composited over it by its alpha. Cached."""
    if "cave" in _CACHE:
        return _CACHE["cave"]
    import trimesh
    m = trimesh.load(CAVE_PLY, process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    z = m.vertices[:, 2]                                                # grapevine z (UP = -z)
    t = (z - z.min()) / (np.ptp(z) + 1e-9)                              # 0 low .. 1 high
    brown = np.array([0.47, 0.36, 0.27]); grey = np.array([0.48, 0.42, 0.35])   # browner muted rock
    col = (1 - t)[:, None] * grey + t[:, None] * brown                 # ceiling browner-grey, floor browner
    rng = np.random.default_rng(0)
    # SMOOTH two-octave value noise (no cell squares): a big ~0.8 m octave for broad rock shading + a ~0.3 m octave
    # for finer relief; a faint per-vertex grain breaks up any residual banding.
    v = m.vertices
    n = 0.68 * _value_noise(v, 0.8, 1) + 0.32 * _value_noise(v, 0.3, 2)
    col = col * (0.70 + 0.60 * n[:, None]) * (1.0 + 0.03 * (rng.random((len(z), 1)) - 0.5))
    vc = np.clip(col, 0, 1)
    m.visual.vertex_colors = (np.concatenate([vc, np.ones((len(z), 1))], 1) * 255).astype(np.uint8)
    _CACHE["cave"] = m
    return m


def _spin_T(center, angle):
    """4x4: translate origin-centered blade to `center` and spin by `angle` about the local up axis (+y)."""
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]], np.float64)
    T = np.eye(4); T[:3, 3] = center
    return T @ R                                                        # rotate at origin, then move to the tip


def _look_at(eye, target, up):
    z = np.asarray(eye, float) - np.asarray(target, float); z /= np.linalg.norm(z) + 1e-9
    x = np.cross(up, z); x /= np.linalg.norm(x) + 1e-9
    y = np.cross(z, x)
    m = np.eye(4); m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, eye
    return m


def _cam_pose(eye, fwd, up):
    """OpenGL camera pose (pyrender looks down -z) at `eye` toward eye+fwd."""
    f = np.asarray(fwd, float); f /= np.linalg.norm(f) + 1e-9
    s = np.cross(f, up); s /= np.linalg.norm(s) + 1e-9
    u = np.cross(s, f)
    m = np.eye(4); m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = s, u, -f, eye
    return m


def _world(p, R, extra=None):
    M = np.eye(4); M[:3, :3] = (R @ ROT_L2B).astype(np.float64); M[:3, 3] = np.asarray(p, float)
    return M @ extra if extra is not None else M


def _mk_drone(green):
    """(body_mesh, logo_mesh, [blade x4], tips) for the real drone, or all solid green for the ghost."""
    import pyrender

    def green_mesh(tm):
        import trimesh
        vc = np.tile([0, 210, 40, 255], (len(tm.vertices), 1)).astype(np.uint8)   # SOLID green, no texture
        g = trimesh.Trimesh(vertices=tm.vertices, faces=tm.faces, vertex_colors=vc, process=False)
        return pyrender.Mesh.from_trimesh(g, smooth=False)
    P = _load_parts()
    if green:                                          # ghost: solid green, logo untextured (just geometry)
        body = green_mesh(P["body"]); logo = green_mesh(P["logo"])
    else:
        body = pyrender.Mesh.from_trimesh(P["body"], smooth=False)
        logo = pyrender.Mesh.from_trimesh(P["logo"], smooth=False)     # carries the SISL UV texture
    blades = [pyrender.Mesh.from_trimesh(P["blade"], smooth=False) for _ in range(4)]
    return body, logo, blades, P["tips"]


def _scenes(W, H, hfov_deg):
    """PERSISTENT pyrender scenes (meshes uploaded ONCE): `cave`, `drone`, `ghost`. One OffscreenRenderer. Per
    frame only node poses / camera change. Keyed by (W,H,hfov); full-resolution meshes (NO decimation)."""
    import pyrender
    key = (int(W), int(H), round(hfov_deg, 3))
    if _CACHE.get("scn_key") == key:
        return _CACHE["scn"]
    if "scn" in _CACHE:
        _CACHE["scn"]["rend"].delete()
    fx = 0.5 * W / np.tan(0.5 * np.radians(hfov_deg))
    yf = 2 * np.arctan(0.5 * H / fx)

    def new_scene(amb):
        return pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[amb, amb, amb])

    # cave MESH (fallback): LOW ambient + a dominant camera headlamp + one off-axis fill, so faces shade by their
    # normal (relief/depth) rather than washing flat. Flat shading (smooth=False) on the scan reads as rock facets.
    cave_sc = new_scene(0.28)
    cave_node = cave_sc.add(pyrender.Mesh.from_trimesh(_load_cave(), smooth=False))
    cave_cam = cave_sc.add(pyrender.PerspectiveCamera(yfov=yf, aspectRatio=W / H, znear=0.05))
    cave_sc.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=2.0),                # fixed off-axis fill
                pose=_look_at(np.array([3.0, -2.0, -4.0]), [0, 0, 0], [0, 0, 1]))
    cave_light = cave_sc.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.5))   # camera headlamp (dominant)

    def drone_scene(green, amb):
        sc = new_scene(amb)
        body, logo, blades, tips = _mk_drone(green)
        bn = sc.add(body); ln_ = sc.add(logo); bl = [sc.add(b) for b in blades]
        cam = sc.add(pyrender.PerspectiveCamera(yfov=yf, aspectRatio=W / H, znear=0.02))
        light = sc.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=4.0))
        return dict(sc=sc, body=bn, logo=ln_, blades=bl, tips=tips, cam=cam, light=light)

    scn = dict(cave=dict(sc=cave_sc, node=cave_node, cam=cave_cam, light=cave_light),
               drone=drone_scene(False, 0.55), ghost=drone_scene(True, 0.7),
               rend=pyrender.OffscreenRenderer(int(W), int(H)))
    _CACHE["scn"] = scn; _CACHE["scn_key"] = key
    return scn


def _set_drone(d, p, R, blade_angle):
    sc = d["sc"]
    sc.set_pose(d["body"], _world(p, R))
    sc.set_pose(d["logo"], _world(p, R))
    for i, bn in enumerate(d["blades"]):
        sc.set_pose(bn, _world(p, R, _spin_T(d["tips"][i], blade_angle)))


def render_view(scene, eye, fwd, up, W, H, hfov_deg, p, R, blade_angle, goal_p, Rg,
                show_drone=True, show_ghost=True, ghost_a=0.25, ceil_cut=None):
    """Render cave (splat) + goal-ghost + drone from (eye,fwd,up) and DEPTH-composite -> rgb (H,W,3) uint8.

    The cave comes from `scene.render_rgbd` (the splat+mesh hybrid, depth in grapevine metres); the drone/ghost
    are pyrender meshes composited OVER it where they are nearer than the cave (so walls occlude the drone).
    `ceil_cut` (m): for the top-down view, clip the cave nearer than this distance from the eye so the ceiling
    above the drone is removed and the overhead camera sees inside (the drone/ghost are not clipped).
    `show_ghost`: the FPV is the drone's own camera -- it gets no ghost (at the goal it would fill the frame green);
    the external chase/top views show the ghost as the target marker."""
    import pyrender
    scn = _scenes(W, H, hfov_deg)
    rend = scn["rend"]
    cp = _cam_pose(eye, fwd, up)
    # SKIP_CULL_FACES: the cave is a one-sided surface whose normals face OUT, so from inside the cavern the walls
    # are back-faces and would cull to black; render double-sided so the walls show.
    flags = pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SKIP_CULL_FACES

    # CAVE = crisp grapevine MESH (pyrender) with the real SPLAT composited over it by the splat's alpha. The mesh is
    # the fallback everywhere; where the splat has coverage (alpha~1) the real reconstruction shows through, where it
    # has none (alpha~0) the mesh shows. c_depth (grape metres, from the mesh) occludes the drone. ceil_cut slices
    # the ceiling for the top-down (mesh via near-plane; splat via its own ceil_cut).
    cav = scn["cave"]
    cav["cam"].camera.znear = float(ceil_cut) if ceil_cut else 0.05
    cav["sc"].set_pose(cav["cam"], cp); cav["sc"].set_pose(cav["light"], cp)
    m_rgb, m_depth = rend.render(cav["sc"], flags=flags)
    s_rgb, s_a, _ = scene.render_splat_rgba(eye, fwd, up, W=W, H=H, hfov=hfov_deg, ceil_cut=ceil_cut)
    a = np.clip((s_a - 0.15) / 0.6, 0.0, 1.0)[..., None]                  # sharpen: ignore faint splat floaters
    out = m_rgb[..., :3].astype(np.float32) * (1 - a) + s_rgb.astype(np.float32) * a
    c_depth = np.where(m_depth > 0, m_depth, np.inf).astype(np.float32)   # mesh is the reliable occluder
    has_cave = m_depth > 0

    # ghost at the goal (translucent), then drone over it (opaque)
    if show_ghost:
        _set_drone(scn["ghost"], goal_p, Rg, 0.0)
        scn["ghost"]["sc"].set_pose(scn["ghost"]["cam"], cp); scn["ghost"]["sc"].set_pose(scn["ghost"]["light"], cp)
        g_rgb, g_depth = rend.render(scn["ghost"]["sc"], flags=flags)
        gvis = ((g_depth > 0) & (~has_cave | (g_depth < c_depth))).astype(np.float32) * ghost_a
        out = out * (1 - gvis[..., None]) + g_rgb[..., :3].astype(np.float32) * gvis[..., None]

    if show_drone:
        _set_drone(scn["drone"], p, R, blade_angle)
        scn["drone"]["sc"].set_pose(scn["drone"]["cam"], cp); scn["drone"]["sc"].set_pose(scn["drone"]["light"], cp)
        d_rgb, d_depth = rend.render(scn["drone"]["sc"], flags=flags)
        dvis = (d_depth > 0) & (~has_cave | (d_depth < c_depth))
        out[dvis] = d_rgb[..., :3][dvis].astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def inspect(out_path, size=512):
    """Render body + 4 props (at arm tips) from 4 views with XYZ axes (R=x,G=y,B=z) for mesh/orientation QA."""
    import trimesh, pyrender
    from PIL import Image, ImageDraw
    P = _load_parts()
    ext = np.ptp(np.concatenate([P["body"].vertices, P["logo"].vertices], 0), axis=0)
    diag = float(np.linalg.norm(ext))
    axis = trimesh.creation.axis(origin_size=0.004, axis_length=0.12, axis_radius=0.0025)
    sc = pyrender.Scene(bg_color=[18, 18, 22, 255], ambient_light=[0.6, 0.6, 0.6])
    sc.add(pyrender.Mesh.from_trimesh(P["body"], smooth=False))
    sc.add(pyrender.Mesh.from_trimesh(P["logo"], smooth=False))          # UV-textured SISL shell
    for tip in P["tips"]:
        b = P["blade"].copy(); b.vertices += tip
        sc.add(pyrender.Mesh.from_trimesh(b, smooth=False))
    sc.add(pyrender.Mesh.from_trimesh(axis, smooth=False))
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 4)
    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=4.0)
    r = pyrender.OffscreenRenderer(size, size)
    views = {"iso": [diag, diag, diag * 0.7], "front +x": [diag * 1.5, 0.01, 0.0],
             "top +z": [0.01, 0.01, diag * 1.6], "side +y": [0.0, diag * 1.5, 0.01]}
    tiles = []
    for name, eye in views.items():
        pose = _look_at(np.array(eye), [0, 0, 0], [0, 0, 1])
        cn = sc.add(cam, pose=pose); ln = sc.add(light, pose=pose)
        color, _ = r.render(sc)
        sc.remove_node(cn); sc.remove_node(ln)
        im = Image.fromarray(color); d = ImageDraw.Draw(im)
        d.rectangle([0, 0, size, 18], fill=(0, 0, 0)); d.text((4, 3), name, fill=(255, 255, 255))
        tiles.append(np.asarray(im))
    r.delete()
    grid = np.concatenate([np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:], 1)], 0)
    Image.fromarray(grid).save(out_path)
    print("wrote", out_path, grid.shape, "drone ext(m)", tuple(round(x, 3) for x in ext), flush=True)


if __name__ == "__main__":
    import sys
    inspect(sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/user_irw/quickdraw/logs/cave/drone_inspect.png")
