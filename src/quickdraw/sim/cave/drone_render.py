"""Sharp all-MESH cave rendering via pyrender (EGL, headless): cave + drone + goal ghost, depth-composited.

NO Gaussian splats. The cave is the `grapevine-exped-3.ply` surface mesh (same metric frame as the control /
collision grid -- its AABB matches the voxel grid exactly), textured in flat rock browns/greys. The drone is the
real SISL quadrotor mesh (body + 4 separable props at the arm tips, spinning); the goal is a translucent green
ghost of the same mesh. Everything renders in ONE metric frame and composites by DEPTH, so walls occlude the
drone correctly and no splat colour (the old purple/blue) leaks in.

Meshes are extracted from cave-maps.blend to .npz (scratchpad/extract_npz2.py). The 4 blade objects in the blend
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
MM = 0.001                                         # mesh units (mm) -> meters
PROP_LIFT = 0.012                                  # m: raise props slightly above the motor tops
# mesh-local (x,z rotor plane, +y up) -> body (x forward, y right, z DOWN): swap y<->z and flip for z-down.
ROT_L2B = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], np.float32)

_CACHE = {}


def _load_parts():
    """body trimesh (centered), ONE blade trimesh (origin-centered), and 4 arm-tip offsets (m, local frame)."""
    if "body" in _CACHE:
        return _CACHE
    import trimesh

    def load(npz):
        d = np.load(os.path.join(ASSETS, npz))
        v = d["verts"].astype(np.float64) * MM
        return trimesh.Trimesh(vertices=v, faces=d["faces"],
                               vertex_colors=(d["colors"] * 255).astype(np.uint8), process=False)
    body = load("drone_body.npz")
    blade = load("drone_blade0.npz")               # all 4 are identical; place copies at the arm tips
    center = body.vertices.mean(0)
    body.vertices -= center
    # 4 motor arm-tips: per x-z quadrant, the outer ring of the body (max radius), lifted slightly up (+y local)
    v = body.vertices
    tips = []
    for sx in (+1, -1):
        for sz in (+1, -1):
            m = (np.sign(v[:, 0]) == sx) & (np.sign(v[:, 2]) == sz)
            sub = v[m]
            rr = np.linalg.norm(sub[:, [0, 2]], axis=1)
            c = sub[rr > np.quantile(rr, 0.995)].mean(0)
            c[1] += PROP_LIFT
            tips.append(c)
    _CACHE.update(body=body, blade=blade, tips=np.array(tips))
    return _CACHE


def _load_cave():
    """grapevine cave surface mesh, flat rock browns/greys (height-tinted). Cached."""
    if "cave" in _CACHE:
        return _CACHE["cave"]
    import trimesh
    m = trimesh.load(CAVE_PLY, process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    z = m.vertices[:, 2]                                                # grapevine z (UP = -z)
    t = (z - z.min()) / (np.ptp(z) + 1e-9)                              # 0 low .. 1 high
    brown = np.array([0.60, 0.49, 0.38]); grey = np.array([0.72, 0.70, 0.67])
    col = (1 - t)[:, None] * grey + t[:, None] * brown                 # ceiling greyer, floor browner
    col += (np.random.default_rng(0).random((len(z), 1)) - 0.5) * 0.05  # faint grain so flats aren't dead
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
    """pyrender Mesh list [body, blade x4-markers] for the drone (real) or green ghost."""
    import pyrender
    P = _load_parts()

    def col(tm):
        c = tm.copy()
        if green:
            vc = np.zeros((len(c.vertices), 4), np.uint8); vc[:, 1] = 230; vc[:, 3] = 255
            c.visual.vertex_colors = vc
        return pyrender.Mesh.from_trimesh(c, smooth=False)
    return col(P["body"]), [col(P["blade"]) for _ in range(4)], P["tips"]


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

    # cave: strong ambient + several FIXED directional lights (directional light does not fall off with distance,
    # so the whole cavern reads, not just the patch a headlamp hits) from varied angles for relief. One extra light
    # follows the camera as a headlamp for local contrast.
    cave_sc = new_scene(0.55)
    cave_node = cave_sc.add(pyrender.Mesh.from_trimesh(_load_cave(), smooth=False))
    cave_cam = cave_sc.add(pyrender.PerspectiveCamera(yfov=yf, aspectRatio=W / H, znear=0.05))
    for d in ([0.3, 0.3, 1.0], [-0.6, 0.2, -0.8], [0.5, -0.7, 0.3]):     # fixed fills (direction = -z of pose)
        cave_sc.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=2.2),
                    pose=_look_at(-np.array(d) * 5, [0, 0, 0], [0, 0, 1]))
    cave_light = cave_sc.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=2.0))   # camera headlamp

    def drone_scene(green, amb):
        sc = new_scene(amb)
        body, blades, tips = _mk_drone(green)
        bn = sc.add(body); bl = [sc.add(b) for b in blades]
        cam = sc.add(pyrender.PerspectiveCamera(yfov=yf, aspectRatio=W / H, znear=0.02))
        light = sc.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=4.0))
        return dict(sc=sc, body=bn, blades=bl, tips=tips, cam=cam, light=light)

    scn = dict(cave=dict(sc=cave_sc, node=cave_node, cam=cave_cam, light=cave_light),
               drone=drone_scene(False, 0.55), ghost=drone_scene(True, 0.7),
               rend=pyrender.OffscreenRenderer(int(W), int(H)))
    _CACHE["scn"] = scn; _CACHE["scn_key"] = key
    return scn


def _set_drone(d, p, R, blade_angle):
    sc = d["sc"]
    sc.set_pose(d["body"], _world(p, R))
    for i, bn in enumerate(d["blades"]):
        sc.set_pose(bn, _world(p, R, _spin_T(d["tips"][i], blade_angle)))


def render_view(eye, fwd, up, W, H, hfov_deg, p, R, blade_angle, goal_p, Rg,
                show_drone=True, show_ghost=True, ghost_a=0.25, ceil_cut=None):
    """Render cave + goal-ghost + drone from (eye,fwd,up) and DEPTH-composite -> rgb (H,W,3) uint8.

    `ceil_cut` (m): for the top-down view, clip the cave nearer than this distance from the eye so the ceiling
    above the drone is removed and the overhead camera sees inside (the drone/ghost are not clipped).
    `show_ghost`: the FPV is the drone's own camera -- it gets no ghost (at the goal it would fill the frame green);
    the external chase/top views show the ghost as the target marker."""
    import pyrender
    scn = _scenes(W, H, hfov_deg)
    rend = scn["rend"]
    cp = _cam_pose(eye, fwd, up)
    # SKIP_CULL_FACES: the cave is a one-sided surface whose normals face OUT, so from inside the cavern the walls
    # are back-faces and get culled (the scene reads as mostly black). Render double-sided so the walls show.
    flags = pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SKIP_CULL_FACES

    # cave (optionally ceiling-sliced via a near-plane cut)
    cav = scn["cave"]
    cav["cam"].camera.znear = float(ceil_cut) if ceil_cut else 0.05
    cav["sc"].set_pose(cav["cam"], cp); cav["sc"].set_pose(cav["light"], cp)
    c_rgba, c_depth = rend.render(cav["sc"], flags=flags)
    out = c_rgba[..., :3].astype(np.float32)
    has_cave = c_depth > 0

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
    parts = [P["body"]]
    for tip in P["tips"]:
        b = P["blade"].copy(); b.vertices += tip; parts.append(b)
    full = trimesh.util.concatenate(parts)
    ext = full.extents; diag = float(np.linalg.norm(ext))
    axis = trimesh.creation.axis(origin_size=0.004, axis_length=0.12, axis_radius=0.0025)
    sc = pyrender.Scene(bg_color=[18, 18, 22, 255], ambient_light=[0.5, 0.5, 0.5])
    sc.add(pyrender.Mesh.from_trimesh(full, smooth=False))
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
