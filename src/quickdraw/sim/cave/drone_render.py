"""Cave control rendering: HYBRID splat cave + crisp drone + goal ghost, depth-composited.

The CAVE is the real scene from `CaveScene.render_rgbd` -- the Gaussian SPLAT reconstruction where we have
coverage, with the grapevine mesh (rock browns/greys, slightly darker) as the baked-lit fallback everywhere
else. The DRONE and GHOST are the real SISL quadrotor MESH (body + 4 separable props at the arm tips, spinning;
translucent solid-green ghost at the goal), rendered crisply in pyrender and composited OVER the cave by DEPTH
(both depths in grapevine metres), so walls occlude the drone correctly. The FPV is the drone's own camera, so
it sees the splat cave exactly as the world model was trained on it.

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

    def drone_scene(green, amb):
        sc = new_scene(amb)
        body, logo, blades, tips = _mk_drone(green)
        bn = sc.add(body); ln_ = sc.add(logo); bl = [sc.add(b) for b in blades]
        cam = sc.add(pyrender.PerspectiveCamera(yfov=yf, aspectRatio=W / H, znear=0.02))
        light = sc.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=4.0))
        return dict(sc=sc, body=bn, logo=ln_, blades=bl, tips=tips, cam=cam, light=light)

    # Only the drone + ghost are pyrender meshes now; the cave comes from the splat (scene.render_rgbd).
    scn = dict(drone=drone_scene(False, 0.55), ghost=drone_scene(True, 0.7),
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
    flags = pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SKIP_CULL_FACES

    # cave from the splat+mesh hybrid; depth in grapevine metres (inf where no geometry). ceil_cut slices the roof.
    c_rgb, c_depth = scene.render_rgbd(eye, fwd, up, W=W, H=H, hfov=hfov_deg, ceil_cut=ceil_cut)
    out = c_rgb.astype(np.float32)
    has_cave = np.isfinite(c_depth)

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
