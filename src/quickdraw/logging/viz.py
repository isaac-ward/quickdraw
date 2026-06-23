"""Torus visualization — single backend (PyVista/VTK offscreen, pixel-perfect z-buffer).

Atlas = GridSpec(4,3): a 3x3 isometric render on top + three 1x1 orthographic axial renders below.
Axial panels are `imshow`n with a data extent so matplotlib draws ticks/numbers/labels (its font).
Surface coloring is a high-res TEXTURE MAP (sharp): segmented hsv bands + an optional grid of grey
squares (visual-OOD). A black arrow at each particle shows its net velocity (all views except FPV).
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

DPI = 220
_PAD = 1.3
N_SEG = 16          # discrete hue bands around the ring
FPV_FOV = 103.5     # egocentric camera FOV (deg); VTK default is 30
_N_THETA, _N_PHI = 420, 210   # torus face density (smooth even up close in FPV)
_TEX: dict = {}

CAPTIONS = {
    "manifold_distance_error": "|signed_dist(p̂)|  ·  how far the prediction floated off the torus (0 = on-manifold)",
    "pointwise_error": "‖p̂ − p‖  ·  distance to the true point at the same step",
    "tangent_velocity_error": "|⟨ṗ̂, n̂(p̂)⟩|  ·  predicted velocity pointing off the surface",
}
_AXIAL_VIEWS = [("x", "y", "z"), ("y", "x", "z"), ("z", "x", "y")]  # (view axis, xlabel, ylabel)


# ------------------------- geometry helpers -------------------------
def _normal_from_point(p, R):
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    th = np.arctan2(y, x)
    ph = np.arctan2(z, np.sqrt(x * x + y * y) - R)
    return np.stack([np.cos(th) * np.cos(ph), np.sin(th) * np.cos(ph), np.sin(ph)], axis=-1)


def action_ambient(p, action, R, r):
    """Map an angular action (a_theta, a_phi) to its ambient tangential push at point p.
    p: (...,3), action: (...,2) -> (...,3). This is what the arrow visualizes (the applied action)."""
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    th = np.arctan2(y, x)
    ph = np.arctan2(z, np.sqrt(x * x + y * y) - R)
    rho = R + r * np.cos(ph)
    dp_dth = np.stack([-rho * np.sin(th), rho * np.cos(th), np.zeros_like(th)], axis=-1)
    dp_dph = np.stack([-r * np.sin(ph) * np.cos(th), -r * np.sin(ph) * np.sin(th), r * np.cos(ph)], axis=-1)
    return action[..., 0:1] * dp_dth + action[..., 1:2] * dp_dph


def _offset_out(xyz, R, r, frac=0.02):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    th = np.arctan2(y, x)
    nx, ny, nz = x - R * np.cos(th), y - R * np.sin(th), z
    n = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2) + 1e-9
    eps = frac * r
    return np.stack([x + eps * nx / n, y + eps * ny / n, z + eps * nz / n], axis=1)


def _texture_array(coloring, w=2048, h=1024):
    key = (coloring, w, h)
    if key in _TEX:
        return _TEX[key]
    u = np.linspace(0, 1, w, endpoint=False)
    v = np.linspace(0, 1, h, endpoint=False)
    U, V = np.meshgrid(u, v)
    seg = np.floor(U * N_SEG) / N_SEG
    rgb = plt.cm.hsv(seg)[..., :3]
    if coloring == "circles":  # grey circles centered in each color band, all the way around the tube
        Rb, rb, M, rho = 0.75, 0.25, 8, 0.06  # base aspect -> circles look round on the base surface
        cu = (np.floor(U * N_SEG) + 0.5) / N_SEG  # band (color-ring) centers in theta
        cv = (np.floor(V * M) + 0.5) / M          # evenly around the tube in phi
        du = (U - cu) * 2 * math.pi * Rb
        dv = (V - cv) * 2 * math.pi * rb
        d = np.sqrt(du * du + dv * dv)
        alpha = np.clip((rho - d) / 0.006, 0.0, 1.0)[..., None]  # smooth (anti-aliased) rim
        rgb = rgb * (1 - alpha) + 0.5 * alpha                    # blend toward grey
    arr = (rgb * 255).astype(np.uint8)
    _TEX[key] = arr
    return arr


# ------------------------- pyvista scene -------------------------
def _pv():
    import pyvista as pv

    pv.OFF_SCREEN = True
    return pv


def _torus(pv, R, r):
    th = np.linspace(0, 2 * math.pi, _N_THETA)
    ph = np.linspace(0, 2 * math.pi, _N_PHI)
    TH, PH = np.meshgrid(th, ph)
    X = (R + r * np.cos(PH)) * np.cos(TH)
    Y = (R + r * np.cos(PH)) * np.sin(TH)
    Z = r * np.sin(PH)
    return pv.StructuredGrid(X, Y, Z), TH.ravel(order="F"), PH.ravel(order="F")


def _add_torus(pl, pv, R, r, coloring):
    grid, thf, phf = _torus(pv, R, r)
    grid.active_texture_coordinates = np.c_[thf / (2 * math.pi), phf / (2 * math.pi)].astype(np.float32)
    pl.add_mesh(grid, texture=pv.Texture(_texture_array(coloring)), show_scalar_bar=False)


def _add_trajs(pl, pv, R, r, trajs, markers=True):
    sc = R + r
    for t in trajs:
        p = _offset_out(np.asarray(t["xyz"]), R, r)
        c = t.get("color", "k")
        if len(p) >= 2:
            pl.add_mesh(pv.lines_from_points(p), color=c, line_width=3)
        if markers:  # start/end markers only in the static summary plot, not the videos
            pl.add_mesh(pv.Sphere(radius=0.035 * sc, center=p[0]), color=c)
            pl.add_mesh(pv.Sphere(radius=0.055 * sc, center=p[-1]), color=c)


def _add_arrows(pl, pv, R, r, arrows):
    """Black applied-action arrows. arrows: list of (point3, action_ambient3). FIXED absolute size
    (same on every torus); raised slightly along the normal to avoid z-fighting. Only LENGTH varies."""
    shaft_r, tip_r, tip_len = 0.01, 0.028, 0.14  # thinner diameter; length unchanged
    for pt, vel in arrows:
        vel = np.asarray(vel, float)
        s = float(np.linalg.norm(vel))
        if s < 1e-6:
            continue
        pt = np.asarray(pt, float)
        n = _normal_from_point(pt[None], R)[0]
        n = n / (np.linalg.norm(n) + 1e-9)
        pt = pt + 0.02 * n  # SLIGHTLY above the surface
        length = float(np.clip(s * 0.15, 0.2, 0.6))
        pl.add_mesh(pv.Arrow(start=pt, direction=vel / s, scale=length,
                             tip_length=tip_len / length, tip_radius=tip_r / length,
                             shaft_radius=shaft_r / length), color="black")


def _render(pv, R, r, coloring, trajs, targets, arrows, view, size, markers=True):
    pl = pv.Plotter(off_screen=True, window_size=(size, size))
    pl.set_background("white")
    _add_torus(pl, pv, R, r, coloring)
    _add_trajs(pl, pv, R, r, trajs, markers=markers)
    _add_arrows(pl, pv, R, r, arrows)
    if targets:
        tp = np.array([np.asarray(pp, float) for _, pp in targets])
        for pp in tp:
            pl.add_mesh(pv.Sphere(radius=0.12 * max(r, 0.12), center=pp), color="black")
        pl.add_point_labels(tp, [n for n, _ in targets], font_size=10, text_color="black",
                            shape=None, show_points=False, always_visible=True)
    L = (R + r) * _PAD
    if view == "iso":
        # back-of-cube only: front faces culled so their edges don't cross the torus; white faces
        # (no lighting) blend into the background, leaving just the black back edges.
        pl.add_mesh(pv.Box(bounds=(-L, L, -L, L, -L, L)), color="white", show_edges=True,
                    edge_color="black", line_width=1.4, culling="front", lighting=False)
        # x/y/z labels at the axis-end CORNERS (world-aligned, so they agree with the axial views)
        c = 1.12 * L
        pl.add_point_labels(np.array([(c, -L, -L), (-L, c, -L), (-L, -L, c)]), ["x", "y", "z"],
                            font_size=18, text_color="black", shape=None, show_points=False,
                            always_visible=True)
        pl.camera_position = "iso"
    else:
        pl.enable_parallel_projection()
        # right = +first label axis, up = +second (no mirror) so axial labels match the world
        pos = {"z": (0, 0, 4 * L), "y": (0, 4 * L, 0), "x": (-4 * L, 0, 0)}[view]
        up = {"z": (0, 1, 0), "y": (0, 0, 1), "x": (0, 0, 1)}[view]
        pl.camera_position = [pos, (0, 0, 0), up]
        pl.camera.parallel_scale = L
    img = pl.screenshot(return_img=True)
    pl.close()
    return img


# ------------------------- public: static atlas -------------------------
def _smooth_seq(seq, alpha=0.12):
    out = np.array(seq, dtype=float)
    for t in range(1, len(out)):
        out[t] = (1 - alpha) * out[t - 1] + alpha * out[t]
    return out


def fig_torus_atlas(R, r, trajs=(), targets=None, arrows=(), coloring="hsv", title="", legend=False,
                    markers=True, iso_size=860, ax_size=580):
    pv = _pv()
    L = (R + r) * _PAD
    iso = _render(pv, R, r, coloring, trajs, targets, arrows, "iso", iso_size, markers=markers)
    fig = plt.figure(figsize=(13, 15))
    gs = GridSpec(4, 3, figure=fig, wspace=0.5, hspace=0.25)
    axm = fig.add_subplot(gs[0:3, :])
    axm.imshow(iso)
    axm.axis("off")
    if legend and any(t.get("label") for t in trajs):
        axm.legend(handles=[Line2D([0], [0], color=t["color"], label=t["label"])
                            for t in trajs if t.get("label")], loc="upper right")
    for col, (view, xl, yl) in enumerate(_AXIAL_VIEWS):
        img = _render(pv, R, r, coloring, trajs, targets, arrows, view, ax_size, markers=markers)
        axp = fig.add_subplot(gs[3, col])
        axp.imshow(img, extent=[-L, L, -L, L])
        axp.set_aspect("equal")
        axp.set_xlabel(xl)
        axp.set_ylabel(yl)
    return fig


def fig_error_vs_step(errors: dict[str, np.ndarray]):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, series in errors.items():
        ax.plot(series, label=name)
    ax.set_xlabel("rollout step")
    ax.set_ylabel("error")
    ax.legend()
    cap = "\n".join(CAPTIONS[k] for k in errors if k in CAPTIONS)
    fig.text(0.01, -0.02, cap, fontsize=7, va="top")
    fig.tight_layout()
    return fig


# ------------------------- public: videos -------------------------
def save_mp4(path, frames, fps):
    import imageio.v2 as imageio

    imageio.mimwrite(path, list(frames), fps=max(1, int(round(fps))), macro_block_size=1)


def _fig_rgb(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    return np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(h, w, 4)[..., :3].copy()


def animate_frames(R, r, coloring, trajs, title="", n_frames=10000):
    """Each frame is the SAME `fig_torus_atlas` as the static plot (identical layout/title), with a
    growing black trail + particle + a smoothed (tweened) applied-action arrow. One frame per sim
    step (n_frames is just a safety cap) -> played at a constant 60 fps = real time."""
    tail = 60  # only the last 60 steps (~1 s) of trail are drawn, so it doesn't linger
    data = [(np.asarray(t["xyz"]),
             _smooth_seq(np.asarray(t["avec"])) if t.get("avec") is not None else None,
             t.get("color", "k")) for t in trajs]
    T = max(len(x) for x, _, _ in data)
    idx = np.linspace(2, T, min(n_frames, T)).astype(int)
    frames = []
    for ti in idx:
        k = int(ti)
        lo = max(0, k - tail)
        pt = [{"xyz": x[lo:k], "color": c} for x, _, c in data]
        arrows = [(x[k - 1], av[k - 1]) for x, av, _ in data if av is not None]
        fig = fig_torus_atlas(R, r, trajs=pt, arrows=arrows, coloring=coloring, title=title,
                              markers=False)  # no start/end markers in videos
        fig.set_dpi(90)
        frames.append(_fig_rgb(fig))
        plt.close(fig)
    return np.stack(frames)


def rollout_video(true_xyz, pred_xyz, R, r, coloring="hsv", n_frames=120):
    return animate_frames(R, r, coloring,
                          [{"xyz": true_xyz, "color": "tab:green"}, {"xyz": pred_xyz, "color": "tab:red"}], n_frames)


def fpv_frames(R, r, coloring, obs, n_frames=10000, fov=FPV_FOV, size=480):
    """Egocentric observation_image: camera at the particle, smoothed heading along tangential
    velocity, up = surface normal, configurable FOV. No velocity arrow here."""
    pv = _pv()
    obs = np.asarray(obs)
    T = len(obs)
    idx = np.linspace(1, T - 1, min(n_frames, T - 1)).astype(int)
    frames, sm = [], None
    for t in idx:
        p, v = obs[t, :3], obs[t, 3:]
        n = _normal_from_point(p[None], R)[0]
        n = n / (np.linalg.norm(n) + 1e-9)
        fwd = v - np.dot(v, n) * n
        nf = np.linalg.norm(fwd)
        cur = fwd / nf if nf > 1e-6 else (sm if sm is not None else np.array([1.0, 0.0, 0.0]))
        sm = cur if sm is None else (0.85 * sm + 0.15 * cur)
        sm = sm / (np.linalg.norm(sm) + 1e-9)
        pl = pv.Plotter(off_screen=True, window_size=(size, size))
        pl.set_background("white")
        _add_torus(pl, pv, R, r, coloring)
        pl.camera_position = [tuple(p + 0.06 * r * n), tuple(p + sm * 2 * r), tuple(n)]
        pl.camera.view_angle = float(fov)
        frames.append(pl.screenshot(return_img=True))
        pl.close()
    return np.stack(frames)
