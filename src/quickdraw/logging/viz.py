"""Torus visualization — single backend (PyVista/VTK offscreen, pixel-perfect z-buffer).

Atlas = GridSpec(4,3): a 3x3 isometric render on top + three 1x1 orthographic axial renders below.
Axial panels are `imshow`n with a data extent so matplotlib draws ticks/numbers/labels (its font).
Surface coloring is a high-res TEXTURE MAP (sharp): segmented hsv bands + an optional grid of grey
squares (visual-OOD). A black arrow at each particle shows its net velocity (all views except FPV).
"""

from __future__ import annotations

import math
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

DPI = 220
VIDEO_DPI = 110     # atlas/compare animation frame dpi (was 45 -> blurry; this is sharp, ~half the PNG)
_PAD = 1.3          # tight framing for summary plots (cube/torus fills the frame)
EVAL_VIEW_PAD = _PAD  # eval plots use the same tight ~±1.3 framing as summaries (fixed, no rescale)
N_SEG = 16          # discrete hue bands around the ring
FPV_FOV = 103.5     # egocentric camera FOV (deg); VTK default is 30
FPV_SIZE = 256      # egocentric video resolution (px, square)
SURFACE_EPS = 0.02  # absolute outward lift for trajectory lines/arrows (no z-fighting, any R,r)
ACTION_SMOOTH_WINDOW = 18  # default boxcar window for action-arrow smoothing (config can override)
TORUS_OPACITY = 0.6  # torus alpha for atlas plots/videos (see prediction through it); FPV stays opaque
_N_THETA, _N_PHI = 420, 210   # torus face density (smooth even up close in FPV)
_TEX: dict = {}

CAPTIONS = {
    "obs_vector_mse":
        "obs_vector_mse  ·  mean over the 6 obs dims of (ô − o)² (normalized — the training loss)  ·  "
        "how large is the whole-state prediction error per step?  ·  [0, ∞)",
    "manifold_distance_error":
        "manifold_distance_error  ·  ρ=√(x²+y²),  |signed_dist(p̂)|/r = |√((ρ−R)² + z²) − r| / r  ·  how far the "
        "predicted point floated off the torus surface, in tube-radii (dimensionless, ÷ this split's r)  ·  [0, ∞)",
    "pointwise_error":
        "pointwise_error  ·  ‖p̂ − p‖ (raw physical units)  ·  how far the predicted point is from the true point "
        "at the same step?  ·  [0, ∞)",
    "tangent_velocity_error":
        "tangent_velocity_error  ·  |⟨ṗ̂, n̂(p̂)⟩| / v_scale  ·  the predicted velocity's off-surface (normal) "
        "component, in characteristic speeds (dimensionless, ÷ this split's v_scale)  ·  [0, ‖ṗ̂‖/v_scale]",
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


def _offset_out(xyz, R, r, eps=SURFACE_EPS):
    # lift trajectory points a fixed ABSOLUTE distance along the outward normal so the line sits
    # clearly outside the surface (no z-fighting) on every torus, incl. the thin geometric-OOD one
    # (a frac*r offset would shrink to ~0.0016 at r=0.08 and sink into the tube).
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    th = np.arctan2(y, x)
    nx, ny, nz = x - R * np.cos(th), y - R * np.sin(th), z
    n = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2) + 1e-9
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
    # Empty meshes (e.g. a zero-length trajectory/arrow set on some eval frame) must be SKIPPED, not
    # crash the whole training run. Newer pyvista raises on empty meshes by default; opt back into the
    # old skip-silently behaviour. (Crashed eval_ood_horizon at epoch-0 eval after a pyvista bump.)
    pv.global_theme.allow_empty_mesh = True
    return pv


def _torus(pv, R, r):
    th = np.linspace(0, 2 * math.pi, _N_THETA)
    ph = np.linspace(0, 2 * math.pi, _N_PHI)
    TH, PH = np.meshgrid(th, ph)
    X = (R + r * np.cos(PH)) * np.cos(TH)
    Y = (R + r * np.cos(PH)) * np.sin(TH)
    Z = r * np.sin(PH)
    return pv.StructuredGrid(X, Y, Z), TH.ravel(order="F"), PH.ravel(order="F")


def _add_torus(pl, pv, R, r, coloring, opacity=1.0):
    grid, thf, phf = _torus(pv, R, r)
    grid.active_texture_coordinates = np.c_[thf / (2 * math.pi), phf / (2 * math.pi)].astype(np.float32)
    pl.add_mesh(grid, texture=pv.Texture(_texture_array(coloring)), show_scalar_bar=False, opacity=opacity)


def _add_trajs(pl, pv, R, r, trajs, markers=True, current=False):
    """Each traj: xyz (+ color, width). Markers: `markers` toggles the default start+end spheres; a
    traj can override with start_sphere / end_sphere (+ marker_color), or set `tip` = {color,
    lighting} for a sphere at the moving head (lighting=False -> flat, no specular)."""
    sc = R + r
    for t in trajs:
        p = _offset_out(np.asarray(t["xyz"]), R, r)
        c = t.get("color", "k")
        if len(p) >= 2:
            # tube (real 3D cylinder) instead of a GL line: depth-correct, so it blends cleanly with
            # the translucent torus (GL lines left white seams where they crossed the surface)
            tube = pv.lines_from_points(p).tube(radius=t.get("radius", 0.008 * sc), n_sides=12)
            pl.add_mesh(tube, color=c, lighting=False)  # flat, unlit -> uniform color, no specular
        mc = t.get("marker_color", c)
        if t.get("start_sphere", markers) and len(p):  # static summary plot defaults to both spheres
            pl.add_mesh(pv.Sphere(radius=0.045 * sc * t.get("start_scale", 1.0), center=p[0]),
                        color=mc, lighting=False)
        if t.get("end_sphere", markers) and len(p):
            pl.add_mesh(pv.Sphere(radius=0.045 * sc, center=p[-1]), color=mc, lighting=False)
        tip = t.get("tip")
        if tip and len(p):  # per-traj moving head sphere (compare video)
            pl.add_mesh(pv.Sphere(radius=0.04125 * sc, center=p[-1]), color=tip.get("color", c), lighting=False)
        if current and len(p):  # summary videos: a single sphere at the current position
            pl.add_mesh(pv.Sphere(radius=0.04125 * sc, center=p[-1]), color=c, lighting=False)


def _add_arrows(pl, pv, R, r, arrows):
    """Applied-action arrows. arrows: list of (point3, action_ambient3) or (point3, action_ambient3,
    color) -- color defaults to black. FIXED absolute size (same on every torus); raised slightly
    along the normal to avoid z-fighting. Only the SHAFT length varies with action magnitude; the
    head is a constant world-space length (tip_len/length cancels the scale)."""
    shaft_r, tip_r, tip_len = 0.01, 0.028, 0.07  # thinner diameter; head half as long, constant length
    for arr in arrows:
        pt, vel = arr[0], arr[1]
        color = arr[2] if len(arr) > 2 else "black"
        vel = np.asarray(vel, float)
        s = float(np.linalg.norm(vel))
        if s < 1e-6:
            continue
        pt = np.asarray(pt, float)
        n = _normal_from_point(pt[None], R)[0]
        n = n / (np.linalg.norm(n) + 1e-9)
        pt = pt + SURFACE_EPS * n  # SLIGHTLY above the surface
        length = float(np.clip(s * 0.15, 0.2, 0.6))
        pl.add_mesh(pv.Arrow(start=pt, direction=vel / s, scale=length,
                             tip_length=tip_len / length, tip_radius=tip_r / length,
                             shaft_radius=shaft_r / length), color=color)


def _tangent_ring(center, R, radius, n=48):
    """A closed loop of points in the tangent plane at `center` (on the torus), used as a target RING
    marker (rendered as a tube). Neutral black ring reads on any hue and is shape-distinct from a dot."""
    c = np.asarray(center, float)
    nrm = _normal_from_point(c[None], R)[0]
    nrm = nrm / (np.linalg.norm(nrm) + 1e-9)
    a = np.array([1.0, 0.0, 0.0]) if abs(nrm[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(nrm, a); u = u / (np.linalg.norm(u) + 1e-9)
    w = np.cross(nrm, u)
    t = np.linspace(0, 2 * math.pi, n)
    return c[None] + radius * (np.cos(t)[:, None] * u[None] + np.sin(t)[:, None] * w[None])


def _add_fan(pl, pv, R, r, fan, opacity=0.5, max_show=48):
    """MPPI candidate fan as thin TUBES (same primitive as the trail — depth-correct + blends cleanly,
    unlike GL lines which can't sub-pixel and don't alpha-blend) colored by return via RdYlGn -> high
    return = green = low cost. fan: {"pts": (K,H,3), "ret": (K,)}.
    SUBSAMPLE to max_show: hundreds of overlapping candidates read as an opaque blob; a sparse subset is
    what actually looks thin + transparent (the candidates are iid noise, so any subset is representative)."""
    pts = np.asarray(fan["pts"], float)              # (K,H,3)
    ret = np.asarray(fan["ret"], float)              # (K,)
    K, H = pts.shape[:2]
    if K == 0 or H < 2:
        return
    if K > max_show:                                 # evenly-spaced subset over the (unordered) candidates
        idx = np.linspace(0, K - 1, max_show).astype(int)
        pts, ret, K = pts[idx], ret[idx], max_show
    conn = np.empty((K, H + 1), dtype=np.int64)      # VTK polyline connectivity: [H, i0..i_{H-1}] per line
    conn[:, 0] = H
    conn[:, 1:] = np.arange(K * H).reshape(K, H)
    poly = pv.PolyData(pts.reshape(-1, 3))
    poly.lines = conn.ravel()
    poly["ret"] = np.repeat(ret, H).astype(np.float32)   # per-point scalar = its candidate's return
    tube = poly.tube(radius=0.0035 * (R + r), n_sides=6)  # ~half the trail thickness; cheap (6 sides)
    lo, hi = float(ret.min()), float(ret.max())
    pl.add_mesh(tube, scalars="ret", cmap="RdYlGn", clim=[lo, hi if hi > lo else lo + 1e-6],
                show_scalar_bar=False, lighting=False, opacity=opacity)


class TorusRenderer:
    """ONE persistent off-screen renderer shared by every video/figure producer. The legacy `_render`
    created a fresh `pv.Plotter` AND rebuilt the 88k-vertex torus mesh + texture on EVERY view of EVERY
    frame (4 views/frame for the atlas) — the dominant cost.

    This builds, ONCE per (view, size), a persistent plotter whose STATIC scene (torus mesh + texture +
    lights + the iso bounding box & x/y/z labels + the fixed camera/parallel-scale) is set up a single
    time. Per frame `view()` adds only the cheap DYNAMIC actors (fan, trajectory tubes, arrows, current
    sphere), screenshots, then removes exactly those — leaving the static scene untouched. Output is
    BYTE-IDENTICAL to the legacy code (proven by smoke/render_golden); only the per-frame plotter
    creation + mesh/box/label rebuild are skipped. (One plotter per VIEW, not just per size: reusing one
    plotter across different cameras leaves VTK state and breaks identity; a per-view plotter only ever
    renders its own fixed camera. Static actor ORDER vs the dynamic actors is irrelevant — verified.)"""

    def __init__(self, R, r, coloring, sizes=(860, 580), reuse=True):
        self.pv = _pv()
        self.R, self.r = R, r
        self.reuse = reuse   # True: persist a plotter per view + only swap dynamic actors (fast). False:
        #                      fresh plotter per view (legacy), still with the cached mesh. The reuse path
        #                      is byte-identical for trails/arrows; the ONE exception is control's stack of
        #                      overlapping translucent actors (agents + colored fan), where reuse leaves a
        #                      single-pixel +1/255 blend LSB — so control uses reuse=False to stay exact.
        grid, thf, phf = _torus(self.pv, R, r)   # cache the mesh + tex coords + texture (built once)
        grid.active_texture_coordinates = np.c_[thf / (2 * math.pi), phf / (2 * math.pi)].astype(np.float32)
        self._grid = grid
        self._tex = self.pv.Texture(_texture_array(coloring))
        self._cache = {}    # (view, size) -> (plotter, static_actor_names) when reuse=True

    def _add_torus(self, pl, opacity):
        pl.add_mesh(self._grid, texture=self._tex, show_scalar_bar=False, opacity=opacity)

    def _build(self, view, size, view_l, torus_opacity):
        """Build a plotter with the STATIC scene (torus + iso box/labels). Returns (plotter,
        static-actor-names, camera-params). The CAMERA is NOT set here — it's applied in view() AFTER the
        dynamic actors, exactly like the legacy code, so the clipping range is computed with all actors
        present (the off-manifold fan extends past the torus; setting the camera with only the torus in
        scene clips it ~1px differently). Per-video constants (view_l, torus_opacity) are baked in."""
        pv, R, r = self.pv, self.R, self.r
        pl = pv.Plotter(off_screen=True, window_size=(int(size), int(size)))
        pl.set_background("white")
        # order-independent transparency for the AXIAL views: without it VTK draws translucent actors in
        # ADD order, so the fan/trajectories paint on top of the translucent torus even when behind it.
        # Depth peeling blends by true depth (fan behind the torus correctly occluded/dimmed).
        pl.enable_depth_peeling(number_of_peels=4, occlusion_ratio=0.0)
        # The ISO torus is OPAQUE; the axial tori keep `torus_opacity` (translucent, see-through). The iso
        # saturation FLASH was VTK depth-peeling intermittently mis-resolving the translucent iso torus over
        # the moving fan (robust to peel count + plotter reuse — neither fixed it). An opaque iso has no
        # translucency to mis-blend, so the flash is gone; the see-through fan stays visible in the axials
        # (which never flickered — principal-axis sightline = few layers).
        self._add_torus(pl, 1.0 if view == "iso" else torus_opacity)
        L = (R + r) * _PAD          # torus reference bound (cube + axis labels)
        vl = view_l if view_l is not None else L  # FIXED view half-extent (>= L shows off-manifold drift)
        # orthographic everywhere + an explicit parallel_scale => framing is fixed, never auto-fit/rescaled
        pl.enable_parallel_projection()
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
            # in the iso projection the cube's top/bottom corners reach ~1.63L and the z label ~1.73L
            # *vertically*, so frame to 2.0L (with text margin) — tighter clips the top & bottom
            cam = ([(4 * vl, 4 * vl, 4 * vl), (0, 0, 0), (0, 0, 1)], max(vl, 2.0 * L))  # fixed iso dir
        else:
            # right = +first label axis, up = +second (no mirror) so axial labels match the world
            pos = {"z": (0, 0, 4 * vl), "y": (0, 4 * vl, 0), "x": (-4 * vl, 0, 0)}[view]
            up = {"z": (0, 1, 0), "y": (0, 0, 1), "x": (0, 0, 1)}[view]
            cam = ([pos, (0, 0, 0), up], vl)
        return pl, set(pl.actors.keys()), cam   # torus + box/labels are static

    def view(self, trajs, targets, arrows, view, size, markers=True, current=False, view_l=None,
             torus_opacity=1.0, fan=None):
        pv, R, r = self.pv, self.R, self.r
        key = (view, int(size))
        if self.reuse:
            if key not in self._cache:
                self._cache[key] = self._build(view, int(size), view_l, torus_opacity)
            pl, static, cam = self._cache[key]
        else:
            pl, static, cam = self._build(view, int(size), view_l, torus_opacity)
        # per-frame DYNAMIC actors only (fan BENEATH trails/arrows so the agents stay on top)
        if fan is not None:
            _add_fan(pl, pv, R, r, fan)
        _add_trajs(pl, pv, R, r, trajs, markers=markers, current=current)
        _add_arrows(pl, pv, R, r, arrows)
        if targets:
            tp = np.array([np.asarray(pp, float) for _, pp in targets])
            for pp in tp:
                pl.add_mesh(pv.Sphere(radius=0.12 * max(r, 0.12), center=pp), color="black", lighting=False)
            pl.add_point_labels(tp, [n for n, _ in targets], font_size=10, text_color="black",
                                shape=None, show_points=False, always_visible=True)
        pl.camera_position = cam[0]            # camera LAST (after all actors) -> clipping includes the
        pl.camera.parallel_scale = cam[1]      # off-manifold fan, byte-matching the legacy _render
        img = pl.screenshot(return_img=True)
        if self.reuse:
            for name in set(pl.actors.keys()) - static:   # remove ONLY this frame's dynamic actors
                pl.remove_actor(name, render=False)
        else:
            pl.close()
        return img

    def close(self):
        for entry in self._cache.values():
            entry[0].close()


# ------------------------- public: static atlas -------------------------
def _smooth_seq(seq, alpha=0.12):
    out = np.array(seq, dtype=float)
    for t in range(1, len(out)):
        out[t] = (1 - alpha) * out[t - 1] + alpha * out[t]
    return out


def _moving_avg(seq, win=ACTION_SMOOTH_WINDOW):
    """Centered boxcar moving average (edge-padded) over a small window, per channel. Used to keep
    the action arrows from whipping around frame-to-frame without adding the lag of a causal EMA."""
    seq = np.asarray(seq, dtype=float)
    if win <= 1 or len(seq) < 2:
        return seq.copy()
    k = win // 2
    pad = np.pad(seq, ((k, k), (0, 0)), mode="edge")
    kernel = np.ones(win) / win
    return np.stack([np.convolve(pad[:, j], kernel, mode="valid") for j in range(seq.shape[1])], axis=1)


def fig_torus_atlas(R, r, trajs=(), targets=None, arrows=(), coloring="hsv", title="", legend=False,
                    markers=True, current=False, iso_size=860, ax_size=580, view_pad=_PAD, torus_opacity=1.0,
                    fan=None, renderer=None):
    # renderer: pass a persistent TorusRenderer to reuse across frames (videos); None -> make + close one
    # for this single figure (static plots). Either way the per-view output is identical.
    own = renderer is None
    rend = renderer if renderer is not None else TorusRenderer(R, r, coloring, sizes=(iso_size, ax_size))
    vl = (R + r) * view_pad  # fixed view half-extent (shared by the render camera and the axial ticks)
    iso = rend.view(trajs, targets, arrows, "iso", iso_size, markers=markers,
                    current=current, view_l=vl, torus_opacity=torus_opacity, fan=fan)
    fig = plt.figure(figsize=(13, 15))
    gs = GridSpec(4, 3, figure=fig, wspace=0.5, hspace=0.25)
    axm = fig.add_subplot(gs[0:3, :])
    axm.imshow(iso)
    axm.axis("off")
    if legend and any(t.get("label") for t in trajs):
        axm.legend(handles=[Line2D([0], [0], color=t["color"], label=t["label"])
                            for t in trajs if t.get("label")], loc="upper right")
    for col, (view, xl, yl) in enumerate(_AXIAL_VIEWS):
        img = rend.view(trajs, targets, arrows, view, ax_size, markers=markers, current=current,
                        view_l=vl, torus_opacity=torus_opacity, fan=fan)
        axp = fig.add_subplot(gs[3, col])
        axp.imshow(img, extent=[-vl, vl, -vl, vl])
        axp.set_aspect("equal")
        axp.set_xlabel(xl)
        axp.set_ylabel(yl)
    if own:
        rend.close()
    return fig


def fig_error_vs_step(errors: dict[str, np.ndarray], colors: dict[str, str] | None = None, vlines=None,
                      yscale: str = "log"):
    fig, ax = plt.subplots(figsize=(11, 6))  # wide enough that the long metric captions don't clip
    for name, series in errors.items():
        ax.plot(series, label=name, color=(colors or {}).get(name))
    # vlines: {color: [step indices]} -> dotted verticals marking events (e.g. goal switches). These
    # explain the sharp jumps in dist-to-current-goal: the distance re-targets when the goal advances.
    for color, steps in (vlines or {}).items():
        for s in steps:
            ax.axvline(float(s), color=color, linestyle=":", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("rollout step")
    ax.set_ylabel(f"error ({yscale} scale)" if yscale == "log" else "distance")
    ax.set_yscale(yscale)  # log spreads small early + late blow-up; linear for bounded curves (control dist)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    # caption goes INSIDE the figure (reserve bottom margin) — wandb.Image ignores bbox_inches="tight",
    # so anything placed below y=0 gets clipped in the wandb logs. Wrap each caption to the figure
    # width so long lines fold onto new lines (readable in full) instead of running past the right edge;
    # grow the bottom margin to fit however many wrapped lines result.
    lines = []
    for k in errors:
        if k in CAPTIONS:
            lines += textwrap.wrap(CAPTIONS[k], width=120, subsequent_indent="      ") or [CAPTIONS[k]]
    fig.subplots_adjust(bottom=min(0.55, 0.10 + 0.028 * len(lines)))
    fig.text(0.02, 0.02, "\n".join(lines), fontsize=7, va="bottom")
    return fig


# ------------------------- public: videos -------------------------
def save_mp4(path, frames, fps, quality=9):
    import imageio.v2 as imageio

    # macro_block_size=2: pad odd dims up to even (libx264 requires divisible-by-2), no 16-px padding.
    # quality (0-10, higher = sharper/larger): default high so summary videos aren't mushy.
    imageio.mimwrite(path, list(frames), fps=max(1, int(round(fps))), macro_block_size=2, quality=quality)


def stitch_grid_video(paths, out_path, grid, fps):
    """Tile `grid`x`grid` already-rendered mp4s into one composite, frame by frame (streaming, so it
    never holds more than one frame per source in memory). All sources must share resolution; the
    composite runs to the shortest source. Cells fill row-major; missing cells stay black."""
    import imageio.v2 as imageio

    paths = list(paths)[: grid * grid]
    readers = [imageio.get_reader(p) for p in paths]
    try:
        probe = readers[0].get_data(0)
        h, w = probe.shape[:2]
        writer = imageio.get_writer(out_path, fps=max(1, int(round(fps))), macro_block_size=2)
        for frames in zip(*(iter(rd) for rd in readers)):  # lockstep; stops at the shortest source
            canvas = np.zeros((grid * h, grid * w, 3), np.uint8)
            for i, fr in enumerate(frames):
                rr, cc = divmod(i, grid)
                canvas[rr * h : (rr + 1) * h, cc * w : (cc + 1) * w] = fr[..., :3]
            writer.append_data(canvas)
        writer.close()
    finally:
        for rd in readers:
            rd.close()


def _fig_rgb(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    return np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(h, w, 4)[..., :3].copy()


def animate_frames(R, r, coloring, trajs, title="", n_frames=10000, smooth_window=ACTION_SMOOTH_WINDOW,
                   torus_opacity=TORUS_OPACITY, log=None):
    """Each frame is the SAME `fig_torus_atlas` as the static plot (identical layout/title), with a
    growing black trail + particle + a smoothed (tweened) applied-action arrow. One frame per sim
    step (n_frames is just a safety cap) -> played at a constant 60 fps = real time."""
    tail = 60  # only the last 60 steps (~1 s) of trail are drawn, so it doesn't linger
    data = [(np.asarray(t["xyz"]),
             _moving_avg(np.asarray(t["avec"]), smooth_window) if t.get("avec") is not None else None,
             t.get("color", "k")) for t in trajs]
    T = max(len(x) for x, _, _ in data)
    idx = np.linspace(2, T, min(n_frames, T)).astype(int)
    frames = []
    every = max(1, len(idx) // 8)  # progress ~every 12% of frames
    rend = TorusRenderer(R, r, coloring)  # one persistent renderer reused across all frames
    try:
        for fi, ti in enumerate(idx):
            if log is not None and fi % every == 0:
                log(f"rendered {fi}/{len(idx)} frames")
            k = int(ti)
            lo = max(0, k - tail)
            pt = [{"xyz": x[lo:k], "color": c} for x, _, c in data]
            arrows = [(x[k - 1], av[k - 1]) for x, av, _ in data if av is not None]
            fig = fig_torus_atlas(R, r, trajs=pt, arrows=arrows, coloring=coloring, title=title,
                                  markers=False, current=True, torus_opacity=torus_opacity, renderer=rend)  # current-position sphere only
            fig.set_dpi(VIDEO_DPI)
            frames.append(_fig_rgb(fig))
            plt.close(fig)
    finally:
        rend.close()
    return np.stack(frames)


def traj_compare_frames(R, r, coloring, true_full, pred_full, avec_true, P, n_frames=120, title="",
                        smooth_window=ACTION_SMOOTH_WINDOW, torus_opacity=TORUS_OPACITY, log=None):
    """Animated truth-vs-prediction. One black sphere (with the applied-action arrow) rides the TRUE
    path (context then ground truth); at the fork (step P) a red sphere (no arrow, flat-shaded) splits
    off along the PREDICTED path. Both carry a solid trailing tail (not persistent). Mirrors the PNG,
    where the tails are instead persistent. true_full/pred_full: (T,3), identical for the first P
    steps then diverging. avec_true: (T,2)->ambient applied action along the true path."""
    tail = 60
    true_full, pred_full = np.asarray(true_full), np.asarray(pred_full)
    avec = _moving_avg(np.asarray(avec_true), smooth_window)
    T = len(true_full)
    idx = np.linspace(2, T, min(n_frames, T)).astype(int)
    frames = []
    every = max(1, len(idx) // 10)  # progress every ~10% of frames
    rend = TorusRenderer(R, r, coloring)  # one persistent renderer reused across all frames
    try:
        for fi, ti in enumerate(idx):
            if log is not None and fi % every == 0:
                log(f"rendered {fi}/{len(idx)} frames")
            k, lo = int(ti), max(0, int(ti) - tail)
            trajs = [{"xyz": true_full[lo:k], "color": "black", "tip": {"color": "black"}}]  # truth: black + arrow
            arrows = [(true_full[k - 1], avec[k - 1])]
            if k >= P:  # after the fork: predicted in dark grey, flat-shaded (no specular), no arrow
                trajs.append({"xyz": pred_full[max(lo, P - 1) : k], "color": "dimgray",
                              "tip": {"color": "dimgray", "lighting": False}})
            fig = fig_torus_atlas(R, r, trajs=trajs, arrows=arrows, coloring=coloring, title=title,
                                  markers=False, view_pad=EVAL_VIEW_PAD,  # wide fixed view: red drift stays visible
                                  torus_opacity=torus_opacity, renderer=rend)
            fig.set_dpi(VIDEO_DPI)
            frames.append(_fig_rgb(fig))
            plt.close(fig)
    finally:
        rend.close()
    return np.stack(frames)


def control_compare_frames(R, r, coloring, agents, n_frames=10000, title="",
                           smooth_window=ACTION_SMOOTH_WINDOW, torus_opacity=TORUS_OPACITY, fan_seq=None,
                           log=None, reuse=False):
    """Animated dual-controller race (eval_control). Each agent = {path (T,3), avec (T,3) ambient
    applied action, goal_seq (T,3) its current goal, color}. Per frame each agent gets a flat moving
    head + trailing tail + a colored action arrow, plus a same-color RING marking ITS current goal ZONE
    on the surface (so it's clear who targets what). Reuses fig_torus_atlas like traj_compare_frames, so
    layout/sizing are unchanged. fan_seq (optional, len ~T): per-step pred candidate fan to overlay."""
    tail = 60
    agents = [{"color": a["color"], "path": np.asarray(a["path"]), "goal_seq": np.asarray(a["goal_seq"]),
               "avec": _moving_avg(np.asarray(a["avec"]), smooth_window)} for a in agents]
    T = min(len(a["path"]) for a in agents)
    idx = np.linspace(2, T, min(n_frames, T)).astype(int)
    frames = []
    every = max(1, len(idx) // 10)  # progress every ~10% of frames
    sc = R + r
    # reuse: the iso saturation flash is fixed at the source by number_of_peels=12 in _build (the iso
    # diagonal pierces >4 translucent layers), so the FAST reuse=True path is stable even with the fan.
    rend = TorusRenderer(R, r, coloring, reuse=reuse)
    try:
        for fi, ti in enumerate(idx):
            if log is not None and fi % every == 0:
                log(f"rendered {fi}/{len(idx)} frames")
            k, lo = int(ti), max(0, int(ti) - tail)
            trajs, arrows = [], []
            for a in agents:
                c = a["color"]
                gi = min(k - 1, len(a["goal_seq"]) - 1)  # goal_seq/avec have one fewer entry than path
                ai = min(k - 1, len(a["avec"]) - 1)
                trajs.append({"xyz": a["path"][lo:k], "color": c, "tip": {"color": c, "lighting": False}})
                trajs.append({"xyz": _tangent_ring(a["goal_seq"][gi], R, 0.0675 * sc), "color": c,
                              "radius": 0.006 * sc, "start_sphere": False, "end_sphere": False})  # goal ring (1.5x agent diam)
                arrows.append((a["path"][k - 1], a["avec"][ai], c))
            fan = fan_seq[min(k - 1, len(fan_seq) - 1)] if fan_seq else None  # this step's candidate fan
            fig = fig_torus_atlas(R, r, trajs=trajs, arrows=arrows, coloring=coloring, title=title,
                                  markers=False, view_pad=EVAL_VIEW_PAD, torus_opacity=torus_opacity,
                                  fan=fan, renderer=rend)
            fig.set_dpi(VIDEO_DPI)
            frames.append(_fig_rgb(fig))
            plt.close(fig)
    finally:
        rend.close()
    return np.stack(frames)


def fpv_frames(R, r, coloring, obs, n_frames=10000, fov=FPV_FOV, size=FPV_SIZE):
    """Egocentric observation_image: camera at the particle, smoothed heading along tangential
    velocity, up = surface normal, configurable FOV. No velocity arrow here."""
    obs = np.asarray(obs)
    T = len(obs)
    # one frame per timestep (aligned 1:1 with obs, as the lerobot image observation needs); n_frames
    # only downsamples if explicitly smaller than T
    idx = np.arange(T) if n_frames >= T else np.linspace(0, T - 1, n_frames).astype(int)
    frames, sm = [], None
    # FPV camera moves EVERY frame; reusing one plotter across changing cameras leaves VTK state (not
    # pixel-identical), so use a FRESH plotter per frame — but with the renderer's CACHED torus mesh +
    # texture so we still skip the per-frame 88k-vertex rebuild. (FPV is 256px + the dataset render is
    # already process-parallel, so per-frame plotter creation here is acceptable.)
    rend = TorusRenderer(R, r, coloring)
    pv = rend.pv
    try:
        for t in idx:
            p, v = obs[t, :3], obs[t, 3:]
            n = _normal_from_point(p[None], R)[0]
            n = n / (np.linalg.norm(n) + 1e-9)
            fwd = v - np.dot(v, n) * n
            nf = np.linalg.norm(fwd)
            cur = fwd / nf if nf > 1e-6 else (sm if sm is not None else np.array([1.0, 0.0, 0.0]))
            sm = cur if sm is None else (0.85 * sm + 0.15 * cur)
            sm = sm / (np.linalg.norm(sm) + 1e-9)
            pl = pv.Plotter(off_screen=True, window_size=(int(size), int(size)))
            pl.set_background("white")
            rend._add_torus(pl, 1.0)
            pl.camera_position = [tuple(p + 0.06 * r * n), tuple(p + sm * 2 * r), tuple(n)]
            pl.camera.view_angle = float(fov)
            frames.append(pl.screenshot(return_img=True))
            pl.close()
    finally:
        rend.close()
    return np.stack(frames)


# ------------------------- diffusion: flow-field viz (design/models/diffusion.md) -------------------------
def diffusion_quiver_frames(R, r, coloring, current, action_amb, per_frame, agent_tail=None,
                            true_next=None, title="", size=860, view_pad=EVAL_VIEW_PAD,
                            torus_opacity=TORUS_OPACITY):
    """Short tau-sweep ANIMATION (tau 1->0, the denoising direction), rendered as the FULL 4-panel atlas
    (iso + 3 axial, like the control/OOD videos) so the see-through axial views are available. The torus +
    moving-agent tail + current dot + action arrow + a black TRUTH ring (the true next position) stay fixed;
    a GREY SWARM of decoded ODE paths flows from off-surface noise onto the torus EACH leaving its own
    growing tail (so the accumulating tails trace the flow field), and the RED committed particle rides its
    own path. per_frame is a list of {particle: (3,), trail: (k,3), swarm: [{particle, trail}..]}."""
    sc = R + r
    ring_r = 0.0225 * sc                                                 # truth ring diameter = 0.5x the agent diameter
    rend = TorusRenderer(R, r, coloring)                                 # reused across frames (atlas, like control)
    frames = []
    cur, act = np.asarray(current), np.asarray(action_amb)
    tail = np.asarray(agent_tail) if agent_tail is not None else None   # moving agent's trajectory tail (static)
    ring = _tangent_ring(true_next, R, ring_r) if true_next is not None else None  # black truth ring (static)
    try:
        for fr in per_frame:
            trajs = [{"xyz": cur[None], "color": "black", "start_sphere": True, "end_sphere": False}]
            if tail is not None and len(tail) >= 2:              # the MOVING AGENT's path up to now (like other plots)
                trajs.append({"xyz": tail, "color": "black", "radius": 0.008 * sc,
                              "start_sphere": False, "end_sphere": False})
            if ring is not None:                                 # ground-truth next position = small black ring
                trajs.append({"xyz": ring, "color": "black", "radius": 0.006 * sc,
                              "start_sphere": False, "end_sphere": False})
            for sp in fr.get("swarm", []):                       # swarm: grey, EACH with its own growing tail
                st = np.asarray(sp["trail"])                     # (traces the flow field as they accumulate)
                if len(st) >= 2:
                    trajs.append({"xyz": st, "color": "dimgray", "radius": 0.008 * sc,  # std line thickness
                                  "start_sphere": False, "end_sphere": False})
                else:
                    trajs.append({"xyz": np.asarray(sp["particle"])[None], "color": "dimgray",
                                  "start_sphere": True, "end_sphere": False, "start_scale": 0.35})
            trail = np.asarray(fr["trail"])                      # MAIN particle: keeps its TAIL (the denoising-path
            if len(trail) >= 2:                                  # history) + red head; std line thickness
                trajs.append({"xyz": trail, "color": "red", "radius": 0.008 * sc, "start_sphere": False,
                              "end_sphere": False, "tip": {"color": "red", "lighting": False}})
            else:
                trajs.append({"xyz": np.asarray(fr["particle"])[None], "color": "red",
                              "start_sphere": True, "end_sphere": False})
            fig = fig_torus_atlas(R, r, trajs=trajs, arrows=[(cur, act)], coloring=coloring, title=title,
                                  markers=False, iso_size=size, ax_size=int(round(size * 0.67)),
                                  view_pad=view_pad, torus_opacity=torus_opacity, renderer=rend)
            fig.set_dpi(VIDEO_DPI)
            frames.append(_fig_rgb(fig))
            plt.close(fig)
    finally:
        rend.close()
    return np.stack(frames)
