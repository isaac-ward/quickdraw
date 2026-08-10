"""Torus visualization — single backend (PyVista/VTK offscreen, pixel-perfect z-buffer).

Atlas = GridSpec(4,3): a 3x3 isometric render on top + three 1x1 orthographic axial renders below.
Axial panels are `imshow`n with a data extent so matplotlib draws ticks/numbers/labels (its font).
Surface coloring is a high-res TEXTURE MAP (sharp): segmented hsv bands + an optional grid of grey
squares (visual-OOD). A black arrow at each particle shows its net velocity (all views except FPV).
"""

from __future__ import annotations

import math
import textwrap
import time

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
TORUS_OPACITY = 0.6  # legacy default still passed by some callers; _build overrides it with the constants below
ISO_TORUS_OPACITY = 0.45    # iso torus opacity for EVERY atlas plot (horizon, control, diffusion, summaries)
AXIAL_TORUS_OPACITY = 0.3   # axial torus opacity (more see-through than iso). Both were 0.6 before.
#   The FAN is rendered OPAQUE (see _add_fan): a transparent fan over a transparent torus was the iso
#   depth-peeling flicker; an opaque fan is one solid layer, so the iso can stay translucent without the flash.
_N_THETA, _N_PHI = 420, 210   # torus face density (smooth even up close in FPV)
_TEX: dict = {}

CAPTIONS = {
    "obs_error":
        "obs_error  ·  mean over the 6 obs dims of (ô − o)² (normalized — the training loss)  ·  "
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
    # image-head rollout metrics (eval_ood_horizon/<head>/metric_vs_step_*): predicted FPV frame x̂ vs ground truth x.
    # psnr is drawn in the TOP panel (dB); ssim/mse/l1 share the BOTTOM [0,1] panel (same x-axis).
    "psnr":
        "psnr  ·  PSNR = −10·log₁₀(MSE),  MSE over pixels in [0,1] (peak=1)  ·  higher = sharper reconstruction  ·  "
        "TOP panel, dB, [0, ∞)",
    "ssim":
        "ssim  ·  SSIM = [(2μx̂μx + c₁)(2σx̂x + c₂)] / [(μx̂² + μx² + c₁)(σx̂² + σx² + c₂)] over local windows "
        "(μ,σ = per-window mean/(co)variance; c₁,c₂ stabilizers)  ·  1 = identical  ·  native [−1,1], shown clamped to [0,1]",
    "mse":
        "mse  ·  MSE = mean_pixels((x̂ − x)²),  x,x̂ ∈ [0,1]  ·  per-pixel L2 (= the image training loss)  ·  lower = closer  ·  [0, 1]",
    "l1":
        "l1  ·  L1 = mean_pixels(|x̂ − x|),  x,x̂ ∈ [0,1]  ·  per-pixel L1, less outlier-sensitive than MSE  ·  lower = closer  ·  [0, 1]",
    "lpips":
        "lpips  ·  LPIPS = Σ_layers ‖w_l ⊙ (φ_l(x̂) − φ_l(x))‖² over a SqueezeNet feature stack φ  ·  PERCEPTUAL "
        "distance: unlike psnr/ssim/mse/l1 (all pixelwise) it separates a prediction BLURRED toward the dataset "
        "mean from one that is sharp but wrong  ·  LOWER = closer (opposite direction to psnr/ssim)  ·  [0, ~1+]",
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


def _add_fan(pl, pv, R, r, fan, opacity=1.0, max_show=48):
    """MPPI candidate fan as thin TUBES (same primitive as the trail — depth-correct + blends cleanly,
    unlike GL lines which can't sub-pixel and don't alpha-blend) colored by return via RdYlGn -> high
    return = green = low cost. fan: {"pts": (K,H,3), "ret": (K,)}. OPAQUE (opacity=1.0): a transparent
    fan over the transparent torus was the iso depth-peeling flicker; one solid fan layer avoids it.
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
        # Atlas torus opacity: iso 0.45, axial 0.3 (the axials stay more see-through). Same for every atlas
        # plot, regardless of the caller's torus_opacity. The iso saturation FLASH came from VTK depth-peeling
        # mis-resolving a TRANSPARENT torus over a TRANSPARENT fan; making the FAN opaque (see _add_fan)
        # removes that transparent-on-transparent stress, so the iso can stay translucent.
        self._add_torus(pl, ISO_TORUS_OPACITY if view == "iso" else AXIAL_TORUS_OPACITY)
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
                      yscale: str = "log", split_top=None, split_bottom=None, caption: str | None = None,
                      linestyles=None, markers=None):
    """Curves vs rollout step, split into up to THREE stacked panels sharing ONE long x-axis, so metrics with
    incompatible ranges are not squashed together:

      TOP     `split_top`     unbounded, own units          (psnr, dB)
      MIDDLE  everything else bounded [0,1]                 (ssim, mse, l1)  -> ylim pinned to [0,1] on linear
      BOTTOM  `split_bottom`  unbounded, DIFFERENT direction (lpips, lower-is-better) -> floors at 0, grows

    LPIPS gets its own panel rather than sharing the bounded one: it is unbounded above (it exceeds 1 exactly
    on the badly-wrong predictions worth seeing, which a [0,1] axis would clip) and it runs the OPPOSITE
    direction to ssim, so overlaying them invites misreading. Any empty group is dropped. Captions (per-curve
    CAPTIONS, plus a free-form `caption`) render below the axes."""
    top = set(split_top or ()) & set(errors)
    bot = (set(split_bottom or ()) & set(errors)) - top
    top_keys = [k for k in errors if k in top]
    bot_keys = [k for k in errors if k in bot]
    mid_keys = [k for k in errors if k not in top and k not in bot]
    groups = [(g, ks) for g, ks in (("top", top_keys), ("mid", mid_keys), ("bottom", bot_keys)) if ks]
    if len(groups) > 1:                                          # EQUAL-height stacked panels, shared x
        fig, axes = plt.subplots(len(groups), 1, figsize=(11, 3.5 * len(groups)), sharex=True,
                                 gridspec_kw={"height_ratios": [1] * len(groups)})
        panels = [(axes[i], ks) for i, (_, ks) in enumerate(groups)]
        kinds = [g for g, _ in groups]
    else:                                                        # single panel
        fig, ax = plt.subplots(figsize=(11, 6))
        panels = [(ax, list(errors))]
        kinds = ["mid" if not groups else groups[0][0]]
    for ax_, keys in panels:
        marked = False
        for name in keys:
            ax_.plot(errors[name], label=name, color=(colors or {}).get(name),
                     linestyle=(linestyles or {}).get(name, "-"))
            if markers and name in markers:   # open circle (line's color) at each marked step, e.g. MPPI replans
                y = np.asarray(errors[name]); xs = [x for x in markers[name] if 0 <= x < len(y)]
                ax_.scatter(xs, y[xs], s=32, facecolors="none", edgecolors=(colors or {}).get(name), zorder=5, linewidths=1.2)
                marked = True
        ax_.set_yscale(yscale)  # log spreads small early + late blow-up; linear for bounded curves (control dist)
        ax_.grid(True, which="both", alpha=0.3)
        ax_.set_ylabel(f"({yscale} scale)" if yscale == "log" else "value")
        handles, labs = ax_.get_legend_handles_labels()
        if marked:                            # ONE generic key for the markers (black circle)
            from matplotlib.lines import Line2D
            handles.append(Line2D([0], [0], marker="o", color="black", linestyle="none",
                                  markerfacecolor="none", label="replanning step"))
        ax_.legend(handles=handles, loc="best")
    ax_bottom = panels[-1][0]
    if yscale == "linear":
        for (ax_, keys), kind in zip(panels, kinds):
            if kind == "mid" and len(panels) > 1:                # ssim/mse/l1 are genuinely bounded -> pin [0,1]
                ax_.set_ylim(0, 1)
            elif kind == "bottom":                               # lpips: floor at 0, grow only if a curve needs it
                _hi = max([float(np.nanmax(errors[k])) for k in keys if len(errors[k])] or [1.0])
                ax_.set_ylim(0, max(1.0, _hi * 1.05))
    # vlines: {color: [step indices]} -> dotted verticals marking events (e.g. goal switches), on the bottom panel
    for color, steps in (vlines or {}).items():
        for s in steps:
            ax_bottom.axvline(float(s), color=color, linestyle=":", linewidth=1.0, alpha=0.6)
    ax_bottom.set_xlabel("rollout step")
    # caption goes INSIDE the figure (reserve bottom margin) — wandb.Image ignores bbox_inches="tight",
    # so anything placed below y=0 gets clipped in the wandb logs. Wrap each caption to the figure
    # width so long lines fold onto new lines (readable in full) instead of running past the right edge;
    # grow the bottom margin to fit however many wrapped lines result.
    lines = []
    for k in errors:
        if k in CAPTIONS:
            lines += textwrap.wrap(CAPTIONS[k], width=190, subsequent_indent="      ") or [CAPTIONS[k]]
    for para in (caption.split("\n") if caption else []):        # free-form footer (e.g. failure decomposition)
        lines += textwrap.wrap(para, width=190, subsequent_indent="      ") or [para]
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


def tile_clips(clips, grid):
    """Array-input analogue of stitch_grid_video (SAME row-major layout): tile up to grid*grid clips, each
    (T,H,W,3) uint8, into one (T, grid*H, grid*W, 3) composite. Cells fill row-major; missing cells stay
    black; truncated to the shortest clip. For clips already in memory (no temp mp4s)."""
    clips = [np.asarray(c) for c in clips][: grid * grid]
    T = min(len(c) for c in clips)
    h, w = clips[0].shape[1:3]
    out = np.zeros((T, grid * h, grid * w, 3), np.uint8)
    for i, c in enumerate(clips):
        rr, cc = divmod(i, grid)
        out[:, rr * h:(rr + 1) * h, cc * w:(cc + 1) * w] = c[:T, ..., :3]
    return out


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
                           log=None, reuse=False, show_goals=True):
    """Animated dual-controller race (eval_control). Each agent = {path (T,3), avec (T,3) ambient
    applied action, goal_seq (T,3) its current goal, color}. Per frame each agent gets a flat moving
    head + trailing tail + a colored action arrow, plus a same-color RING marking ITS current goal ZONE
    on the surface (so it's clear who targets what). Reuses fig_torus_atlas like traj_compare_frames, so
    layout/sizing are unchanged. fan_seq (optional, len ~T): per-step pred candidate fan to overlay.
    show_goals=False drops the goal rings (language steering has no target point, only a reward direction)."""
    tail = 60
    agents = [{"color": a["color"], "path": np.asarray(a["path"]), "goal_seq": np.asarray(a["goal_seq"]),
               "avec": _moving_avg(np.asarray(a["avec"]), smooth_window)} for a in agents]
    T = min(len(a["path"]) for a in agents)
    idx = np.linspace(2, T, min(n_frames, T)).astype(int)
    frames = []
    every = max(1, len(idx) // 10)  # progress every ~10% of frames
    sc = R + r
    # reuse: the iso saturation flash is fixed at the source by number_of_peels=4 in _build (the iso
    # diagonal pierces several translucent layers), so the FAST reuse=True path is stable even with the fan.
    rend = TorusRenderer(R, r, coloring, reuse=reuse)
    _t0 = time.perf_counter()
    try:
        for fi, ti in enumerate(idx):
            if log is not None and fi % every == 0:
                log(_eta_str(_t0, fi, len(idx)))
            k, lo = int(ti), max(0, int(ti) - tail)
            trajs, arrows = [], []
            for a in agents:
                c = a["color"]
                gi = min(k - 1, len(a["goal_seq"]) - 1)  # goal_seq/avec have one fewer entry than path
                ai = min(k - 1, len(a["avec"]) - 1)
                trajs.append({"xyz": a["path"][lo:k], "color": c, "tip": {"color": c, "lighting": False}})
                if show_goals:
                    trajs.append({"xyz": _tangent_ring(a["goal_seq"][gi], R, 0.135 * sc), "color": c,
                                  "radius": 0.008 * sc, "start_sphere": False, "end_sphere": False})  # goal ring (2x'd 2026-08-05 to match control.tol=0.30)
                    #                                     diam; tube thickness == the main agent tail thickness)
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


class FPVRenderer:
    """FAST egocentric FPV for the control loop / previews: ONE persistent offscreen plotter + torus mesh
    (built once), then per state just move the camera and screenshot. ~order-of-magnitude faster than
    fpv_frames (which rebuilds plotter + mesh every frame). NOT byte-identical to the data renderer (VTK
    state persists across camera moves), which is fine for control. Maintains per-episode heading smoothing
    across successive render() calls, so a streaming control loop gets the same smoothed heading as data."""

    def __init__(self, R, r, coloring, fov=FPV_FOV, size=FPV_SIZE):
        self.R, self.r, self.fov = R, r, float(fov)
        self.rend = TorusRenderer(R, r, coloring)
        self.pl = self.rend.pv.Plotter(off_screen=True, window_size=(int(size), int(size)))
        self.pl.set_background("white")
        self.rend._add_torus(self.pl, 1.0)
        self.pl.camera.view_angle = self.fov
        self._sm = None                                  # per-episode smoothed heading (persists across calls)

    def render(self, obs):                               # obs (B,6) physical states -> (B,size,size,3) uint8
        obs = np.asarray(obs)
        B = len(obs)
        if self._sm is None or len(self._sm) != B:
            self._sm = [None] * B
        out = []
        for i in range(B):
            p, v = obs[i, :3], obs[i, 3:]
            n = _normal_from_point(p[None], self.R)[0]
            n = n / (np.linalg.norm(n) + 1e-9)
            fwd = v - np.dot(v, n) * n
            nf = np.linalg.norm(fwd)
            cur = fwd / nf if nf > 1e-6 else (self._sm[i] if self._sm[i] is not None else np.array([1.0, 0.0, 0.0]))
            self._sm[i] = cur if self._sm[i] is None else (0.85 * self._sm[i] + 0.15 * cur)
            self._sm[i] = self._sm[i] / (np.linalg.norm(self._sm[i]) + 1e-9)
            self.pl.camera_position = [tuple(p + 0.06 * self.r * n), tuple(p + self._sm[i] * 2 * self.r), tuple(n)]
            self.pl.camera.view_angle = self.fov
            self.pl.render()                                 # force VTK to apply the camera move; screenshot() alone reuses the prior render (frozen frames)
            out.append(self.pl.screenshot(return_img=True))
        return np.stack(out)

    def close(self):
        self.pl.close()
        self.rend.close()


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
def _quiver_static_trajs(cur, agent_tail, future_path, true_next, R, sc):
    """The fixed (non-animated) trajs for a quiver frame: current dot + the agent's HISTORY tail and FUTURE
    path (both black, std thickness) + a small black TRUTH ring. Shared by the single + multistep quivers."""
    out = [{"xyz": np.asarray(cur)[None], "color": "black", "start_sphere": True, "end_sphere": False}]
    for path in (agent_tail, future_path):
        p = np.asarray(path) if path is not None else None
        if p is not None and len(p) >= 2:
            out.append({"xyz": p, "color": "black", "radius": 0.008 * sc,
                        "start_sphere": False, "end_sphere": False})
    if true_next is not None:
        out.append({"xyz": _tangent_ring(true_next, R, 0.0225 * sc), "color": "black", "radius": 0.006 * sc,
                    "start_sphere": False, "end_sphere": False})
    return out


def _quiver_swarm_trajs(fr, sc):
    """The animated grey swarm trajs for one frame: each member a growing tail (tracing the flow field)."""
    out = []
    for sp in fr.get("swarm", []):
        st = np.asarray(sp["trail"])
        if len(st) >= 2:
            out.append({"xyz": st, "color": "dimgray", "radius": 0.008 * sc,
                        "start_sphere": False, "end_sphere": False})
        else:
            out.append({"xyz": np.asarray(sp["particle"])[None], "color": "dimgray",
                        "start_sphere": True, "end_sphere": False, "start_scale": 0.35})
    return out


def _eta_str(t0, done, total):
    """'{done}/{total} frames | elapsed Xs | ETA Ys (~HH:MM:SS)' for a render loop's progress callback."""
    el = time.perf_counter() - t0
    rem = el / max(1, done) * max(0, total - done)
    return f"{done}/{total} frames | elapsed {el:.0f}s | ETA {rem:.0f}s (~{time.strftime('%H:%M:%S', time.localtime(time.time() + rem))})"


def diffusion_quiver_sequential_frames(R, r, coloring, current, action_amb, agent_tail, future_path, steps,
                                       title="", size=860, view_pad=EVAL_VIEW_PAD, torus_opacity=TORUS_OPACITY, log=None):
    """N SEQUENTIAL swarms at a FIXED agent. The current dot, the action arrow, the black history tail and
    the black future path are STATIC the whole time; each step's grey swarm denoises (converges) to its own
    target ring — which advances along the fixed future line — one swarm after the next. steps: a list of
    {per_frame: [{swarm: [{particle, trail}..]}, ...], true_next: (3,)} (one denoising sub-animation each)."""
    sc = R + r
    rend = TorusRenderer(R, r, coloring)
    base = _quiver_static_trajs(current, agent_tail, future_path, None, R, sc)   # ring is per-step, added below
    cur, act, frames = np.asarray(current), np.asarray(action_amb), []
    total = sum(len(s["per_frame"]) for s in steps)
    every = max(1, total // 10)                                       # progress every ~10% of frames
    t0 = time.perf_counter()
    try:
        for s in steps:
            target = {"xyz": np.asarray(s["true_next"])[None], "color": "black", "marker_color": "black",
                      "start_sphere": True, "end_sphere": False, "start_scale": 0.5}   # solid black sphere, radius 0.0225*sc (was a ring)
            for fr in s["per_frame"]:
                if log is not None and len(frames) % every == 0:
                    log(_eta_str(t0, len(frames), total))
                fig = fig_torus_atlas(R, r, trajs=base + [target] + _quiver_swarm_trajs(fr, sc),
                                      arrows=[(cur, act)], coloring=coloring, title=title, markers=False,
                                      iso_size=size, ax_size=int(round(size * 0.67)), view_pad=view_pad,
                                      torus_opacity=torus_opacity, renderer=rend)
                fig.set_dpi(VIDEO_DPI)
                frames.append(_fig_rgb(fig))
                plt.close(fig)
        if steps:   # 0.5s hold (30 frames @ 60fps) on the clean scene — last swarm gone, grey disappeared
            last_tgt = {"xyz": np.asarray(steps[-1]["true_next"])[None], "color": "black", "marker_color": "black",
                        "start_sphere": True, "end_sphere": False, "start_scale": 0.5}
            fig = fig_torus_atlas(R, r, trajs=base + [last_tgt], arrows=[(cur, act)], coloring=coloring,
                                  title=title, markers=False, iso_size=size, ax_size=int(round(size * 0.67)),
                                  view_pad=view_pad, torus_opacity=torus_opacity, renderer=rend)
            fig.set_dpi(VIDEO_DPI)
            frames.extend([_fig_rgb(fig)] * 30)
            plt.close(fig)
    finally:
        rend.close()
    return np.stack(frames)


def _pad3(P, frac=0.05):
    """Per-axis padded (lo,hi) for an (N,3) cloud -> ((xlo,xhi),(ylo,yhi),(zlo,zhi)). Autoscaled box for the
    geometry-free 3D renderers (no torus R/r to derive limits from)."""
    P = np.asarray(P, float).reshape(-1, 3)
    lo, hi = P.min(0), P.max(0); pad = frac * (hi - lo + 1e-6)
    return tuple((float(lo[i] - pad[i]), float(hi[i] + pad[i])) for i in range(3))


def diffusion_swarm_plain_frames(current, agent_tail, future_path, steps, title="", lims=None,
                                 size=6.0, dpi=VIDEO_DPI, log=None):
    """Geometry-FREE fallback for diffusion_quiver_sequential_frames: the SAME N-sequential-swarm denoising
    animation (static agent dot + black history tail + black future line; each step's swarm denoises to its
    target, tails growing then collapsing) but in plain matplotlib 3D with axes AUTOSCALED from the data and
    NO torus surface mesh. Used when the env supplies no geometry (recorded datasets). `steps`/`current`/
    `agent_tail`/`future_path` are the SAME structures the torus renderer consumes."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers the 3d projection)
    cur = np.asarray(current, float).reshape(3)
    tail = np.asarray(agent_tail, float).reshape(-1, 3)
    fut = np.asarray(future_path, float).reshape(-1, 3)
    if lims is None:                                        # autoscale over EVERY point ever drawn
        allpts = [cur[None], tail, fut]
        for s in steps:
            allpts.append(np.asarray(s["true_next"], float).reshape(1, 3))
            for fr in s["per_frame"]:
                for pcl in fr["swarm"]:
                    allpts.append(np.asarray(pcl["trail"], float).reshape(-1, 3))
        lims = _pad3(np.concatenate([a for a in allpts if len(a)], axis=0))
    (xl, yl, zl) = lims
    total = sum(len(s["per_frame"]) for s in steps) or 1
    every = max(1, total // 10); t0 = time.perf_counter(); done = 0; frames = []

    def draw(swarm, target):
        fig = plt.figure(figsize=(size, size))
        ax = fig.add_subplot(111, projection="3d"); ax.set_proj_type("ortho")
        ax.set_xlim(xl); ax.set_ylim(yl); ax.set_zlim(zl)
        ax.set_box_aspect((xl[1] - xl[0], yl[1] - yl[0], zl[1] - zl[0]))
        if len(fut) > 1:
            ax.plot(fut[:, 0], fut[:, 1], fut[:, 2], color="0.6", lw=1.2)          # static future line
        if len(tail) > 1:
            ax.plot(tail[:, 0], tail[:, 1], tail[:, 2], color="black", lw=1.4)     # static history tail
        ax.scatter([cur[0]], [cur[1]], [cur[2]], s=60, c="black", depthshade=False)
        if target is not None:
            ax.scatter([target[0]], [target[1]], [target[2]], s=45, c="black", depthshade=False)
        for pcl in (swarm or []):
            tr = np.asarray(pcl["trail"], float).reshape(-1, 3)
            if len(tr) > 1:
                ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color="0.5", lw=0.6, alpha=0.7)   # denoising trail
            p = np.asarray(pcl["particle"], float).reshape(3)
            ax.scatter([p[0]], [p[1]], [p[2]], s=8, c="tab:blue", depthshade=False)     # swarm particle
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.set_title(title, fontsize=10)
        fig.set_dpi(dpi); rgb = _fig_rgb(fig); plt.close(fig)
        return rgb

    for s in steps:
        tgt = np.asarray(s["true_next"], float).reshape(3)
        for fr in s["per_frame"]:
            if log is not None and done % every == 0:
                log(_eta_str(t0, done, total))
            frames.append(draw(fr["swarm"], tgt)); done += 1
    if steps:                                              # 0.5s hold on the clean scene (last swarm gone)
        frames.extend([draw([], np.asarray(steps[-1]["true_next"], float).reshape(3))] * 30)
    return np.stack(frames) if frames else np.zeros((1, int(size * dpi), int(size * dpi), 3), np.uint8)


# ------------------------- diffusion: recovered-manifold point clouds (matplotlib 3D) -------------------------
_POINT_PURPLE = "#8E44AD"   # flat fill when no scalar `color` is given (color everything purple)


def _draw_hull_2d(ax, hull_pts):
    """Black convex-hull outline of the request cluster (the caller pre-trims which points count)."""
    from scipy.spatial import ConvexHull
    p = np.asarray(hull_pts)[:, :2]
    if len(p) < 3:
        return
    v = p[ConvexHull(p).vertices]
    v = np.vstack([v, v[:1]])
    ax.fill(v[:, 0], v[:, 1], facecolor="black", alpha=0.05, zorder=1)
    ax.plot(v[:, 0], v[:, 1], color="black", lw=1.8, zorder=5)


def _draw_hull_3d(ax, hull_pts):
    """Black convex hull of the request cluster (the caller pre-trims which points count). If the points are
    COPLANAR (e.g. a 3-class LDA gives only 2 axes, zero-padded to 3D — no 3D volume), draw the 2D hull polygon
    in that plane instead of asking Qhull for an impossible 3D hull."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from scipy.spatial import ConvexHull
    p = np.asarray(hull_pts)[:, :3]
    if len(p) < 4:
        return
    if np.linalg.matrix_rank(p - p.mean(0), tol=1e-6) < 3:            # coplanar/degenerate -> 2D hull in-plane
        v = p[ConvexHull(p[:, :2]).vertices]
        v = np.vstack([v, v[:1]])
        ax.plot(v[:, 0], v[:, 1], v[:, 2], color="black", lw=1.6, alpha=0.7)
        return
    tris = [p[s] for s in ConvexHull(p).simplices]
    ax.add_collection3d(Poly3DCollection(tris, facecolor="black", edgecolor="black",
                                         alpha=0.06, linewidths=0.4))


def _draw_agent_2d(ax, trail, pos):
    """Agent trail (line) + current position (sphere) on a 2D axis -> the created artists (for blit remove)."""
    arts = []
    tr = np.asarray(trail)
    if len(tr) > 1:
        arts += ax.plot(tr[:, 0], tr[:, 1], color="black", lw=1.3, alpha=0.8, zorder=10)
    arts.append(ax.scatter([pos[0]], [pos[1]], s=140, c="black", marker="o",
                           edgecolors="white", linewidths=1.0, zorder=11))
    return arts


def _draw_agent_3d(ax, trail, pos):
    """3D agent: trail as a 3d line + the current-position SPHERE as a 2D circle projected onto the current
    view. A 3d scatter marker doesn't blit reliably via draw_artist (it reads as just the trail line), so the
    sphere is a text2D circular bbox (same trick as the C/M marks) -> always visible, on top. Returns the artists."""
    from mpl_toolkits.mplot3d import proj3d
    arts = []
    tr = np.asarray(trail)
    if len(tr) > 1:
        arts += ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color="black", lw=1.6, alpha=0.9)
    xp, yp, _ = proj3d.proj_transform(pos[0], pos[1], pos[2], ax.get_proj())
    fx, fy = ax.transAxes.inverted().transform(ax.transData.transform((xp, yp)))
    arts.append(ax.text2D(fx, fy, " ", transform=ax.transAxes, fontsize=7, zorder=1e6,
                          bbox=dict(boxstyle="circle,pad=0.45", facecolor="black", edgecolor="white", linewidth=1.2)))
    return arts


def _marks_on_top_3d(ax, marks):
    """Draw C/M as 2D overlays projected onto the CURRENT (fixed) view so they always sit ON TOP of the 3d
    cloud — 3d has no true zorder (it depth-sorts), so an in-cloud marker is otherwise occluded. Each is a
    black circle (a circular text bbox) with a white letter, matching the agent sphere. MUST be called AFTER
    view_init + lims are set (so ax.get_proj() is final); it reads back like the static PNG otherwise."""
    from mpl_toolkits.mplot3d import proj3d
    for mk in (marks or []):
        mp = np.asarray(mk["pos"])
        xp, yp, _ = proj3d.proj_transform(mp[0], mp[1], mp[2], ax.get_proj())     # data -> 2D projected
        fx, fy = ax.transAxes.inverted().transform(ax.transData.transform((xp, yp)))   # -> axes fraction (bbox-invariant)
        ax.text2D(fx, fy, mk["text"], transform=ax.transAxes, ha="center", va="center",
                  fontsize=8, fontweight="bold", color="white", zorder=1e6,
                  bbox=dict(boxstyle="circle,pad=0.35", facecolor="black", edgecolor="white", linewidth=1.0))


def _overlay_2d(ax, marks, agent, hull=None):
    """Static latent-plot overlays on a 2D axis: request-region hull + C/M lettered black circles (+ agent if
    given one-shot, e.g. a static PNG). For animations the agent is drawn per-frame via _draw_agent_2d instead."""
    if hull is not None:
        _draw_hull_2d(ax, hull)
    for mk in (marks or []):                                       # black sphere (like the agent) + white C/M letter
        mp = np.asarray(mk["pos"])
        ax.scatter([mp[0]], [mp[1]], s=140, c="black", marker="o", edgecolors="white", linewidths=1.0, zorder=7)
        ax.text(mp[0], mp[1], mk["text"], ha="center", va="center", fontsize=8, fontweight="bold",
                color="white", zorder=9)
    if agent is not None:
        _draw_agent_2d(ax, agent["trail"], agent["pos"])


def _overlay_3d(ax, agent, hull=None):
    """3D backdrop overlays: request-region hull + agent (if given, one-shot). C/M are drawn separately via
    _marks_on_top_3d AFTER the view is set, so they always sit on top of the cloud."""
    if hull is not None:
        _draw_hull_3d(ax, hull)
    if agent is not None:
        _draw_agent_3d(ax, agent["trail"], agent["pos"])


def animate_latent(fig, traj, is3d, idx, log=None, tail=60):
    """Efficient agent-overlay animation over a STATIC latent-plot backdrop `fig` (built once by
    fig_points_2d/fig_points_9view with the cloud + C/M marks + hull). Mirrors the persistent-renderer pattern
    of the torus videos: the expensive backdrop (10k+ points x up-to-9 views) is rasterized ONCE, then each
    frame only BLITS the agent (trail + sphere) on top — so a full-length 9-view clip costs ~one backdrop
    raster, not one per frame. traj: (T, dim) agent path in the embedding; idx: 1-based step indices to render.
    `tail`: only the last `tail` steps of trail are drawn, so it fades out BEHIND the agent (like the control
    videos). Returns (len(idx), H, W, 3) uint8."""
    fig.canvas.draw()
    bg = fig.canvas.copy_from_bbox(fig.bbox)                          # cache the rendered backdrop
    axes = list(fig.axes)
    draw_agent = _draw_agent_3d if is3d else _draw_agent_2d
    w, h = fig.canvas.get_width_height()
    frames = []
    every = max(1, len(idx) // 10)
    for fi, k in enumerate(idx):
        if log is not None and fi % every == 0:
            log(f"{fi}/{len(idx)} frames")
        fig.canvas.restore_region(bg)
        arts = []
        for ax in axes:
            for a in draw_agent(ax, traj[max(0, k - tail):k], traj[k - 1]):   # only the last `tail` steps of trail
                ax.draw_artist(a)
                arts.append(a)
        fig.canvas.blit(fig.bbox)
        frames.append(np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(h, w, 4)[..., :3].copy())
        for a in arts:
            a.remove()
    return np.stack(frames)


def fig_points_4view(pts, color=None, title="", lims=None, point_size=4.0, cmap="plasma", cbar_label="",
                     depthshade=True):
    """A 3D point cloud from 4 ORTHOGRAPHIC views in a 2x2 GridSpec (plain matplotlib 3D scatter — no torus
    mesh, the POINTS are the surface). Views: side (x-z), top-down (x-y), and two obliques. pts: (N,3).
    color: (N,) scalar array -> colormap + colorbar; None -> flat purple, no colorbar. lims: (lo,hi) cube
    OR ((xlo,xhi),(ylo,yhi),(zlo,zhi)) per-axis — box aspect from the lims extents so the cloud FILLS."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers the 3d projection)
    pts = np.asarray(pts)
    if lims is not None and np.ndim(lims) == 1:
        lims = (tuple(lims), tuple(lims), tuple(lims))
    from matplotlib.ticker import MaxNLocator
    views = [(8, -90, "side  (x–z)"), (89, -90, "top-down  (x–y)"), (26, 45, "oblique A"), (26, 135, "oblique B")]
    fig = plt.figure(figsize=(12, 12))
    gs = GridSpec(2, 2, figure=fig, wspace=0.0, hspace=0.0)
    sc = None
    for i, (elev, azim, lbl) in enumerate(views):
        ax = fig.add_subplot(gs[i // 2, i % 2], projection="3d")
        ax.set_proj_type("ortho")                                  # orthographic (no perspective foreshortening)
        sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=point_size,
                        c=(color if color is not None else _POINT_PURPLE), cmap=cmap,
                        depthshade=depthshade, linewidths=0)
        ax.view_init(elev=elev, azim=azim)
        if lims is not None:
            (xl, yl, zl) = lims
            ax.set_xlim(xl); ax.set_ylim(yl); ax.set_zlim(zl)
            ax.set_box_aspect((xl[1] - xl[0], yl[1] - yl[0], zl[1] - zl[0]))  # proportional -> fills, undistorted
        else:
            ax.set_box_aspect((1, 1, 1))
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):    # keep the ticks (marks); drop tick AND axis labels
            axis.set_major_locator(MaxNLocator(5))
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
    fig.subplots_adjust(left=0.0, right=0.88, top=0.93, bottom=0.0, wspace=0.0, hspace=0.0)  # use the whitespace
    if isinstance(color, np.ndarray) and sc is not None:           # colorbar only for a scalar field
        cax = fig.add_axes([0.905, 0.30, 0.015, 0.40])             # dedicated right-side colorbar
        fig.colorbar(sc, cax=cax, label=cbar_label)
    fig.suptitle(title, fontsize=11, y=0.995)
    return fig


def fig_points_9view(pts, color=None, title="", lims=None, point_size=4.0, cmap="plasma", cbar_label="",
                     depthshade=True, legend=None, marks=None, agent=None, alpha=1.0, hull=None):
    """A 3D point cloud from 9 ORTHOGRAPHIC views in a 3x3 grid — SAME points + SAME embedding.
    Row 1: ISOMETRIC (elev≈35.26°) rotated +0/+15/+30° about the VERTICAL axis (azimuth).
    Row 2: ISOMETRIC rotated +0/+15/+30° about a HORIZONTAL axis (elevation / tilt).
    Row 3: axial views — front on, side on, top down.
    Axes are forced to EQUAL length + equal aspect (a cube), so long-thin embeddings render at their TRUE
    shape. pts: (N,3). legend: list of (label, color) for a CATEGORICAL scatter (per-point (N,3) RGB array)
    -> a legend box instead of a colorbar."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers the 3d projection)
    from matplotlib.ticker import MaxNLocator
    pts = np.asarray(pts)
    if lims is not None and np.ndim(lims) == 1:
        lims = (tuple(lims), tuple(lims), tuple(lims))
    if lims is not None:                                           # EQUAL-length axes (cube) centered per-axis -> true aspect
        ctr = [(lo + hi) / 2 for lo, hi in lims]; half = max(hi - lo for lo, hi in lims) / 2
        lims = tuple((c - half, c + half) for c in ctr)
    _ISO = 35.264                                                  # true isometric elevation (atan(1/sqrt(2)))
    views = [(_ISO, 45, "iso +0° abt vertical"), (_ISO, 60, "iso +15° abt vertical"), (_ISO, 75, "iso +30° abt vertical"),
             (_ISO, 45, "iso +0° abt horizontal"), (_ISO + 15, 45, "iso +15° abt horizontal"), (_ISO + 30, 45, "iso +30° abt horizontal"),
             (0, -90, "front on"), (0, 0, "side on"), (90, -90, "top down")]
    fig = plt.figure(figsize=(18, 18))
    gs = GridSpec(3, 3, figure=fig, wspace=0.0, hspace=0.08)
    sc = None
    _cmap = None if (isinstance(color, np.ndarray) and color.ndim == 2) else cmap   # RGB array -> no colormap
    for i, (elev, azim, lbl) in enumerate(views):
        ax = fig.add_subplot(gs[i // 3, i % 3], projection="3d")
        ax.set_proj_type("ortho")                                  # orthographic (no perspective foreshortening)
        sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=point_size,
                        c=(color if color is not None else _POINT_PURPLE), cmap=_cmap,
                        depthshade=depthshade, linewidths=0, alpha=alpha)
        _overlay_3d(ax, agent, hull=hull)                          # hull + (one-shot) agent
        ax.view_init(elev=elev, azim=azim)
        if lims is not None:
            (xl, yl, zl) = lims
            ax.set_xlim(xl); ax.set_ylim(yl); ax.set_zlim(zl)
        ax.set_box_aspect((1, 1, 1))                               # equal aspect always (lims are already a cube)
        _marks_on_top_3d(ax, marks)                                # C/M projected onto this view -> always on top
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_major_locator(MaxNLocator(5))
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
        ax.set_title(lbl, fontsize=10, y=0.97)                     # named view (top row + rotated bottom row)
    fig.subplots_adjust(left=0.0, right=0.9, top=0.92, bottom=0.0, wspace=0.0, hspace=0.06)
    if legend is not None:
        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(facecolor=c, edgecolor="black", linewidth=0.5, label=str(l)) for l, c in legend],
                   loc="center left", bbox_to_anchor=(0.915, 0.5), frameon=False, fontsize=11)  # gap from the 0.9 grid edge
    elif isinstance(color, np.ndarray) and color.ndim == 1 and sc is not None:
        cax = fig.add_axes([0.915, 0.30, 0.012, 0.40])
        fig.colorbar(sc, cax=cax, label=cbar_label)
    fig.suptitle(title, fontsize=11, y=0.99)
    return fig


def fig_points_2d(pts, color=None, title="", lims=None, point_size=4.0, cmap="plasma", cbar_label="", legend=None,
                  marks=None, agent=None, alpha=1.0, hull=None, annotations=None):
    """The 2D analogue of fig_points_6view: a single scatter, same styling (no tick/axis labels). Axes are
    EQUAL length + equal aspect, so a long-thin embedding renders at its TRUE shape (not stretched to fill).
    pts: (N,2). lims: ((xlo,xhi),(ylo,yhi)) or None. legend: list of (label, color) for a CATEGORICAL scatter
    (color is a per-point (N,3) RGB array) -> legend box.
    marks: list of {"pos": (2,), "text": str} -> C/M lettered circles; agent: {"pos","trail"} one-shot overlay;
    hull: points whose convex hull outlines the request region (latent-animation extras).
    annotations: list of {"pos": (2,), "text": str} -> a star + a leader line to a text label (e.g. where a
    query phrasing / bucket word lands in the cloud)."""
    from matplotlib.ticker import MaxNLocator
    pts = np.asarray(pts)
    fig = plt.figure(figsize=(11.5, 9))                            # extra width so the right-side legend isn't clipped
    ax = fig.add_subplot(1, 1, 1)
    _cmap = None if (isinstance(color, np.ndarray) and color.ndim == 2) else cmap   # RGB array -> no colormap
    sc = ax.scatter(pts[:, 0], pts[:, 1], s=point_size,
                    c=(color if color is not None else _POINT_PURPLE), cmap=_cmap, linewidths=0, alpha=alpha)
    _overlay_2d(ax, marks, agent, hull=hull)
    anns = annotations or []
    if anns:                                                       # query/bucket labels: star + short leader line to a text box
        span = float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))) or 1.0
        cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
        for j, a in enumerate(anns):
            x, y = float(a["pos"][0]), float(a["pos"][1])
            dx, dy = x - cx, y - cy                                 # push the label just OUTWARD from its own cluster (short line)
            n = float(np.hypot(dx, dy)) or 1.0
            ux, uy = (dx / n, dy / n) if n > 1e-6 else (np.cos(2 * np.pi * j / len(anns)), np.sin(2 * np.pi * j / len(anns)))
            tx, ty = x + 0.16 * span * ux, y + 0.16 * span * uy
            ax.scatter([x], [y], s=point_size * 48, c="black", marker="*", zorder=21, linewidths=0)
            ax.annotate(str(a["text"]), xy=(x, y), xycoords="data", xytext=(tx, ty), textcoords="data",
                        fontsize=18, fontweight="bold", zorder=22, ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", lw=0.6, alpha=0.92),
                        arrowprops=dict(arrowstyle="-", color="black", lw=0.7, alpha=0.8))
    if lims is not None:
        ctr = [(lo + hi) / 2 for lo, hi in lims]; half = max(hi - lo for lo, hi in lims) / 2   # equal-length axes (square)
        ax.set_xlim(ctr[0] - half, ctr[0] + half); ax.set_ylim(ctr[1] - half, ctr[1] + half)
    ax.set_aspect("equal", adjustable="box")                       # equal aspect -> true shape (thin looks thin)
    ax.xaxis.set_major_locator(MaxNLocator(5)); ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.set_xticklabels([]); ax.set_yticklabels([])
    fig.subplots_adjust(left=0.03, right=0.80, top=0.93, bottom=0.03)  # leave the right ~20% for the legend/colorbar
    if legend is not None:
        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(facecolor=c, edgecolor="black", linewidth=0.5, label=str(l)) for l, c in legend],
                   loc="center left", bbox_to_anchor=(0.815, 0.5), frameon=False, fontsize=11)
    elif isinstance(color, np.ndarray) and color.ndim == 1:          # colorbar only for a scalar field
        cax = fig.add_axes([0.82, 0.30, 0.013, 0.40])                # dedicated right-side colorbar
        fig.colorbar(sc, cax=cax, label=cbar_label)
    fig.suptitle(title, fontsize=11, y=0.98)
    return fig


def fig_confusion(mat, labels, title="", xlabel="VLM", ylabel="analytic"):
    """A confusion-count heatmap (rows=analytic, cols=VLM) with the count printed in each cell. mat: (K,K)
    ints, labels: the K bucket names. Feeds eval_interpret's VLM-vs-analytic trust check."""
    mat = np.asarray(mat)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(1.6 + n, 1.6 + n))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=9)
    thr = mat.max() / 2 if mat.max() else 1
    for i in range(n):
        for j in range(n):
            ax.text(j, i, int(mat[i, j]), ha="center", va="center", fontsize=9,
                    color="white" if mat[i, j] > thr else "black")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def _img_u8(x):
    """(H,W,3) or (...,H,W,3) -> uint8. Passes uint8 through; treats float as [0,1]."""
    x = np.asarray(x)
    return x if x.dtype == np.uint8 else (np.clip(x, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def fig_image_filmstrip(pred_future, true_future, n_cols=8, title=""):
    """A 2-row filmstrip over the prediction horizon: TOP = predicted frames, BOTTOM = ground truth, sampled
    at n_cols evenly-spaced future steps. pred_future / true_future: (F,H,W,3) uint8 or float[0,1]. Minimal
    text — only a 'pred'/'true' row label and the horizon offset above each column."""
    pred_future, true_future = _img_u8(pred_future), _img_u8(true_future)
    f = min(len(pred_future), len(true_future))
    idx = np.unique(np.linspace(0, f - 1, min(n_cols, f)).round().astype(int))
    fig, axes = plt.subplots(2, len(idx), figsize=(1.6 * len(idx), 3.4), squeeze=False)
    for col, k in enumerate(idx):
        for row, (frames, lbl) in enumerate(((pred_future, "pred"), (true_future, "true"))):
            ax = axes[row, col]
            ax.imshow(frames[k]); ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(lbl, fontsize=9)
            if row == 0:
                ax.set_title(f"+{int(k) + 1}", fontsize=7)        # horizon offset (steps ahead)
    fig.suptitle(title, fontsize=10)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.88, bottom=0.01, wspace=0.04, hspace=0.04)
    return fig


def image_rollout_video(true_full, pred_future, context_len, sep_px=2):
    """Stacked predicted-over-true rollout video, NO text. BOTTOM plays the full GT sequence (context THEN
    future); TOP is black through the context, then the predicted future — so once context has played out
    the two play in sync. true_full: (T,H,W,3); pred_future: (F,H,W,3), F = T - context_len. Returns
    (T, 2H+sep_px, W, 3) uint8, ready for writer.video."""
    true_full, pred_future = _img_u8(true_full), _img_u8(pred_future)
    T, H, W = true_full.shape[:3]
    black = np.zeros((context_len, H, W, 3), np.uint8)
    top = np.concatenate([black, pred_future], axis=0)[:T]            # black during context, then predictions
    sep = np.zeros((T, sep_px, W, 3), np.uint8)                       # thin divider (no text)
    return np.concatenate([top, sep, true_full], axis=1)             # vstack: pred on top, GT on bottom


def points_collapse_frames(paths, color=None, title="", n_frames=60, lims=None, point_size=4.0,
                           cmap="plasma", cbar_label="", depthshade=False, ease=True, dpi=110, log=None):
    """Animate a cloud collapsing onto the recovered manifold: paths (N, T, 3) are the per-point positions
    over the T denoising steps; each frame is fig_points_4view at an interpolated time. ease=True uses a
    trapezoidal velocity profile: constant-speed collapse for most of the window, then a linear deceleration
    into a soft stop (the cloud eases to a settle, no hard halt), then held still. depthshade defaults
    False here (much faster for the many-frame render)."""
    paths = np.asarray(paths)
    T = paths.shape[1]
    frames = []
    every = max(1, n_frames // 10)                           # progress every ~10% of frames
    t0 = time.perf_counter()
    for k in range(n_frames):
        if log is not None and k % every == 0:
            log(_eta_str(t0, k, n_frames))
        u = k / (n_frames - 1) if n_frames > 1 else 1.0
        if ease:
            move = 0.85                                       # collapse completes by ~85% of frames (~6.8s of 8s), then hold still
            tt = min(u / move, 1.0)                           # 0->1 collapse progress in linear time
            k = 0.8                                           # constant speed for the first k, then linear velocity ramp-down to 0 -> eases into the settle
            integ = tt if tt <= k else k + (tt - k) - (tt - k) ** 2 / (2.0 * (1.0 - k))
            u = integ / ((1.0 + k) / 2.0)                     # trapezoidal velocity (flat, then decel over the last ~1.4s); no hard stop
        f = u * (T - 1)
        j0 = int(np.floor(f)); j1 = min(j0 + 1, T - 1); w = f - j0
        pts = (1 - w) * paths[:, j0] + w * paths[:, j1]
        fig = fig_points_4view(pts, color=color, title=title, lims=lims, point_size=point_size,
                               cmap=cmap, cbar_label=cbar_label, depthshade=depthshade)
        fig.set_dpi(dpi)
        frames.append(_fig_rgb(fig))
        plt.close(fig)
    return np.stack(frames)



# samples per timestep for the action-distribution preview + eval (4x the 256 train trajectories, so the
# shape/peaks/weights read clearly above Monte-Carlo noise). The eval samples the LEARNED head this many too.
ACTION_DIST_N_SAMPLES = 4096   # total pooled samples for the action-dist eval; K=this//n_ep drawn per context


def fig_action_distribution(act, a_max, sampler_name="", timesteps=None):
    """8 magnitude-histogram tiles of the data's ACTION distribution at 8 timesteps (2x4 grid).

    act: (n_traj, steps, action_dim) float. Shows |a| pooled over trajectories at each timestep, so the
    (possibly time-varying) shape — e.g. the two-basin bimodal magnitude — is directly visible. Regenerated
    on every data_generation run and referenced by the HF dataset card.
    """
    act = np.asarray(act)
    act = np.asarray(act); n_traj = act.shape[0]
    mag = np.linalg.norm(act, axis=-1)            # (n_traj, steps)
    steps = mag.shape[1]
    if timesteps is None:                          # 8 timesteps spanning the episode (near-start ... end)
        timesteps = [int(round(f * (steps - 1))) for f in (0.02, 0.06, 0.12, 0.25, 0.4, 0.6, 0.8, 1.0)]
    data_hi = float(np.nanmax(mag)) * 1.2                 # fit the data, 20% headroom
    hi = data_hi if a_max is None else min(float(a_max), data_hi)   # a_max: torus-only x-limit knob; None -> data-derived
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, t in zip(axes.ravel(), timesteps):
        ax.hist(mag[:, t], bins=60, range=(0, hi), color="steelblue")
        ax.set_title(f"|a| @ t={t}", fontsize=11); ax.set_xlabel("|a|")
    ttl = (f"action-magnitude distribution over time  ·  each tile pooled over {n_traj} trajectories"
           + (f"  ·  sampler={sampler_name}" if sampler_name else ""))
    fig.suptitle(ttl, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def anim_action_distribution(true_acts, pred_acts, a_max, bins=60, max_frames=None, dpi=90, window=0):
    """Animated 3-panel action-magnitude histogram over time for eval_action_distribution: LEFT = true (green),
    MIDDLE = pred (red), RIGHT = the two overlaid at 50% opacity. Axes (x=|a| range, y=density) are LOCKED across
    every frame AND identical on all three panels, computed up front over ALL data. Densities (not counts), so
    true/pred are comparable even with different sample counts. `true_acts`/`pred_acts`: (N, steps, action_dim).
    max_frames=None -> one frame per timestep (no cap). window>0 -> each frame pools timesteps [t-w, t+w] (the
    dist changes slowly -> ~(2w+1)x more samples/frame at negligible bias; the way to densify past #episodes)."""
    tm = np.linalg.norm(np.asarray(true_acts), axis=-1)     # (N, steps)
    pm = np.linalg.norm(np.asarray(pred_acts), axis=-1)
    steps = tm.shape[1]
    def win(a, t):                                          # samples pooled over [t-w, t+w]
        return a[:, max(0, t - window): t + window + 1].reshape(-1)
    data_hi = max(float(tm.max()), float(pm.max())) * 1.05                # x-lim over ALL data
    hi = data_hi if a_max is None else min(float(a_max), data_hi)   # a_max: torus-only x-limit knob; None -> data-derived
    edges = np.linspace(0.0, hi, bins + 1)
    ymax = 0.0                                                              # y-lim = worst-case density over ALL frames
    for t in range(steps):
        ymax = max(ymax, float(np.histogram(win(tm, t), bins=edges, density=True)[0].max()),
                   float(np.histogram(win(pm, t), bins=edges, density=True)[0].max()))
    ymax = ymax * 1.08 or 1.0
    ts = (range(steps) if (max_frames is None or steps <= max_frames)
          else np.linspace(0, steps - 1, max_frames).round().astype(int))
    green, red = (0.20, 0.60, 0.25), (0.85, 0.20, 0.20)
    frames = []
    for t in ts:
        t = int(t); tw, pw = win(tm, t), win(pm, t)
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
        for a in ax:
            a.set_xlim(0, hi); a.set_ylim(0, ymax); a.set_xlabel("|a|")
        ax[0].hist(tw, bins=edges, density=True, color=green, alpha=0.5); ax[0].set_title(f"true ({tw.size} samples)")
        ax[1].hist(pw, bins=edges, density=True, color=red, alpha=0.5); ax[1].set_title(f"pred ({pw.size} samples)")
        ax[2].hist(tw, bins=edges, density=True, color=green, alpha=0.5, label="true")
        ax[2].hist(pw, bins=edges, density=True, color=red, alpha=0.5, label="pred")
        ax[2].set_title("both"); ax[2].legend(loc="upper right", fontsize=9)
        fig.suptitle(f"action-magnitude distribution  ·  t={t}/{steps - 1}"
                     + (f"  (±{window} pooled)" if window else ""), fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.set_dpi(dpi)
        frames.append(_fig_rgb(fig)); plt.close(fig)
    return np.stack(frames)


def anim_action_by_state(true_acts, pred_acts, split, a_max, low_name="slow", high_name="fast", bins=55,
                          max_frames=None, dpi=90, window=0):
    """Animated BY-STATE action-magnitude histograms over time: 2 rows (TOP=low_name, BOTTOM=high_name) x 3 cols
    (true green | pred red | both). Splitting by a meaningful state feature (see WorldEnv.action_dist_split,
    environments/base.py) keeps each group's mode crisp — pooling over all rows smears the state-dependent scale
    together. Densities (comparable despite differing per-frame counts); x-range and y-range are locked across
    every frame and panel. true/pred: (N, steps, 2); split: (N, steps) bool (True -> low_name group).
    window>0 -> each frame pools timesteps [t-w, t+w] (~(2w+1)x more samples/frame, the way to densify past #episodes)."""
    tm = np.linalg.norm(np.asarray(true_acts), axis=-1)      # (N, steps)
    pm = np.linalg.norm(np.asarray(pred_acts), axis=-1)
    split = np.asarray(split, dtype=bool)
    steps = tm.shape[1]
    def wsel(a, t, low):                                      # samples in [t-w,t+w] on the requested group
        lo, hiw = max(0, t - window), t + window + 1
        aw, sw = a[:, lo:hiw], split[:, lo:hiw]
        return aw[sw if low else ~sw]
    data_hi = max(float(tm.max()), float(pm.max())) * 1.05
    hi = data_hi if a_max is None else min(float(a_max), data_hi)   # a_max: torus-only x-limit knob; None -> data-derived
    edges = np.linspace(0.0, hi, bins + 1)
    ymax = 0.0                                                # locked y (density) over all frames/rows/panels
    for t in range(steps):
        for low in (True, False):
            for m in (wsel(tm, t, low), wsel(pm, t, low)):
                if m.size:
                    ymax = max(ymax, float(np.histogram(m, bins=edges, density=True)[0].max()))
    ymax = ymax * 1.08 or 1.0
    ts = (range(steps) if (max_frames is None or steps <= max_frames)
          else np.linspace(0, steps - 1, max_frames).round().astype(int))
    green, red = (0.20, 0.60, 0.25), (0.85, 0.20, 0.20)
    rows = ((low_name, 0, True), (high_name, 1, False))
    frames = []
    for t in ts:
        t = int(t)
        fig, ax = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
        for lbl, r, low in rows:
            tt, pp = wsel(tm, t, low), wsel(pm, t, low)
            for c in range(3):
                ax[r, c].set_xlim(0, hi); ax[r, c].set_ylim(0, ymax)
            if tt.size: ax[r, 0].hist(tt, bins=edges, density=True, color=green, alpha=0.6)
            if pp.size: ax[r, 1].hist(pp, bins=edges, density=True, color=red, alpha=0.6)
            if tt.size: ax[r, 2].hist(tt, bins=edges, density=True, color=green, alpha=0.5, label="true")
            if pp.size: ax[r, 2].hist(pp, bins=edges, density=True, color=red, alpha=0.5, label="pred")
            ax[r, 0].set_title(f"{lbl} · true (n={tt.size})", fontsize=10)
            ax[r, 1].set_title(f"{lbl} · pred (n={pp.size})", fontsize=10)
            ax[r, 2].set_title(f"{lbl} · both", fontsize=10); ax[r, 2].legend(loc="upper right", fontsize=8)
        for c in range(3):
            ax[1, c].set_xlabel("|a|")
        fig.suptitle(f"action-magnitude by state ({low_name}/{high_name})  ·  t={t}/{steps - 1}"
                     + (f"  (±{window} pooled)" if window else ""), fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.set_dpi(dpi)
        frames.append(_fig_rgb(fig)); plt.close(fig)
    return np.stack(frames)


def fig_action_by_state(act, split, a_max, low_name="slow", high_name="fast", sampler_name="", n_cols=4, window=0):
    """Action-magnitude distribution split by a meaningful state feature (see WorldEnv.action_dist_split,
    environments/base.py): TOP row = low_name group, BOTTOM row = high_name group; columns are uniformly-spaced
    timesteps. Same column across rows -> same timestep, so the two groups are directly comparable. Histograms
    are densities (comparable across tiles despite differing counts). `act` (n_traj, steps, 2), `split`
    (n_traj, steps) bool (True -> low_name group). window>0 -> each column pools timesteps [t-w, t+w]
    (~(2w+1)x more samples/tile)."""
    a = np.asarray(act); n_traj = a.shape[0]
    mag = np.linalg.norm(a, axis=-1)                 # (n_traj, steps)
    split = np.asarray(split, dtype=bool)             # (n_traj, steps)
    steps = mag.shape[1]
    cols = [int(round(f * (steps - 1))) for f in np.linspace(0.05, 1.0, n_cols)]   # uniform timesteps
    data_hi = float(mag.max()) * 1.15
    hi = data_hi if a_max is None else min(float(a_max), data_hi)   # a_max: torus-only x-limit knob; None -> data-derived
    def wsel(t, low):                                # samples in [t-w,t+w] on the requested group
        lo, hiw = max(0, t - window), t + window + 1
        mw, sw = mag[:, lo:hiw], split[:, lo:hiw]
        return mw[sw if low else ~sw]
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 7), sharex=True, sharey=True)
    for r, (lbl, low) in enumerate(((low_name, True), (high_name, False))):
        for c, t in enumerate(cols):
            m = wsel(t, low)
            ax = axes[r, c]
            ax.hist(m, bins=55, range=(0, hi), density=True, color="steelblue")
            ax.set_title(f"{lbl},  t={t}  (n={m.size})", fontsize=10)
            if r == 1:
                ax.set_xlabel("|a|")
    ttl = (f"action-magnitude distribution — top: {low_name}  |  bottom: {high_name}  ·  "
           f"pooled over {n_traj} trajectories" + (f"  ·  sampler={sampler_name}" if sampler_name else ""))
    fig.suptitle(ttl, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


_AGREEN, _ARED = (0.20, 0.60, 0.25), (0.85, 0.20, 0.20)   # true=green, pred=red (shared by pooled anim + marginals)


def _action_marginal_spec(true_a, discrete_max=10, q=(0.001, 0.999)):
    """Per-dim classification + LOCKED bins/values computed once from the POOLED true actions, so the PNG and the
    video frames are all directly comparable. Returns a list of {kind, ...} dicts, one per action dim."""
    A = true_a.shape[-1]; spec = []
    for i in range(A):
        t = true_a[..., i].reshape(-1); uniq = np.unique(t)
        if uniq.size <= 1:
            spec.append({"kind": "constant", "val": float(uniq[0]) if uniq.size else 0.0})
        elif uniq.size <= discrete_max:
            spec.append({"kind": "discrete", "uniq": uniq})
        else:
            lo, hi = np.quantile(t, q[0]), np.quantile(t, q[1]); pad = (hi - lo) * 0.05 or 1.0
            spec.append({"kind": "continuous", "edges": np.linspace(lo - pad, hi + pad, 41)})
    return spec


def _draw_action_marginal(ax, t, p, name, s, ylim=None):
    """Draw ONE dim's panel from true samples `t` and pred samples `p` using the locked spec `s`. STYLING (shared by
    the PNG and the video, matching anim_action_distribution): true GREEN + pred RED, both FILLED at alpha 0.5 and
    OVERLAID — no step outline. `ylim` locks the y-axis (video frames); None -> autoscale (PNG)."""
    if s["kind"] == "constant":                               # don't fake a histogram
        ax.text(0.5, 0.5, f"{name} constant @ {s['val']:.3f}", ha="center", va="center", fontsize=11,
                transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([]); return
    if s["kind"] == "discrete":                               # value-frequency bars, OVERLAID at the same x (alpha)
        uniq = s["uniq"]; xs = np.arange(uniq.size)
        t_freq = np.array([(t == v).mean() for v in uniq]) if t.size else np.zeros(uniq.size)
        near = np.abs(p[:, None] - uniq[None, :]).argmin(axis=1) if p.size else np.array([], int)   # snap pred
        p_freq = np.bincount(near, minlength=uniq.size).astype(float) / max(p.size, 1)
        ax.bar(xs, t_freq, 0.8, color=_AGREEN, alpha=0.5, label="true")
        ax.bar(xs, p_freq, 0.8, color=_ARED, alpha=0.5, label="pred")
        ax.set_xticks(xs); ax.set_xticklabels([f"{v:.2g}" for v in uniq], fontsize=8)
        ax.set_title(f"{name}  (discrete, n={uniq.size})", fontsize=10)
    else:                                                     # continuous: shared bins, both FILLED + translucent
        edges = s["edges"]
        ax.hist(t, bins=edges, density=True, color=_AGREEN, alpha=0.5, label="true")
        ax.hist(p, bins=edges, density=True, color=_ARED, alpha=0.5, label="pred")
        ax.set_xlim(edges[0], edges[-1]); ax.set_title(f"{name}  (continuous)", fontsize=10)
    if ylim:
        ax.set_ylim(0, ylim)
    ax.legend(loc="upper right", fontsize=8)


def fig_action_marginals(true_a, pred_a, names=None, max_cols=4, discrete_max=10, q=(0.001, 0.999)):
    """Per-dim action marginals (STATIC): recorded (green) vs head (red), both FILLED + translucent + OVERLAID on
    SHARED bins, one panel per dim, pooling ALL timesteps for density. Dataset/env-agnostic. Per-dim behaviour
    (from TRUE): constant -> text tile; discrete (<=discrete_max unique) -> overlaid value-freq bars; continuous ->
    overlaid density histograms on the true dim's robust q-quantile range. Same styling as anim_action_marginals."""
    true_a, pred_a = np.asarray(true_a), np.asarray(pred_a); A = true_a.shape[-1]
    names = list(names) if names and len(names) == A else [f"a[{i}]" for i in range(A)]
    spec = _action_marginal_spec(true_a, discrete_max, q)
    n_cols = max(1, min(max_cols, A)); n_rows = int(np.ceil(A / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows)); axes = np.atleast_1d(axes).ravel()
    for i in range(A):
        _draw_action_marginal(axes[i], true_a[..., i].reshape(-1), pred_a[..., i].reshape(-1), names[i], spec[i])
    for j in range(A, len(axes)):
        axes[j].axis("off")
    fig.suptitle("per-dim action marginals — recorded (green) vs head (red), overlaid (pooled over all t)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def anim_action_marginals(true_a, pred_a, names=None, window=0, max_frames=None, dpi=90, max_cols=4,
                          discrete_max=10, q=(0.001, 0.999)):
    """Per-dim marginals ANIMATED over timesteps — the per-dim analogue of anim_action_distribution, same styling
    (green+red filled/overlaid) and the same `window` (each frame pools timesteps [t-w, t+w] to densify past the
    #episodes ceiling). Bins/values are LOCKED from the pooled true (via _action_marginal_spec) and per-dim y-limits
    are locked to the worst-case over all frames, so panels are comparable frame-to-frame. Returns (F,H,W,3)."""
    true_a, pred_a = np.asarray(true_a), np.asarray(pred_a); A = true_a.shape[-1]; steps = true_a.shape[1]
    names = list(names) if names and len(names) == A else [f"a[{i}]" for i in range(A)]
    spec = _action_marginal_spec(true_a, discrete_max, q)

    def win(a, i, t):                                          # dim i, timesteps [t-w, t+w] pooled over episodes
        return a[:, max(0, t - window): t + window + 1, i].reshape(-1)

    ylim = [None] * A                                         # lock each dim's y-axis = worst-case density over frames
    for i in range(A):
        if spec[i]["kind"] == "constant":
            continue
        m = 0.0
        for t in range(steps):
            tw, pw = win(true_a, i, t), win(pred_a, i, t)
            if spec[i]["kind"] == "discrete":
                uq = spec[i]["uniq"]
                m = max(m, max((tw == v).mean() for v in uq) if tw.size else 0.0)
                near = np.abs(pw[:, None] - uq[None, :]).argmin(axis=1) if pw.size else np.array([], int)
                m = max(m, float((np.bincount(near, minlength=uq.size) / max(pw.size, 1)).max()) if pw.size else 0.0)
            else:
                e = spec[i]["edges"]
                m = max(m, float(np.histogram(tw, bins=e, density=True)[0].max()) if tw.size else 0.0,
                        float(np.histogram(pw, bins=e, density=True)[0].max()) if pw.size else 0.0)
        ylim[i] = m * 1.08 or 1.0
    ts = (range(steps) if (max_frames is None or steps <= max_frames)
          else np.linspace(0, steps - 1, max_frames).round().astype(int))
    n_cols = max(1, min(max_cols, A)); n_rows = int(np.ceil(A / n_cols))
    frames = []
    for t in ts:
        t = int(t)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows)); axes = np.atleast_1d(axes).ravel()
        for i in range(A):
            _draw_action_marginal(axes[i], win(true_a, i, t), win(pred_a, i, t), names[i], spec[i], ylim=ylim[i])
        for j in range(A, len(axes)):
            axes[j].axis("off")
        fig.suptitle(f"per-dim action marginals  ·  t={t}/{steps - 1}" + (f"  (±{window} pooled)" if window else ""),
                     fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.set_dpi(dpi)
        frames.append(_fig_rgb(fig)); plt.close(fig)
    return np.stack(frames)
