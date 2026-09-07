"""RoboCasa specifics kept OUT of the env file — the `torus_utils` / `owm_physics` pattern.

WHY THIS FILE EXISTS. `examples/robocasa.py` is meant to be readable as the reference for "I have a heavy
third-party simulator" (docs/byo.md). The two things that would bury that — the 16-dim observation packing
and the diagnostic scene projection/drawing — live here instead, exactly as `torus_utils.py` (324 lines)
keeps `examples/torus.py` down to 208.

THE OBSERVATION LAYOUT IS NOT GUESSABLE AND WAS NOT DOCUMENTED. `madang6/quickdraw-robocasa-scene4-4h`
ships a 16-dim `observation.state` with NO `names` metadata. It was recovered by fingerprinting 29,294 val
frames and then confirmed against a live `robosuite.make` at layout 4 / style 4:

    dims  0:3   robot0_base_pos           world xyz; z pinned 0.70 (base on the floor)
    dims  3:7   robot0_base_quat          |q| = 1.00000 exactly, dims 3,4 IDENTICALLY zero -> yaw-only
    dims  7:10  robot0_base_to_eef_pos    sub-metre, BASE frame -- see the trap below
    dims 10:14  robot0_base_to_eef_quat   |q| = 1.00000, w (dim 13) never negative -> canonical sign
    dims 14:16  robot0_gripper_qpos       dims 14,15 are mirror images (the two jaw joints)

THE TRAP: `robot0_eef_pos` also exists, is world-frame, and looks perfectly reasonable. Using it produces
an env that resets, steps and renders happily while emitting observations the trained model cannot
interpret -- world eef y tracks the base at ~-3.5 where the data has values near 0. The dataset wants
`robot0_base_to_eef_pos`. There is no error to catch this; only the value ranges give it away.
"""

from __future__ import annotations

import numpy as np

# The recovered layout, as (obs key, width) in concat order. THE single source of truth for the mapping;
# `obs_slices` derives the offsets so nothing hardcodes an index twice.
OBS_LAYOUT: tuple[tuple[str, int], ...] = (
    ("robot0_base_pos", 3),
    ("robot0_base_quat", 4),
    ("robot0_base_to_eef_pos", 3),
    ("robot0_base_to_eef_quat", 4),
    ("robot0_gripper_qpos", 2),
)
OBS_DIM = sum(w for _, w in OBS_LAYOUT)          # 16


def obs_slices() -> dict[str, slice]:
    """{obs key: slice into the 16-dim vector}. Derived, never written out by hand."""
    out, i = {}, 0
    for k, w in OBS_LAYOUT:
        out[k], i = slice(i, i + w), i + w
    return out


def pack_obs(raw: dict) -> np.ndarray:
    """One robosuite obs dict -> the (16,) float32 vector the dataset stores.

    Raises on a missing key rather than padding: a silently short observation would train/evaluate against
    garbage, and every key here is present in every robocasa kitchen env."""
    missing = [k for k, _ in OBS_LAYOUT if k not in raw]
    if missing:
        raise KeyError(f"robocasa obs is missing {missing}; got {sorted(raw)[:12]}...")
    return np.concatenate([np.asarray(raw[k], dtype=np.float32).ravel() for k, _ in OBS_LAYOUT])


def quat_angle_error(q_pred: np.ndarray, q_true: np.ndarray) -> np.ndarray:
    """Geodesic angle (radians) between two quaternion batches (..., 4), sign-invariant.

    theta = 4 * atan2(min(a, b), max(a, b))  with  a = ||q_p - q_t||,  b = ||q_p + q_t||.

    NOT the textbook `2*acos(|<q_p, q_t>|)`. That form is ILL-CONDITIONED exactly where it matters most:
    acos has infinite derivative at 1, so float32 error in the dot product of two IDENTICAL unit
    quaternions (1 +- 1e-7) comes out as ~1e-3 radians of angle that is not there. Measured -- it is what
    made `smoke/robocasa_env.py`'s "identical obs -> zero error" check fail. On a metric we intend to
    READ per horizon step, a phantom 1e-3 rad noise floor is a real defect, not a rounding curiosity.

    atan2 is well conditioned everywhere, and taking min/max of the sum and difference norms gets
    sign-invariance for free -- q and -q are the same rotation, and a model's prediction does not respect
    the dataset's canonical-w convention. Identity in, exact zero out."""
    a = np.linalg.norm(q_pred - q_true, axis=-1)
    b = np.linalg.norm(q_pred + q_true, axis=-1)
    return 4.0 * np.arctan2(np.minimum(a, b), np.maximum(a, b))


# -------------------------------------------------------------------------------------------------------
# Diagnostic scene: two 2-D panels drawn from the WORLD-space overlay paths.
#
# WHY 2-D PANELS AND NOT A CAMERA RENDER WITH PROJECTED PATHS. Projecting world xyz into a simulator
# camera needs that camera's intrinsics AND the frame it rendered, per step, which means one offscreen
# render per diagnostic frame -- at ~117 ms/step that dominates the eval it is meant to diagnose. The
# question these panels answer ("did the base drive where the model thought it did?") is a plan-view
# question, and a plan view of the kitchen's own extent answers it at zero render cost.
# -------------------------------------------------------------------------------------------------------
def scene_panels(overlay, views, extent: tuple[float, float, float, float],
                 role_style: dict, title: str | None = None) -> dict:
    """`render_diagnostics` body: {view name: (H,W,3) uint8} for the views requested.

    "floor" = plan view (world x-y), which is where base drive error shows. "side" = elevation (x-z),
    where the arm's height and the base's flatness show. Paths are styled by ROLE (base.ROLE_STYLE), so
    true/oracle come out black and pred/learned grey, matching every other renderer in the project.
    """
    want = [v for v in ("floor", "side") if v in views]
    if not want or not (overlay.agents or overlay.markers):
        return {}
    from matplotlib.backends.backend_agg import FigureCanvasAgg      # lazy: keep this module import-light
    from matplotlib.figure import Figure

    x0, x1, y0, y1 = extent
    out = {}
    for view in want:
        fig = Figure(figsize=(4.8, 4.8), dpi=100)
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        ai, bi = (0, 1) if view == "floor" else (0, 2)               # which world axes this panel shows
        for role, path in (overlay.agents or {}).items():
            p = np.asarray(path.detach().cpu() if hasattr(path, "detach") else path, dtype=float)
            if p.ndim != 2 or p.shape[-1] < 3:
                continue
            st = role_style.get(role, {"color": "tab:blue"})
            ax.plot(p[:, ai], p[:, bi], color=st.get("color", "tab:blue"), lw=1.8, label=role, zorder=3)
            ax.scatter(p[:1, ai], p[:1, bi], s=28, color=st.get("color", "tab:blue"), zorder=4)  # start
        for role, pts in (overlay.markers or {}).items():
            m = np.asarray(pts.detach().cpu() if hasattr(pts, "detach") else pts, dtype=float)
            if m.ndim == 1:
                m = m[None]
            st = role_style.get(role, {"color": "gold"})
            ax.scatter(m[:, ai], m[:, bi], s=70, facecolors="none",
                       edgecolors=st.get("color", "gold"), lw=2.0, zorder=5, label=role)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1) if view == "floor" else ax.set_ylim(0.0, 2.2)
        ax.set_xlabel("world x (m)")
        ax.set_ylabel("world y (m)" if view == "floor" else "world z (m)")
        ax.set_title(f"{title + ' — ' if title else ''}{'plan (x-y)' if view == 'floor' else 'elevation (x-z)'}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)
        if overlay.agents or overlay.markers:
            ax.legend(loc="upper right", fontsize=7)
        fig.tight_layout()
        fig.canvas.draw()
        out[view] = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    return out
