"""One Swoosh right-arm recording -> (states, actions, frames) on a single uniform grid.

Written for the `longhand` corpus: an xArm7 right arm mounted 45 deg clockwise from vertical,
teleoperated with an Xbox pad, four USB cameras (two scene, two on the gripper). The collection
side lives in the `xarm7-data-collection` repo; this file only READS what it wrote.

WHAT A RUN DIRECTORY CONTAINS
  run.json                       provenance, per-camera summary, 32 validation checks
  raw/controller.jsonl    100 Hz the Xbox pad, post-deadzone/expo    <- THE ACTION
  raw/commanded.jsonl     100 Hz the integrated target pose the pad asked for
  raw/xarm_command.jsonl  100 Hz the literal SDK arguments sent
  raw/arm_state.jsonl      50 Hz what the arm reported back          <- THE OBSERVATION
  raw/tick.jsonl          100 Hz loop timing
  video/<label>.mp4        30 Hz + <label>_frame_times.json (one stamp per mp4 frame)

EVERY stream shares one monotonic clock origin (`t_loop0`), cameras included, so no cross-stream
alignment is needed here -- only resampling. Camera stamps come from the V4L2 kernel buffer at
capture, so the ~21 ms userspace read lag is ALREADY REMOVED and nothing should be shifted.

RESAMPLING IS NEAREST-IN-TIME, NEVER INTERPOLATED. Every stream runs at 1.7-3.3x the 30 Hz grid,
so interpolation would invent precision; and `joints_real_deg`, quaternion signs and the gripper's
20 Hz poll all interpolate badly. The grid spans the window where ALL FOUR cameras have frames --
the cameras open staggered (0.25 s apart by config) so the arm streams start ~1-2 s earlier, and a
step with no frame is useless to a visual world model. `sync_error_s` reports the worst nearest-
neighbour gap actually incurred, so a bad run is visible rather than silently smoothed.

THREE FIELD CHOICES THAT ARE NOT OBVIOUS, all learned the hard way on the lego corpus:
  * `joints_real_deg`, not `joints_deg`. The latter is the controller's PLANNED angle and was
    measured diverging from the real one by up to 5.02 deg.
  * orientation as 6D (Zhou et al. -- first two columns of the rotation matrix), not euler. Euler
    wrapped 702 times in a single lego arm stream, and every wrap is a discontinuity the model has
    to spend capacity memorising.
  * `target_yaw_world_deg` is NOT absolute -- it resets on re-home, clear-errors and servo recovery.
    Absolute orientation only ever comes from `pose_base_mm_deg[3:6]`.
"""

from __future__ import annotations

import json
import os

import numpy as np

TARGET_HZ_DEFAULT = 30.0          # = camera fps = recording.export_rate_hz on the collection side
OUT_HW_DEFAULT = (144, 192)       # 4:3 like the 640x480 source, both sides divisible by 16

ALL_CAMERAS = ("scene_left", "scene_right", "gripper_right_bottom", "gripper_right_top")

# The 45-degree mount, from swoosh_collect/frames.py. Every run.json carries this under
# provenance.R_WORLD_FROM_BASE and we assert against it rather than trusting this copy.
_S = float(np.sqrt(0.5))
R_WORLD_FROM_BASE = np.array([[1.0, 0.0, 0.0], [0.0, _S, -_S], [0.0, _S, _S]])

# state layout, documented once so a consumer can slice it
STATE_KEYS = (
    ["ee_x_mm", "ee_y_mm", "ee_z_mm"]                       # 0:3   world frame, mm
    + [f"ee_rot6_{i}" for i in range(6)]                    # 3:9   world frame, 6D rotation
    + ["gripper"]                                           # 9     0 = closed, 1 = open
    + [f"joint{i + 1}_rad" for i in range(7)]               # 10:17 joints_real_deg, radians
)
ACTION_KEYS = ["move_x", "move_y", "height", "yaw", "gripper"]   # the Xbox pad, per spec


def _jsonl(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def _nearest(src_t: np.ndarray, grid: np.ndarray, bound: float) -> tuple[np.ndarray, dict]:
    """-> (index of the nearest sample to each grid point, displacement stats).

    Reporting the MAX alone is misleading: one dropped sample anywhere in a two-minute run puts
    the max at 3x the typical value and makes a clean run look broken. `over` -- the fraction of
    steps displaced by more than `bound` (half the source's own sampling period, the best any
    nearest-neighbour resample can do) -- is what actually distinguishes a hiccup from a stream
    that is systematically too slow for the grid.
    """
    j = np.searchsorted(src_t, grid)
    lo = np.clip(j - 1, 0, len(src_t) - 1)
    hi = np.clip(j, 0, len(src_t) - 1)
    pick = np.where(np.abs(src_t[lo] - grid) <= np.abs(src_t[hi] - grid), lo, hi)
    err = np.abs(src_t[pick] - grid)
    return pick, {"max": float(err.max()), "p99": float(np.percentile(err, 99)),
                  "over": float(np.mean(err > bound)), "bound": float(bound)}


def _half_period(t: np.ndarray) -> float:
    """Half the stream's own median sampling period -- its irreducible nearest-neighbour error."""
    return 0.5 * float(np.median(np.diff(t))) if len(t) > 1 else 0.0


def _rpy_to_rot6_world(rpy_deg: np.ndarray, R_wb: np.ndarray) -> np.ndarray:
    """(T,3) roll/pitch/yaw in the ARM BASE frame -> (T,6) 6D rotation in the WORLD frame.

    The xArm reports RPY as an X-Y-Z fixed-axis (extrinsic) triple, i.e. R = Rz @ Ry @ Rx.
    6D = the first two columns of R, which is what Zhou et al. show is continuous; the third
    column is recoverable by cross product, so nothing is lost.
    """
    r, p, y = np.radians(rpy_deg).T
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    R = np.empty((len(r), 3, 3))
    R[:, 0, 0], R[:, 0, 1], R[:, 0, 2] = cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr
    R[:, 1, 0], R[:, 1, 1], R[:, 1, 2] = sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr
    R[:, 2, 0], R[:, 2, 1], R[:, 2, 2] = -sp, cp * sr, cp * cr
    Rw = np.einsum("ij,tjk->tik", R_wb, R)
    return np.concatenate([Rw[:, :, 0], Rw[:, :, 1]], axis=1)


def _decode_at(mp4: str, want: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Pull mp4 frames at the (non-decreasing) indices `want`, resized to out_hw.

    Sequential single pass: `want` is monotonic because the grid is, so the decoder never seeks
    backwards. Repeats are copied rather than re-decoded.
    """
    import cv2
    import imageio.v2 as imageio

    h, w = out_hw
    out = np.zeros((len(want), h, w, 3), np.uint8)
    rdr = imageio.get_reader(mp4)
    try:
        cur, frame, k = -1, None, 0
        for idx, fr in enumerate(rdr):
            while k < len(want) and want[k] == idx:
                if cur != idx:
                    frame = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
                    cur = idx
                out[k] = frame
                k += 1
            if k >= len(want):
                break
        if k < len(want):        # mp4 ended early: hold the last decoded frame
            out[k:] = out[k - 1] if k else 0
    finally:
        rdr.close()
    return out


def read_run(run_dir: str, target_hz: float = TARGET_HZ_DEFAULT,
             out_hw: tuple[int, int] = OUT_HW_DEFAULT,
             cameras: tuple[str, ...] = ALL_CAMERAS,
             want_frames: bool = True) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """-> (states (T,17) f32, actions (T,5) f32, {camera: (T,h,w,3) uint8} | None, info)."""
    run = json.load(open(os.path.join(run_dir, "run.json")))

    R_wb = np.asarray(run.get("provenance", {}).get("R_WORLD_FROM_BASE") or R_WORLD_FROM_BASE)
    assert np.allclose(R_wb, R_WORLD_FROM_BASE, atol=1e-9), (
        f"{run_dir}: R_WORLD_FROM_BASE differs from the mount this reader assumes:\n{R_wb}")

    ctl = _jsonl(os.path.join(run_dir, "raw", "controller.jsonl"))
    arm = _jsonl(os.path.join(run_dir, "raw", "arm_state.jsonl"))
    if not ctl or not arm:
        raise ValueError(f"{run_dir}: empty controller or arm_state stream")

    cam_t: dict[str, np.ndarray] = {}
    for c in cameras:
        p = os.path.join(run_dir, "video", f"{c}_frame_times.json")
        if not os.path.exists(p):
            raise ValueError(f"{run_dir}: camera {c} has no frame_times.json")
        d = json.load(open(p))
        if d.get("error"):
            raise ValueError(f"{run_dir}: camera {c} recorded an error: {d['error']}")
        t = np.asarray(d["t"], dtype=np.float64)
        if len(t) < 2:
            raise ValueError(f"{run_dir}: camera {c} has {len(t)} frames")
        cam_t[c] = t

    ct = np.array([r["t"] for r in ctl], dtype=np.float64)
    at = np.array([r["t"] for r in arm], dtype=np.float64)

    # The grid spans where EVERY stream is live: cameras open staggered, so this is normally
    # bounded below by the last camera to start and above by the first to stop.
    t_lo = max([ct[0], at[0]] + [v[0] for v in cam_t.values()])
    t_hi = min([ct[-1], at[-1]] + [v[-1] for v in cam_t.values()])
    if t_hi - t_lo < 2.0:
        raise ValueError(f"{run_dir}: only {t_hi - t_lo:.2f}s of overlap across streams")
    grid = np.arange(t_lo, t_hi, 1.0 / target_hz)
    T = len(grid)

    i_ctl, e_ctl = _nearest(ct, grid, _half_period(ct))
    i_arm, e_arm = _nearest(at, grid, _half_period(at))

    # -- actions: the Xbox pad, exactly the five axes the operator drove --------------------
    actions = np.array([[ctl[i][k] for k in ACTION_KEYS] for i in i_ctl], dtype=np.float32)

    # -- states: what the arm actually reported ---------------------------------------------
    a = [arm[i] for i in i_arm]
    ee = np.array([r["pose_world_xyz_mm"] for r in a], dtype=np.float64)
    rot6 = _rpy_to_rot6_world(np.array([r["pose_base_mm_deg"][3:6] for r in a]), R_wb)
    # gripper_pos is the RAW servo count and its polarity is opposite the action's: 850 = open,
    # 0 = closed, while action gripper 1 = squeeze. Normalise to 1 = open so the state is
    # monotonic in aperture, and leave the action alone -- they are different quantities.
    gcfg = run.get("provenance", {}).get("config", {}).get("gripper", {})
    g_open = float(gcfg.get("open_position", 850) or 850)
    g_closed = float(gcfg.get("closed_position", 0) or 0)
    grip = np.array([r["gripper_pos"] for r in a], dtype=np.float64)
    grip = np.clip((grip - g_closed) / max(g_open - g_closed, 1e-6), 0.0, 1.0)
    joints = np.radians(np.array([r["joints_real_deg"] for r in a], dtype=np.float64))

    states = np.concatenate([ee, rot6, grip[:, None], joints], axis=1).astype(np.float32)
    assert states.shape[1] == len(STATE_KEYS), (states.shape, len(STATE_KEYS))

    # -- frames ------------------------------------------------------------------------------
    frames, e_cam = None, {}
    for c in cameras:
        idx, err = _nearest(cam_t[c], grid, _half_period(cam_t[c]))
        e_cam[c] = err
        if want_frames:
            mp4 = os.path.join(run_dir, "video", f"{c}.mp4")
            frames = frames or {}
            frames[c] = _decode_at(mp4, idx, out_hw)

    info = {
        "run": os.path.basename(run_dir),
        "steps": T,
        "seconds": float(t_hi - t_lo),
        "target_hz": target_hz,
        # Per-stream nearest-neighbour displacement: max, p99, and the fraction past that
        # stream's own half-period. A camera sitting at its bound is arithmetic, not a fault;
        # a large `over` is a stream too slow (or too gappy) for the grid.
        "sync_error_s": {"controller": e_ctl, "arm_state": e_arm, **e_cam},
        "validation_all_green": bool(run.get("validation", {}).get("all_green", False)),
        "git_sha": run.get("provenance", {}).get("git_sha"),
    }
    return states, actions, frames, info


def run_seconds(run_dir: str) -> float:
    """Overlap-window length WITHOUT decoding anything -- what the split rule sorts on.

    Deliberately not run.json's `duration_s`: that is wall time from A to B, which includes the
    1-2 s before the last camera opens. The split has to rank episodes by what will actually be
    in the dataset.
    """
    try:
        los, his = [], []
        for s in ("controller", "arm_state"):
            rows = _jsonl(os.path.join(run_dir, "raw", f"{s}.jsonl"))
            if not rows:
                return 0.0
            los.append(rows[0]["t"])
            his.append(rows[-1]["t"])
        for p in os.listdir(os.path.join(run_dir, "video")):
            if p.endswith("_frame_times.json"):
                t = json.load(open(os.path.join(run_dir, "video", p)))["t"]
                if len(t) < 2:
                    return 0.0
                los.append(t[0])
                his.append(t[-1])
        return max(0.0, min(his) - max(los))
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return 0.0
