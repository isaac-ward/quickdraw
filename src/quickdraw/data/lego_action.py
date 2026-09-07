"""Recover the COMMANDED arm pose for swoosh-data/lego_assemblies.

THE PROBLEM. The exported `action` is the raw Quest controller pose in the HEADSET's world frame
(`right_action_reference_xarm == "raw_world"`, and `right_action_xarm` is byte-identical to the first 8
of `quest_observation_state_xarm`). `observation.state` is the measured TCP in the ROBOT BASE frame, mm.
Nothing fixed relates them, so action-conditioning trains on noise -- see quickdraw#15 and
wizard/records/lego.md.

THE TRANSFORM, from the collection code shipped in the dataset's own `code/` directory
(`xarm/quest_anchor.py` states it outright):

    arm_target = robot_neutral + M_side @ (position_scale * (quest_now - quest_at_calibration))

Every term is recoverable:
  * `M_side`, `position_scale=2.0`   -- constants in `xarm/direct_teleop.py`, reproduced below.
  * `quest_now - quest_at_calibration` -- NOT something we must re-derive: the Quest app already
    publishes it per frame as `Pose_to_CalibrationBase_{R,L}` in `raw_streams/`.
  * `robot_neutral` -- the arm pose captured at each calibrate/deadman press. `_recapture_neutral`
    fires on EVERY deadman re-press, so it is PIECEWISE CONSTANT, not one value per episode. That is
    the whole difficulty: fitting a single offset per episode leaves a 116 mm residual; solving one per
    deadman segment leaves 5.5 mm.

MEASURED on session_13-36-22 (right arm, 5991 frames @100 Hz, TCP motion range 342/370/693 mm):

    do-nothing (constant TCP)      239.6 mm
    single offset for the episode  116.4 mm
    per-deadman-segment offset       5.5 mm median, 19.4 mm p90

So the command is recovered to about half a centimetre, and the action becomes
`arm_target - measured_tcp` -- the same quantity ABC-130k publishes directly as
`/{side}-arm-action.position - /{side}-arm-state.position`.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np

# --- constants lifted verbatim from code/src/retriever_data_collection/xarm/direct_teleop.py ---
# Per-arm Quest-world -> arm-base linear transforms; rows are arm-base axes in Quest coords. Each is the
# exact-fit Kabsch from that arm's 3 calibration translation deltas (2026-06-17 recording).
POSITION_AXES = {
    "right": np.array([[0.0, 0.0, 1.0],
                       [-0.7071, 0.7071, 0.0],
                       [0.7071, 0.7071, 0.0]]),
    "left":  np.array([[0.0, 0.0, 1.0],
                       [-0.7071, -0.7071, 0.0],
                       [-0.7071, 0.7071, 0.0]]),
}
POSITION_SCALE = 2.0        # teleop moves the arm 2x the hand
# THE DEADMAN IS THE HAND TRIGGER (grip), not the index trigger. direct_teleop._deadman_pressed reads
# `Axis_HandTrigger_{R,L}` against `deadman_threshold = 0.5`. Using the index trigger instead gives a
# FLATTERING residual (5.5 vs 8.4 mm on session_13-36-22) while covering only 23% of frames against the
# grip's 43% -- it selects a subset of genuinely-commanded frames rather than the right ones.
DEADMAN_KEY = {"right": "Axis_HandTrigger_R", "left": "Axis_HandTrigger_L"}
P2CB_KEY = {"right": "Pose_to_CalibrationBase_R", "left": "Pose_to_CalibrationBase_L"}
DEADMAN_ON = 0.5            # trigger axis threshold
MIN_SEG = 50                # frames; shorter engaged runs are noise, not a teleop segment
# THE ARM LAGS THE COMMAND BY 50 ms. Measured by sweeping the offset over 5 episodes: the residual
# minimises sharply at 5 frames @100 Hz (17.0 -> 6.0 mm mean, and 31.2 -> 11.0 on the worst episode),
# rising steeply either side. 50 ms is EXACTLY `pose_smoothing_tau_sec = 0.05` in direct_teleop.py, so
# this is the documented command smoothing showing up in the data, not a fitted fudge factor.
LAG_FRAMES = 5
M_TO_MM = 1000.0


def load_session(session_dir: str):
    """One raw session -> dict of parallel arrays at the 100 Hz collection rate."""
    f = glob.glob(os.path.join(session_dir, "lerobot_jsonl/data/chunk-000/*.jsonl"))
    if not f:
        raise FileNotFoundError(f"no episode jsonl under {session_dir}")
    out = {k: [] for k in ("p2cb_right", "p2cb_left", "tcp_right", "tcp_left",
                           "quat_right", "quat_left", "rpy_right", "rpy_left",
                           "grip_right", "grip_left", "cgrip_right", "cgrip_left",
                           "dead_right", "dead_left", "state")}
    for line in open(f[0]):
        d = json.loads(line)
        fr, rs = d["frame"], d.get("robot_state_xarm") or {}
        r, l = rs.get("right"), rs.get("left")
        if not (r and l and r.get("position") and l.get("position")):
            continue
        if len(r["position"]) < 6 or len(l["position"]) < 6:
            continue
        for side, rs_ in (("right", r), ("left", l)):
            out[f"p2cb_{side}"].append(fr[P2CB_KEY[side]][:3])
            out[f"quat_{side}"].append(fr[P2CB_KEY[side]][3:7])   # controller orientation
            out[f"tcp_{side}"].append(rs_["position"][:3])
            out[f"rpy_{side}"].append(rs_["position"][3:6])       # measured TCP rpy, degrees (xyz)
            out[f"grip_{side}"].append(rs_.get("gripper_position_norm", 0.0) or 0.0)
        # commanded grip: the Quest hand-trigger axis, already 0..1 (direct_teleop notes it shares the
        # observation's scale), which is what `actions/<side>/grip` carries in the export.
        out["cgrip_right"].append(fr.get("Axis_HandTrigger_R", 0.0) or 0.0)
        out["cgrip_left"].append(fr.get("Axis_HandTrigger_L", 0.0) or 0.0)
        out["dead_right"].append(fr.get(DEADMAN_KEY["right"], 0.0))
        out["dead_left"].append(fr.get(DEADMAN_KEY["left"], 0.0))
        out["state"].append(d.get("robot_observation_state_xarm") or [np.nan] * 28)
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def segments(engaged: np.ndarray, min_len: int = MIN_SEG):
    """Maximal runs where the deadman is held. Each gets its OWN robot_neutral."""
    on = engaged > DEADMAN_ON
    edges = np.flatnonzero(np.diff(on.astype(int)) != 0) + 1
    return [s for s in np.split(np.arange(len(on)), edges) if len(s) >= min_len and on[s[0]]]


MAX_RESID_MM = 20.0     # a reconstructed command further than this from the TCP it produced is not
#                         trustworthy: a Quest re-anchor (quest_anchor.py), a max_pose_jump_m clamp, or a
#                         tracking dropout. Masking on the RESIDUAL catches all three without having to
#                         classify which -- and an unmasked bad frame is a wrong label, not a missing one.
MAX_HOLD_SPEED_MM_S = 5.0   # with the deadman released the arm holds: median TCP speed is exactly 0.00
#                         and 107.3 m of 115.1 m total path is travelled under deadman. The 0.9% of
#                         released frames that DO move are release-edge transitions, so they are masked
#                         rather than labelled as commanded stillness.


def reconstruct(p2cb: np.ndarray, tcp: np.ndarray, engaged: np.ndarray, side: str,
                lag: int = LAG_FRAMES):
    """-> (arm_target mm, valid mask, per-segment residuals).

    `arm_target` is only DEFINED while the deadman is held: with it released the arm holds position and
    the controller keeps moving, so there is no command to recover. Those frames are masked out rather
    than extrapolated, and the right action for them is a ZERO delta (the arm was told to stay put)."""
    pred_rel = (POSITION_AXES[side] @ (POSITION_SCALE * p2cb).T).T * M_TO_MM
    target = np.full_like(tcp, np.nan)
    resid = []
    for s in segments(engaged):
        s = s[s + lag < len(tcp)]                           # the TCP that answers this command is `lag` later
        if len(s) < MIN_SEG:
            continue
        off = np.median(tcp[s + lag] - pred_rel[s], axis=0)  # robot_neutral for THIS segment
        target[s] = off + pred_rel[s]
        resid.append(np.linalg.norm(tcp[s + lag] - target[s], axis=1))
    valid = ~np.isnan(target[:, 0])
    # drop frames the reconstruction cannot vouch for
    bad = valid.copy()
    bad[valid] = np.linalg.norm(tcp[np.flatnonzero(valid) + lag] - target[valid], axis=1) > MAX_RESID_MM
    target[bad] = np.nan
    valid = ~np.isnan(target[:, 0])
    return target, valid, (np.concatenate(resid) if resid else np.array([]))


def reconstruct_rot(quat: np.ndarray, rpy: np.ndarray, engaged: np.ndarray, side: str,
                    lag: int = LAG_FRAMES):
    """Commanded TCP ORIENTATION, per deadman segment. -> (target rotations, valid mask).

    From `_orientation_target_rpy`: delta = q_now * conj(q_neutral), the rotvec is re-expressed in base
    coords by the SAME axis map used for translation, then NEGATED because those matrices are
    reflections (det = -1.0000, verified) so they map the axis correctly but invert the rotation sense.
    `orientation_scale` is 1.0, so it drops out. Both neutrals are read at the segment's first frame,
    which is where `_recapture_neutral` captures them -- a direct read, not a fit.
    Measured: 0.96 deg (right) / 1.58 deg (left) median-of-medians, every episode under 10 deg."""
    from scipy.spatial.transform import Rotation as Rot

    M = POSITION_AXES[side]
    tgt = np.full((len(rpy), 3, 3), np.nan)
    for seg in segments(engaged):
        seg = seg[seg + lag < len(rpy)]
        if len(seg) < MIN_SEG:
            continue
        q0 = Rot.from_quat(quat[seg[0]])
        R0 = Rot.from_euler("xyz", rpy[seg[0] + lag], degrees=True)
        rv = -(M @ (Rot.from_quat(quat[seg]) * q0.inv()).as_rotvec().T).T
        tgt[seg] = (Rot.from_rotvec(rv) * R0).as_matrix()
    return tgt, ~np.isnan(tgt[:, 0, 0])


def hold_mask(tcp: np.ndarray, engaged: np.ndarray, dt: float = 0.01) -> np.ndarray:
    """Frames where the deadman is RELEASED and the arm is genuinely stationary -> action is a ZERO delta.

    Released-but-moving frames are excluded: the arm is being moved by something the command does not
    explain, so labelling them "commanded to hold" would be a wrong label."""
    speed = np.zeros(len(tcp))
    speed[1:] = np.linalg.norm(np.diff(tcp, axis=0), axis=1) / dt
    return (engaged <= DEADMAN_ON) & (speed <= MAX_HOLD_SPEED_MM_S)


def episode_action(session_dir: str, n_out: int) -> tuple[np.ndarray, np.ndarray]:
    """One raw session -> (action (n_out, 20) float32, valid (n_out,) bool) on the dataset's 30 Hz grid.

    Per arm: [dxyz_mm(3), 6D(dR)(6), dgrip(1)] = 10, so 20 for the pair. The delta is
    COMMANDED-minus-MEASURED in the ROBOT BASE frame -- the same quantity ABC-130k publishes directly as
    `/{side}-arm-action.position - /{side}-arm-state.position`, which is what makes the two datasets
    comparable. Held frames get an exact zero translation delta and an IDENTITY rotation (the arm was
    told to stay put); frames the reconstruction cannot vouch for are marked invalid, never guessed.

    A frame is usable only if BOTH arms are usable: the model consumes one 20-dim vector, so a
    half-valid row would silently feed a wrong label for one arm.
    """
    from scipy.spatial.transform import Rotation as Rot

    from .rotations import matrix_to_6d

    S = load_session(session_dir)
    n_raw = len(S["tcp_right"])
    out = np.zeros((n_raw, 20), dtype=np.float64)
    eye6 = matrix_to_6d(np.eye(3))
    ok_side = {}

    for k, side in enumerate(("right", "left")):
        tcp, rpy = S[f"tcp_{side}"], S[f"rpy_{side}"]
        dead, p2cb, quat = S[f"dead_{side}"], S[f"p2cb_{side}"], S[f"quat_{side}"]
        tgt, vpos, _ = reconstruct(p2cb, tcp, dead, side)
        rtgt, vrot = reconstruct_rot(quat, rpy, dead, side)
        held = hold_mask(tcp, dead)
        cmd = vpos & vrot
        b = k * 10

        out[:, b + 3:b + 9] = eye6                                    # default: no rotation delta
        out[cmd, b:b + 3] = tgt[cmd] - tcp[cmd]                       # commanded - measured, mm
        meas_R = Rot.from_euler("xyz", rpy[cmd], degrees=True)
        dR = Rot.from_matrix(rtgt[cmd]) * meas_R.inv()                # relative rotation, base frame
        out[cmd, b + 3:b + 9] = matrix_to_6d(dR.as_matrix())
        out[cmd, b + 9] = S[f"cgrip_{side}"][cmd] - S[f"grip_{side}"][cmd]
        out[held, b:b + 3] = 0.0                                      # explicit hold
        out[held, b + 3:b + 9] = eye6
        out[held, b + 9] = 0.0
        ok_side[side] = cmd | held

    ok = ok_side["right"] & ok_side["left"]
    # 100 Hz -> the dataset's 30 Hz grid, nearest sample (the export uses a zero-order hold)
    idx = np.clip((np.arange(n_out) * (n_raw / max(n_out, 1))).astype(int), 0, n_raw - 1)
    return out[idx].astype(np.float32), ok[idx]
