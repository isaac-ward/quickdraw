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
MIN_SEG = 10                # frames; the neutral is READ at the segment start, not fitted, so a
#                             short run is still reconstructable. Measured: runs below 50 frames are 6%
#                             of runs but only 0.1% of held frames, so this threshold barely binds.
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


# NO RESIDUAL MASK, and NO SPEED MASK -- both were tried and both were CONCEPTUALLY WRONG.
#
# The residual is |reconstructed_command - measured_TCP|. A large value does NOT mean the reconstruction
# is bad; it means the ARM DID NOT KEEP UP, which happens during fast motion and is a robot-side fact.
# Dropping those frames threw away exactly the fast-motion frames a dynamics model most needs. Measured:
# it invalidated 2.6% of held frames per arm.
#
# Likewise a released deadman means the command IS "hold", whether or not the arm drifted afterwards.
# Masking the 0.6% of released-but-moving frames removed correct labels on the grounds that the ROBOT
# misbehaved.
#
# Together those two masks cost only ~3.2% per arm, but ANDed across both arms they scattered validity
# frame-to-frame: 12,442 contiguous valid runs with a MEDIAN LENGTH OF 5 FRAMES, which left 18.4% of
# windows intact and made the channel unusable for training. The lesson is that a per-frame mask on a
# windowed dataset is far more expensive than its own percentage suggests.
#
# The residual is still computed and REPORTED -- it is the quality measure for the recovery (3.4 mm) --
# it just no longer deletes data.


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


def hold_mask(engaged: np.ndarray) -> np.ndarray:
    """Frames where the deadman is RELEASED -> the command is "hold", i.e. a ZERO delta.

    This is a statement about what the robot was TOLD, so it does not depend on what the arm then did.
    Measured: median TCP speed with the deadman released is exactly 0.00 mm/s and 107.3 m of the 115.1 m
    of total path is travelled under deadman, so the arm does overwhelmingly hold -- but the label is
    correct even on the 0.6% of frames where it drifts."""
    return engaged <= DEADMAN_ON


def episode_action(session_dir: str, n_out: int) -> tuple[np.ndarray, np.ndarray]:
    """One raw session -> (action (n_out, 20) float32, valid (n_out,) bool) on the dataset's 30 Hz grid.

    Per arm: [tcp_xyz_mm(3), 6D(R)(6), grip(1)] = 10, so 20 for the pair -- the ABSOLUTE COMMANDED TCP
    POSE in the ROBOT BASE frame, the same frame and units as `observation.state`.

    WHY ABSOLUTE AND NOT A DELTA, having first built the delta version. `cmd(t) - meas(t)` is better
    conditioned for conditioning (zero-centred, small) but it does not SUBSAMPLE: summing six such
    deltas adds six differences taken against six different reference positions, which means nothing,
    and `last` silently discards the five commands in between. The correct decimated delta would be
    `cmd(t+s) - meas(t)`, which is neither aggregation. Absolute has no such problem -- `action_aggregate
    = "last"` is exactly right, because the command standing at the end of the group IS the command --
    and the delta stays available downstream at zero cost, since the model already holds the state in
    the same frame. It is also what ABC-130k publishes, so the two datasets stay comparable.

    HOLDS NEED NO SPECIAL CASE. With the deadman released the command is "stay here", so the commanded
    pose IS the measured pose. That falls out of the representation instead of being encoded as a magic
    zero, and it is why coverage is 99.9% rather than the 92% the delta version managed.
    """
    from scipy.spatial.transform import Rotation as Rot

    from .rotations import matrix_to_6d

    S = load_session(session_dir)
    n_raw = len(S["tcp_right"])
    out = np.zeros((n_raw, 20), dtype=np.float64)
    ok_side = {}

    for k, side in enumerate(("right", "left")):
        tcp, rpy = S[f"tcp_{side}"], S[f"rpy_{side}"]
        dead, p2cb, quat = S[f"dead_{side}"], S[f"p2cb_{side}"], S[f"quat_{side}"]
        tgt, vpos, _ = reconstruct(p2cb, tcp, dead, side)
        rtgt, vrot = reconstruct_rot(quat, rpy, dead, side)
        held = hold_mask(dead)
        cmd = vpos & vrot & ~held           # a reconstructed command
        b = k * 10

        # commanded pose while the deadman is held
        out[cmd, b:b + 3] = tgt[cmd]
        out[cmd, b + 3:b + 9] = matrix_to_6d(rtgt[cmd])
        out[cmd, b + 9] = S[f"cgrip_{side}"][cmd]
        # "hold" == commanded pose is the CURRENT pose
        out[held, b:b + 3] = tcp[held]
        out[held, b + 3:b + 9] = matrix_to_6d(
            Rot.from_euler("xyz", rpy[held], degrees=True).as_matrix())
        out[held, b + 9] = S[f"grip_{side}"][held]
        ok_side[side] = cmd | held

    ok = ok_side["right"] & ok_side["left"]
    idx = np.clip((np.arange(n_out) * (n_raw / max(n_out, 1))).astype(int), 0, n_raw - 1)
    return out[idx].astype(np.float32), ok[idx]
