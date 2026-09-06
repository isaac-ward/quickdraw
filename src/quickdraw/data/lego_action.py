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
                           "dead_right", "dead_left", "state")}
    for line in open(f[0]):
        d = json.loads(line)
        fr, rs = d["frame"], d.get("robot_state_xarm") or {}
        r, l = rs.get("right"), rs.get("left")
        if not (r and l and r.get("position") and l.get("position")):
            continue
        out["p2cb_right"].append(fr[P2CB_KEY["right"]][:3])
        out["p2cb_left"].append(fr[P2CB_KEY["left"]][:3])
        out["tcp_right"].append(r["position"][:3])
        out["tcp_left"].append(l["position"][:3])
        out["dead_right"].append(fr.get(DEADMAN_KEY["right"], 0.0))
        out["dead_left"].append(fr.get(DEADMAN_KEY["left"], 0.0))
        out["state"].append(d.get("robot_observation_state_xarm") or [np.nan] * 28)
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def segments(engaged: np.ndarray, min_len: int = MIN_SEG):
    """Maximal runs where the deadman is held. Each gets its OWN robot_neutral."""
    on = engaged > DEADMAN_ON
    edges = np.flatnonzero(np.diff(on.astype(int)) != 0) + 1
    return [s for s in np.split(np.arange(len(on)), edges) if len(s) >= min_len and on[s[0]]]


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
