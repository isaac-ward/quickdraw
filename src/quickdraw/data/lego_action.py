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

from .rotations import EULER_SEQ

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
MIN_SEG = 10
HOLD_MOVE_MM = 20.0   # a "hold" in which the arm travels further than this is not a hold at all -- see
                      # episode_pose. 20 mm is loose against the 3.4 mm recovery precision and tight
                      # against the 240-640 mm excursions the check actually catches.                # frames; the neutral is READ at the segment start, not fitted, so a
#                             short run is still reconstructable. Measured: runs below 50 frames are 6%
#                             of runs but only 0.1% of held frames, so this threshold barely binds.
# THE ARM LAGS THE COMMAND BY 50 ms. Measured by sweeping the offset over 5 episodes: the residual
# minimises sharply at 5 frames @100 Hz (17.0 -> 6.0 mm mean, and 31.2 -> 11.0 on the worst episode),
# rising steeply either side. 50 ms is EXACTLY `pose_smoothing_tau_sec = 0.05` in direct_teleop.py, so
# this is the documented command smoothing showing up in the data, not a fitted fudge factor.
LAG_FRAMES = 5
M_TO_MM = 1000.0


def load_session(session_dir: str):
    """One raw session -> dict of parallel arrays at the ~29.5 Hz collection rate.

    (Not 100 Hz, as this said while I believed it. Measured median dt is 0.0339 s.)"""
    f = glob.glob(os.path.join(session_dir, "lerobot_jsonl/data/chunk-000/*.jsonl"))
    if not f:
        raise FileNotFoundError(f"no episode jsonl under {session_dir}")
    out = {k: [] for k in ("p2cb_right", "p2cb_left", "tcp_right", "tcp_left",
                           "quat_right", "quat_left", "rpy_right", "rpy_left",
                           "grip_right", "grip_left", "cgrip_right", "cgrip_left",
                           "dead_right", "dead_left", "state", "t")}
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
        out["t"].append(d["timestamp"])
    out = {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}
    out["t"] -= out["t"][0]                 # session-relative seconds
    return out


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
        R0 = Rot.from_euler(EULER_SEQ, rpy[seg[0] + lag], degrees=True)
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


def align_to_dataset(S: dict, dts: np.ndarray, st: np.ndarray,
                     search: float = 4.0) -> tuple[np.ndarray, float, float]:
    """Map each DATASET row onto a RAW-SESSION row. -> (idx (n,), delta_seconds, exact_fraction).

    THE RAW SESSION AND THE EXPORTED EPISODE DO NOT SHARE A TIME ORIGIN. The session log starts before
    the episode does, by a per-episode offset that is ~0.75-0.85 s on most episodes and ~0 on a few. I
    originally mapped the two grids PROPORTIONALLY (`arange(n_out) * n_raw / n_out`), which is only
    correct when both cover the same span, and it silently placed every action row about 24 frames away
    from the observation it belongs to. Nothing downstream could detect it: the recovery residual is
    computed entirely in raw index space, so it still read 3.1 mm, while the action column landed on the
    wrong frames. It surfaced only as an action that failed to predict the motion it had caused --
    partial R-squared 0.003 against 0.018 for the raw controller pose it was supposed to beat.

    The offset is recovered from the DATA rather than assumed, by grid-searching delta for the value that
    maximises exact agreement between the dataset's `observation.state` and the raw log's
    `robot_observation_state_xarm` -- the same field the export was built from, so a correct delta gives
    a bit-exact match. Measured across all 74 episodes this lifts exact agreement from 3-8 percent to
    85-90 percent with a median error of 0.0000, and the residual disagreement is rows of the 30 Hz grid
    that fall between two irregular raw samples, where nearest-in-time is the right answer anyway.
    """
    rel, rst = S["t"], S["state"]

    def lookup(delta):
        want = dts + delta
        j = np.clip(np.searchsorted(rel, want), 0, len(rel) - 1)
        jm = np.clip(j - 1, 0, len(rel) - 1)
        return np.where(np.abs(rel[j] - want) <= np.abs(rel[jm] - want), j, jm)

    def exact(delta):
        idx = lookup(delta)
        return float((np.abs(rst[idx] - st).max(1) < 1e-3).mean())

    coarse = np.arange(-search, search, 1 / 120)
    best = float(coarse[int(np.argmax([exact(d) for d in coarse]))])
    fine = np.arange(best - 1 / 120, best + 1 / 120, 1 / 2400)      # refine within one coarse cell
    best = float(fine[int(np.argmax([exact(d) for d in fine]))])
    return lookup(best), best, exact(best)


def episode_pose(session_dir: str, dts: np.ndarray, st: np.ndarray) -> dict:
    """The COMMANDED TCP POSE, recovered from one raw session onto the dataset's own frame grid.

    `dts` and `st` are the episode's `timestamp` and `observation.state` columns; they are what pins the
    raw log to the exported grid (see `align_to_dataset`).

    -> {"xyz": (n,2,3) mm, "R": (n,2,3,3), "grip": (n,2), "valid": (n,)}, arms ordered (right, left),
    in the ROBOT BASE frame -- the same frame and units as `observation.state`.

    This is the single source of truth for the recovery; `episode_action` (6D, for training) and the
    published v2 `action` column (mm/deg, for humans) are both thin wrappers over it.

    WHY ABSOLUTE AND NOT A DELTA, having first built the delta version. `cmd(t) - meas(t)` is better
    conditioned for conditioning (zero-centred, small) but it does not SUBSAMPLE: summing s such deltas
    adds s differences taken against s different reference positions, which means nothing, and `last`
    silently discards the s-1 commands in between. The correct decimated delta would be
    `cmd(t+s) - meas(t)`, which is neither aggregation. Absolute has no such problem -- `last` is exactly
    right, because the command standing at the end of a group IS the command -- and the delta stays
    available downstream at zero cost, since the model already holds the state in the same frame. It is
    also what ABC-130k publishes, so the two datasets stay comparable.

    HOLDS ARE A FORWARD FILL, NOT THE MEASURED POSE. `direct_teleop.py` line 363 `continue`s the whole
    per-side block when the deadman is up, so NO command is sent and the arm holds the last one -- the
    gripper included, since it lives inside the same skipped block. So the command during a hold is the
    last commanded pose, held CONSTANT.

    I first wrote this as `cmd = measured TCP(t)`, reasoning that "stay here" means the current pose.
    That is wrong twice. It is not what was commanded -- the latched target does not drift, but the
    measurement does. And it makes the action a near-copy of the state on the ~70 percent of frames that
    are holds, so the action carries almost nothing the state does not already have. Measured as partial
    R-squared (what the action adds GIVEN the state) it cost real signal at short horizons, losing to
    v1's raw controller pose at 33 and 100 ms. The forward fill fixes both.

    BEFORE AN ARM'S FIRST COMMAND it is parked, not commanded, so the faithful target is its own
    resting pose held constant. That is not a guess: measured across every session the TCP excursion
    before the first command is a median of 0.0 mm and a maximum of 1.7 mm, so the arm demonstrably does
    not move. One session never commands its right arm at all, and gets a constant right action for the
    whole episode -- which is exactly what happened. Using tcp[0] rather than tcp[t] keeps this a single
    constant per arm per episode, so no measurement drift leaks into the action.

    NOTHING IS DROPPED. `valid` is all-True; `estimated` marks the frames whose command came from the
    future-pose estimator rather than the teleop transform. Episodes therefore stay whole, which keeps
    both the training-window count and the evaluable rollout horizon intact."""
    from scipy.spatial.transform import Rotation as Rot

    S = load_session(session_dir)
    n_raw = len(S["tcp_right"])
    idx, delta, frac = align_to_dataset(S, np.asarray(dts, dtype=np.float64),
                                        np.asarray(st, dtype=np.float64))
    xyz = np.zeros((n_raw, 2, 3))
    # IDENTITY, not zeros: rows with no reconstructable command are masked out by `valid` and then
    # dropped by the episode split, but a zero matrix is not a rotation and scipy's from_matrix
    # rejects it outright, so the placeholder has to be a legal one.
    R = np.tile(np.eye(3), (n_raw, 2, 1, 1))
    grip = np.zeros((n_raw, 2))
    ok = np.ones(n_raw, dtype=bool)
    est = np.zeros((n_raw, 2), dtype=bool)      # True where the command is ESTIMATED from tcp(t+lag)

    for k, side in enumerate(("right", "left")):
        dead = S[f"dead_{side}"]
        tgt, vpos, _ = reconstruct(S[f"p2cb_{side}"], S[f"tcp_{side}"], dead, side)
        rtgt, vrot = reconstruct_rot(S[f"quat_{side}"], S[f"rpy_{side}"], dead, side)
        cmd = vpos & vrot & ~hold_mask(dead)            # a freshly issued command

        # Forward-fill the last issued command over every frame that is not itself a fresh command.
        # `ff[i]` is the index of the most recent commanded frame at or before i, or -1 if none yet.
        tcp_s = S[f"tcp_{side}"]
        ff = np.maximum.accumulate(np.where(cmd, np.arange(len(cmd)), -1))
        have = ff >= 0
        src = ff.copy()
        src[~have] = 0                                  # placeholder; those rows are overwritten below
        xyz[:, k] = tgt[src]
        R[:, k] = rtgt[src]
        grip[:, k] = S[f"cgrip_{side}"][src]
        # Before the first command the arm is PARKED: its own resting pose, one constant. Not a guess --
        # the measured excursion over that stretch is a median of 0.0 mm and a max of 1.7 mm.
        if (~have).any():
            xyz[~have, k] = tcp_s[0]
            R[~have, k] = Rot.from_euler(EULER_SEQ, tcp_s[0:1] * 0 + S[f"rpy_{side}"][0:1],
                                         degrees=True).as_matrix()[0]
            grip[~have, k] = S[f"grip_{side}"][0]

        # WHERE THE FILL IS FALSE, ESTIMATE THE COMMAND FROM THE ARM'S OWN FUTURE POSE.
        # The forward fill above asserts "the arm is sitting at its last commanded target". That is
        # true almost always -- with the deadman up nothing is streamed and the arm holds -- so the fill
        # IS the true command and must be kept. But occasionally something outside teleop moves the arm
        # (a reset, a physical reposition), and there the assertion is simply false.
        #
        # TEST THE ASSERTION PER FRAME, which is the whole point. An earlier version tested it per RUN
        # and invalidated the entire hold whenever the arm moved anywhere within it, then OR-ed the mask
        # across both arms. Measured, that flagged 27 percent of arm-frames to handle the 0.82 percent
        # that are genuinely unexplained -- a 33x inflation -- and cost 46 percent of the dataset,
        # 3.5 dB of codec and 5x the eval horizon. Frame-level:
        #
        #     commanded (deadman held)              42.6 percent  -> reconstructed
        #     released, arm AT its last command     56.6 percent  -> forward fill, which is exact
        #     released, arm moved elsewhere          0.8 percent  -> estimated from tcp(t+lag)
        #
        # Using the estimator on the 56.6 percent would be a real loss, not a wash: the fill is the
        # actual commanded target and is CONSTANT, whereas tcp(t+lag) tracks the measured pose, which is
        # sample-and-held at ~9 Hz. Substituting it would push the state's staleness into the action and
        # make the action a near-copy of the state on most frames.
        # The test is MOTION, not distance-to-target. During a hold the streamed target is the last one
        # sent, and that IS the command even though the arm sits a little short of it -- servo tracking
        # error is normal and does not make the fill wrong. What makes it wrong is the arm being MOVED,
        # which only an agent outside teleop can do. Testing `|tcp - fill| > tol` instead flagged 52
        # percent of frames, because it was measuring tracking error.
        w = 3                                               # +-0.1 s, wide enough to clear the ~9 Hz
        lo = np.clip(np.arange(len(tcp_s)) - w, 0, len(tcp_s) - 1)   # sample-and-hold in the state
        hi = np.clip(np.arange(len(tcp_s)) + w, 0, len(tcp_s) - 1)
        moving = np.linalg.norm(tcp_s[hi] - tcp_s[lo], axis=1) > HOLD_MOVE_MM
        bad = (~cmd) & moving
        if bad.any():
            k_ = np.clip(np.arange(len(cmd))[bad] + LAG_FRAMES, 0, len(tcp_s) - 1)
            xyz[bad, k] = tcp_s[k_]
            R[bad, k] = Rot.from_euler(EULER_SEQ, S[f"rpy_{side}"][k_], degrees=True).as_matrix()
            grip[bad, k] = S[f"grip_{side}"][k_]
            est[bad, k] = True

    # Nearest raw sample per dataset row, at the offset `align_to_dataset` recovered -- NOT an
    # interpolation: a slerp between two commands invents a command that was never issued.
    return {"xyz": xyz[idx], "R": R[idx], "grip": grip[idx], "valid": ok[idx],
            "estimated": est[idx].any(1), "delta": delta, "align_exact": frac}


def episode_action(session_dir: str, dts: np.ndarray, st: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """TRAINING form of `episode_pose`: (action (n, 20) float32, valid (n,) bool), n = len(dts).

    Per arm [tcp_xyz_mm(3), 6D(R)(6), grip(1)] = 10, so 20 for the pair. 6D because rpy wraps 702/802
    times per arm on this data and the quaternion sign flips 29 times; see `rotations.py`."""
    from .rotations import matrix_to_6d

    P = episode_pose(session_dir, dts, st)
    out = np.concatenate([
        np.concatenate([P["xyz"][:, k], matrix_to_6d(P["R"][:, k]), P["grip"][:, k, None]], axis=1)
        for k in (0, 1)], axis=1)
    return out.astype(np.float32), P["valid"]


def episode_action_native(session_dir: str, dts: np.ndarray, st: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PUBLISHED form of `episode_pose`: (action (n, 14) float32, valid (n,) bool), n = len(dts).

    Per arm [tcp_xyz_mm(3), tcp_rpy_deg(3), gripper_norm(1)] = 7, so 14 for the pair -- deliberately the
    SAME convention and units as the first seven columns of each arm's `observation.state` block, so a
    reader can subtract the two without a conversion. Euler is fine HERE, where the column is read by
    humans one frame at a time; it is not fine for training, which is what `episode_action` is for."""
    from scipy.spatial.transform import Rotation as Rot

    P = episode_pose(session_dir, dts, st)
    out = np.concatenate([
        np.concatenate([P["xyz"][:, k],
                        Rot.from_matrix(P["R"][:, k]).as_euler(EULER_SEQ, degrees=True),
                        P["grip"][:, k, None]], axis=1)
        for k in (0, 1)], axis=1)
    return out.astype(np.float32), P["valid"]
