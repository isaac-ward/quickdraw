"""MOLA AUV: simplified 6-DOF dynamics, 6 actions, heavy damping.

THE ACTION IS SIX NUMBERS IN [-1,1]: three body-frame forces (surge, sway, heave) and three body-frame
torques (roll, pitch, yaw). Full control authority, including negative axial thrust, as specified.

DAMPING IS THE POINT, not an afterthought. A real AUV in water is massively over-damped: let go of the
thrusters and it stops within a body length. That is modelled as quadratic drag plus a linear term,
            F_drag = -(c1 * v + c2 * |v| * v)
applied in the world frame, and the angular equivalent on omega. Quadratic alone is too soft at low
speed (momentum lingers); linear alone is unphysical at speed. Both, with the linear term sized so the
e-folding time at cruise is well under a second.

Written against MuJoCo's `xfrc_applied` rather than as actuators because the drag has to be recomputed
from the current velocity every step, and because the same function then serves the control benchmark:
`step()` is the true plant, and the learned world model is swapped in for it unchanged.
"""
from __future__ import annotations

import numpy as np

# thrust authority, body frame. Surge is strongest; a torpedo hull sways and heaves poorly.
# TUNED AS A PAIR, not independently. Damping sets the stopping time (tau = m/c_lin) and thrust then
# has to be raised to keep a sensible cruise speed against it: the first pass had tau = 1.1 s and took
# 2.6 s to coast down, which is not "momentum dies quickly". At c_lin = 200 on a 60 kg hull tau is
# 0.30 s, so 3 tau ~ 0.9 s to 5% of cruise, and F_MAX 330 N buys ~1.2 m/s back.
# LATERAL AUTHORITY IS DELIBERATE. A bare torpedo hull (F_sway 170 N against C_sway 400) tops out at
# 0.42 m/s sideways, and a circle in a tilted plane then demands lateral force the vehicle cannot
# produce: the sway action rails, the error grows, and the episode diverges -- measured, at k=150 of a
# 900-step circle. MBARI's docking AUVs carry lateral thrusters for exactly this reason, so the model
# gets them: sway and heave are weaker than surge but no longer token.
F_MAX = np.array([330.0, 300.0, 300.0], np.float32)     # newtons at |action| = 1
T_MAX = np.array([45.0, 90.0, 90.0], np.float32)        # newton-metres
C_LIN = np.array([200.0, 260.0, 260.0], np.float32)     # linear drag, body axes (still anisotropic)
C_QUAD = np.array([60.0, 90.0, 90.0], np.float32)       # quadratic drag, body axes
C_ANG_LIN = np.array([40.0, 67.0, 67.0], np.float32)    # I_zz ~ 20 kg m^2 -> tau ~ 0.3 s as well
C_ANG_QUAD = np.array([15.0, 25.0, 25.0], np.float32)


def quat_to_mat(q):
    """MuJoCo quaternion (w,x,y,z) -> 3x3 body->world rotation."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]], np.float32)


def wrench(action, qvel, quat):
    """6 actions in [-1,1] + current velocity -> (force_world, torque_world) including drag.

    Drag is computed in BODY axes (an AUV's resistance is anisotropic -- it slips forward and fights
    sideways) and rotated out, which is the whole reason the rotation matrix is needed here."""
    a = np.clip(np.asarray(action, np.float32), -1.0, 1.0)
    R = quat_to_mat(quat)
    # VELOCITIES ARE CLAMPED before the quadratic term. Nothing in this tank physically reaches 20 m/s
    # or 20 rad/s, but a transient during an unstable step squares into float32 overflow and then NaN,
    # which is how the first circle episodes died. Clamping bounds the drag instead of the simulation.
    v_w = np.clip(np.asarray(qvel[:3], np.float32), -20.0, 20.0)
    # MUJOCO FREE-JOINT CONVENTION: qvel[:3] is LINEAR velocity in the WORLD frame, but qvel[3:6] is
    # ANGULAR velocity in the BODY frame. Rotating the angular part by R^T as if it were world applies
    # the rotation twice. Invisible while the body is near identity -- which is why an isolated
    # attitude test from qpos=(1,0,0,0) converged perfectly while every circle diverged.
    w_b = np.clip(np.asarray(qvel[3:6], np.float32), -20.0, 20.0)
    v_b = R.T @ v_w

    f_b = F_MAX * a[:3] - (C_LIN * v_b + C_QUAD * np.abs(v_b) * v_b)
    t_b = T_MAX * a[3:] - (C_ANG_LIN * w_b + C_ANG_QUAD * np.abs(w_b) * w_b)
    return R @ f_b, R @ t_b


def apply(model, data, action):
    """Write the wrench into xfrc_applied for the mola body. Call once per physics step."""
    import mujoco

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mola")
    f, t = wrench(action, data.qvel[:6], data.qpos[3:7])
    data.xfrc_applied[bid, :3] = f
    data.xfrc_applied[bid, 3:] = t


def state(data):
    """The 13-dim observation: position(3) WORLD, quaternion(4) wxyz, linear vel(3) WORLD,
    angular vel(3) BODY -- the last one is MuJoCo's free-joint convention and is kept rather than
    converted, so the state is exactly what the simulator integrates.

    Quaternion rather than euler because the AUV rolls and pitches freely and euler would introduce a
    discontinuity the world model would have to learn around for no reason."""
    return np.concatenate([data.qpos[:3], data.qpos[3:7], data.qvel[:3], data.qvel[3:6]]).astype(np.float32)





STATE_NAMES = ["x", "y", "z", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "wx_b", "wy_b", "wz_b"]
ACTION_NAMES = ["surge", "sway", "heave", "roll", "pitch", "yaw"]
