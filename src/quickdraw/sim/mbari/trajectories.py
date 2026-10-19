"""The three nominal behaviour classes, and the OOD action corruptions.

SOLVED ANALYTICALLY, THEN TRACKED. For the circle classes the world-frame path is closed form -- a
circle of radius R in a plane rotated by a fixed rotation, traversed at constant speed -- so position,
velocity and acceleration are all exact at any t. The action that produces it comes from inverting the
plant in `mola.py`:

    F_body = R^T (m * a_desired) + (C_LIN * v_b + C_QUAD * |v_b| v_b)
    action[:3] = F_body / F_MAX

which is exact, because that IS the plant. A small PD correction rides on top, and it is there for one
honest reason: RK4 at 2 ms plus action clipping accumulates a slow drift, and without correction a
30 s episode leaves the circle. The feedforward does the work; the PD keeps the episode on the path it
is labelled with. Both are recorded in the action, so the dataset's action-conditional is exact
regardless of which term produced what.

RADII ARE CHOSEN AGAINST THE TANK, not picked and hoped. `fit_radius` shrinks the circle until the
whole swept path plus a margin is inside the box and clear of the crane, so no nominal episode ever
collides -- collisions are not in the nominal distribution and would quietly become an OOD signal.
"""
from __future__ import annotations

import numpy as np

from .mola import C_ANG_LIN, C_ANG_QUAD, C_LIN, C_QUAD, F_MAX, T_MAX, quat_to_mat
from .tank import KEEPOUT, TANK_D, TANK_L, TANK_W, VEHICLE_R, WATER_Z

MASS = 60.0
MARGIN = 1.6                      # metres of clearance from every wall
PLAN_SLACK = 0.30                 # extra clearance a PLANNED path must keep beyond the vehicle
#                                   radius, so tracking error cannot turn a legal plan into a
#                                   collision. `clear` is the physical test (clearance >= 0); a
#                                   circle grazing a keep-out by a millimetre passed it and then
#                                   truncated on the controller's own 0.04 m tracking error.
CRANE_KEEPOUT = 1.2               # radius around the crane line at x=y=0
CLASSES = ("random", "circle", "circle_yaw")


# ---------------------------------------------------------------- geometry
def rot_from_axis_angle(axis, ang):
    a = np.asarray(axis, np.float64)
    a = a / max(1e-9, np.linalg.norm(a))
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def mat_to_quat(R):
    """3x3 -> MuJoCo (w,x,y,z). Branchy Shepperd form, for numerical safety near 180 deg."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.asarray(q, np.float64)
    return q / np.linalg.norm(q)


def frame_from_forward(fwd, up_hint=(0, 0, 1)):
    """Body frame with +x along `fwd`, +z as close to `up_hint` as orthogonality allows.

    CAUTION: the `up_hint` fallback below is a DISCONTINUITY. When the forward axis swings through the
    hint, the frame snaps by 90 degrees in one step, the attitude error jumps to ~pi, the torque rails
    and the episode diverges -- which is exactly what killed the first tilted-circle episodes. Safe for
    a one-off pose; for anything swept over time use `frame_on_plane`, which cannot discontinuity."""
    x = np.asarray(fwd, np.float64)
    n = np.linalg.norm(x)
    x = np.array([1.0, 0, 0]) if n < 1e-9 else x / n
    up = np.asarray(up_hint, np.float64)
    if abs(float(x @ up)) > 0.98:                       # degenerate: pick any perpendicular
        up = np.array([0.0, 1.0, 0.0])
    y = np.cross(up, x)
    y /= max(1e-9, np.linalg.norm(y))
    return np.stack([x, y, np.cross(x, y)], axis=1)     # columns are the body axes in world


def frame_on_plane(fwd, normal):
    """Body frame for travel along `fwd` within a plane of the given `normal`: +x forward, +z the
    plane normal, +y completing it. Continuous for any sweep of `fwd` inside the plane, because the
    normal is fixed -- no hint, no fallback, no snap. The AUV banks into the circle, which is also
    what a real vehicle does."""
    x = np.asarray(fwd, np.float64)
    x = x / max(1e-9, np.linalg.norm(x))
    z = np.asarray(normal, np.float64)
    z = z - float(z @ x) * x                       # re-orthogonalise against drift
    z = z / max(1e-9, np.linalg.norm(z))
    return np.stack([x, np.cross(z, x), z], axis=1)


def clear(p, r=VEHICLE_R):
    """True where a vehicle of radius `r` touches NOTHING -- not a wall, the floor, the surface, or
    any of the tank's furniture.

    Separate from `inside`, which exists for PLANNING and carries a generous 1.6 m margin so a
    circle never comes near a wall. This is the PHYSICAL test: the real clearance, used to end an
    episode on contact and to reject target poses that cannot be reached without one. Measured on
    the v6 data before it existed: 6% of all steps were out of bounds, 20 of 210 sampled episodes
    touched a wall and one reached |x| = 9.18 m in a tank that stops at 7.5."""
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    ok = ((np.abs(x) < TANK_L / 2 - r) & (np.abs(y) < TANK_W / 2 - r)
          & (z < WATER_Z - r) & (z > -TANK_D + r))
    for cx, cy, cr, z0, z1 in KEEPOUT:
        hit = (np.hypot(x - cx, y - cy) < cr + r) & (z > z0 - r) & (z < z1 + r)
        ok = ok & ~hit
    return ok


def inside(p, margin=MARGIN):
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    ok = ((np.abs(x) < TANK_L / 2 - margin) & (np.abs(y) < TANK_W / 2 - margin)
          & (z < -margin) & (z > -TANK_D + margin))
    near_crane = (np.hypot(x, y) < CRANE_KEEPOUT) & (z > -TANK_D / 4 - 1.0)
    return ok & ~near_crane


# ---------------------------------------------------------------- the paths
def circle_spec(rng):
    """A circle in a plane of random orientation, random radius, random centre, random speed."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    R = rot_from_axis_angle(axis, rng.uniform(0, 2 * np.pi))
    # the DEPTH axis is the binding one: a circle of radius r needs MARGIN + r of headroom at both
    # ends of a 10 m column, so r > (TANK_D/2 - MARGIN) cannot be placed at all
    rad = rng.uniform(1.0, min(3.6, TANK_D / 2 - MARGIN - 0.1))
    speed = rng.uniform(0.22, 0.85)   # SPREAD, not two values. Capped at 0.85 because at 1.05 m/s
    #                                   drag alone eats 0.84 of the surge budget and the loop has
    #                                   nothing left to correct with -- measured, not guessed
    centre = np.array([rng.uniform(-1, 1) * (TANK_L / 2 - MARGIN - rad),
                       rng.uniform(-1, 1) * (TANK_W / 2 - MARGIN - rad),
                       -rng.uniform(MARGIN + rad, max(MARGIN + rad + 1e-3, TANK_D - MARGIN - rad))])
    return dict(R=R, rad=rad, speed=speed, centre=centre, phase=rng.uniform(0, 2 * np.pi),
                sweep_phase=rng.uniform(0, 2 * np.pi, 3))


def circle_at(spec, t):
    """Exact position / velocity / acceleration on the circle at time t."""
    w = spec["speed"] / spec["rad"]
    a = w * t + spec["phase"]
    loc = np.stack([spec["rad"] * np.cos(a), spec["rad"] * np.sin(a), np.zeros_like(a)], -1)
    dloc = np.stack([-spec["rad"] * w * np.sin(a), spec["rad"] * w * np.cos(a), np.zeros_like(a)], -1)
    ddloc = np.stack([-spec["rad"] * w * w * np.cos(a), -spec["rad"] * w * w * np.sin(a),
                      np.zeros_like(a)], -1)
    Rm = spec["R"]
    return spec["centre"] + loc @ Rm.T, dloc @ Rm.T, ddloc @ Rm.T


def fit_circle(rng, T, dt, tries=60):
    """Draw circles until one stays inside the tank for the whole episode. Shrinks on failure rather
    than rejecting outright, so the radius distribution keeps its upper tail near the walls."""
    ts = np.arange(0, T, dt)
    for _ in range(tries):
        s = circle_spec(rng)
        for _ in range(6):
            p, _, _ = circle_at(s, ts)
            # BOTH: `inside` keeps a planning margin off the walls, `clear` is the physical test
            # that also rejects a path through a buoy tether, the ladder or the crane hook. Without
            # the second, a planned circle truncated on contact as often as a random episode did
            # (48% vs 83% measured on v6) -- the planner simply did not know the furniture existed.
            if inside(p).all() and clear(p, VEHICLE_R + PLAN_SLACK).all():
                return s
            s["rad"] *= 0.82
            s["centre"][2] = np.clip(s["centre"][2], -TANK_D + MARGIN + s["rad"], -MARGIN - s["rad"])
    return None


# ---------------------------------------------------------------- inverse dynamics
# BODY-FRAME inertia about x/y/z, not MuJoCo's principal triple. `body_inertia` is reported in the
# body-INERTIAL frame and has to be rotated by `body_iquat` to get body axes: the principal values are
# (22.45, 20.16, 8.86) but in body axes they are (8.86, 22.45, 20.16). Using the principal order
# directly told the controller roll had 2.5x its real inertia and pitch half of its, which is most of
# why the attitude loop railed.
INERTIA = np.array([8.86, 22.45, 20.16], np.float32)
TAU_V = 0.25                                             # velocity tracker time constant, seconds
# TAU_W against the achievable angular acceleration T_MAX/I = (5.1, 4.0, 4.5) rad/s^2: reaching the
# 0.9 rad/s cap takes ~0.22 s, so a 0.30 s constant was asking for torque the hull does not have.
TAU_W = 0.45


def thrust_for(v_cmd, v_w, quat):
    """Action that drives world velocity toward `v_cmd` with time constant TAU_V.

    FIRST ORDER ON VELOCITY, not a PD on position. The acceleration-level PD this replaces was
    second-order against saturating actuators and diverged on every tilted circle: once an axis rails
    it cannot recover, so the error grows and rails harder. A first-order velocity law asks for a
    bounded force by construction -- m*dv/tau with dv itself bounded -- so saturation degrades it
    gracefully instead of destabilising it.

    `apply()` subtracts drag from F_MAX*action, so the drag term is added back here and the net force
    on the hull is exactly m*(v_cmd - v)/tau."""
    Rb = quat_to_mat(quat)
    v_b = Rb.T @ np.asarray(v_w, np.float32)
    dv_b = Rb.T @ (np.asarray(v_cmd, np.float32) - np.asarray(v_w, np.float32))
    f_b = MASS * dv_b / TAU_V + (C_LIN * v_b + C_QUAD * np.abs(v_b) * v_b)
    return f_b / F_MAX


def torque_for(w_cmd_b, quat, omega_b):
    """Action that drives BODY angular velocity toward `w_cmd_b` with time constant TAU_W.

    THE GYROSCOPIC TERM IS NOT OPTIONAL HERE. Euler's equation is I w_dot + w x (I w) = tau, and the
    hull's body-frame inertia is (8.86, 22.45, 20.16): a circle in a tilted plane puts the rotation
    about body z, which is the INTERMEDIATE principal axis, and rotation about the intermediate axis
    is unstable (the tennis-racket / Dzhanibekov effect). Without compensation the episodes tracked
    beautifully for ~250 steps at 0.04 m error and then spun up without any actuator railing or any
    numerical fault -- it was the rigid body doing what rigid bodies do. Cancelling w x (I w) makes
    the plant behave like the decoupled first-order system the controller assumes."""
    # omega_b is ALREADY body-frame (MuJoCo free-joint qvel[3:6]) -- do not rotate it again.
    w_b = np.asarray(omega_b, np.float32)
    gyro = np.cross(w_b, INERTIA * w_b)
    t_b = (INERTIA * (np.asarray(w_cmd_b, np.float32) - w_b) / TAU_W
           + (C_ANG_LIN * w_b + C_ANG_QUAD * np.abs(w_b) * w_b) + gyro)
    return t_b / T_MAX


def orientation_error(quat, R_des):
    """Rotation FROM current TO desired, as a rotation vector in BODY coordinates.

    The order matters and the first version had it backwards: `q_des* (x) q_cur` expresses the error in
    the DESIRED frame, and feeding that to a body-frame rate command is only correct while the two
    frames nearly coincide. With a static target it still converges, which is why the isolated attitude
    test passed at 150 degrees; with a target that rotates along the path the mismatch compounds and
    the episode diverges. `q_cur* (x) q_des` is the body-frame error, so +k * e is the right rate.

    Quaternion rather than matrix log because the latter is singular at 180 degrees."""
    q_cur = np.asarray(quat, np.float64)
    q_des = mat_to_quat(np.asarray(R_des, np.float64))
    w1, x1, y1, z1 = q_cur[0], -q_cur[1], -q_cur[2], -q_cur[3]
    w2, x2, y2, z2 = q_des
    q_e = np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])
    return 2.0 * np.sign(q_e[0] if abs(q_e[0]) > 1e-12 else 1.0) * q_e[1:]


# ---------------------------------------------------------------- the policies
# The largest angular rate the hull can HOLD is T_MAX / C_ANG_LIN -- 1.13 rad/s in roll, 1.34 in
# pitch and yaw. Commanding past it rails the torque permanently, which is what pinned the attitude
# action at the limit for 91% of the first rate-controlled episodes.
W_CAP = 0.9


# Roll, pitch and yaw excursions for the third class, as (amplitude, frequency). Three different
# frequencies so the attitude never repeats a simple cycle. Sized against the ACTUATOR, not for
# looks: peak body rate is the sum of A*2*pi*f = 0.20 + 0.22 + 0.54 = 0.95 rad/s worst case against
# the 1.34 rad/s the yaw torque can hold, and they peak together only rarely because the
# frequencies are incommensurate. An earlier 60 deg at 0.22 Hz on yaw alone peaked at 1.45 and
# saturated the actuator, which is what the sizing is avoiding.
SWEEP = ((np.radians(20.0), 0.09), (np.radians(18.0), 0.11), (np.radians(35.0), 0.14))


def _excursion(t, phase):
    """The periodic attitude offset and its rate, as body-frame rotation vectors."""
    w = np.array([2 * np.pi * f for _, f in SWEEP])
    A = np.array([a for a, _ in SWEEP])
    return A * np.sin(w * t + phase), A * w * np.cos(w * t + phase)


def policy_circle(spec, yaw_sweep=False, k_pos=1.1, k_att=2.6, v_cap=1.4):
    """Returns f(t, state) -> action. `yaw_sweep` periodically rotates the body off the path frame
    about ALL THREE axes -- roll, pitch and yaw -- so the vehicle tumbles slowly while still
    thrusting along the circle. That is the third class.

    Both loops are first order (see `thrust_for`). The position term enters as a VELOCITY correction,
    capped, rather than as an acceleration, so a large excursion asks for a bounded speed back onto the
    path instead of an unbounded force."""
    n_hat = spec["R"][:, 2]
    w_path = spec["speed"] / spec["rad"]            # the desired frame's own rotation rate about n_hat

    def act(t, s):
        p_d, v_d, _ = circle_at(spec, np.array(t))
        p, q, v, w = s[:3], s[3:7], s[7:10], s[10:13]
        corr = k_pos * (p_d - p)
        nc = np.linalg.norm(corr)
        v_cmd = v_d + (corr if nc < v_cap else corr * (v_cap / nc))
        lin = thrust_for(v_cmd, v, q)

        fwd = v_d / max(1e-6, np.linalg.norm(v_d))
        R_des = frame_on_plane(fwd, n_hat)
        # FEEDFORWARD + correction. Without the feedforward the loop can only generate the path's own
        # turn rate by holding a standing error (w_des / k_att = 5.5 deg here), which is both sloppy
        # and the seed the intermediate-axis instability grows from.
        w_ff = (quat_to_mat(q).T @ n_hat) * w_path
        if yaw_sweep:
            e, de = _excursion(float(t), spec["sweep_phase"])
            ne = np.linalg.norm(e)
            if ne > 1e-9:
                R_des = R_des @ rot_from_axis_angle(e / ne, ne)
            w_ff = w_ff + de                     # the excursion rate is already in body axes
        w_cmd = w_ff + k_att * orientation_error(q, R_des)
        nw = np.linalg.norm(w_cmd)
        if nw > W_CAP:
            w_cmd = w_cmd * (W_CAP / nw)
        return np.clip(np.concatenate([lin, torque_for(w_cmd, q, w)]), -1, 1).astype(np.float32)
    return act


HOLD_RANGE = (0.25, 1.60)         # seconds; log-uniform, resampled at every switch


def policy_random(rng, dt, hold=None, scale=1.0):
    """Piecewise-constant random actions, each held for a RANDOM duration.

    Held rather than per-step white noise: at 30 Hz, independent samples average to nothing against a
    0.3 s time constant and the AUV would just sit there vibrating. Measured -- at hold 0.4 s and scale
    0.75 a 30 s episode wandered a 2 m box; at 0.7 s and full scale it covers the tank, which is what
    the world model needs to see.

    A DISTRIBUTION OF HOLDS, not one constant. A fixed 0.7 s puts a single timescale into every
    random episode, and a model can learn that period rather than the dynamics. Holds are drawn
    log-uniform over 0.25-1.6 s so short jabs and long sustained pushes both appear, with the
    geometric mean near the old constant. Pass `hold` to pin it for a test."""
    lo, hi = HOLD_RANGE
    state = {"k": 10 ** 9, "n": 1, "a": np.zeros(6, np.float32)}

    def act(t, s):
        if state["k"] >= state["n"]:
            h = hold if hold is not None else float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            state["n"] = max(1, int(round(h / dt)))
            state["a"] = (rng.uniform(-1, 1, 6) * scale).astype(np.float32)
            state["k"] = 0
        state["k"] += 1
        return state["a"]
    return act


# ---------------------------------------------------------------- OOD corruptions
def glitch(action, rng, p=0.30):
    """Thruster fault: with probability p the command is replaced by a wild one, per step.

    Chosen over a constant 1/5 derate (the first plan) because a derate is just a different LINEAR
    plant, which a world model can re-fit and which would therefore show up as low uncertainty after a
    few steps. An intermittent replacement is a shift in the action-conditional itself: the action
    logged is what was COMMANDED, the plant receives something else, and no amount of fitting makes the
    pair consistent. That is precisely what predictive uncertainty should catch."""
    if rng.random() < p:
        return (rng.uniform(-1, 1, 6)).astype(np.float32)
    return action
