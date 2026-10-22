"""Prospector cave-quadrotor dynamics prior — VELOCITY-COMMANDED vehicle, not a force-controlled one.

The cave data (prospector `Dynamics3D`, velocity mode) is flown by commanding a velocity SETPOINT, so the
force law in `owm_physics.dynamics_prior` (a = R(q)F/m, action = force) does NOT apply. The prior here is
"the vehicle executes its command": every channel is grounded on the (bounded) action or on the exact
kinematic integral of a now-bounded state, and the WM predicts a RESIDUAL on top (prediction = prior +
residual_head, residual zero-init). The residual then learns only the DEVIATION from perfect
command-following -- the controller lag, the tilt needed to achieve a velocity, the small body rates.

    v' = v_cmd                      action[vel_slice]; grounds velocity on the command (BOUNDED)
    p' = p + v · dt                 exact kinematic integration of the (now bounded) velocity -> 0.9% resid
    q' = q ⊗ exp(½ w_prev dt)       attitude integrates the OBSERVED previous body rate (variant A only)
    w' = w_prev                     body rate persists (clamped); the residual corrects it (variant A only)

VARIANT A vs B. Pass `bodyrate_idx` -> variant A: attitude is integrated from the observed previous body
rate w_prev = prev_obs[bodyrate_idx] (ALL three axes, so pitch/roll tilt is captured), exactly as the owm
chaser prior does (`owm_physics.dynamics_prior`, kinematics-only path). Pass `bodyrate_idx=None` -> variant
B: the prior touches only position + velocity and leaves the quaternion (and body rate) to the black-box
decode head, which already tracks attitude to ~4 deg across the horizon.

WHY OBSERVED BODY RATE, NOT COMMANDED YAW (fixed 2026-10-19, variant A). The first A-candidate integrated
`[0,0,ψ̇_cmd]` (commanded yaw only). A quadrotor steers by TILTING (pitch/roll), so a yaw-only integrator
applies none of the dominant attitude change and the error ramps monotonically (~105 deg by 3 s on val).
The observed body rate includes the tilt rates, so attitude tracks the maneuver -- this is the path verified
at 0.056 deg/step on val. w is a persisted channel, but attitude lives on the unit sphere (quat_integrate
multiplies by a UNIT dq, preserving |q|) so direction can only walk, capped at 180 deg -- unlike velocity,
an open integral whose persistence blew position up quadratically (obs_error 470, fixed by grounding v=v_cmd).
`rate_clamp` bounds the persisted body-rate channel itself (owm's RATE_CLAMP; 1.0 rad/s >> ~0.03 data range).

WHY VELOCITY STAYS GROUNDED. v'=v_cmd (bounded) in BOTH variants: velocity cannot be left to a persist+
residual path because that is the unbounded random walk that exploded position. The residual learns the
small controller lag Δv = v_true - v_cmd.

The attitude integrator `quat_integrate` is a generic quaternion-kinematics helper; it lives in
`owm_physics` and is reused here rather than duplicated.
"""
import torch

from .owm_physics import quat_integrate


def velocity_kinematics(prev_obs: torch.Tensor, action: torch.Tensor, pos_idx, vel_idx, quat_idx,
                        bodyrate_idx=None, vel_cmd_slice=(0, 3),
                        dt: float = 1.0 / 30.0, rate_clamp: float = 1.0) -> torch.Tensor:
    """Grounded kinematic next-state for a velocity-commanded vehicle. See module docstring.

    action = [vx, vy, vz (world m/s) | yaw_rate (rad/s)] -> vel_cmd_slice (0,3).
    bodyrate_idx set -> variant A (owm-style attitude from observed body rate); None -> variant B (no attitude
    prior, black-box decodes the quaternion). prev_obs (...,obs_dim) ABSOLUTE; action (...,act_dim) RAW."""
    pos_idx, vel_idx, quat_idx = list(pos_idx), list(vel_idx), list(quat_idx)
    v_cmd = action[..., vel_cmd_slice[0]:vel_cmd_slice[1]]
    out = prev_obs.clone()
    out[..., pos_idx] = prev_obs[..., pos_idx] + prev_obs[..., vel_idx] * dt   # exact integration of bounded v
    out[..., vel_idx] = v_cmd                                                  # ground velocity on the command
    if bodyrate_idx is not None:                                              # variant A: owm-style attitude
        bodyrate_idx = list(bodyrate_idx)
        omega = prev_obs[..., bodyrate_idx].clamp(-rate_clamp, rate_clamp)    # OBSERVED body rate (captures tilt)
        out[..., quat_idx] = quat_integrate(prev_obs[..., quat_idx], omega, dt)
        out[..., bodyrate_idx] = omega                                        # persist (clamped); residual corrects
    # variant B (bodyrate_idx=None): quat + body rate persist (= prev via clone); black-box residual handles both
    return out
