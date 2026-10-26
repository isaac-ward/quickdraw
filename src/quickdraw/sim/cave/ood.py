"""Generate cave OOD *onset* trajectories for the world-model anomaly-detection study.

Each trajectory is ~30 s (900 steps @ 30 Hz) flown by the SAME policy that made the training data (reuses
prospector `record.py` helpers -> the nominal segment is in-distribution, so a detector's pre-onset surprise is
the floor). At a known onset step t* (mid-flight, 10-20 s) the anomaly switches ON and stays on. t* is logged so
a detector can be scored on detection latency (steps after t* to fire) and false positives (firing before t*).

THREE OOD conditions (one battery of latent-surprise detectors is run against all three later):
  dynamics : at t*, the drone gets LIGHTER (payload falls off) -> the plant is swapped to a reduced mass/inertia
             Dynamics3D; the commanded action is unchanged, so the WM (trained on the heavy plant) mispredicts
             the response. PROPRIO-detectable.
  glitch   : from t*, ~10% of steps the EXECUTED action is corrupted (erratic rotor command) while the RECORDED
             action is the intended one -- the WM predicts from the intended action, so a glitch step surprises
             it. PROPRIO-detectable.
  person   : from t*, a red person MESH is composited into the FPV (handled in the render/detector stage; proprio
             is identical). VISION-only.

Nominal (mode="nominal") is the matched control: same policy/seed, no anomaly -> the false-positive baseline.
"""
from __future__ import annotations

import numpy as np

CONFIG = "/home/ubuntu/user_irw/prospector/src/prospector/conf/config.yaml"

# FPV camera params -- MUST match the training re-record (prospector record.py) so OOD frames are rendered by the
# SAME mesh renderer the WM learned (drone_render.render_fpv: crisp grapevine MESH + mesh person, NOT splats).
_UP_G = np.array([0.0, 0.0, -1.0], np.float32)   # grapevine up
_HFOV = 96.7                                      # CaveScene default hfov


def build_payload_drop_dynamics(mass_phys_scale=0.5, dt=1.0 / 30.0):
    """A quadrotor whose PHYSICAL mass has suddenly dropped (payload fell off) but whose cascaded controller still
    computes its thrust feedforward for the OLD, heavier mass. This mismatch is the detectable OOD: a *consistently*
    lighter drone is INVARIANT under this controller (thrust = m*(g+a_des), accel = thrust/m -> mass cancels), so a
    clean payload drop must decouple the controller mass (kept nominal) from the physics mass (reduced). The result
    is a thrust surplus -> an upward lurch at the drop, then the velocity loop fights it back.

    Implemented by overriding only the translational mass in `_state_derivative` and re-jitting; the controller
    (`_velocity_controller`, via `self.mass`) is untouched, so its feedforward lags reality exactly as a real one
    would until an adaptive estimate catches up."""
    import jax
    import jax.numpy as jnp
    from prospector.cave_wm.record import build_dynamics

    d = build_dynamics(dt)                                  # controller + physics both at nominal mass
    m_phys = d.mass * mass_phys_scale                       # physics only: lighter; controller keeps d.mass
    kd, g, Ix, Iy, Iz = d.drag_force_coef, d.g, d.Ix, d.Iy, d.Iz

    def _state_derivative(s, w_sq):                         # copy of Dynamics3D._state_derivative, m -> m_phys
        psi, theta, phi = s[3], s[4], s[5]
        vx, vy, vz = s[6], s[7], s[8]
        p, q, r = s[9], s[10], s[11]
        ft, tx, ty, tz = d._wrench(w_sq)
        s_psi, c_psi = jnp.sin(psi), jnp.cos(psi)
        s_th, c_th = jnp.sin(theta), jnp.cos(theta)
        s_ph, c_ph = jnp.sin(phi), jnp.cos(phi)
        t_th = s_th / c_th
        psi_dot = (q * s_ph + r * c_ph) / c_th
        theta_dot = q * c_ph - r * s_ph
        phi_dot = p + (q * s_ph + r * c_ph) * t_th
        m = m_phys
        ax = (ft / m) * (c_psi * s_th * c_ph + s_psi * s_ph) - (kd / m) * vx
        ay = (ft / m) * (s_psi * s_th * c_ph - c_psi * s_ph) - (kd / m) * vy
        az = (ft / m) * (c_th * c_ph) - g - (kd / m) * vz
        pdot = ((Iy - Iz) * q * r + tx) / Ix
        qdot = ((Iz - Ix) * p * r + ty) / Iy
        rdot = ((Ix - Iy) * p * q + tz) / Iz
        return jnp.stack([vx, vy, vz, psi_dot, theta_dot, phi_dot, ax, ay, az, pdot, qdot, rdot])

    d._state_derivative = _state_derivative                 # swap physics, keep controller -> re-jit the step
    d._step_jit = jax.jit(d._step_pure)
    d._rollout_batch_jit = jax.jit(jax.vmap(d._rollout_pure, in_axes=(None, 0)))
    return d


def _ground(scene, p, max_drop=5.0, step=0.05):
    """Drop a world point to the cave FLOOR so the person's FEET sit on the ground, not float at flight altitude.
    Grapevine up is -z, so the floor is toward +z: march down until clearance collapses to ~0 (entered the floor)
    and return the last free point. Falls back to the input if no floor is found within max_drop (open shaft)."""
    p = np.asarray(p, float)
    feet, d = None, 0.0
    while d <= max_drop:
        q = np.array([p[0], p[1], p[2] + d])
        if scene.clearance(q) <= 0.1:              # touching the floor
            break
        feet, d = q, d + step
    return feet if feet is not None else p


def ood_rollout(scene, kind, seed, mode, t_star, steps=900, dt=1.0 / 30.0,
                mass_scale=0.65, glitch_p=0.10, want_frames=False):
    """One onset trajectory. mode in {nominal, dynamics, glitch, person}. Returns a dict with obs (steps,13),
    act (steps,4, the INTENDED/recorded command), t_star, meta -- and frames (steps,H,W,3) if want_frames.
    For mode=="person" a correctly-scaled red person MESH stands FEET-ON-THE-FLOOR at a fixed point ~60% along the
    planned path and is rendered EVERY frame (no teleport pop-in): it grows in view naturally as the drone flies
    toward it, exactly like the recorded dataset. The returned t_star is then the FIRST step the person is clearly
    visible (for detection-latency scoring). Returns None if the episode collides, the plan fails, or the drone
    never gets a clear look at the person (caller retries with another seed)."""
    import jax.numpy as jnp
    from prospector.cave_wm.record import (plan_corridor, _goto, state_to_obs, forward_of, build_dynamics,
                                           VLIM, DRONE_R)

    rng = np.random.default_rng(seed)
    path = None
    center = None
    rad = 1.5
    if kind in ("corridor", "explore"):
        path = plan_corridor(scene, rng)
        if path is None:
            return None
        start = path[0]
    elif kind == "orbit":
        ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        for _ in range(40):
            c = scene.sample_flyable(1, rng)[0]
            ring = c + np.stack([rad * np.cos(ang), rad * np.sin(ang), np.zeros_like(ang)], 1)
            if all(scene.clearance(p) > DRONE_R + 0.3 for p in ring):
                center = c
                break
        if center is None:
            return None
        start = center + np.array([rad, 0, 0.0])
    else:  # station
        start = scene.sample_flyable(1, rng)[0]
    s = np.zeros(12, np.float32)
    s[:3] = start
    s[3] = rng.uniform(0, 2 * np.pi)

    dyn_nom = build_dynamics(dt)
    dyn_light = build_payload_drop_dynamics(mass_scale, dt=dt) if mode == "dynamics" else None

    from . import drone_render
    obs = np.zeros((steps, 13), np.float32)
    act = np.zeros((steps, 4), np.float32)
    frames = np.zeros((steps, scene.H, scene.W, 3), np.uint8) if want_frames else None
    # mode=="person": a FIXED, GROUNDED red person the drone flies toward (no pop-in). Feet on the floor under a
    # point ~60% along the path, correctly scaled (person_mesh = 1.7 m), rendered every frame from the start.
    person_place = None
    if mode == "person":
        anchor = path[int(0.6 * len(path))] if path is not None else (center if center is not None else start)
        person_place = _ground(scene, anchor)
    redness = np.zeros(steps, np.float32) if (mode == "person" and want_frames) else None
    wp_i = 0
    for k in range(steps):
        if not np.isfinite(s).all() or scene.clearance(s[:3]) <= DRONE_R:
            return None                                              # collided / diverged -> drop, retry seed
        if kind == "corridor":
            d = np.linalg.norm(path[:, :2] - s[:2], axis=1); wp_i = max(wp_i, int(d.argmin()))
            a = _goto(s[:3], s[3], path[min(wp_i + 12, len(path) - 1)])
        elif kind == "explore":
            d = np.linalg.norm(path[:, :2] - s[:2], axis=1); wp_i = max(wp_i, int(d.argmin()))
            a = _goto(s[:3], s[3], path[min(wp_i + 10, len(path) - 1)], k=0.8); a[3] = 0.4 * np.sin(0.03 * k)
        elif kind == "orbit":
            t = k / 30.0; w = 0.45
            tgt = center + np.array([rad * np.cos(w * t), rad * np.sin(w * t), 0.4 * np.sin(0.3 * w * t)])
            a = _goto(s[:3], s[3], tgt, k=1.2); a[3] = 0.3
        else:  # station
            a = np.array([0, 0, 0, 0.45 * np.sin(0.02 * k)], np.float32)

        obs[k] = state_to_obs(s)
        act[k] = a                                                  # RECORD the intended command
        if want_frames:                                            # person is rendered every frame (fixed, grounded)
            fr = drone_render.render_fpv(scene, s[:3], forward_of(s), _UP_G, scene.W, scene.H, _HFOV,
                                         person_place=person_place)
            frames[k] = fr
            if redness is not None:                                # track how strongly the person is in view
                r = fr[:, :, 0].astype(np.float32) - np.maximum(fr[:, :, 1], fr[:, :, 2]).astype(np.float32)
                redness[k] = float((r > 100).sum())

        a_plant = a
        dyn = dyn_nom
        if k >= t_star:                                            # anomaly ON
            if mode == "glitch" and rng.random() < glitch_p:
                a_plant = (rng.uniform(-1, 1, 4) * VLIM).astype(np.float32)
            elif mode == "dynamics":
                dyn = dyn_light
        s = np.asarray(dyn.step(jnp.asarray(s), jnp.asarray(a_plant))[0], np.float32)

    if redness is not None:                                         # person: onset = first clearly-visible frame
        if redness.max() < 120:                                     # drone never got a clear look -> retry seed
            return None
        vis = np.where(redness > 40)[0]
        t_star = int(vis[0]) if len(vis) else int(t_star)

    meta = {"kind": kind, "seed": int(seed), "mode": mode, "t_star": int(t_star), "steps": int(steps),
            "mass_scale": float(mass_scale) if mode == "dynamics" else 1.0,
            "glitch_p": float(glitch_p) if mode == "glitch" else 0.0}
    out = {"obs": obs, "act": act, "t_star": int(t_star), "meta": meta}
    if want_frames:
        out["frames"] = {"fpv": frames}
    return out
