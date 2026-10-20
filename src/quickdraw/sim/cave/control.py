"""Point-to-point pose control benchmark in the Grapevine cave: LEARNED world model vs TRUE dynamics.

ONE CONTROLLER, THE DYNAMICS MODEL IS THE VARIABLE. The planner (MPPI) is held fixed -- same horizon,
samples, cost, seed -- and only the function that predicts the next state is swapped, so any difference is
the model, not the search. Execution is ALWAYS the true plant (prospector `Dynamics3D`, velocity mode); the
MPPI *prediction* model is one of:

    TrueDyn      a Dynamics3D rollout (the ceiling; Dynamics3D.step is a pure fn so one instance serves both)
    LearnedDyn   the red proprio WM (prior-free + velocity-integral position), rolled via imagine_shared
    AnalyticDyn  perfect command-following kinematics (p+=v dt, v=v_cmd, yaw+=yaw_rate dt) -- a cheap reference

POSE = position (x,y,z) + yaw. A quadrotor is velocity-commanded and cannot independently hold roll/pitch, so
the controllable, holdable pose is position + heading. The goal is a target position + yaw; roll/pitch settle.

STATE FRAMES. Dynamics3D state is 12-D ZYX-Euler [x y z | yaw pitch roll | vx vy vz | p q r] (NO quaternion).
The WM obs is 13-D [x y z | qw qx qy qz | vx vy vz | p q r]. The learned arm converts the executed 12-D state
to the 13-D quat obs (record.state_to_obs) for its context. Everything is in the grapevine metric frame; in
velocity mode the controller tracks the commanded world velocity despite the z-up/z-down gravity mismatch
(positions are grapevine-consistent; attitude is not gravity-correct -- a documented later refinement).

Runs in the BRIDGE: prospector venv + PYTHONPATH=quickdraw/src, JAX_PLATFORMS=cpu (Dynamics3D on CPU), WM on
cuda. CaveScene is built collision-only (its gsplat render palette is patched out -- control needs clearance,
not color). Collision is an IMMEDIATE failure (clearance <= DRONE_R).
"""
from __future__ import annotations

import numpy as np

VOX = 0.2                                    # CaveScene voxel size (prospector.cave_wm.scene.VOX)
DRONE_R = 0.3                                # collision radius (record.DRONE_R)
ALIM = np.array([[-2, 2], [-2, 2], [-1, 1], [-0.5, 0.5]], np.float32)   # Dynamics3D velocity action limits


# ------------------------------------------------------------------ frame helpers
def quat_to_yaw(q):
    """yaw (ZYX) from quaternion [w,x,y,z]."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def batch_clearance(scene, pts):
    """Vectorised clearance-field lookup, pts (...,3) grapevine -> (...,) distance-to-wall [m]."""
    idx = np.clip(np.floor((pts - scene.gmin) / VOX).astype(int), 0, np.array(scene.occ.shape) - 1)
    return scene.clear[idx[..., 0], idx[..., 1], idx[..., 2]]


# ------------------------------------------------------------------ prediction models (MPPI plant)
class TrueDyn:
    """Dynamics3D used as the predictive model -- the ceiling. One instance (step is a pure fn)."""
    name = "true"

    def __init__(self, dyn):
        self.dyn = dyn

    def rollout_batch(self, s12, acts):
        """s12 (12,), acts (K,H,4) -> pos (K,H,3), yaw (K,H), vel (K,H,3)."""
        import jax.numpy as jnp
        tr = np.asarray(self.dyn.rollout_batch(jnp.asarray(s12), jnp.asarray(acts)))   # (K,H,12)
        return tr[..., :3], tr[..., 3], tr[..., 6:9]

    def observe(self, s12, action=None):
        pass


class AnalyticDyn:
    """Perfect command-following kinematics: p+=v dt (prev v), v=v_cmd, yaw+=yaw_rate dt. Cheap reference."""
    name = "analytic"

    def __init__(self, dt=1.0 / 30.0):
        self.dt = dt

    def rollout_batch(self, s12, acts):
        K, H = acts.shape[:2]
        p = np.tile(s12[:3], (K, 1)).astype(np.float64)
        v = np.tile(s12[6:9], (K, 1)).astype(np.float64)
        yaw = np.full(K, s12[3], np.float64)
        pos = np.zeros((K, H, 3)); yout = np.zeros((K, H)); vout = np.zeros((K, H, 3))
        for t in range(H):
            p = p + v * self.dt                       # integrate the (bounded) previous velocity
            v = acts[:, t, :3].astype(np.float64)     # velocity becomes the command
            yaw = yaw + acts[:, t, 3] * self.dt
            pos[:, t] = p; yout[:, t] = yaw; vout[:, t] = v
        return pos.astype(np.float32), yout.astype(np.float32), vout.astype(np.float32)

    def observe(self, s12, action=None):
        pass


class LearnedDyn:
    """The red proprio WM (prior-free + velocity-integral position). Context is the executed state stream."""
    name = "learned"

    def __init__(self, core, norm, P=8, dev="cuda"):
        from collections import deque
        from ..mbari.record import FPS  # noqa: F401  (unused; keeps the import graph explicit)
        self.core, self.norm, self.P, self.dev = core, norm, P, dev
        self.o = deque(maxlen=P)        # 13-D quat obs
        self.a = deque(maxlen=P - 1)    # 4-D actions

    def observe(self, s12, action=None):
        from prospector.cave_wm.record import state_to_obs
        o = state_to_obs(np.asarray(s12, np.float32))
        if not self.o:                                   # warm start: hold the first obs, zero past actions
            for _ in range(self.P):
                self.o.append(o)
            for _ in range(self.P - 1):
                self.a.append(np.zeros(4, np.float32))
        else:
            self.o.append(o)
            if action is not None:
                self.a.append(np.asarray(action, np.float32))

    def rollout_batch(self, s12, acts):
        import torch
        K, H = acts.shape[:2]
        past = np.stack([np.asarray(self.a)] * K)                             # (K,P-1,4)
        seq = torch.from_numpy(np.concatenate([past, acts], 1))              # (K,P-1+H,4)
        seq = self.norm.norm_act(seq).float().to(self.dev)
        ctx = {"proprio": self.norm.norm_obs(torch.from_numpy(np.stack(self.o))[None]).float().to(self.dev)}
        with torch.no_grad():
            out = self.core.imagine_shared(ctx, seq, H, K, heads=["proprio"], norm=self.norm)
        pro = self.norm.denorm_obs(out["proprio"].cpu()).numpy()             # (K,H,13)
        return pro[..., :3], quat_to_yaw(pro[..., 3:7]), pro[..., 7:10]


# ------------------------------------------------------------------ planner
def _cost(pos, yaw, vel, acts, goal_p, goal_yaw, scene, w_term=1.6, w_path=0.15, w_yaw=0.3, w_wall=50.0,
          w_vel=0.5, r_settle=0.5, beta_ctrl=0.02):
    """Trajectory cost (K,): terminal + mean-path position, terminal yaw, a hard clearance barrier, a SMOOTH
    near-goal velocity-damping term, and a small control-effort penalty. The wall term uses the clearance SDF on
    the PREDICTED path, so MPPI avoids geometry for both arms identically (the WM has no collision knowledge; the
    cost supplies it).

    The velocity term is weighted by a SMOOTH proximity bump exp(-(d/r_settle)^2) rather than a hard d<r gate: a
    hard gate makes the controller stall at the mode boundary (it stops braking the instant it steps outside r,
    then over-shoots back in -- the classic ~0.37 m floor). The control-effort term penalises ||a|| which, in
    velocity mode, is deviation from hover (a=0) -- it damps the near-goal jitter without a speed gate that stalls
    the approach (friend's recipe: penalise effort, not a boundary-gated speed)."""
    term = np.linalg.norm(pos[:, -1] - goal_p, axis=1)
    path = np.linalg.norm(pos - goal_p, axis=2).mean(1)
    yerr = np.abs(wrap(yaw[:, -1] - goal_yaw))
    clr = batch_clearance(scene, pos)                                        # (K,H)
    wall = np.maximum(0.0, DRONE_R + 0.15 - clr).sum(1)                      # penalise entering the margin
    prox = np.exp(-(term / r_settle) ** 2)                                   # smooth 1-near-goal -> 0-far (no gate)
    vdamp = w_vel * prox * np.linalg.norm(vel[:, -1], axis=1)                # damp terminal speed, smoothly
    effort = beta_ctrl * np.linalg.norm(acts, axis=2).mean(1)                # ||a|| = deviation from hover (vel mode)
    return w_term * term + w_path * path + w_yaw * yerr + w_wall * wall + vdamp + effort


def mppi_step(model, s12, goal_p, goal_yaw, scene, mean, rng,
              horizon=15, repeat=6, samples=48, sigma=np.array([0.6, 0.6, 0.4, 0.3], np.float32), temp=0.5,
              w_vel=0.5, w_term=1.6, r_settle=0.5, mean_decay=0.8):
    """One MPPI step -> (action (4,), warm-started mean (H,4)). `mean` is the warm-started nominal plan (H,4).

    LOOKAHEAD = horizon x repeat control steps. Each of the `horizon` planned actions is HELD for `repeat`
    control steps in the rollout, so a 15x6 plan sees 90 steps = 3 s ahead with only 15 free params per axis --
    coherent multi-second motions (a far goal now differentiates candidates), not a per-step random walk.

    WARM START = SHIFT, ZERO-PAD, DECAY (never np.roll). np.roll WRAPS the last action back to the front, so near
    the goal -- where the cost is flat, the softmax weights go uniform and the mean barely updates -- the approach
    thrust gets recycled forever and the vehicle vibrates in place. Shifting and zero-padding retires consumed
    actions; the decay bleeds the stale plan toward hover so a flat cost landscape relaxes to rest, not to a loop."""
    noise = rng.normal(0, 1, (samples, horizon, 4)).astype(np.float32) * sigma
    acts = np.clip(mean[None] + noise, ALIM[:, 0], ALIM[:, 1])          # (K,H,4) planned
    fut = np.repeat(acts, repeat, axis=1)                                # (K,H*repeat,4) held -> rollout
    pos, yaw, vel = model.rollout_batch(s12, fut)                        # over the full H*repeat horizon
    cost = _cost(pos, yaw, vel, acts, goal_p, goal_yaw, scene, w_vel=w_vel, w_term=w_term, r_settle=r_settle)
    w = np.exp(-(cost - cost.min()) / temp); w /= w.sum() + 1e-9
    plan = np.clip((w[:, None, None] * acts).sum(0), ALIM[:, 0], ALIM[:, 1])
    nxt = np.concatenate([plan[1:], np.zeros((1, 4), np.float32)], 0) * mean_decay   # shift + zero-pad + decay
    return plan[0].copy(), nxt


# ------------------------------------------------------------------ closed loop
# TWO PLANNING MODES switched on range-to-goal. COARSE: long coherent lookahead (15 actions x 6 held = 3 s) to
# reach a far goal. FINE: within NEAR, repeat=1 one-step actions + low noise so the planner can PLACE the vehicle
# to tolerance rather than lob it (the 0.2 s coarse impulse is the near-goal error floor). Both arms get the
# identical schedule, so the only variable stays the dynamics model.
# SINGLE MODE. The warm-start fix (shift+zero-pad+decay, see mppi_step) is what makes the drone ARRIVE: it drops
# the final error from ~0.37-0.67 m (np.roll recycled the approach thrust into a hover-short / vibrate loop) to
# ~0.12 m, SR 0.90, zero collisions. A separate FINE repeat=1 station-keeping mode does NOT help a quadrotor:
# with no hydrodynamic drag (unlike the AUV the recipe came from) a one-step plan either freezes short of the goal
# (decay bleeds the creep) or cannot arrest the approach momentum. COARSE's chunky held actions give the reach
# authority to close the last metre; the smooth near-goal velocity damping (w_vel, r_settle) settles the residual.
NEAR = 0.0               # FINE disabled: COARSE (with the warm-start fix) does the whole reach-and-hold
NEAR_EXIT = 0.6          # (hysteresis bound; unused while NEAR=0)
V_SETTLE = 0.15          # m/s; "settled" = speed below this (velocity-settling requirement)
POS_TOL = 0.20           # m; position-hold tolerance (the warm-start fix clears the old ~0.37 m floor; final ~0.12)
HOLD_WIN = 20            # consecutive steps (~0.67 s) the drone must stay settled-in-tol to count as success
# COARSE = reach + hold: long coherent lookahead (15 actions x 6 held = 3 s); exec_n small so it REPLANS often
# (cheapest accuracy, point 3); warm-start decay bleeds stale thrust toward hover where the near-goal cost goes
# flat; smooth velocity damping (w_vel, weighted by exp(-(d/r_settle)^2)) nulls residual speed at the goal.
COARSE = dict(horizon=15, repeat=6, samples=48, exec_n=3, sigma=np.array([0.6, 0.6, 0.4, 0.3], np.float32),
              w_vel=0.5, w_term=1.6, r_settle=0.5, mean_decay=0.9)
FINE = dict(horizon=18, repeat=1, samples=96, exec_n=1, sigma=np.array([0.12, 0.12, 0.09, 0.09], np.float32),
            w_vel=0.6, w_term=2.0, r_settle=0.35, mean_decay=0.7)


def run_episode(plant, model, s0_12, goal_p, goal_yaw, scene, geodesic,
                steps=360, pos_tol=POS_TOL, yaw_tol=np.radians(20), hold_win=HOLD_WIN, hold_extra=0,
                seed=0, log=None):
    """Closed loop: `model` plans (MPPI), the true `plant` (Dynamics3D) executes. Returns a trace+metrics dict.

    geodesic = plan_fast path length start->goal (the shortest-path denominator). Collision (clearance<=DRONE_R)
    is an immediate failure. `hold_extra` = control steps to KEEP station-keeping after success is first confirmed
    (0 = stop at success, as the benchmark; >0 = keep holding, e.g. for a video that shows the drone parked)."""
    import jax.numpy as jnp
    s = np.asarray(s0_12, np.float32).copy()
    model.observe(s)                                     # warm-start context (learned arm)
    rng = np.random.default_rng(seed)
    traj, acts, clrs = [], [], []
    collided = False
    hold = 0                                             # consecutive settled-in-tol steps (station-keeping)
    succeeded = False; success_k = 0
    done = False
    k = 0
    mode, mean = None, None
    while k < steps and not done:
        # COARSE far = reach (long coherent lookahead); FINE within NEAR = station-keep (one-step + decay settle).
        # Hysteresis: enter FINE under NEAR, leave only past NEAR_EXIT -- jitter around NEAR keeps the settle plan.
        d2g = np.linalg.norm(s[:3] - goal_p)
        if mode is None:
            want = FINE if d2g < NEAR else COARSE
        elif mode is COARSE:
            want = FINE if d2g < NEAR else COARSE
        else:
            want = COARSE if d2g > NEAR_EXIT else FINE
        if want is not mode:                             # horizon changed -> reset the warm-started plan
            mode, mean = want, np.zeros((want["horizon"], 4), np.float32)
        a, mean = mppi_step(model, s, goal_p, goal_yaw, scene, mean, rng,
                            horizon=mode["horizon"], repeat=mode["repeat"],
                            samples=mode["samples"], sigma=mode["sigma"], w_vel=mode["w_vel"],
                            w_term=mode["w_term"], r_settle=mode["r_settle"], mean_decay=mode["mean_decay"])
        for _ in range(min(mode["exec_n"], steps - k)):
            c = scene.clearance(s[:3])
            traj.append(s.copy()); acts.append(a); clrs.append(c)
            if c <= DRONE_R:                            # collision = immediate failure
                collided = True; done = True; break
            if np.linalg.norm(s[:3] - goal_p) < pos_tol and np.linalg.norm(s[6:9]) < V_SETTLE:
                hold += 1                               # arrived AND settled; require a sustained hold
                if hold >= hold_win and not succeeded:
                    succeeded = True; success_k = k      # first confirmed hold -> success
            elif not succeeded:
                hold = 0
            if succeeded and (k - success_k) >= hold_extra:   # stop at success (+ optional extra hold for video)
                done = True; break
            s = np.asarray(plant.dyn.step(jnp.asarray(s), jnp.asarray(a))[0], np.float32)
            model.observe(s, a)
            k += 1
        if log and k % 60 < mode["exec_n"]:
            log(f"      k={k:3d} dist {np.linalg.norm(s[:3]-goal_p):.2f} spd {np.linalg.norm(s[6:9]):.2f} clr {scene.clearance(s[:3]):.2f}")
    traj = np.asarray(traj, np.float32); clrs = np.asarray(clrs, np.float32)
    dist = np.linalg.norm(traj[:, :3] - goal_p, axis=1)
    speed = np.linalg.norm(traj[:, 6:9], axis=1)
    yawerr = np.abs(wrap(traj[:, 3] - goal_yaw))
    inside = dist < pos_tol
    settled = inside & (speed < V_SETTLE)
    hold_start = _run_start(settled, hold_win)                               # first index of a hold_win-long run
    reached = (hold_start >= 0) and not collided
    t_complete = (int(np.argmax(inside)) / 30.0) if inside.any() else -1.0
    t_settle = (hold_start / 30.0) if reached else -1.0
    path_len = float(np.linalg.norm(np.diff(traj[:, :3], axis=0), axis=1).sum()) if len(traj) > 1 else 0.0
    yaw_at = yawerr[inside].min() if inside.any() else yawerr[-1]            # best yaw error while on-goal
    acts_a = np.asarray(acts, np.float32)
    tail = slice(max(0, len(traj) - 60), len(traj))                         # last ~2 s: thrash diagnostics at goal
    v_goal = float(speed[tail].mean()) if len(traj) else float("nan")       # mean speed at goal (should -> 0)
    a_goal = float(np.linalg.norm(acts_a[tail], axis=1).mean()) if len(acts_a) else float("nan")  # effort at goal
    da_goal = (float(np.linalg.norm(np.diff(acts_a[tail], axis=0), axis=1).mean())
               if len(acts_a) > 1 else float("nan"))                        # action chatter at goal
    wander = float(dist[tail].max() - dist[tail].min()) if len(traj) else float("nan")  # distance spread (jitter)
    return {"traj": traj, "act": acts_a, "dist": dist, "clear": clrs,
            "success": reached, "collided": collided,
            "time_to_complete": t_complete, "settle_time": t_settle,
            "final_dist": float(dist[-1]), "min_dist": float(dist.min()) if len(traj) else float("nan"),
            "yaw_err_deg": float(np.degrees(yaw_at)),
            "v_goal": v_goal, "a_goal": a_goal, "da_goal": da_goal, "wander": wander,
            "path_len": path_len, "geodesic": float(geodesic),
            "path_mult": (path_len / geodesic) if geodesic > 1e-6 else float("nan"),
            "closest_obstacle": float(clrs.min()) if len(clrs) else float("nan")}


def quat_yaw12(s12):
    """yaw directly from the Dynamics3D 12-D state (Euler yaw is s[3])."""
    return float(s12[3])


def _run_start(mask, win):
    """Earliest index i with `win` consecutive True (a sustained settled-in-tol hold); -1 if none."""
    n = len(mask)
    for i in range(n - win + 1):
        if mask[i:i + win].all():
            return i
    return -1
