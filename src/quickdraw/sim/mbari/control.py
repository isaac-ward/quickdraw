"""Control benchmark: reach a pose using TRUE dynamics vs the LEARNED world model.

ONE CONTROLLER, TWO PLANTS. The point of the benchmark is to isolate the dynamics model, so the
planner is held fixed (MPPI, same horizon, same samples, same cost, same seed) and only the function
that predicts `state_{t+1} = f(state_t, action_t)` is swapped. Any difference in the result is the
model, not the search.

    TrueDynamics      a second MuJoCo instance, stepped exactly as the plant is
    LearnedDynamics   the trained world model's proprio head, rolled forward on the same actions

MPPI rather than a shooting PD because the learned model's errors are correlated over a horizon, and
a sampling planner that averages over many perturbed sequences degrades gracefully as the model drifts,
where a gradient or a single-shot plan chases its own model error.

TWO TASKS, as specified: reach the dock entry pose, and reach a random free pose in the tank. The dock
one is harder than it looks -- the entry is in a corner at the surface, so the approach has to come
from inside the tank rather than straight through a wall.
"""
from __future__ import annotations

import numpy as np

from . import mola, trajectories as tj
from .record import FPS


# ----------------------------------------------------------------- plants
class TrueDynamics:
    """The simulator itself, used as a predictive model. The ceiling any learned model is measured against."""

    name = "true"

    def __init__(self, model):
        import mujoco

        self.model = model
        self._d = mujoco.MjData(model)
        self.sub = int(round((1.0 / FPS) / model.opt.timestep))

    def rollout(self, state, actions, repeat=1):
        """state (13,), actions (H,6) -> (H,13) predicted states, each action held `repeat` steps."""
        import mujoco

        d = self._d
        mujoco.mj_resetData(self.model, d)
        d.qpos[:3], d.qpos[3:7] = state[:3], state[3:7]
        d.qvel[:3], d.qvel[3:6] = state[7:10], state[10:13]
        mujoco.mj_forward(self.model, d)
        out = np.zeros((len(actions), 13), np.float32)
        for k, a in enumerate(actions):
            for _ in range(self.sub * repeat):
                mola.apply(self.model, d, a)
                mujoco.mj_step(self.model, d)
            out[k] = mola.state(d)
        return out


class AnalyticDynamics:
    """The same equations as `mola.wrench`, integrated semi-implicitly at the control rate.

    Not a learned model -- a cheap, exact-form reference that shows how much of the gap to TrueDynamics
    is *model class* (this has the right equations and a coarse integrator) versus *learning*."""

    name = "analytic"

    def __init__(self, model=None, mass=60.0, dt=1.0 / FPS):
        self.mass, self.dt = mass, dt
        self.I = np.array([8.86, 22.45, 20.16], np.float32)

    def rollout_batch(self, state, actions):
        """(N,H,6) -> (N,H,13). Vectorised over candidates; the per-candidate loop is the same maths."""
        return np.stack([self.rollout(state, a) for a in actions])

    def rollout(self, state, actions, repeat=1):
        s = np.asarray(state, np.float64).copy()
        out = np.zeros((len(actions), 13), np.float32)
        for k, a in enumerate(np.repeat(np.asarray(actions), repeat, axis=0)):
            R = mola.quat_to_mat(s[3:7]).astype(np.float64)
            f_w, t_w = mola.wrench(a, np.concatenate([s[7:10], s[10:13]]), s[3:7])
            s[7:10] += self.dt * np.asarray(f_w, np.float64) / self.mass
            t_b = R.T @ np.asarray(t_w, np.float64)
            w_b = s[10:13]
            s[10:13] = w_b + self.dt * (t_b - np.cross(w_b, self.I * w_b)) / self.I
            s[:3] += self.dt * s[7:10]
            w = s[10:13]
            q = s[3:7]
            dq = 0.5 * np.array([-q[1]*w[0] - q[2]*w[1] - q[3]*w[2],
                                  q[0]*w[0] + q[2]*w[2] - q[3]*w[1],
                                  q[0]*w[1] - q[1]*w[2] + q[3]*w[0],
                                  q[0]*w[2] + q[1]*w[1] - q[2]*w[0]])
            q = q + self.dt * dq
            s[3:7] = q / max(1e-9, np.linalg.norm(q))
            if (k + 1) % repeat == 0:
                out[k // repeat] = s
        return out


# ----------------------------------------------------------------- planner
def pose_cost(states, goal_p, goal_q=None, w_pos=1.0, w_att=0.35, w_vel=0.08, w_wall=4.0):
    """Cost of a predicted trajectory. Terminal-weighted, plus a wall barrier.

    The wall term is not decoration: without it MPPI happily plans straight through a tank wall,
    because the plant has no collision geometry for the AUV (contype=0) and nothing else penalises it.
    """
    p = states[:, :3]
    c = w_pos * np.linalg.norm(p[-1] - goal_p) + 0.15 * w_pos * np.linalg.norm(p - goal_p, axis=1).mean()
    c += w_vel * np.linalg.norm(states[-1, 7:10])
    if goal_q is not None:
        d = abs(float(np.dot(states[-1, 3:7], goal_q)))
        c += w_att * 2.0 * np.arccos(min(1.0, d))
    out = ~tj.inside(p, margin=0.8)
    c += w_wall * float(out.mean())
    return float(c)


def mppi(dyn, state, goal_p, goal_q=None, horizon=15, samples=48, iters=1, sigma=0.5,
         temp=0.35, seed=0, prev=None, repeat=6):
    """One MPPI step -> (action, planned_mean). Actions are clipped to the box the plant accepts.

    BUDGET, because the true plant is the expensive one. Each candidate costs horizon x 17 MuJoCo
    substeps, so the first attempt (160 samples, horizon 25, 2 iterations) was 136,000 mj_steps PER
    CONTROL STEP and did not finish. 48 x 15 x 1 is 12,000, and the warm start -- the previous plan,
    shifted one step -- recovers most of what the smaller search gives up, because consecutive
    problems differ by one step. The learned model is batched and far cheaper, but it uses the same
    budget: the benchmark only means something if the planner is identical on both sides.

    LOOKAHEAD IS IN SECONDS, NOT STEPS. `repeat` holds each planned action for that many control
    steps, so horizon 15 x repeat 6 at 30 Hz is a 3-second plan at the cost of a 15-step one. The
    first version planned 15 raw steps = 0.5 s and every trial closed about 3 m and stalled whatever
    the start distance -- at half a second of lookahead the terminal cost barely moves, so there is
    almost nothing to descend."""
    rng = np.random.default_rng(seed)
    mean = np.zeros((horizon, 6), np.float32) if prev is None else prev.copy()
    for _ in range(iters):
        noise = rng.normal(0, sigma, (samples, horizon, 6)).astype(np.float32)
        cand = np.clip(mean[None] + noise, -1, 1)
        costs = np.array([pose_cost(dyn.rollout(state, c, repeat=repeat), goal_p, goal_q)
                          for c in cand])
        wts = np.exp(-(costs - costs.min()) / temp)
        wts /= wts.sum()
        mean = np.clip((wts[:, None, None] * cand).sum(0), -1, 1)
    return mean[0].copy(), np.roll(mean, -1, axis=0)


def run_episode(plant_model, dyn, start, goal_p, goal_q=None, steps=240, seed=0, log=None,
                repeat=6, samples=24, horizon=12):
    """Closed loop: `dyn` plans, the MuJoCo `plant_model` executes. Returns a trace dict.

    EACH PLANNED ACTION IS EXECUTED FOR `repeat` CONTROL STEPS, which is what it was costed as: the
    planner evaluates candidates at that granularity, so replanning every single step both wastes the
    other five-sixths of the plan and multiplies the cost by six. Measured on the true plant at
    24 samples / horizon 12 / repeat 6: 993 ms per MPPI call, so replanning every step is 9.9 min per
    600-step trial against 1.7 min when the plan is actually used."""
    import mujoco

    d = mujoco.MjData(plant_model)
    d.qpos[:3], d.qpos[3:7] = start[:3], start[3:7]
    mujoco.mj_forward(plant_model, d)
    sub = int(round((1.0 / FPS) / plant_model.opt.timestep))
    traj, acts, prev = [], [], None
    k = 0
    while k < steps:
        s = mola.state(d)
        if not np.isfinite(s).all():
            break
        a, prev = mppi(dyn, s, goal_p, goal_q, seed=seed + k, prev=prev,
                       samples=samples, horizon=horizon, repeat=repeat)
        for _ in range(min(repeat, steps - k)):
            s = mola.state(d)
            traj.append(s)
            acts.append(a)
            for _ in range(sub):
                mola.apply(plant_model, d, a)
                mujoco.mj_step(plant_model, d)
            k += 1
        if log and k % 120 < repeat:
            log(f"    k={k:3d} dist {np.linalg.norm(s[:3]-goal_p):.2f} m")
    traj = np.asarray(traj, np.float32)
    dist = np.linalg.norm(traj[:, :3] - goal_p, axis=1)
    return {"traj": traj, "act": np.asarray(acts, np.float32), "dist": dist,
            "final_dist": float(dist[-1]), "best_dist": float(dist.min()),
            "steps_to_0p5m": int(np.argmax(dist < 0.5)) if (dist < 0.5).any() else -1}
