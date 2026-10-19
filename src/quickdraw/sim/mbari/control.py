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

    def rollout_batch(self, state, actions, repeat=1):
        """(N,H,6) -> (N,H,13). Vectorised over candidates; the per-candidate loop is the same maths."""
        return np.stack([self.rollout(state, a, repeat=repeat) for a in actions])

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


class LearnedDynamics:
    """The trained world model's proprio head, rolled on the same actions as the other two.

    CONTEXT COMES FROM THE PLANT, NOT FROM THE PLANNER. The model cannot predict from a 13-vector --
    it conditions on P past (FPV frame, proprio, action) triples. In closed loop those are free: the
    plant is right there producing them, so `observe(d, a)` is called once per executed control step
    with the rendered frame and the true state, exactly the stream the model saw in training. The
    `state` argument that `mppi` passes is therefore ignored here, and that asymmetry is the point of
    the benchmark -- TrueDynamics is handed perfect state, this one has to have been watching.

    WARM START BY HOLDING THE FIRST OBSERVATION. At k=0 there is no history, so the initial frame and
    state fill all P slots and the context actions are zero. The vehicle starts at rest, so a static
    context is physically consistent rather than a lie the model has to unlearn.

    HORIZON IS H x repeat STEPS, SUBSAMPLED BACK TO H. The planner costs trajectories at the repeat
    granularity (see `mppi`), so to stay identical the model rolls every 30 Hz step the plan implies
    and returns every repeat-th one. heads=["proprio"] skips the image decoder, which is the whole
    cost; 24 candidates x 72 latent steps is cheaper than one true-dynamics candidate.

    `norm` is passed to `imagine_shared` on purpose -- without it the rollout scores the black-box
    latent dynamics instead of the physics-anchored proprio rollout, which `multimodal.py` warns
    about in the method's own docstring."""

    name = "learned"

    def __init__(self, core, norm, plant_model, key="fpv", P=8, dev="cuda", size=(144, 192)):
        from collections import deque

        from .water import UnderwaterRenderer

        self.core, self.norm, self.key, self.P, self.dev = core, norm, key, P, dev
        self.rend = UnderwaterRenderer(plant_model, *size)
        self.o = deque(maxlen=P)
        self.f = deque(maxlen=P)
        self.a = deque(maxlen=P - 1)

    def observe(self, d, action=None):
        """One executed control step: push the rendered frame, the true state and the action taken."""
        import numpy as _np

        s = mola.state(d)
        fr = self.rend.render(d, self.key)
        if not self.o:                                   # warm start: hold the first observation
            for _ in range(self.P):
                self.o.append(s); self.f.append(fr)
            for _ in range(self.P - 1):
                self.a.append(_np.zeros(6, _np.float32))
        else:
            self.o.append(s); self.f.append(fr)
            if action is not None:
                self.a.append(_np.asarray(action, _np.float32))

    def rollout(self, state, actions, repeat=1):
        return self.rollout_batch(state, np.asarray(actions)[None], repeat=repeat)[0]

    def rollout_batch(self, state, actions, repeat=1):
        import torch

        acts = np.asarray(actions, np.float32)           # (K,H,6)
        K, H = acts.shape[:2]
        fut = np.repeat(acts, repeat, axis=1)            # (K,H*repeat,6)
        past = np.stack([np.asarray(self.a)] * K)        # (K,P-1,6)
        seq = torch.from_numpy(np.concatenate([past, fut], 1))
        seq = self.norm.norm_act(seq).float().to(self.dev)
        ctx = {"proprio": self.norm.norm_obs(torch.from_numpy(np.stack(self.o))[None]).float().to(self.dev),
               self.key: torch.from_numpy(np.stack(self.f))[None].float().div(255.0).to(self.dev)}
        with torch.no_grad():
            out = self.core.imagine_shared(ctx, seq, H * repeat, K, heads=["proprio"], norm=self.norm)
        pro = self.norm.denorm_obs(out["proprio"].cpu()).numpy()     # (K,H*repeat,13)
        return pro[:, repeat - 1::repeat][:, :H]


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


def _rollout_fn(dyn, state, repeat, goal_p, goal_q, w_att, w_wall):
    """Adapt a dynamics object to controller.mppi's rollout_fn: (G,K,H,A) -> (p_xyz, v_xyz, dist).

    ATTITUDE RIDES IN `dist`, and it has to. `_score` builds its observation as cat(p_xyz, v_xyz)
    -- six dims, no quaternion -- so a reward_fn physically cannot see orientation. The shared
    module's other scoring path takes a per-step `dist` (G,K,H) override, which is what language
    steering uses, and `_score` then applies the same velocity gating and control cost to it. So
    `dist` here is the full per-step POSE error: position + w_att * angle + a wall barrier.

    Without this the benchmark was position-only on both tasks -- including "reach a random POSE",
    which is then not the task that was asked for, and including docking, where an AUV that arrives
    at the entry pointing the wrong way has not docked."""
    import torch

    def fn(cand):
        c = cand.detach().cpu().numpy()[0]                      # (K,H,A); G is always 1 here
        preds = (dyn.rollout_batch(state, c, repeat=repeat) if hasattr(dyn, "rollout_batch")
                 else np.stack([dyn.rollout(state, x, repeat=repeat) for x in c]))
        t = torch.as_tensor(preds, dtype=cand.dtype, device=cand.device)[None]   # (1,K,H,13)
        p, v, q = t[..., :3], t[..., 7:10], t[..., 3:7]
        d = (p - torch.as_tensor(goal_p, dtype=p.dtype, device=p.device)).norm(dim=-1)
        if goal_q is not None:
            gq = torch.as_tensor(np.asarray(goal_q), dtype=q.dtype, device=q.device)
            qn = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            ang = 2.0 * torch.arccos((qn * gq).sum(-1).abs().clamp(max=1.0))     # radians, 0..pi
            d = d + w_att * ang
        lim = torch.tensor([tj.TANK_L / 2 - 0.8, tj.TANK_W / 2 - 0.8], device=p.device, dtype=p.dtype)
        out = ((p[..., :2].abs() > lim).any(-1) |
               (p[..., 2] > -0.8) | (p[..., 2] < -(tj.TANK_D - 0.8)))
        return p, v, d + w_wall * out.to(d.dtype)
    return fn


def mppi(dyn, state, goal_p, goal_q=None, horizon=15, samples=48, iters=1, sigma=0.5,
         temp=0.35, seed=0, prev=None, repeat=6, fan=None,
         w_att=1.2, w_vel=0.0, w_wall=4.0, r_settle=0.5, beta_ctrl=0.02):
    """One MPPI step -> (action, planned_mean), THROUGH `controller.mppi`.

    This used to be a bespoke forty-line sampler, which was a mistake: the shared module's core
    (_mppi_step / _score / _commit) is generic given a rollout_fn and a reward_fn, and writing a
    second one meant reinventing its `chunk` as `exec_n` and keeping a near-goal velocity penalty
    that module had already measured and switched OFF ("it made the controller stall at the gate
    boundary"). Routing through it also brings `beta_ctrl`, which damps the settling jitter, and
    the Proposal interface -- so the learned action prior can later propose candidates instead of
    white noise, which the bespoke version could not do at all.

    `fan` still collects {"pts": (K,H,3), "ret": (K,)} per call for mppi_viz."""
    import torch

    from ...controller.mppi import MPPIConfig, _mppi_step
    from ...evaluation.steering import GaussianProposal

    cfg = MPPIConfig(horizon=horizon, num_samples=samples, noise_sigma=sigma, lambda_=temp,
                     beta_vel=w_vel, r_settle=r_settle, beta_ctrl=beta_ctrl)
    mean = (torch.zeros(1, horizon, 6) if prev is None
            else torch.as_tensor(prev, dtype=torch.float32)[None].clone())
    g = torch.Generator().manual_seed(int(seed))
    plan, a0, p_xyz, ret = _mppi_step(
        _rollout_fn(dyn, state, repeat, goal_p, goal_q, w_att, w_wall), mean,
        torch.as_tensor(goal_p, dtype=torch.float32)[None], cfg, GaussianProposal(sigma), g)
    if fan is not None:
        fan.append({"pts": p_xyz[0].numpy(), "ret": ret[0].numpy()})
    plan = plan[0].numpy()
    return plan[0].copy(), np.roll(plan, -1, axis=0)


# TWO PLANNING MODES, SWITCHED ON RANGE. Far from the goal the problem is reach: you need seconds
# of lookahead, which `repeat` buys cheaply by holding each planned action for 6 control steps.
# Close in, that same coarseness is the error floor -- the finest correction the planner can command
# is one 0.2 s impulse, and with perfect dynamics the true-dynamics arm still settled at 0.16 m
# mean. Inside NEAR the planner drops to one-step actions, a longer horizon in steps (but shorter in
# seconds) and a third of the action noise, so it can place the vehicle rather than lob it.
#
# This costs less per call than it looks: a fine call is 48 x 18 x 17 = 14,688 mj_steps against the
# coarse 24 x 12 x 6 x 17 = 29,376, so roughly half. It is called six times as often during the
# final approach, so the net is about 1.7x -- and only over the last metre.
# `exec_n` DECOUPLES the action hold from the replan cadence. They were the same number because
# run_episode executed exactly what it planned, but they answer different questions: `repeat` is
# how coarse the ACTIONS are, `exec_n` is how much of a plan is spent before replanning. Dropping
# the hold to 1 while lengthening the horizon to 72 keeps both the lookahead (2.4 s) and the cost
# (samples x horizon x repeat x substeps = 29,376 mj_steps, identical) -- what it actually spends
# is SEARCH, because the plan now has 72 free parameters per axis instead of 12 for the same 24
# samples. Whether that trade is worth it is measured in scratch/_ceiling.py, not assumed.
NEAR = 1.2
COARSE = dict(repeat=6, horizon=12, samples=24, sigma=0.50)
FINE = dict(repeat=1, horizon=18, samples=48, sigma=0.18)


def run_episode(plant_model, dyn, start, goal_p, goal_q=None, steps=240, seed=0, log=None,
                repeat=6, samples=24, horizon=12, fan=None, render=None):
    """Closed loop: `dyn` plans, the MuJoCo `plant_model` executes. Returns a trace dict.

    EACH PLANNED ACTION IS EXECUTED FOR `repeat` CONTROL STEPS, which is what it was costed as: the
    planner evaluates candidates at that granularity, so replanning every single step both wastes the
    other five-sixths of the plan and multiplies the cost by six. Measured on the true plant at
    24 samples / horizon 12 / repeat 6: 993 ms per MPPI call, so replanning every step is 9.9 min per
    600-step trial against 1.7 min when the plan is actually used.

    The mode switch above overrides `repeat`/`samples`/`horizon` once inside NEAR of the goal. Both
    arms of the benchmark get the identical schedule, so the comparison is still only the dynamics."""
    import mujoco

    d = mujoco.MjData(plant_model)
    d.qpos[:3], d.qpos[3:7] = start[:3], start[3:7]
    mujoco.mj_forward(plant_model, d)
    observe = getattr(dyn, "observe", None)      # the learned model needs the plant's own stream
    if observe is not None:
        observe(d)
    sub = int(round((1.0 / FPS) / plant_model.opt.timestep))
    # `render` = an UnderwaterRenderer; frames are captured from the PLANT's own MjData each
    # executed step, so a control episode can be watched in the tank from every camera rather than
    # read off a plot.
    frames = {c: [] for c in (render.cams if render is not None else ())}
    traj, acts, prev = [], [], None
    k, mode = 0, None
    while k < steps:
        s = mola.state(d)
        if not np.isfinite(s).all():
            break
        want = FINE if np.linalg.norm(s[:3] - goal_p) < NEAR else COARSE
        if want is not mode:                  # the warm start has the old horizon; drop it
            mode, prev = want, None
        a, prev = mppi(dyn, s, goal_p, goal_q, seed=seed + k, prev=prev,
                       samples=mode["samples"], horizon=mode["horizon"],
                       sigma=mode["sigma"], repeat=mode["repeat"], fan=fan)
        for _ in range(min(mode.get("exec_n", mode["repeat"]), steps - k)):
            s = mola.state(d)
            traj.append(s)
            acts.append(a)
            if observe is not None:
                observe(d, a)
            if render is not None:
                for c in render.cams:
                    frames[c].append(render.render(d, c))
            for _ in range(sub):
                mola.apply(plant_model, d, a)
                mujoco.mj_step(plant_model, d)
            k += 1
        if log and k % 120 < repeat:
            log(f"    k={k:3d} dist {np.linalg.norm(s[:3]-goal_p):.2f} m")
    traj = np.asarray(traj, np.float32)
    dist = np.linalg.norm(traj[:, :3] - goal_p, axis=1)
    return {"traj": traj, "act": np.asarray(acts, np.float32), "dist": dist, "fan": fan,
            "frames": {c: np.stack(v) for c, v in frames.items() if v},
            "final_dist": float(dist[-1]), "best_dist": float(dist.min()),
            "steps_to_0p5m": int(np.argmax(dist < 0.5)) if (dist < 0.5).any() else -1}
