"""MPPI control through a random sequence of 8 torus goals (design/training.md eval/control/).

Two controllers race the SAME task (same random init + same random goal permutation), both
executing on the TRUE env and differing only in the dynamics used to score MPPI candidates:
  - true: the true TorusEnv dynamics (oracle baseline -- best achievable control)
  - pred: the learned world model (the controller under test)
Each advances its own goal pointer when IT settles within `tol` of its current goal for
`settle_steps` steps. All episodes x candidates are batched into one rollout per control step.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from ..environments.torus import TorusConfig, TorusEnv, control_goals


@dataclass
class MPPIConfig:
    horizon: int = 24
    chunk: int = 4             # execute this many steps of each plan before replanning (action chunking)
    num_samples: int = 512
    noise_sigma: float = 0.5
    lambda_: float = 1.0
    mean_decay: float = 1.0
    tol: float = 0.15          # within this ambient distance of the goal counts as "at" it
    settle_steps: int = 4      # consecutive in-tol steps before advancing to the next goal
    max_steps: int = 800       # per-episode step budget for the whole goal sequence
    beta_vel: float = 0.3      # velocity penalty weight, gated to near-goal (encourages settling)
    r_settle: float = 0.5      # distance under which the velocity penalty turns on
    beta_ctrl: float = 0.0     # control (action-magnitude) cost weight: penalizes sum_h ||a_h||^2 over the
    #                            horizon, so the planner prefers cheaper thrust (and settles with less jitter).
    #                            Applies to BOTH controllers (shared _score). 0 = off (no control cost).
    n_episodes: int = 16       # parallel control episodes (random inits/orders); video is episode 0
    n_goals: int = 5           # goals visited per episode (random subset of the 8 NESW in/out goals)


def _score(p_xyz, v_xyz, cand, goal, mppi):
    """MPPI return for each candidate: -distance, a near-goal velocity penalty so it settles, and an
    optional control (action-magnitude) cost. p_xyz/v_xyz: (G,K,H,3); cand: (G,K,H,2); goal: (G,3) -> (G,K)."""
    d = (p_xyz - goal[:, None, None]).norm(dim=-1)            # (G,K,H)
    gate = (d < mppi.r_settle).float()
    ret = (-d - mppi.beta_vel * gate * v_xyz.norm(dim=-1)).sum(dim=-1)
    if mppi.beta_ctrl > 0.0:                                   # cheaper thrust preferred (energy/jitter)
        ret = ret - mppi.beta_ctrl * cand.pow(2).sum(dim=-1).sum(dim=-1)   # sum_h ||a_h||^2  (G,K)
    return ret


def _mppi_step(rollout_fn, mean, goal, mppi, a_max, g):
    """One MPPI update: sample candidates, score via rollout_fn, return the new weighted mean (G,H,2)
    and the first action (G,2). rollout_fn(cand) -> (p_xyz, v_xyz), both (G,K,H,3)."""
    G, H = mean.shape[0], mppi.horizon
    K = mppi.num_samples
    noise = torch.randn(G, K, H, 2, device=mean.device, generator=g) * mppi.noise_sigma
    cand = (mean[:, None] + noise).clamp(-a_max, a_max)        # (G,K,H,2)
    p_xyz, v_xyz = rollout_fn(cand)
    ret = _score(p_xyz, v_xyz, cand, goal, mppi)              # (G,K) higher = better (lower cost)
    w = torch.softmax(ret / max(mppi.lambda_, 1e-6), dim=1)   # (G,K)
    new_mean = (w[..., None, None] * cand).sum(dim=1)         # (G,H,2)
    return new_mean, new_mean[:, 0], p_xyz, ret               # p_xyz/ret expose the candidate fan


def _true_rollout_fn(env: TorusEnv, cfg: TorusConfig, device):
    """Roll candidate action sequences through the TRUE dynamics from env's current state."""
    def fn(cand):                                             # cand: (G,K,H,2)
        G, K, H = cand.shape[:3]
        sim = TorusEnv(cfg, batch=G * K, device=device)
        sim.theta = env.theta.repeat_interleave(K)
        sim.phi = env.phi.repeat_interleave(K)
        sim.theta_dot = env.theta_dot.repeat_interleave(K)
        sim.phi_dot = env.phi_dot.repeat_interleave(K)
        a = cand.reshape(G * K, H, 2)
        obs = torch.stack([sim.step(a[:, h]) for h in range(H)], dim=1)   # (G*K,H,6)
        obs = obs.view(G, K, H, 6)
        return obs[..., :3], obs[..., 3:]
    return fn


def _model_rollout_fn(model, normalizer, ctx, pa):
    """Roll candidate action sequences through the LEARNED model from the real context (ctx,pa)."""
    def fn(cand):                                            # cand: (G,K,H,2)
        G, K, H = cand.shape[:3]
        p = ctx.shape[1]
        ctxK = ctx[:, None].expand(G, K, p, 6).reshape(G * K, p, 6)
        paK = pa[:, None].expand(G, K, pa.shape[1], 2).reshape(G * K, pa.shape[1], 2)
        actK = torch.cat([paK, cand.reshape(G * K, H, 2)], dim=1)        # (G*K, p-1+H, 2)
        pr = normalizer.denorm_obs(model.imagine_eval(normalizer.norm_obs(ctxK),
                                                      normalizer.norm_act(actK), H))
        pr = pr.view(G, K, H, 6)
        return pr[..., :3], pr[..., 3:]
    return fn


def _init_controller(cfg, B, device, seed):
    env = TorusEnv(cfg, batch=B, device=device)
    env.reset(torch.Generator(device=device).manual_seed(seed))
    return {"env": env, "obs": [env.observe()], "act": [],
            "gidx": torch.zeros(B, dtype=torch.long, device=device),
            "settle": torch.zeros(B, dtype=torch.long, device=device), "goal_log": []}


@torch.no_grad()
def run_control(model, normalizer, env_cfg: TorusConfig, mppi: MPPIConfig, device="cpu", log=None):
    """Race the oracle (true-dynamics) and the learned controller through 8 goals. Returns both
    controllers' per-step paths/actions/goals (episode 0 for the video) and aggregate stats."""
    goals = control_goals(env_cfg.R, env_cfg.r, device=device)
    names = [n for n, _ in goals]
    n_goals = min(mppi.n_goals, len(goals))                   # visit this many per episode (subset of the 8)
    tgt = torch.stack([p for _, p in goals]).to(device)       # (8,3) all goal points
    B, P, H, a_max = mppi.n_episodes, model.window, mppi.horizon, env_cfg.a_max
    g = torch.Generator(device=device).manual_seed(0)         # candidate-noise stream
    # per episode: a random n_goals-subset of the 8 goals, in random order (variety across episodes)
    order = torch.rand(B, len(goals), generator=torch.Generator(device=device).manual_seed(1),
                       device=device).argsort(dim=1)[:, :n_goals]    # (B, n_goals)
    chunk = max(1, min(mppi.chunk, H))
    ctrls = {"true": _init_controller(env_cfg, B, device, 2),    # oracle: plans with true dynamics
             "pred": _init_controller(env_cfg, B, device, 2)}    # learned: plans with the model (SAME init)
    arange = torch.arange(B, device=device)
    for c in ctrls.values():
        c["mean"] = torch.zeros(B, H, 2, device=device)
        c["done_step"] = torch.full((B,), -1, dtype=torch.long, device=device)
        c["dist_log"] = []  # per executed step: distance of each episode to its current goal

    t0 = time.perf_counter()
    step = 0
    n_chunks = 0        # number of MPPI replans (one per action chunk) — for per-step timing
    next_log = 100
    fan_log = []        # per executed step: episode-0 pred candidate fan {pts (K,H,3), ret (K,)} for the viz
    cur_fan = None
    while step < mppi.max_steps:
        n_chunks += 1
        for kind, c in ctrls.items():  # plan once per chunk (re-grounded on the latest true state)
            cur = tgt[order[arange, c["gidx"].clamp(max=n_goals - 1)]]
            if kind == "pred":
                ctx = torch.stack(c["obs"][-P:], dim=1)
                pa = torch.stack(c["act"][-(P - 1):], dim=1) if c["act"] else torch.zeros(B, 0, 2, device=device)
                rollout = _model_rollout_fn(model, normalizer, ctx, pa)
            else:
                rollout = _true_rollout_fn(c["env"], env_cfg, device)
            c["plan"], _, p_xyz, ret = _mppi_step(rollout, c["mean"], cur, mppi, a_max, g)
            if kind == "pred":  # episode-0 candidate fan, ANCHORED at the current known position: prepend
                # the dot (last true obs) so the first segment joins where-we-are -> first prediction.
                anchor = c["obs"][-1][0, :3].cpu().numpy()                      # (3,) ep0 current position
                pts = p_xyz[0].cpu().numpy()                                    # (K, H, 3)
                anchored = np.concatenate([np.broadcast_to(anchor, (pts.shape[0], 1, 3)), pts], axis=1)
                cur_fan = {"pts": anchored, "ret": ret[0].cpu().numpy()}        # (K, H+1, 3)
        for j in range(chunk):  # execute `chunk` actions of each plan open-loop, then replan
            if step >= mppi.max_steps:
                break
            fan_log.append(cur_fan)  # same plan's fan governs each of the chunk's executed steps
            for kind, c in ctrls.items():
                cur = tgt[order[arange, c["gidx"].clamp(max=n_goals - 1)]]   # (B,3)
                c["goal_log"].append(cur.cpu().numpy())
                new_obs = c["env"].step(c["plan"][:, j])
                c["obs"].append(new_obs)
                c["act"].append(c["plan"][:, j])
                d = (new_obs[:, :3] - cur).norm(dim=-1)                 # (B,)
                c["dist_log"].append(d.cpu().numpy())
                c["settle"] = torch.where(d < mppi.tol, c["settle"] + 1, torch.zeros_like(c["settle"]))
                advance = (c["settle"] >= mppi.settle_steps) & (c["gidx"] < n_goals)
                c["gidx"] = c["gidx"] + advance.long()
                c["settle"] = torch.where(advance, torch.zeros_like(c["settle"]), c["settle"])
                just_done = (c["gidx"] >= n_goals) & (c["done_step"] < 0)
                c["done_step"] = torch.where(just_done, torch.full_like(c["done_step"], step + 1), c["done_step"])
            step += 1
        for c in ctrls.values():  # warm-start: shift the executed chunk off the plan
            c["mean"] = torch.cat([c["plan"][:, chunk:], torch.zeros(B, chunk, 2, device=device)], dim=1) * mppi.mean_decay
        if log is not None and step >= next_log:  # periodic progress (so eval_control time is visible live)
            gr = {k: float(c["gidx"].clamp(max=n_goals).float().mean()) for k, c in ctrls.items()}
            log(f"step {step}/{mppi.max_steps} ({time.perf_counter() - t0:.0f}s) "
                f"mean goals true={gr['true']:.1f} pred={gr['pred']:.1f} / {n_goals}")
            next_log += 100
        if all((c["gidx"] >= n_goals).all() for c in ctrls.values()):
            break
    elapsed = time.perf_counter() - t0
    dt = env_cfg.dt

    out = {"goals": [(n, p.cpu().numpy()) for n, p in goals], "n_goals": n_goals, "n_chunks": n_chunks,
           "n_steps": step, "dt": dt, "fan_seq": fan_log}  # pred candidate fan per executed step (ep 0)
    for kind, c in ctrls.items():
        done = c["done_step"]
        completed = done >= 0
        steps_tc = float(done[completed].float().mean()) if completed.any() else float("nan")
        out[kind] = {
            "path": np.stack([o[0, :3].cpu().numpy() for o in c["obs"]]),       # episode 0 (T,3)
            "actions": np.stack([a[0].cpu().numpy() for a in c["act"]]),        # episode 0 (T-1,2)
            "goal_seq": np.stack(c["goal_log"])[:, 0],                          # episode 0 (T,3)
            "dist_curve": np.stack(c["dist_log"])[:, 0],                        # episode 0 (T,) — smooth,
            # matches the video + goal-change markers. (Mean over episodes was jagged: 16 episodes switch
            # goals at different steps, so each switch-jump lands at a different step -> spurious sawtooth.)
            "success_rate": float(completed.float().mean()),                   # frac reaching all 8
            "mean_goals_reached": float(c["gidx"].clamp(max=n_goals).float().mean()),
            "mean_steps_to_complete": steps_tc,                                # over completed episodes
            "mean_seconds_to_complete": steps_tc * dt,
        }
    return out, names
