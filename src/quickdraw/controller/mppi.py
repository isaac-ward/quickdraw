"""MPPI control with the world model as dynamics (design/training.md eval/control/).

The world model imagines candidate rollouts; the true TorusEnv executes the chosen first action.
All 16 targets and `num_samples` candidates are batched into one model rollout per control step.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from ..environments.torus import TorusConfig, TorusEnv, control_targets


@dataclass
class MPPIConfig:
    horizon: int = 24
    num_samples: int = 512
    noise_sigma: float = 0.5
    lambda_: float = 1.0
    mean_decay: float = 1.0
    tol: float = 0.15
    max_steps: int = 400
    beta_vel: float = 0.3
    r_settle: float = 0.5


@torch.no_grad()
def run_control(model, normalizer, env_cfg: TorusConfig, mppi: MPPIConfig, device="cpu"):
    """Returns dict: per-target trajectories/timings + aggregate Hz/success_rate."""
    targets = control_targets(env_cfg.R, env_cfg.r, device=device)
    names = [n for n, _ in targets]
    G = len(targets)                                  # 16
    tgt = torch.stack([p for _, p in targets]).to(device)   # (G,3)
    P, H, K = model.window, mppi.horizon, mppi.num_samples

    env = TorusEnv(env_cfg, batch=G, device=device)
    env.reset(torch.Generator(device=device).manual_seed(0))
    obs_buf = [env.observe()]                          # list of (G,6) real obs
    act_buf = []                                       # list of (G,2) real actions
    mean = torch.zeros(G, H, 2, device=device)
    paths = {n: [obs_buf[0][i, :3].cpu().numpy()] for i, n in enumerate(names)}
    done_step = [None] * G

    t0 = time.perf_counter()
    for step in range(mppi.max_steps):
        # context: last P real obs and last P-1 real actions (left-padded with zeros early on)
        ctx = torch.stack(obs_buf[-P:], dim=1)         # (G, p<=P, 6)
        pa = torch.stack(act_buf[-(P - 1):], dim=1) if act_buf else torch.zeros(G, 0, 2, device=device)
        # sample candidate action sequences
        noise = torch.randn(G, K, H, 2, device=device) * mppi.noise_sigma
        cand = (mean[:, None] + noise).clamp(-env_cfg.a_max, env_cfg.a_max)   # (G,K,H,2)
        # build model inputs replicated over K
        p_now = ctx.shape[1]
        ctxK = ctx[:, None].expand(G, K, p_now, 6).reshape(G * K, p_now, 6)
        paK = pa[:, None].expand(G, K, pa.shape[1], 2).reshape(G * K, pa.shape[1], 2)
        actK = torch.cat([paK, cand.reshape(G * K, H, 2)], dim=1)            # (G*K, p-1+H, 2)
        preds = model.imagine_eval(normalizer.norm_obs(ctxK), normalizer.norm_act(actK), H)
        pr = normalizer.denorm_obs(preds)                                    # (G*K,H,6)
        p_xyz, v_xyz = pr[..., :3], pr[..., 3:]
        d = (p_xyz - tgt.repeat_interleave(K, 0)[:, None]).norm(dim=-1)       # (G*K,H)
        gate = (d < mppi.r_settle).float()
        reward = (-d - mppi.beta_vel * gate * v_xyz.norm(dim=-1)).sum(dim=1)  # (G*K,)
        ret = reward.view(G, K)
        w = torch.softmax(ret / max(mppi.lambda_, 1e-6), dim=1)               # (G,K)
        mean = (w[..., None, None] * cand).sum(dim=1)                         # (G,H,2)
        a0 = mean[:, 0]
        new_obs = env.step(a0)
        obs_buf.append(new_obs)
        act_buf.append(a0)
        # shift mean for receding horizon
        mean = torch.cat([mean[:, 1:], torch.zeros(G, 1, 2, device=device)], dim=1) * mppi.mean_decay
        # bookkeeping
        cur_d = (new_obs[:, :3] - tgt).norm(dim=-1)
        for i, n in enumerate(names):
            paths[n].append(new_obs[i, :3].cpu().numpy())
            if done_step[i] is None and cur_d[i].item() < mppi.tol:
                done_step[i] = step + 1
        if all(s is not None for s in done_step):
            break
    elapsed = time.perf_counter() - t0
    n_steps = step + 1
    dt = env_cfg.dt
    results = {
        "hz": n_steps / elapsed,
        "success_rate": float(np.mean([s is not None for s in done_step])),
        "mean_time_to_completion": float(np.mean([(s or n_steps) * dt for s in done_step])),
        "per_target": {
            n: {
                "success": done_step[i] is not None,
                "time_to_completion": (done_step[i] or n_steps) * dt,
                "final_distance": float((obs_buf[-1][i, :3] - tgt[i]).norm().item()),
            }
            for i, n in enumerate(names)
        },
        "paths": {n: np.stack(paths[n]) for n in names},
        "targets": [(n, p.cpu().numpy()) for n, p in targets],
    }
    return results
