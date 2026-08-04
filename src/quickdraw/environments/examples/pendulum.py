"""PendulumEnv — the MINIMAL `WorldEnv` example (environments/base.py): a batched torch pendulum.

The classic swing-up task (gym `Pendulum-v1` conventions: theta=0 is UP), written directly against the
`WorldEnv` protocol — a real batched-tensor env, not a gym wrapper (for that, see gym_adapter.py).
Copy this file to bring your own env; the torus (examples/torus.py) shows the optional hooks.

What it implements — ONLY the four REQUIRED contract members:
  reset / step   : batched simulation, deterministic given a torch.Generator.
  reward         : the env's OWN per-step return (goal ignored) -> the control eval runs REWARD-ONLY
                   (no `control_goals` -> controller.mppi.run_control_reward_only maximizes this).
  render_obs     : a simple rod drawing, (B, obs_dim) -> (B, H, W, 3) uint8 — the image modality.
Everything optional is OMITTED on purpose: no `render_diagnostics` (eval-viz falls back to the
render_obs filmstrip), no `control_goals` (reward-only), no `rollout_metrics` (generic pointwise L2),
no `physical_loss`, no `POLICIES` (datagen policy 'random' is always available)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass
class PendulumConfig:
    dt: float = 0.05          # integration step (gym Pendulum-v1 default)
    max_torque: float = 2.0   # action clip: torque in [-max_torque, max_torque]
    g: float = 10.0           # gravity
    m: float = 1.0            # mass
    l: float = 1.0            # rod length


class PendulumEnv:
    """Batched pendulum dynamics. Holds state for `batch` parallel envs as tensors.

    obs = [cos(theta), sin(theta), theta_dot] (gym-Pendulum convention, theta=0 = upright);
    action = (B, 1) torque. Implements the `WorldEnv` protocol (environments/base.py)."""

    action_dim = 1   # torque
    obs_dim = 3      # [cos(theta), sin(theta), theta_dot]

    def __init__(self, cfg: PendulumConfig, batch: int, device="cpu"):
        self.cfg = cfg
        self.batch = batch
        self.device = torch.device(device)
        self.a_max = float(cfg.max_torque)                      # action range, read by policies + MPPI
        self.theta = torch.zeros(batch, device=self.device)     # angle from upright, wrapped to [-pi, pi]
        self.theta_dot = torch.zeros(batch, device=self.device)
        self._torque = torch.zeros(batch, device=self.device)   # last applied torque (for the reward term)

    # ---- WorldEnv contract ----
    def reset(self, generator: torch.Generator | None = None) -> Tensor:
        """Random start: theta ~ U[-pi, pi], theta_dot ~ U[-1, 1] (gym Pendulum-v1 init)."""
        rand = lambda: torch.rand(self.batch, device=self.device, generator=generator)
        self.theta = (rand() * 2.0 - 1.0) * math.pi
        self.theta_dot = rand() * 2.0 - 1.0
        self._torque = torch.zeros(self.batch, device=self.device)
        return self.observe()

    def step(self, action: Tensor) -> Tensor:
        """action: (B, 1) torque, clipped to +-max_torque. Pendulum dynamics
        theta_dotdot = (3g / 2l) sin(theta) + 3 / (m l^2) * tau, semi-implicit Euler, wrap theta."""
        cfg = self.cfg
        tau = action[:, 0].clamp(-cfg.max_torque, cfg.max_torque)
        acc = (3.0 * cfg.g / (2.0 * cfg.l)) * torch.sin(self.theta) + 3.0 / (cfg.m * cfg.l**2) * tau
        self.theta_dot = self.theta_dot + cfg.dt * acc                              # velocity first, then
        self.theta = _wrap(self.theta + cfg.dt * self.theta_dot)                    # position (semi-implicit)
        self._torque = tau
        return self.observe()

    def observe(self) -> Tensor:
        return torch.stack((torch.cos(self.theta), torch.sin(self.theta), self.theta_dot), dim=-1)

    def reward(self, obs: Tensor, goal: Tensor | None = None) -> Tensor:
        """The gym Pendulum reward: -(theta_wrapped^2 + 0.1 theta_dot^2 + 0.001 tau^2), higher = better,
        max 0 when balanced upright at rest. `goal` is IGNORED — the task is intrinsic, which is exactly
        what makes this env the reward-only-control example. Broadcasts over leading dims (..., 3) -> (...);
        the tau^2 term uses the LAST applied torque and so only applies to live (B, 3) calls — MPPI's
        imagined candidates get the state cost (its own beta_ctrl covers control cost there)."""
        theta = torch.atan2(obs[..., 1], obs[..., 0])           # wrapped angle back out of [cos, sin]
        cost = theta**2 + 0.1 * obs[..., 2] ** 2
        if self._torque.shape == cost.shape:                    # live env call (not planner candidates)
            cost = cost + 0.001 * self._torque**2
        return -cost

    def render_obs(self, obs: Tensor, size: int = 96) -> Tensor:
        """The IMAGE MODALITY: draw the rod, (B, 3) -> (B, size, size, 3) uint8. Pure numpy (no plotting
        lib): per pixel, distance to the rod segment from the pivot (image center) to the tip; theta=0 UP."""
        o = obs.detach().cpu().numpy()
        span = 1.4 * self.cfg.l                                          # world half-extent shown
        ax = (np.arange(size) / (size - 1)) * 2.0 * span - span
        X, Y = ax[None, None, :], -ax[None, :, None]                     # world coords per pixel, y up
        tx = (o[:, 1] * self.cfg.l)[:, None, None]                       # rod tip = l * (sin, cos):
        ty = (o[:, 0] * self.cfg.l)[:, None, None]                       # theta measured from upright
        t = ((X * tx + Y * ty) / (self.cfg.l**2)).clip(0.0, 1.0)         # projection onto the segment
        d = np.hypot(X - t * tx, Y - t * ty)                             # (B, size, size) distance to rod
        img = np.full((o.shape[0], size, size, 3), 255, dtype=np.uint8)  # white background
        img[d < 0.10 * self.cfg.l] = (204, 77, 77)                       # the rod (gym's reddish bar)
        img[np.broadcast_to(np.hypot(X, Y) < 0.06 * self.cfg.l, d.shape)] = (0, 0, 0)   # pivot dot
        return torch.from_numpy(img).to(self.device)


def _wrap(theta: Tensor) -> Tensor:
    """Wrap angle to [-pi, pi]."""
    return (theta + math.pi) % (2.0 * math.pi) - math.pi
