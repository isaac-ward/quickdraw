"""Torus manifold environment: geometry, metrics, dynamics, targets.

Single source of truth for the surface math and the three rollout errors. Imported by
data generation, training, evaluation, control, and logging — nothing reimplements these.

Conventions (see design/environment.md):
  state         : (theta, phi, theta_dot, phi_dot)
  observation   : observation_vector = [p; p_dot] in R^6  (ambient position + velocity)
  action        : (a_theta, a_phi), tangential thrust, clamped to a_max
  theta         : toroidal angle (around the big ring),  phi: poloidal (around the tube)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------------------
# Geometry (all functions are batched over leading dims; angles in radians)
# --------------------------------------------------------------------------------------
def point(theta: Tensor, phi: Tensor, R: float, r: float) -> Tensor:
    """Surface point p(theta, phi) -> (..., 3)."""
    rho = R + r * torch.cos(phi)
    x = rho * torch.cos(theta)
    y = rho * torch.sin(theta)
    z = r * torch.sin(phi)
    return torch.stack((x, y, z), dim=-1)


def velocity(theta: Tensor, phi: Tensor, theta_dot: Tensor, phi_dot: Tensor, R: float, r: float) -> Tensor:
    """Ambient velocity p_dot = theta_dot * dp/dtheta + phi_dot * dp/dphi -> (..., 3)."""
    rho = R + r * torch.cos(phi)
    dp_dtheta = torch.stack(
        (-rho * torch.sin(theta), rho * torch.cos(theta), torch.zeros_like(theta)), dim=-1
    )
    dp_dphi = torch.stack(
        (
            -r * torch.sin(phi) * torch.cos(theta),
            -r * torch.sin(phi) * torch.sin(theta),
            r * torch.cos(phi),
        ),
        dim=-1,
    )
    return theta_dot.unsqueeze(-1) * dp_dtheta + phi_dot.unsqueeze(-1) * dp_dphi


def observation_vector(theta: Tensor, phi: Tensor, theta_dot: Tensor, phi_dot: Tensor, R: float, r: float) -> Tensor:
    """[p; p_dot] -> (..., 6)."""
    p = point(theta, phi, R, r)
    v = velocity(theta, phi, theta_dot, phi_dot, R, r)
    return torch.cat((p, v), dim=-1)


def angles_from_point(p: Tensor, R: float) -> tuple[Tensor, Tensor]:
    """Nearest-surface (theta, phi) for any ambient point p (..., 3)."""
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    theta = torch.atan2(y, x)
    rho = torch.sqrt(x * x + y * y)
    phi = torch.atan2(z, rho - R)
    return theta, phi


def normal(theta: Tensor, phi: Tensor) -> Tensor:
    """Outward unit normal n_hat(theta, phi) -> (..., 3)."""
    return torch.stack(
        (torch.cos(theta) * torch.cos(phi), torch.sin(theta) * torch.cos(phi), torch.sin(phi)),
        dim=-1,
    )


def signed_dist(p: Tensor, R: float, r: float) -> Tensor:
    """Signed distance to the torus surface (>0 outside the tube). p: (..., 3) -> (...,)."""
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    rho = torch.sqrt(x * x + y * y)
    return torch.sqrt((rho - R) ** 2 + z * z) - r


# --------------------------------------------------------------------------------------
# The three rollout errors (design/environment.md). o_* are observation_vectors (...,6).
# manifold & tangent are made DIMENSIONLESS by normalizing per geometry (./r, ./v_scale) so they're
# comparable across splits with different geometry/dynamics; pointwise stays in raw physical units.
# --------------------------------------------------------------------------------------
def split_obs(o: Tensor) -> tuple[Tensor, Tensor]:
    return o[..., :3], o[..., 3:]


def manifold_distance_error(o_hat: Tensor, R: float, r: float) -> Tensor:
    """Off-surface distance normalized by tube radius r -> dimensionless (in tube-radii)."""
    p_hat, _ = split_obs(o_hat)
    return signed_dist(p_hat, R, r).abs() / r


def pointwise_error(o_hat: Tensor, o_true: Tensor) -> Tensor:
    """Position error in raw physical units (accuracy metric; not normalized)."""
    p_hat, _ = split_obs(o_hat)
    p_true, _ = split_obs(o_true)
    return (p_hat - p_true).norm(dim=-1)


def tangent_velocity_error(o_hat: Tensor, R: float, v_scale: float) -> Tensor:
    """Velocity's normal (off-surface) component normalized by characteristic speed v_scale ->
    dimensionless (in characteristic speeds)."""
    p_hat, v_hat = split_obs(o_hat)
    th, ph = angles_from_point(p_hat, R)
    n = normal(th, ph)
    return (v_hat * n).sum(dim=-1).abs() / v_scale


def _wrap(a: Tensor) -> Tensor:
    """Wrap angle differences to (-pi, pi]."""
    return (a + math.pi) % TWO_PI - math.pi


def phase_drift(o_hat: Tensor, o_true: Tensor, R: float) -> tuple[Tensor, Tensor]:
    p_hat, _ = split_obs(o_hat)
    p_true, _ = split_obs(o_true)
    th_h, ph_h = angles_from_point(p_hat, R)
    th_t, ph_t = angles_from_point(p_true, R)
    return _wrap(th_h - th_t), _wrap(ph_h - ph_t)


# --------------------------------------------------------------------------------------
# Control goals: 8 points = NESW (theta in {0,90,180,270}) on BOTH the outer ring (phi=0) and the
# inner ring (phi=pi). All lie in the z=0 plane (outer radius R+r, inner radius R-r). The control eval
# subsamples 3 of these 8 per episode (see run_control), for variety across episodes.
# --------------------------------------------------------------------------------------
_GOAL_THETAS = (0.0, math.pi / 2, math.pi, 3 * math.pi / 2)  # E, N, W, S
_GOAL_DIRS = ("E", "N", "W", "S")
_GOAL_PHIS = ((0.0, "out"), (math.pi, "in"))                 # outer ring (phi=0), inner ring (phi=pi)


def control_goals(R: float, r: float, device=None) -> list[tuple[str, Tensor]]:
    """List of (name, goal_point[3]) for the 8 control goals: {out,in} x {E,N,W,S}."""
    out = []
    for phi, ring in _GOAL_PHIS:
        for theta, d in zip(_GOAL_THETAS, _GOAL_DIRS):
            th = torch.tensor(theta, device=device)
            ph = torch.tensor(phi, device=device)
            out.append((f"{ring}_{d}", point(th, ph, R, r)))
    return out


# --------------------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------------------
@dataclass
class TorusConfig:
    R: float = 0.75
    r: float = 0.25
    dt: float = 1.0 / 60.0
    gamma: float = 0.1  # damping
    a_max: float = 2.0
    init_speed: float = 1.0  # magnitude of the (nonzero) initial angular velocity
    mass: float = 1.3  # inertia: action is divided by mass, so higher = harder to push around


class TorusEnv:
    """Batched torus dynamics. Holds state for `batch` parallel envs as tensors.

    Pure-tensor (no nn.Module needed); all ops run on `device`. Deterministic given a
    torch.Generator. The true state is always exactly on the torus.
    """

    def __init__(self, cfg: TorusConfig, batch: int, device="cpu"):
        self.cfg = cfg
        self.batch = batch
        self.device = torch.device(device)
        self.theta = torch.zeros(batch, device=self.device)
        self.phi = torch.zeros(batch, device=self.device)
        self.theta_dot = torch.zeros(batch, device=self.device)
        self.phi_dot = torch.zeros(batch, device=self.device)

    def reset(self, generator: torch.Generator | None = None) -> Tensor:
        g = generator
        rand = lambda *s: torch.rand(*s, device=self.device, generator=g)
        self.theta = rand(self.batch) * TWO_PI
        self.phi = rand(self.batch) * TWO_PI
        # nonzero initial speed: fixed magnitude, random direction in (theta_dot, phi_dot)
        ang = rand(self.batch) * TWO_PI
        self.theta_dot = self.cfg.init_speed * torch.cos(ang)
        self.phi_dot = self.cfg.init_speed * torch.sin(ang)
        return self.observe()

    def step(self, action: Tensor) -> Tensor:
        """action: (batch, 2) = (a_theta, a_phi). Semi-implicit Euler. Returns observation."""
        a = action.clamp(-self.cfg.a_max, self.cfg.a_max)
        dt, gamma, m = self.cfg.dt, self.cfg.gamma, self.cfg.mass
        self.theta_dot = self.theta_dot + dt * (a[:, 0] / m - gamma * self.theta_dot)
        self.phi_dot = self.phi_dot + dt * (a[:, 1] / m - gamma * self.phi_dot)
        self.theta = (self.theta + dt * self.theta_dot) % TWO_PI
        self.phi = (self.phi + dt * self.phi_dot) % TWO_PI
        return self.observe()

    def observe(self) -> Tensor:
        return observation_vector(self.theta, self.phi, self.theta_dot, self.phi_dot, self.cfg.R, self.cfg.r)


class OUActionSampler:
    """Ornstein-Uhlenbeck action process for temporally-correlated exploration."""

    def __init__(self, batch: int, a_max: float, theta_ou: float = 0.15, sigma: float = 0.4, device="cpu"):
        self.a_max = a_max
        self.theta_ou = theta_ou
        self.sigma = sigma
        self.device = torch.device(device)
        self.state = torch.zeros(batch, 2, device=self.device)

    def reset(self, generator: torch.Generator | None = None):
        self.state = torch.zeros_like(self.state)

    def sample(self, generator: torch.Generator | None = None) -> Tensor:
        noise = torch.randn(self.state.shape, device=self.device, generator=generator)
        self.state = self.state + self.theta_ou * (-self.state) + self.sigma * noise
        return (self.state * self.a_max).clamp(-self.a_max, self.a_max)
