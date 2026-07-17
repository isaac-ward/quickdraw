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


class BimodalActionSampler:
    """Play-data action process with a BIMODAL action-MAGNITUDE distribution and temporal smoothing.

    A black box (the world model never sees its internals) used only for data generation. Each trajectory:
      - carries a binary latent `mode` (low- vs high-thrust) that flips with small per-step prob `p_switch`
        -> occasional smooth HOPS between the two basins (not a scripted schedule);
      - relaxes its magnitude toward the current mode's target via an OU step (temporal smoothing), so a
        single sequence is smooth and DWELLS in a basin;
      - walks the thrust DIRECTION smoothly (OU/random walk on the angle).
    The magnitude starts at 0 and relaxes into its basin, so across many trajectories the magnitude
    distribution is unimodal-near-0 early and separates into TWO peaks later -> the time-variation EMERGES
    from the dynamics rather than being hardcoded. Draw many trajectories and histogram |a| at any timestep
    to see the (evolving) bimodal shape. Action = magnitude * (cos ang, sin ang), clamped to a_max.
    """

    def __init__(self, batch: int, a_max: float, *, mu_lo: float = 0.6, mu_hi: float = 1.6,
                 theta_mag: float = 0.12, sigma_mag: float = 0.06, p_switch: float = 0.004,
                 sigma_ang: float = 0.25, device="cpu"):
        self.batch, self.a_max = batch, a_max
        self.mu = torch.tensor([mu_lo, mu_hi], device=device)
        self.theta_mag, self.sigma_mag, self.p_switch, self.sigma_ang = theta_mag, sigma_mag, p_switch, sigma_ang
        self.device = torch.device(device)
        self.mode = torch.zeros(batch, dtype=torch.long, device=self.device)
        self.mag = torch.zeros(batch, device=self.device)
        self.ang = torch.zeros(batch, device=self.device)

    def reset(self, generator: torch.Generator | None = None):
        r = lambda *s: torch.rand(*s, device=self.device, generator=generator)
        self.mode = (r(self.batch) < 0.5).long()                 # 50/50 low vs high basin
        self.mag = torch.zeros(self.batch, device=self.device)    # start at 0 -> bimodality EMERGES as it relaxes
        self.ang = r(self.batch) * TWO_PI

    def sample(self, generator: torch.Generator | None = None) -> Tensor:
        r = lambda: torch.rand(self.batch, device=self.device, generator=generator)
        n = lambda: torch.randn(self.batch, device=self.device, generator=generator)
        self.mode = torch.where(r() < self.p_switch, 1 - self.mode, self.mode)   # occasional basin hop
        target = self.mu[self.mode]
        self.mag = self.mag + self.theta_mag * (target - self.mag) + self.sigma_mag * n()   # OU toward basin
        self.ang = self.ang + self.sigma_ang * n()                                          # smooth direction walk
        mag = self.mag.clamp(0.0, self.a_max)
        a = torch.stack([mag * torch.cos(self.ang), mag * torch.sin(self.ang)], dim=-1)
        return a.clamp(-self.a_max, self.a_max)


# --------------------------------------------------------------------------------------
# Surface COLOR + vertical POSITION — the analytic ground truth (single source of truth),
# matching the RENDERER exactly (logging/viz.py `_texture_array`): the ring is painted with
# plt.cm.hsv in N_SEG discrete bands by the major angle theta (texture U = theta/2pi, no offset).
# These replace the old hand-set, miscalibrated `hue_centers`. Everything is derived from the paint
# + the named-color hexes; nothing hand-tuned. Reused by eval_interpret analytic labels, the
# language-reward ground-truth check, and any controller wanting an analytic color/position reward.
# numpy in/out (xyz world points, shape (...,3)); validated in smoke/torus_env.py against the renderer.
# --------------------------------------------------------------------------------------
import colorsys as _colorsys  # noqa: E402

import numpy as _np  # noqa: E402

N_SEG = 16  # discrete hue bands around the ring — MUST equal logging/viz.N_SEG (asserted in the smoke)
# 9 human color names -> hex (must match conf/interpret/torus.yaml factors.color.colors). NOTE the paint is
# pure-hue in 16 bands, so only ~7 of these are producible: "light green" (shares green's hue) and "purple"
# (falls between the blue-violet and magenta bands) are NEVER the nearest to a real band — a lossy naming,
# surfaced (not hidden) by the smoke.
NAMED_COLORS = {
    "red": "#ff0000", "orange": "#ffa500", "yellow": "#ffff00", "light green": "#90ee90",
    "green": "#228b22", "cyan": "#00ffff", "blue": "#0000ff", "purple": "#800080", "pink": "#ff69b4",
}


def _hex_rgb(h: str) -> _np.ndarray:
    h = h.lstrip("#")
    return _np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def hue_at(xyz) -> _np.ndarray:
    """Painted hue at a world point = texture U = (theta/2pi) mod 1 (this IS the hsv colormap input)."""
    xyz = _np.asarray(xyz, dtype=float)
    return (_np.arctan2(xyz[..., 1], xyz[..., 0]) / TWO_PI) % 1.0


def band_at(xyz, n_seg: int = N_SEG) -> _np.ndarray:
    return (_np.floor(hue_at(xyz) * n_seg).astype(int)) % n_seg


def rgb_at(xyz, n_seg: int = N_SEG) -> _np.ndarray:
    """Exact rendered RGB at a world point: plt.cm.hsv of the quantized band — identical to viz._texture_array."""
    import matplotlib.cm as cm
    return _np.asarray(cm.hsv(band_at(xyz, n_seg) / n_seg))[..., :3]


def color_at(xyz, colors: dict | None = None, n_seg: int = N_SEG):
    """Nearest NAMED color to the actual painted RGB — 'what it looks like'. Rarely returns
    'light green'/'purple' (the paint can't produce them distinctly). Scalar name, or list for a batch."""
    colors = {k: _hex_rgb(v) for k, v in (colors or NAMED_COLORS).items()}
    rgb = rgb_at(xyz, n_seg)
    names = list(colors)
    mat = _np.stack([colors[n] for n in names])
    idx = _np.linalg.norm(rgb[..., None, :] - mat, axis=-1).argmin(-1)
    return names[int(idx)] if _np.ndim(idx) == 0 else [names[int(i)] for i in _np.asarray(idx).ravel()]


def position_band(xyz, r: float, top_frac: float = 0.1):
    """Vertical band from ambient z: |z| > (1-2*top_frac)*r -> outer top/bottom frac; else middle."""
    z = _np.asarray(xyz, dtype=float)[..., 2]
    thr = (1.0 - 2.0 * top_frac) * r
    f = lambda zz: "top" if zz > thr else "bottom" if zz < -thr else "middle"
    return f(float(z)) if _np.ndim(z) == 0 else [f(float(zz)) for zz in _np.asarray(z).ravel()]


def color_reward(xyz, name: str, colors: dict | None = None) -> _np.ndarray:
    """Graded [0,1]: 1 - (circular hue distance to the target color's true hue)/0.5. Smooth around the ring."""
    colors = colors or NAMED_COLORS
    t = _colorsys.rgb_to_hsv(*_hex_rgb(colors[name]))[0]
    d = _np.abs(hue_at(xyz) - t) % 1.0
    d = _np.minimum(d, 1.0 - d)
    return _np.clip(1.0 - d / 0.5, 0.0, 1.0)


def position_reward(xyz, r: float, band: str) -> _np.ndarray:
    """Graded [0,1]: 1 at the target band's z-center (top:+r, middle:0, bottom:-r), decaying over the tube."""
    z = _np.asarray(xyz, dtype=float)[..., 2]
    ctr = {"top": r, "middle": 0.0, "bottom": -r}[band]
    return _np.clip(1.0 - _np.abs(z - ctr) / (2.0 * r), 0.0, 1.0)
