"""FlowField — rectified-flow velocity field for the latent diffusion world model.

The ONLY new module the diffusion model adds (design/models/diffusion.md): a small MLP velocity field
`v_theta(x_tau, tau, h)` conditioned on the transformer context `h`. It owns the two operations that
define the diffusion head, both target-agnostic (the caller decides whether the diffused target is the
residual Delta-z or the absolute z_{t+1}):

  - `loss(h, target)`  : rectified flow-matching loss ||v - (eps - target)||^2 (+ optional shortcut
                         self-consistency, which unlocks accurate K=1 sampling — Frans et al. 2024).
  - `sample(h, ...)`   : K-step Euler integration of dx/dtau = v from tau=1 (noise) -> tau=0 (data),
                         returning the integrated prediction. eps=0 (the noise mean) gives the
                         deterministic, reproducible "committed" sample used for the headline metrics.

enc / dec / fuser / transformer backbone are all reused from `SequenceWorldModel`; this is the one
genuinely new net. It operates on arbitrary leading dims (so it serves both the parallel teacher-forced
forward `(B,T,*)` and the per-step rollout `(B,*)`), flattening internally.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _time_features(t: Tensor, freqs: Tensor) -> Tensor:
    """Scalar t in [0,1] (any leading shape, last dim 1) -> [sin, cos] Fourier features (..., 2*F)."""
    ang = t * freqs                                   # (..., F)
    return torch.cat([ang.sin(), ang.cos()], dim=-1)  # (..., 2F)


class FlowField(nn.Module):
    """Velocity MLP `v_theta([x_tau || emb(tau) || h (|| emb(d))]) -> dz`, shared across rollout steps."""

    def __init__(self, dz: int, h_dim: int, hidden: int, *, cond: str = "concat",
                 shortcut: bool = False, n_freq: int = 16, time_dim: int = 32):
        super().__init__()
        if cond != "concat":
            # adaln (per-layer FiLM, the DiT/SD3 default) is the documented stronger upgrade; concat is
            # the simple default and all the greenlight tests use it. Fail loud rather than silently
            # ignore an unimplemented knob.
            raise NotImplementedError(f"FlowField cond={cond!r} not implemented; use 'concat' (adaln is a future upgrade).")
        self.dz, self.shortcut = dz, shortcut
        # fixed log-spaced frequencies (deterministic -> reproducible embeddings / golden-testable)
        self.register_buffer("freqs", 2.0 * math.pi * torch.logspace(0.0, 2.0, n_freq), persistent=False)
        self.tau_mlp = nn.Sequential(nn.Linear(2 * n_freq, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim))
        self.d_mlp = (nn.Sequential(nn.Linear(2 * n_freq, time_dim), nn.GELU(),
                                    nn.Linear(time_dim, time_dim)) if shortcut else None)
        in_dim = dz + time_dim + h_dim + (time_dim if shortcut else 0)
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, dz))

    # ---- the field itself ----
    def velocity(self, x: Tensor, tau: Tensor, h: Tensor, d: Tensor | None = None) -> Tensor:
        """x (...,dz), tau (...,1) in [0,1], h (...,h_dim), d (...,1) step size (shortcut only) -> (...,dz)."""
        parts = [x, self.tau_mlp(_time_features(tau, self.freqs)), h]
        if self.shortcut:
            if d is None:
                d = torch.zeros_like(tau)            # d=0 -> the instantaneous (flow-matching) field
            parts.append(self.d_mlp(_time_features(d, self.freqs)))
        return self.net(torch.cat(parts, dim=-1))

    # ---- training: flow-matching (+ shortcut self-consistency) ----
    def _sample_time(self, shape, device, dtype, time_sampling: str) -> Tensor:
        if time_sampling == "logit_normal":          # SD3-style: weight mid-noise levels more
            return torch.sigmoid(torch.randn(shape, device=device, dtype=dtype))
        return torch.rand(shape, device=device, dtype=dtype)   # uniform (default)

    def loss(self, h: Tensor, target: Tensor, *, time_sampling: str = "uniform") -> tuple[Tensor, Tensor | None]:
        """Rectified flow-matching loss on the (already detached if needed) target velocity, computed
        teacher-forced at every supplied position. Returns (L_flow, L_consistency_or_None)."""
        tau = self._sample_time(target.shape[:-1] + (1,), target.device, target.dtype, time_sampling)
        eps = torch.randn_like(target)
        x_tau = (1.0 - tau) * target + tau * eps      # straight (rectified) path
        u = eps - target                              # constant velocity along it (the regression target)
        d0 = torch.zeros_like(tau) if self.shortcut else None   # flow-matching = the d->0 field
        v = self.velocity(x_tau, tau, h, d0)
        l_flow = F.mse_loss(v, u)
        if not self.shortcut:
            return l_flow, None
        l_consistency = self._consistency(h, target)
        return l_flow, l_consistency

    def _consistency(self, h: Tensor, target: Tensor) -> Tensor:
        """Shortcut self-consistency: one step of size 2d must equal two chained steps of size d
        (bootstrap target stop-gradded). Teaches accurate large/single steps -> K=1 sampling works."""
        # d in {1/2,1/4,1/8}; tau uniform in [2d, 1] so a 2d step stays in-range. Per-position scalars.
        lead = target.shape[:-1]
        k = torch.randint(1, 4, lead + (1,), device=target.device)          # 1..3
        d = (0.5 ** k.float())                                              # 1/2, 1/4, 1/8
        tau = 2.0 * d + (1.0 - 2.0 * d) * torch.rand(lead + (1,), device=target.device, dtype=target.dtype)
        eps = torch.randn_like(target)
        x = (1.0 - tau) * target + tau * eps
        with torch.no_grad():                                              # bootstrap target (stop-grad)
            v1 = self.velocity(x, tau, h, d)
            x2 = x - v1 * d
            v2 = self.velocity(x2, tau - d, h, d)
            s_target = 0.5 * (v1 + v2)
        v_2d = self.velocity(x, tau, h, 2.0 * d)                           # the large step, with grad
        return F.mse_loss(v_2d, s_target)

    # ---- inference: integrate the ODE ----
    def sample(self, h: Tensor, *, steps: int, deterministic: bool, eps: Tensor | None = None,
               record_path: bool = False):
        """Euler-integrate dx/dtau = v from tau=1 (x=eps) to tau=0 over `steps` steps. eps=0 (the noise
        mean, when deterministic and no eps given) -> a reproducible committed prediction. Returns the
        tau=0 prediction (...,dz); if record_path, also the list of every intermediate x (for viz)."""
        lead = h.shape[:-1]
        if eps is None:
            eps = h.new_zeros(lead + (self.dz,)) if deterministic else torch.randn(
                lead + (self.dz,), device=h.device, dtype=h.dtype)
        x = eps
        path = [x]
        d = h.new_full(lead + (1,), 1.0 / steps) if self.shortcut else None   # step-size conditioning
        for k in range(steps):
            tau = h.new_full(lead + (1,), 1.0 - k / steps)
            x = x - self.velocity(x, tau, h, d) * (1.0 / steps)               # dtau = -1/steps
            if record_path:
                path.append(x)
        return (x, path) if record_path else x
