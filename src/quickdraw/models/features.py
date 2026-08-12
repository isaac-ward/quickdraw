"""Fourier (sin/cos) feature expansion — ONE implementation, used everywhere.

The flow head has always Fourier-encoded its diffusion time (`_time_features`), because a scalar tau in [0,1]
whose small differences matter is exactly what a frequency ladder is for. The ACTION vector has the same
character and got a raw `nn.Linear` instead: robocasa's 12-dim action is effectively ~4 dims, consecutive
actions differ slightly, and a linear map of nearly-identical inputs gives nearly-collinear outputs. This
module exists so the expansion lives in one place instead of being re-derived per call site.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def symlog(x: Tensor) -> Tensor:
    """sign(x) * log(1 + |x|) -- monotone, INVERTIBLE (symexp undoes it), near-identity for |x| < 1.

    For z-scored inputs. data/dataset.Normalizer does (x - mean)/std, which centers and scales but does NOT
    bound: a 10-sigma value becomes 10.0, and robocasa action dims whose std floors near 1e-6 turn a real 0.5
    deviation into 5e5 (measured). symlog compresses that to ~13.1 while PRESERVING ORDER, where a hard clamp or
    a unit-ball projection would map all of that dim's variation onto one value and destroy it. Needs no dataset
    statistics at all -- normalization_stats.json carries only mean/std, no min/max, so a true [0,1] rescale
    would mean regenerating it. GAIA-2 (arXiv:2503.20523) applies symlog to actions for the same reason;
    DreamerV3 symlogs observations/rewards and uses a/max(1,|a|) for actions."""
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: Tensor) -> Tensor:
    """Inverse of symlog."""
    return torch.sign(x) * (x.abs().expm1())


def fourier_freqs(n_freq: int, f_max: float = 100.0) -> Tensor:
    """The frequency ladder: 2*pi * logspace(1 .. f_max), `n_freq` bands. Matches the flow head's historical
    `2*pi*logspace(0,2,16)` exactly at n_freq=16, f_max=100."""
    return 2.0 * math.pi * torch.logspace(0.0, math.log10(f_max), n_freq)


def fourier_dim(n_in: int, n_freq: int) -> int:
    """Output width of `fourier_features` for an `n_in`-dim input: sin AND cos, per input dim, per band."""
    return 2 * n_in * n_freq


def fourier_features(x: Tensor, freqs: Tensor, squash: float | None = None) -> Tensor:
    """(..., n) -> (..., 2*n*F). Per-element sin/cos at every frequency.

    `squash`: divide by this and clamp to [-1, 1] FIRST. REQUIRED for anything z-scored. Actions and proprio
    are normalized as (x - mean)/std (data/dataset.Normalizer), so they are unbounded -- and because some dims
    are near-constant (robocasa's action std floors around 1e-6, so a 0.5 deviation normalizes to 5e5) a raw
    expansion would alias catastrophically: at the top band a z-score of 3 gives a phase of 600*pi. Clamping to
    [-1, 1] both bounds the phase and neutralizes the 1e6 amplification. Pass squash=None ONLY for inputs
    already in [0,1] (the diffusion time tau).
    """
    if squash is not None:
        x = (x / squash).clamp(-1.0, 1.0)
    ang = x.unsqueeze(-1) * freqs            # (..., n, F)
    ang = ang.flatten(-2)                    # (..., n*F)
    return torch.cat([ang.sin(), ang.cos()], dim=-1)
