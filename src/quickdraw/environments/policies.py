"""Behavior policies for data generation (design/gym_refactor.md Phase 2): `policy.sample(obs, generator)
-> action (B, action_dim)`, used to mine play data from ANY batched WorldEnv.

`random` is the only env-agnostic policy (uniform over the env's action range) — the default for a BYO env.
`ou`/`bimodal` wrap the EXISTING torus action samplers (environments/torus.py) unchanged — same math, same
RNG draw order (`sample(obs, g)` forwards as the legacy `sampler.sample(g, state=obs)` call), so torus
datasets stay byte-identical (proven by smoke/refactor_parity_datagen)."""

from __future__ import annotations

import torch
from torch import Tensor

from .torus import BimodalActionSampler, OUActionSampler


class RandomPolicy:
    """Env-agnostic: uniform over the env's action range, (B, action_dim) in [-a_max, a_max]. Stateless."""

    def __init__(self, action_dim: int, a_max: float, device="cpu"):
        self.action_dim, self.a_max = int(action_dim), float(a_max)
        self.device = torch.device(device)

    def reset(self, generator: torch.Generator | None = None):
        pass

    def sample(self, obs: Tensor, generator: torch.Generator | None = None) -> Tensor:
        u = torch.rand(obs.shape[0], self.action_dim, device=self.device, generator=generator)
        return (2.0 * u - 1.0) * self.a_max


class SamplerPolicy:
    """Adapter: an existing torus action sampler (OU/Bimodal) as a Policy, without touching its math or RNG."""

    def __init__(self, sampler):
        self.sampler = sampler

    def reset(self, generator: torch.Generator | None = None):
        self.sampler.reset(generator)

    def sample(self, obs: Tensor, generator: torch.Generator | None = None) -> Tensor:
        return self.sampler.sample(generator, state=obs)   # exact legacy call (state = current obs)


def make_policy(name: str, env, device="cpu"):
    """Behavior policy by name: 'random' (env-agnostic) | 'ou' | 'bimodal' (torus samplers as policies).
    Batch and action range come from the env (a batched WorldEnv exposing `batch` + `cfg.a_max`, e.g. TorusEnv)."""
    n = str(name).lower()
    a_max = float(env.cfg.a_max)
    if n == "random":
        return RandomPolicy(env.action_dim, a_max, device=device)
    if n == "ou":
        return SamplerPolicy(OUActionSampler(env.batch, a_max, device=device))
    if n == "bimodal":
        return SamplerPolicy(BimodalActionSampler(env.batch, a_max, device=device))
    raise ValueError(f"unknown policy {name!r} (expected 'random', 'ou' or 'bimodal')")
