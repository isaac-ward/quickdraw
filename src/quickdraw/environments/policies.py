"""Behavior policies for data generation (design/gym_refactor.md Phase 2): `policy.sample(obs, generator)
-> action (B, action_dim)`, used to mine play data from ANY batched WorldEnv.

`random` is the only env-agnostic policy (uniform over the env's action range) — the default for a BYO env.
Env-specific policies ship WITH their env via a `POLICIES` class registry (name -> factory(env, device));
torus registers `ornstein_uhlenbeck`/`bimodal` (environments/examples/torus.py) wrapping its EXISTING action samplers
unchanged — same math, same RNG draw order (`sample(obs, g)` forwards as the legacy `sampler.sample(g,
state=obs)` call), so torus datasets stay byte-identical (proven by smoke/refactor_parity_datagen)."""

from __future__ import annotations

import torch
from torch import Tensor


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
    """Behavior policy by name: 'random' (env-agnostic) or any name the env registers in its `POLICIES`
    class attr (name -> factory(env, device); torus ships 'ornstein_uhlenbeck' + 'bimodal'). A BYO env
    without a registry only gets 'random'."""
    n = str(name).lower()
    if n == "random":
        # action range: env-level `a_max` (GymBatchAdapter, from the action_space) else cfg (torus)
        a_max = getattr(env, "a_max", None)
        return RandomPolicy(env.action_dim, float(env.cfg.a_max if a_max is None else a_max), device=device)
    env_policies = getattr(env, "POLICIES", {})
    if n in env_policies:
        return env_policies[n](env, device)
    raise ValueError(f"unknown policy {name!r} (generic: 'random'; "
                     f"{type(env).__name__} registers: {sorted(env_policies)})")
