"""Env registry: name -> WorldEnv factory (design/gym_refactor.md). The pipeline builds envs through
`make_env` so a run selects its environment purely by config (`environments.name`), never by import.

Registered: `torus_world` (the reference env — batched TorusEnv). `gym:<EnvId>` (Phase 6) will wrap an
arbitrary gymnasium.Env via GymBatchAdapter."""

from __future__ import annotations

from .base import WorldEnv


def make_env(name: str, cfg, batch: int, device="cpu") -> WorldEnv:
    """Construct a batched `WorldEnv` by name. `cfg` is the resolved `environments` config group."""
    n = str(name).lower()
    if n in ("torus_world", "torus", "torusworld-v0"):
        from .torus import TorusEnv, TorusConfig
        tc = TorusConfig(R=cfg.R, r=cfg.r, dt=cfg.dt, gamma=cfg.gamma, a_max=cfg.a_max,
                         init_speed=cfg.init_speed, mass=cfg.mass)
        return TorusEnv(tc, batch, device=device)
    if n.startswith("gym:"):
        from .gym_adapter import GymBatchAdapter          # Phase 6 (not yet implemented)
        return GymBatchAdapter(n.split("gym:", 1)[1], cfg, batch, device=device)
    raise ValueError(f"unknown environment name: {name!r} (known: torus_world, gym:<EnvId>)")
