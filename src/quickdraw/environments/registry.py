"""Env registry: name -> WorldEnv factory (design/gym_refactor.md). The pipeline builds envs through
`make_env` so a run selects its environment purely by config (`environments.name`), never by import.

Registered: `torus_world` (the reference env — batched TorusEnv). `gym:<EnvId>` (Phase 6) wraps an
arbitrary gymnasium.Env via GymBatchAdapter."""

from __future__ import annotations

from .base import WorldEnv


def is_torus_name(name) -> bool:
    """True if `name` (case-insensitive) is one of the aliases `make_env` resolves to the torus reference
    env. The single source of truth for "is this the torus env" by name — anything gating torus-specific
    behavior on `environments.name` (e.g. training/lit.py's val metrics) should call this rather than
    duplicating the alias list."""
    return str(name).lower() in ("torus_world", "torus", "torusworld-v0")


def make_env(name: str, cfg, batch: int, device="cpu") -> WorldEnv:
    """Construct a batched `WorldEnv` by name. `cfg` is the resolved `environments` config group."""
    n = str(name).lower()
    if is_torus_name(name):
        from .torus import TorusEnv, TorusConfig
        tc = TorusConfig(R=cfg.R, r=cfg.r, dt=cfg.dt, gamma=cfg.gamma, a_max=cfg.a_max,
                         init_speed=cfg.init_speed, mass=cfg.mass)
        return TorusEnv(tc, batch, device=device)
    if n.startswith("gym:"):
        from .gym_adapter import GymBatchAdapter          # Phase 6
        env_id = str(name).split(":", 1)[1]               # from the ORIGINAL name — gym ids are case-sensitive
        return GymBatchAdapter(env_id, cfg, batch, device=device)
    raise ValueError(f"unknown environment name: {name!r} (known: torus_world, gym:<EnvId>)")
