"""Env registry: name -> WorldEnv factory (design/gym_refactor.md). The pipeline builds envs through
`make_env` so a run selects its environment purely by config (`environments.name`), never by import.

Registered: `torus_world` (the reference env — batched TorusEnv, environments/examples/torus.py).
`pendulum` (the minimal reward-only example — environments/examples/pendulum.py). `gym:<EnvId>`
(Phase 6) wraps an arbitrary gymnasium.Env via GymBatchAdapter."""

from __future__ import annotations

from .base import WorldEnv


def make_env(name: str, cfg, batch: int, device="cpu") -> WorldEnv:
    """Construct a batched `WorldEnv` by name. `cfg` is the resolved `environments` config group."""
    n = str(name).lower()
    if n in ("torus_world", "torus", "torusworld-v0"):
        from .examples.torus import TorusEnv
        from .torus_utils import TorusConfig
        tc = TorusConfig(R=cfg.R, r=cfg.r, dt=cfg.dt, gamma=cfg.gamma, a_max=cfg.a_max,
                         init_speed=cfg.init_speed, mass=cfg.mass)
        # optional render params (texture coloring + camera fov for render_obs) — per split in data
        # generation (hsv vs circles); absent from most cfgs -> the env's defaults (hsv, 100).
        return TorusEnv(tc, batch, device=device, coloring=getattr(cfg, "coloring", "hsv"),
                        fov=float(getattr(cfg, "fov", 100.0)))
    if n == "pendulum":
        from .examples.pendulum import PendulumConfig, PendulumEnv
        pc = PendulumConfig(dt=float(cfg.dt), max_torque=float(cfg.max_torque),
                            g=float(cfg.g), m=float(cfg.m), l=float(cfg.l))
        return PendulumEnv(pc, batch, device=device)
    if n == "recorded":
        from .recorded import RecordedEnv
        return RecordedEnv(cfg, batch, device=device)
    if n.startswith("gym:"):
        from .gym_adapter import GymBatchAdapter          # Phase 6
        env_id = str(name).split(":", 1)[1]               # from the ORIGINAL name — gym ids are case-sensitive
        return GymBatchAdapter(env_id, cfg, batch, device=device)
    raise ValueError(f"unknown environment name: {name!r} (known: torus_world, pendulum, recorded, gym:<EnvId>)")
