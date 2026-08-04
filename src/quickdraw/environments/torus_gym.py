"""`TorusWorld-v0`: the public single-env gymnasium wrapper around a batch-1 `TorusEnv`
(design/gym_refactor.md Phase 1). numpy in/out per the gym API; the batched pipeline talks to `TorusEnv`
directly via `registry.make_env` — this wrapper is the standard gym entry point.

Importing this module registers the id (guarded against double-registration):
    import quickdraw.environments.torus_gym  # noqa: F401
    env = gymnasium.make("TorusWorld-v0")
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from .examples.torus import TorusEnv
from .torus_utils import TorusConfig


class TorusWorldEnv(gym.Env):
    """Single-env gym view of the torus: obs = [p; p_dot] in R^6, action = (a_theta, a_phi) in
    [-a_max, a_max]^2. Reward = `TorusEnv.reward` against an optional goal point — pass
    `reset(options={"goal": (3,) world point})`; no goal -> 0 reward. render() -> the FPV image
    modality (rgb_array)."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}
    # same env-shipped behavior policies as the batched env (factories fall back to batch=1 here)
    POLICIES = TorusEnv.POLICIES

    def __init__(self, cfg: TorusConfig | None = None, render_mode: str = "rgb_array"):
        self.cfg = cfg if cfg is not None else TorusConfig()
        self.render_mode = render_mode
        self.env = TorusEnv(self.cfg, batch=1)
        a = float(self.cfg.a_max)
        self.action_space = gym.spaces.Box(low=-a, high=a, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self._obs = None
        self._goal = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        g = torch.Generator().manual_seed(seed) if seed is not None else None
        self._obs = self.env.reset(g)
        goal = (options or {}).get("goal")
        self._goal = None if goal is None else torch.as_tensor(np.asarray(goal, dtype=np.float32))
        return self._obs[0].numpy().astype(np.float32), {}

    def step(self, action):
        a = torch.as_tensor(np.asarray(action, dtype=np.float32)).reshape(1, 2)
        self._obs = self.env.step(a)
        reward = float(self.env.reward(self._obs, self._goal)[0])
        return self._obs[0].numpy().astype(np.float32), reward, False, False, {}

    def render(self):
        return self.env.render_obs(self._obs).numpy()[0]


if "TorusWorld-v0" not in gym.registry:
    gym.register(id="TorusWorld-v0", entry_point="quickdraw.environments.torus_gym:TorusWorldEnv")
