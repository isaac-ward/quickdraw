"""`GymBatchAdapter`: any `gymnasium.Env` as a batched `WorldEnv` (design/gym_refactor.md Phase 6) — the
"bring your own env" entry point, built by `registry.make_env("gym:<EnvId>", ...)`.

Semantics:
  - B independent `gymnasium.make(env_id, render_mode="rgb_array")` copies, stepped in a loop (numpy in/out
    per env; torch (B, ...) tensors at the boundary).
  - obs: `observation_space` flattened (`gym.spaces.flatten`) -> obs_dim. action: flat Box dim, or `n` for
    Discrete — a Discrete env takes a (B, n) score row and steps its argmax.
  - `reset(generator)`: env i is seeded `base + i`, base drawn from `generator` (0 if None) -> reproducible.
  - `step` stores each env's reward + done; a done env is AUTO-RESET (standard vector semantics: the stored
    reward/done belong to the step that ended, the returned obs is the fresh reset obs). Auto-resets draw from
    the env's own np_random, already seeded at `reset` -> still deterministic.
  - `reward(obs=None, goal=None)`: the gym-native reward from the LAST `step` (zeros before any step);
    obs/goal are accepted only for `WorldEnv` protocol compatibility and ignored.
  - `render_obs`: stacks each env's `render()` (rgb_array) -> (B, H, W, 3) uint8. No `render_diagnostics`
    (a generic env has no bespoke scene) -> `wants_diagnostics` is False -> eval-viz falls back to the
    `render_obs` filmstrip."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor


class GymBatchAdapter:
    """Batched `WorldEnv` view of B copies of a single-env `gymnasium.Env`."""

    def __init__(self, env_id: str, cfg, batch: int, device="cpu"):
        self.cfg = cfg
        self.batch = int(batch)
        self.device = torch.device(device)
        self.envs = [gym.make(env_id, render_mode="rgb_array") for _ in range(self.batch)]
        self._obs_space = self.envs[0].observation_space
        self._act_space = self.envs[0].action_space
        self.obs_dim = gym.spaces.flatdim(self._obs_space)
        if isinstance(self._act_space, gym.spaces.Discrete):
            self.action_dim = int(self._act_space.n)
            self.a_max = 1.0   # random-policy scale; argmax of iid U[-1,1]^n is uniform over actions
        elif isinstance(self._act_space, gym.spaces.Box):
            self.action_dim = gym.spaces.flatdim(self._act_space)
            hi = np.abs(np.asarray(self._act_space.high, dtype=np.float64))
            self.a_max = float(hi.max()) if np.all(np.isfinite(hi)) else 1.0
        else:
            raise ValueError(f"unsupported action space for {env_id!r}: {self._act_space} (Box/Discrete only)")
        self._rewards = torch.zeros(self.batch)
        self._dones = torch.zeros(self.batch, dtype=torch.bool)

    def _flat(self, obs) -> np.ndarray:
        return np.asarray(gym.spaces.flatten(self._obs_space, obs), dtype=np.float32)

    def reset(self, generator: torch.Generator | None = None) -> Tensor:
        base = int(torch.randint(2**31 - 1, (1,), generator=generator).item()) if generator is not None else 0
        obs = [self._flat(e.reset(seed=base + i)[0]) for i, e in enumerate(self.envs)]
        self._rewards.zero_()
        self._dones.zero_()
        return torch.from_numpy(np.stack(obs)).to(self.device)

    def _to_env_action(self, row: np.ndarray):
        if isinstance(self._act_space, gym.spaces.Discrete):
            return int(np.argmax(row))
        a = row.reshape(self._act_space.shape).astype(self._act_space.dtype)
        return np.clip(a, self._act_space.low, self._act_space.high)

    def step(self, action: Tensor) -> Tensor:
        act = action.detach().cpu().numpy()
        obs = []
        for i, e in enumerate(self.envs):
            o, r, terminated, truncated, _ = e.step(self._to_env_action(act[i]))
            self._rewards[i], self._dones[i] = float(r), bool(terminated or truncated)
            if terminated or truncated:
                o, _ = e.reset()   # auto-reset (env's np_random already seeded -> deterministic)
            obs.append(self._flat(o))
        return torch.from_numpy(np.stack(obs)).to(self.device)

    def reward(self, obs: Tensor | None = None, goal: Tensor | None = None) -> Tensor:
        """Gym-native reward from the LAST `step` (obs/goal ignored — protocol compatibility only)."""
        return self._rewards.clone().to(self.device)

    def render_obs(self, obs: Tensor | None = None) -> Tensor:
        """(B, H, W, 3) uint8 from each env's render(). `obs` is ignored: gym renders its CURRENT state."""
        frames = []
        for e in self.envs:
            try:
                f = e.render()
            except Exception as err:   # e.g. classic-control without pygame installed
                raise RuntimeError(
                    f"render() failed for {e.spec.id if e.spec else e!r} — install the env's render "
                    f"backend (classic control: `pip install pygame`) and use render_mode='rgb_array'"
                ) from err
            if f is None:
                raise RuntimeError(f"render() returned None for {e.spec.id if e.spec else e!r} — "
                                   f"the env must support render_mode='rgb_array'")
            frames.append(np.asarray(f, dtype=np.uint8))
        return torch.from_numpy(np.stack(frames)).to(self.device)
