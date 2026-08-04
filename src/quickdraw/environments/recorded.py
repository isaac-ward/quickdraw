"""Recorded-data "environment": collected trajectories with NO simulator behind them.

The world model trains purely on the dataset (data.processors run folder); there is nothing to
step, reset or render, so those raise. Implements just enough of `WorldEnv` for the training path:
dims/dt from config, a flat zero `reward`, the generic `rollout_metrics`. No `render_diagnostics`
-> `wants_diagnostics` is False -> eval-viz falls back to the filmstrip."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .base import default_rollout_metrics, not_provided

_NO_SIM = "recorded data has no simulator — WM trains on the dataset; control/interp need a live env"


@dataclass
class RecordedConfig:
    obs_dim: int = 16
    action_dim: int = 4
    dt: float = 1.0 / 30.0
    # inert placeholders: train_world_model reads e.R / e.r / e.init_speed unconditionally (torus geometry
    # knobs for variations + LoggingCallback); nothing consumes them on a recorded run.
    R: float = 1.0
    r: float = 1.0
    init_speed: float = 1.0


class RecordedEnv:
    """Batched `WorldEnv` facade over recorded data (see module docstring)."""

    def __init__(self, cfg, batch: int, device="cpu"):
        self.cfg = cfg
        self.batch = batch
        self.device = torch.device(device)
        self.obs_dim = int(cfg.obs_dim)
        self.action_dim = int(cfg.action_dim)
        self.dt = float(cfg.dt)

    # `not_provided` marks the raising stubs / generic defaults so `log_env_capabilities` reports them ✗
    # (reporting only — call behavior is unchanged).
    @not_provided
    def reset(self, generator=None) -> Tensor:
        raise NotImplementedError(_NO_SIM)

    @not_provided
    def step(self, action: Tensor) -> Tensor:
        raise NotImplementedError(_NO_SIM)

    @not_provided
    def reward(self, obs: Tensor, goal: Tensor | None = None) -> Tensor:
        return torch.zeros(obs.shape[0], device=obs.device)   # no goal semantics in recorded data

    @not_provided
    def render_obs(self, obs: Tensor) -> Tensor:
        raise NotImplementedError(_NO_SIM)   # images come from the dataset, not a renderer

    @not_provided
    def rollout_metrics(self, pred_obs: Tensor, true_obs: Tensor) -> dict[str, Tensor]:
        return default_rollout_metrics(pred_obs, true_obs)   # the generic default, nothing env-specific
