"""DataSpaceAR — data-space autoregressor (predicts the observation directly).

Subclass of SequenceWorldModel. The carried state IS the observation, so encode_state/to_obs are
identity: the obs encoder runs inside to_token each step and the delta head reads out directly in
obs space. This is the former `BaseWorldModel`, refactored onto the shared ancestor with no math
change (design/models/data_space_autoregressor.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn
from torch import Tensor

from .fuser import TokenStreamFuser
from .sequence import SequenceWorldModel, mlp


@dataclass
class BaseModelConfig:
    obs_dim: int = 6
    action_dim: int = 2
    d: int = 128
    depth: int = 4
    heads: int = 4
    window: int = 64
    mlp_ratio: float = 4.0
    rope_theta: float = 10000.0


class DataSpaceAR(SequenceWorldModel):
    def __init__(self, cfg: BaseModelConfig):
        super().__init__(cfg.action_dim, cfg.d, cfg.depth, cfg.heads, cfg.window, cfg.mlp_ratio, cfg.rope_theta)
        self.cfg = cfg
        self.obs_enc = mlp(cfg.obs_dim, cfg.d)
        self.fuser = TokenStreamFuser(input_dims=[cfg.d, cfg.d], output_dim=cfg.d, pre_fuse_dim=cfg.d, post_fuse_dim=cfg.d)
        self.delta_head = nn.Linear(cfg.d, cfg.obs_dim)

    # ---- hooks: state == observation ----
    def encode_state(self, obs: Tensor) -> Tensor:
        return obs

    def to_token(self, state: Tensor, act: Tensor) -> Tensor:
        return self.fuser([self.obs_enc(state), self.act_enc(act)])

    def readout(self, h: Tensor, prev_state: Tensor) -> Tensor:
        return prev_state + self.delta_head(h)

    def to_obs(self, state: Tensor) -> Tensor:
        return state

    # No model-specific loss terms: DSAR's whole objective is the unified loss_pred_obs (computed in
    # the LightningModule as MSE(to_obs(preds), future_obs); to_obs is identity here, so this is exactly
    # the former obs_vector_mse / MSE-on-delta objective). pred_obs_in_loss=True (inherited).
