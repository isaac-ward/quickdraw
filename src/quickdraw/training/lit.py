"""LightningModule: p_tf rollout training, MSE on delta, env-space metrics (design/training.md)."""

from __future__ import annotations

import lightning as L
import torch

from ..environments import torus as T
from ..models.base import BaseWorldModel


class LitWorldModel(L.LightningModule):
    def __init__(self, model: BaseWorldModel, normalizer, R: float, r: float, P: int, F: int,
                 p_tf: float, lr: float, weight_decay: float, detach_every: int = 8):
        super().__init__()
        self.model = model
        self.norm = normalizer
        self.R, self.r, self.P, self.F = R, r, P, F
        self.p_tf, self.lr, self.weight_decay, self.detach_every = p_tf, lr, weight_decay, detach_every

    # ---- shared: produce future predictions (normalized) for a batch window ----
    def _future_preds(self, obs_seq, act_seq):
        P, F, L = self.P, self.F, self.P + self.F
        if self.p_tf >= 1.0:  # parallel teacher forcing
            preds = self.model(obs_seq[:, :-1], act_seq[:, :-1])  # predict obs[:,1:]
            return preds[:, P - 1 :], obs_seq[:, P:]
        ctx, actions, future = obs_seq[:, :P], act_seq[:, : L - 1], obs_seq[:, P:]
        preds = self.model.rollout_train(ctx, actions, future, self.p_tf, self.detach_every)
        return preds, future

    def _step(self, batch, tag):
        obs_seq, act_seq = batch["obs_seq"], batch["act_seq"]
        preds, future = self._future_preds(obs_seq, act_seq)
        loss = torch.nn.functional.mse_loss(preds, future)
        self.log(f"{tag}/loss_total", loss, prog_bar=(tag == "train"))
        self.log(f"{tag}/loss_obs_vector", loss)
        with torch.no_grad():  # metrics in real (denormalized) observation space
            p_hat = self.norm.denorm_obs(preds)
            p_true = self.norm.denorm_obs(future)
            self.log(f"{tag}/manifold_distance_error", T.manifold_distance_error(p_hat, self.R, self.r).mean())
            self.log(f"{tag}/pointwise_error", T.pointwise_error(p_hat, p_true).mean())
            self.log(f"{tag}/tangent_velocity_error", T.tangent_velocity_error(p_hat, self.R).mean())
        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    def configure_optimizers(self):
        fused = torch.cuda.is_available()
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, fused=fused)
