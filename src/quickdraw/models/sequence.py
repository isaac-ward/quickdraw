"""SequenceWorldModel: shared ancestor for DSAR (data-space) and LSAR (latent-space).

Owns the action encoder + causal Transformer backbone + the ONE autoregressive rollout (p_tf teacher
forcing + truncated BPTT). The rollout is written against four hooks the subclass fills in; the
carried "state" is the observation for DSAR and a learned latent for LSAR. See design/models/.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .transformer import Transformer, pad_block_mask


def mlp(din: int, dout: int) -> nn.Module:
    return nn.Sequential(nn.Linear(din, dout), nn.GELU(), nn.Linear(dout, dout))


class SequenceWorldModel(nn.Module):
    """Backbone + rollout; subclasses implement the four state hooks (and compute_losses)."""

    def __init__(self, action_dim: int, d: int, depth: int, heads: int, window: int,
                 mlp_ratio: float, rope_theta: float):
        super().__init__()
        self.act_enc = mlp(action_dim, d)
        self.transformer = Transformer(d, depth, heads, window, mlp_ratio, rope_theta)
        self.window = window

    # ---- hooks (subclass implements) -------------------------------------------------
    def encode_state(self, obs: Tensor) -> Tensor:
        """obs (B,T,6) -> carried state (B,T,state_dim). DSAR: identity; LSAR: enc(obs)."""
        raise NotImplementedError

    def to_token(self, state: Tensor, act: Tensor) -> Tensor:
        """(state, action) -> fused step-token (B,T,d)."""
        raise NotImplementedError

    def readout(self, h: Tensor, prev_state: Tensor) -> Tensor:
        """hidden + previous state -> next state (residual). Works for (B,*) and (B,T,*)."""
        raise NotImplementedError

    def to_obs(self, state: Tensor) -> Tensor:
        """carried state -> observation (B,*,6). DSAR: identity; LSAR: dec(state)."""
        raise NotImplementedError

    # ---- composability seams for the train-time variations (design/models/variations.md) ----
    # These live on the ANCESTOR so every model (DSAR/LSAR/future RSSM/vision) gets them via the hooks,
    # and the variation code never reaches into a concrete model.
    def physical_state(self, pred: Tensor) -> Tensor | None:
        """Physical 6-vector [p; p_dot] for the physical-loss variation. Default: `to_obs` (correct when
        the observation IS the physical state, e.g. DSAR / vector LSAR). Models whose `to_obs` is a
        learned decoder override to FREEZE it (grad to the state, not the decoder weights). A vision
        model returns its physical-readout head's output, or None to signal "physics unavailable" ->
        the variation auto-skips."""
        return self.to_obs(pred)

    def one_step_states(self, state_win: Tensor, act_win: Tensor, attn_eager: bool = False) -> Tensor:
        """One advance of the shared rollout as a pure state->state map: (state_win (B,W,state),
        act_win (B,W,2)) -> next state (B,state), predicted at the LAST position. Built only from the
        hooks + backbone, so it is identical for every model. `attn_eager=True` routes the backbone
        through the differentiable eager sdpa(MATH) attention path (FlexAttention can't double-back),
        used by the contraction penalty's Jacobian power-iteration."""
        x = self.to_token(state_win, act_win)
        h = self.transformer(x, attn_eager=attn_eager)
        return self.readout(h[:, -1], state_win[:, -1])

    # Whether the unified obs-rollout loss (loss_pred_obs, computed in the LightningModule) is a real
    # loss term that shapes this model (True: DSAR, LSAR-reconstruction) or a detached eval-only readout
    # probe that trains the decoder without shaping the representation (False: JEPA variants).
    pred_obs_in_loss: bool = True
    lambda_pred_obs: float = 1.0

    def loss_terms(self, pred_states: Tensor, future_obs: Tensor, obs_seq: Tensor, p_tf: float):
        """Model-SPECIFIC RAW loss terms + their weights, as (raw_dict, weight_dict). The LightningModule
        logs the raw terms (pre-scaling, comparable across methods) and minimizes sum(weight*term), then
        adds the unified obs term. DSAR: ({}, {}) (its only term is loss_pred_obs); LSAR: pred_latent[, reg]."""
        return {}, {}

    def on_optimizer_step(self) -> None:
        """Called after each optimizer step (EMA target update for LSAR-EMA; no-op otherwise)."""
        pass

    # ---- teacher-forced parallel forward (predict next state at every position) ------
    def forward(self, obs: Tensor, act: Tensor) -> Tensor:
        s = self.encode_state(obs)
        h = self.transformer(self.to_token(s, act))
        return self.readout(h, s)

    # ---- shared autoregressive rollout (returns STATES; call to_obs() to read out) ---
    def _rollout(self, ctx_obs: Tensor, actions: Tensor, horizon: int, p_tf: float,
                 true_future: Tensor | None, detach_every: int) -> Tensor:
        """ctx_obs (B,P,6); actions (B,P+horizon-1,2); true_future (B,horizon,6) or None.
        Returns predicted states (B,horizon,state_dim)."""
        B = ctx_obs.shape[0]
        W = self.window
        state_buf = list(self.encode_state(ctx_obs).unbind(dim=1))  # P states of (B,state_dim)
        tf_future = self.encode_state(true_future) if true_future is not None else None
        preds = []
        for h in range(horizon):
            L = len(state_buf)
            real = min(L, W)                                # real tokens in the window (sliding cap W)
            pad = W - real                                  # front-padding to keep length == W (fixed shape)
            s_win = torch.stack(state_buf[-real:], dim=1)   # (B,real,state) most-recent states
            a_win = actions[:, L - real:L]                  # (B,real,2)  token t consumes a_t
            if pad:                                         # front-pad (masked out) so T==W every step ->
                s_win = F.pad(s_win, (0, 0, pad, 0))        # FlexAttention sees ONE shape, compiles once
                a_win = F.pad(a_win, (0, 0, pad, 0))
            x = self.to_token(s_win, a_win)                 # (B,W,d)
            bm = pad_block_mask(W, pad, x.device)           # causal + mask the `pad` front positions
            h_last = self.transformer(x, block_mask=bm)[:, -1]  # hidden at the current (last) position
            s_pred = self.readout(h_last, state_buf[-1])    # predicted next state
            preds.append(s_pred)
            if tf_future is not None and p_tf > 0.0:        # teacher-force per-batch with prob p_tf
                tf = (torch.rand(B, 1, device=s_pred.device) < p_tf).float()
                s_feed = tf * tf_future[:, h] + (1.0 - tf) * s_pred
            else:
                s_feed = s_pred
            if detach_every and ((h + 1) % detach_every == 0):
                s_feed = s_feed.detach()
            state_buf.append(s_feed)
        return torch.stack(preds, dim=1)

    @torch.no_grad()
    def imagine_eval(self, ctx_obs: Tensor, actions: Tensor, horizon: int) -> Tensor:
        # bf16 autocast for the eval rollout (no grad): ~2x throughput on the long-horizon/OOD rollouts
        # and the MPPI control candidates. Output cast back to fp32 for the obs-space metrics/plots.
        with torch.autocast(device_type=ctx_obs.device.type, dtype=torch.bfloat16, enabled=ctx_obs.is_cuda):
            out = self.to_obs(self._rollout(ctx_obs, actions, horizon, 0.0, None, 0))
        return out.float()

    def rollout_train(self, ctx_obs: Tensor, actions: Tensor, true_future: Tensor, p_tf: float,
                      detach_every: int = 8) -> Tensor:
        """Predicted STATES (B,horizon,state_dim); the LightningModule computes the loss in that space."""
        return self._rollout(ctx_obs, actions, true_future.shape[1], p_tf, true_future, detach_every)
