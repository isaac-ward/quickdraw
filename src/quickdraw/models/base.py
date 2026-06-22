"""BaseWorldModel: causal transformer, data-space delta prediction (design/models/base.md).

Full fusion: every modality is a stream -> encoder -> one fused token per timestep
`x_t = fuse(enc_o(o_t), enc_a(a_t))`. Causal transformer -> hidden `h_t` -> delta head ->
`o_hat_{t+1} = o_t + delta`. One `imagine` rollout fn serves training (p_tf), eval, and MPPI.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from .fuser import TokenStreamFuser
from .transformer import Transformer


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


def _mlp(din: int, dout: int) -> nn.Module:
    return nn.Sequential(nn.Linear(din, dout), nn.GELU(), nn.Linear(dout, dout))


class BaseWorldModel(nn.Module):
    def __init__(self, cfg: BaseModelConfig):
        super().__init__()
        self.cfg = cfg
        self.obs_enc = _mlp(cfg.obs_dim, cfg.d)
        self.act_enc = _mlp(cfg.action_dim, cfg.d)
        self.fuser = TokenStreamFuser(input_dims=[cfg.d, cfg.d], output_dim=cfg.d, pre_fuse_dim=cfg.d, post_fuse_dim=cfg.d)
        self.transformer = Transformer(cfg.d, cfg.depth, cfg.heads, cfg.window, cfg.mlp_ratio, cfg.rope_theta)
        self.delta_head = nn.Linear(cfg.d, cfg.obs_dim)
        self.window = cfg.window

    # ---- token construction ----
    def tokens(self, obs: Tensor, act: Tensor) -> Tensor:
        """obs (B,T,6), act (B,T,2) -> fused tokens (B,T,d)."""
        return self.fuser([self.obs_enc(obs), self.act_enc(act)])

    # ---- parallel teacher-forced forward (p_tf = 1) ----
    def forward(self, obs: Tensor, act: Tensor) -> Tensor:
        """Predict next obs at every position. obs/act (B,T,*) -> next-obs preds (B,T,6),
        where preds[:, t] approximates obs at t+1."""
        h = self.transformer(self.tokens(obs, act))  # (B,T,d)
        return obs + self.delta_head(h)

    # ---- shared autoregressive rollout (p_tf in [0,1]) ----
    @torch.no_grad()
    def imagine_eval(self, ctx_obs: Tensor, actions: Tensor, horizon: int) -> Tensor:
        return self._rollout(ctx_obs, actions, horizon, p_tf=0.0, true_future=None, detach_every=0)

    def _rollout(self, ctx_obs: Tensor, actions: Tensor, horizon: int, p_tf: float, true_future: Tensor | None, detach_every: int) -> Tensor:
        """Autoregressive rollout.

        Indices (per design/data.md Subtrajectory convention):
          ctx_obs : (B, P, 6)            = o_0 .. o_{P-1}            (true context)
          actions : (B, P+horizon-1, 2)  = a_0 .. a_{P+horizon-2}    (token t consumes a_t)
          true_future : (B, horizon, 6)  = o_P .. o_{P+horizon-1}    (for teacher forcing; may be None)
        Returns preds (B, horizon, 6) = predicted o_P .. o_{P+horizon-1}.
        """
        B, P, _ = ctx_obs.shape
        obs_buf = list(ctx_obs.unbind(dim=1))  # P tensors of (B,6)
        preds = []
        for h in range(horizon):
            L = len(obs_buf)  # = P + h ; we predict o_L from token at position L-1
            o_stack = torch.stack(obs_buf, dim=1)           # (B, L, 6)
            a_stack = actions[:, :L]                        # (B, L, 2)  uses a_0..a_{L-1}
            lo = max(0, L - self.window)                    # sliding window
            x = self.tokens(o_stack[:, lo:], a_stack[:, lo:])
            h_last = self.transformer(x)[:, -1]             # hidden at position L-1
            o_pred = obs_buf[-1] + self.delta_head(h_last)  # prediction of o_L
            preds.append(o_pred)
            # next fed observation: teacher-force per-batch with prob p_tf
            if true_future is not None and p_tf > 0.0:
                tf = (torch.rand(B, 1, device=o_pred.device) < p_tf).float()
                o_feed = tf * true_future[:, h] + (1.0 - tf) * o_pred
            else:
                o_feed = o_pred
            if detach_every and ((h + 1) % detach_every == 0):
                o_feed = o_feed.detach()
            obs_buf.append(o_feed)
        return torch.stack(preds, dim=1)

    def rollout_train(self, ctx_obs: Tensor, actions: Tensor, true_future: Tensor, p_tf: float, detach_every: int = 8) -> Tensor:
        """Training rollout with gradients (used when p_tf < 1)."""
        return self._rollout(ctx_obs, actions, true_future.shape[1], p_tf, true_future, detach_every)
