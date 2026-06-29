"""Diffusion — latent autoregressive flow-matching world model (design/models/diffusion.md).

A fourth model class for the shoot-out. Where DSAR/LSAR predict a point estimate of the next state,
this predicts the *distribution* by learning a velocity field that transports noise onto the next-state
manifold. It is **latent diffusion**: the field lives entirely in the latent z; enc/dec own the
modality (a vector MLP today). The diffusion never touches obs space directly.

It is a thin `SequenceWorldModel` subclass that is **reconstruction-grounded** (exactly like
`LSAR-reconstruction`: loss_pred_obs flows full-grad into the encoder — the flow loss alone is NOT
anti-collapse, a constant latent is trivially flow-predictable) and plugs a `FlowField` into:
  - `readout`  : integrate the ODE -> z_t + Delta-z-hat   (the SAMPLE, replacing the deterministic delta)
  - `loss_terms`: the rectified flow-matching loss (teacher-forced; the field's training signal)
Everything else — action encoder, fuser, transformer backbone, the ONE autoregressive rollout, the
p_tf curriculum, truncated BPTT, the obs-grounding, metrics — is reused unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .collapse import latent_diagnostics
from .flow import FlowField
from .fuser import TokenStreamFuser
from .sequence import SequenceWorldModel, mlp


@dataclass
class DiffusionConfig:
    obs_dim: int = 6
    action_dim: int = 2
    d: int = 96           # backbone / token width (shared substrate with DSAR/LSAR)
    dz: int = 16          # latent width (dz < d)
    depth: int = 4
    heads: int = 6
    window: int = 64
    mlp_ratio: float = 4.0
    rope_theta: float = 10000.0
    dec_hidden: int = 64  # decoder hidden width; the flow field mirrors it (flow_hidden)
    lambda_pred_obs: float = 1.0   # weight on the unified obs-rollout (reconstruction) loss
    lambda_flow: float = 1.0       # weight on the flow-matching prediction loss
    lambda_consistency: float = 1.0  # weight on the shortcut self-consistency loss (shortcut only)
    # --- flow head (diffusion.md "Config") ---
    cond: str = "concat"           # concat (default) | adaln (future upgrade)
    shortcut: bool = False         # false = plain flow (sampling_steps 4-8); true = step-size cond + consistency -> K works
    sampling_steps: int = 6        # K Euler steps / AR step at inference (>=1)
    predict: str = "residual"      # residual (diffuse Delta-z; default) | absolute (diffuse z_{t+1})
    stochastic_eval: bool = False  # false = deterministic ODE from eps=0 (reproducible metrics); true = fresh noise
    time_sampling: str = "uniform" # uniform (default) | logit_normal (SD3-style)
    flow_hidden: int = 0           # 0 -> mirror dec_hidden


def _ln(x: Tensor) -> Tensor:  # non-affine LayerNorm over the latent dim -> scale-free (matches LSAR)
    return F.layer_norm(x, (x.shape[-1],))


class Diffusion(SequenceWorldModel):
    def __init__(self, cfg: DiffusionConfig):
        super().__init__(cfg.action_dim, cfg.d, cfg.depth, cfg.heads, cfg.window, cfg.mlp_ratio, cfg.rope_theta)
        assert cfg.predict in ("residual", "absolute"), f"diffusion.predict={cfg.predict!r}"
        self.cfg = cfg
        self.predict_residual = cfg.predict == "residual"
        self.sampling_steps = int(cfg.sampling_steps)
        self.stochastic_eval = bool(cfg.stochastic_eval)
        self.time_sampling = cfg.time_sampling
        self.enc = mlp(cfg.obs_dim, cfg.dz)                 # 6 -> dz (the latent; LayerNorm-bounded)
        self.fuser = TokenStreamFuser(input_dims=[cfg.dz, cfg.d], output_dim=cfg.d,
                                      pre_fuse_dim=cfg.d, post_fuse_dim=cfg.d)
        self.dec = nn.Sequential(nn.Linear(cfg.dz, cfg.dec_hidden), nn.GELU(),
                                 nn.Linear(cfg.dec_hidden, cfg.obs_dim))  # dz -> 6 (grounding + readout)
        self.flow = FlowField(cfg.dz, h_dim=cfg.d, hidden=(cfg.flow_hidden or cfg.dec_hidden),
                              cond=cfg.cond, shortcut=cfg.shortcut)
        # reconstruction grounding: loss_pred_obs is a REAL loss shaping the encoder (anti-collapse),
        # exactly as LSAR-reconstruction (the flow loss alone admits the collapse solution).
        self.pred_obs_in_loss = True
        self.lambda_pred_obs = cfg.lambda_pred_obs

    # ---- hooks: state == latent z (LayerNorm-bounded, like LSAR-reconstruction) ----
    def encode_state(self, obs: Tensor) -> Tensor:
        return _ln(self.enc(obs))

    def to_token(self, state: Tensor, act: Tensor) -> Tensor:
        return self.fuser([state, self.act_enc(act)])

    def readout(self, h: Tensor, prev_state: Tensor) -> Tensor:
        # the SAMPLE: integrate the ODE to get Delta-z (residual) or z_{t+1} (absolute). Deterministic
        # (eps=0, reproducible) at eval unless stochastic_eval; stochastic during training (generative).
        deterministic = (not self.training) and (not self.stochastic_eval)
        out = self.flow.sample(h, steps=self.sampling_steps, deterministic=deterministic)
        z = (prev_state + out) if self.predict_residual else out
        return _ln(z)

    def to_obs(self, state: Tensor) -> Tensor:
        return self.dec(state)

    def physical_state(self, pred: Tensor) -> Tensor:
        # physical-loss seam: decode with a FROZEN decoder (functional_call), so the gradient flows to
        # the latent but NOT the decoder weights — identical to LSAR. Lets physical_loss + diffusion compose.
        from torch.func import functional_call
        pb = {n: p.detach() for n, p in self.dec.named_parameters()}
        pb.update({n: b.detach() for n, b in self.dec.named_buffers()})
        return functional_call(self.dec, pb, (pred,))

    # ---- model-specific loss: the flow-matching prediction (teacher-forced) ----
    def loss_terms(self, pred_states, future_obs, obs_seq, p_tf, act_seq=None):
        """Rectified flow-matching loss, computed TEACHER-FORCED at every transition in the window: encode
        the true window -> z, run the transformer over the true (z, action) sequence -> context h at each
        position, and regress the velocity toward the true (detached) transition target. This is the
        "one-step" training (the field sees true contexts); grounding (loss_pred_obs, added by the
        LightningModule) + the metrics use the SAMPLER via `readout`, so the encoder/decoder are grounded
        and the rollout still exposes the model to its own drift. Returns RAW terms + weights."""
        assert act_seq is not None, "Diffusion.loss_terms needs act_seq (the LightningModule passes it)"
        z = self.encode_state(obs_seq)                  # (B, L, dz) LN'd, L = P+F
        L = obs_seq.shape[1]
        s = z[:, :-1]                                   # contexts for transitions t -> t+1, t in 0..L-2
        a = act_seq[:, : L - 1]                         # token t consumes action t
        h = self.transformer(self.to_token(s, a))       # (B, L-1, d) teacher-forced context
        z_next = z[:, 1:]
        # detach the regression TARGET (the "data") — standard flow matching, and keeps the flow loss from
        # shaping the encoder (recon is the anti-collapse force). The CONTEXT s is NOT detached, so the
        # backbone is trained to produce predictive contexts (as DSAR/LSAR train it via prediction).
        target = (z_next - s).detach() if self.predict_residual else z_next.detach()
        l_flow, l_cons = self.flow.loss(h, target, time_sampling=self.time_sampling)
        raw = {"flow": l_flow}
        w = {"flow": self.cfg.lambda_flow}
        if l_cons is not None:
            raw["flow_consistency"] = l_cons
            w["flow_consistency"] = self.cfg.lambda_consistency
        return raw, w

    @torch.no_grad()
    def collapse_diagnostics(self, obs_seq: Tensor) -> dict:
        # confirm reconstruction grounding holds (eff_rank > 1). Shared with LSAR (models/collapse.py).
        with torch.autocast(device_type=obs_seq.device.type, enabled=False):
            z = self.enc(obs_seq).reshape(-1, self.cfg.dz).float()
            return latent_diagnostics(z)
