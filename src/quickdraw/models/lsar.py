"""LatentSpaceAR — latent-space autoregressor (JEPA/Dreamer style).

Encodes obs -> latent z (dz < d), predicts the next latent as a residual, and computes its loss in
latent space; a decoder reads latents back to obs for metrics (and, in the reconstruction mechanism,
shapes the encoder). The collapse mechanism is a composed CollapseStrategy. Shares the entire backbone
+ rollout with DSAR via SequenceWorldModel (design/models/latent_space_autoregressor.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .collapse import CollapseStrategy, Reconstruction
from .fuser import TokenStreamFuser
from .sequence import SequenceWorldModel, mlp


@dataclass
class LSARConfig:
    obs_dim: int = 6
    action_dim: int = 2
    d: int = 128          # backbone / token width
    dz: int = 16          # latent width (dz < d)
    depth: int = 4
    heads: int = 4
    window: int = 64
    mlp_ratio: float = 4.0
    rope_theta: float = 10000.0
    dec_hidden: int = 64  # small decoder hidden width (kept small + identical across mechanisms)
    lambda_pred_obs: float = 1.0  # weight on the unified obs-rollout loss (added by the LightningModule)
    lambda_reg: float = 1.0       # weight on the collapse regularizer (SIGReg/VICReg)
    expander_hidden: int = 256    # VICReg expander MLP width (VICReg only; discarded at eval)
    expander_dim: int = 256       # VICReg expander output (embedding) dim where var/cov are computed


def _ln(x: Tensor) -> Tensor:  # non-affine LayerNorm over the latent dim -> scale-free
    return F.layer_norm(x, (x.shape[-1],))


def pred_loss(pred: Tensor, target: Tensor, metric: str) -> Tensor:
    target = target.detach()  # stop-grad on the target (online or EMA)
    if metric == "mse":
        return F.mse_loss(pred, target)
    if metric == "normed_mse":
        return F.mse_loss(pred, _ln(target))
    return F.mse_loss(_ln(pred), _ln(target))  # cosine (default): standardize both


class LatentSpaceAR(SequenceWorldModel):
    def __init__(self, cfg: LSARConfig, collapse: CollapseStrategy | None = None):
        super().__init__(cfg.action_dim, cfg.d, cfg.depth, cfg.heads, cfg.window, cfg.mlp_ratio, cfg.rope_theta)
        self.cfg = cfg
        self.collapse = collapse or Reconstruction()
        self.enc = mlp(cfg.obs_dim, cfg.dz)                 # 6 -> dz  (role-a: the latent)
        self.fuser = TokenStreamFuser(input_dims=[cfg.dz, cfg.d], output_dim=cfg.d,
                                      pre_fuse_dim=cfg.d, post_fuse_dim=cfg.d)
        self.predictor = nn.Linear(cfg.d, cfg.dz)           # mirror of DSAR's delta_head (d -> dz)
        self.dec = nn.Sequential(nn.Linear(cfg.dz, cfg.dec_hidden), nn.GELU(),
                                 nn.Linear(cfg.dec_hidden, cfg.obs_dim))  # dz -> 6
        # how the unified obs-rollout loss is applied: a real loss for reconstruction (Dreamer), a
        # detached decoder-only readout probe for the JEPA variants (faithful: rep is latent-only).
        self.pred_obs_in_loss = self.collapse.obs_grounds_encoder
        self.lambda_pred_obs = cfg.lambda_pred_obs
        # LayerNorm the carried latent (encode_state + readout + target) to stop magnitude drift over the
        # rollout — but ONLY for variants WITHOUT a variance-based reg. LN fixes per-sample total variance,
        # while VICReg's var/cov and SIGReg's Epps-Pulley constrain per-dim/projection variance -> they
        # fight. So LN is for naked/recon/ema; vicreg/sigreg rely on their reg instead.
        self.latent_norm = not self.collapse.has_reg
        assert not (self.latent_norm and self.collapse.has_reg), \
            "LayerNorm-the-latent conflicts with a variance-based reg (vicreg/sigreg) — never combine them."
        # VICReg-only expander: var/cov act on this high-dim embedding (not the dz latent), so the harsh
        # constraints don't deform the dynamics latent. Built ONLY for VICReg; None elsewhere -> the
        # backbone + other variants are untouched. Discarded at eval (used only inside reg_loss).
        self.expander = None
        if getattr(self.collapse, "needs_expander", False):
            h, o = cfg.expander_hidden, cfg.expander_dim
            self.expander = nn.Sequential(
                nn.Linear(cfg.dz, h), nn.BatchNorm1d(h), nn.ReLU(),
                nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Linear(h, o))
        # BYOL predictor q (EMA-only): a dedicated 2-layer MLP the ONLINE prediction passes through
        # before being compared to the (no-predictor) EMA target. This online/target asymmetry is BYOL's
        # actual anti-collapse mechanism ("remove q -> it collapses"). Training-only: the carried latent
        # is still the raw predicted latent; q is applied solely inside the prediction loss.
        self.predictor_q = None
        if getattr(self.collapse, "needs_predictor", False):
            h = cfg.dec_hidden
            self.predictor_q = nn.Sequential(nn.Linear(cfg.dz, h), nn.BatchNorm1d(h), nn.ReLU(),
                                             nn.Linear(h, cfg.dz))
        # EMA mechanism: a slow, non-trainable copy of enc, registered so it moves with the model and
        # is checkpointed (the strategy updates it post-step and reads it as the target encoder).
        self.ema_enc = None
        if getattr(self.collapse, "needs_ema", False):
            import copy
            self.ema_enc = copy.deepcopy(self.enc)
            for p in self.ema_enc.parameters():
                p.requires_grad_(False)

    # ---- hooks: state == latent z (LayerNorm-bounded for the non-reg variants) ----
    def encode_state(self, obs: Tensor) -> Tensor:
        z = self.enc(obs)
        return _ln(z) if self.latent_norm else z

    def to_token(self, state: Tensor, act: Tensor) -> Tensor:
        return self.fuser([state, self.act_enc(act)])       # fuser up-projects dz -> d

    def readout(self, h: Tensor, prev_state: Tensor) -> Tensor:
        z = prev_state + self.predictor(h)
        return _ln(z) if self.latent_norm else z

    def to_obs(self, state: Tensor) -> Tensor:
        return self.dec(state)

    # ---- model-specific losses: latent prediction (+ optional collapse regularizer). Returns RAW
    # (pre-scaling) terms + their weights; the LightningModule logs the raw terms (comparable across
    # methods) and minimizes sum(weight*term) (+ the unified obs term). No decoder/autoencoding term:
    # the decoder is grounded by loss_pred_obs (full grad for reconstruction, detached probe otherwise).
    def loss_terms(self, pred_states, future_obs, obs_seq, p_tf):
        target = self.collapse.target_encoder(self)(future_obs)     # enc / enc_ema of the future obs
        if self.latent_norm:                                        # match the LN'd carried latent
            target = _ln(target)
        online = pred_states
        if self.predictor_q is not None:                            # BYOL: online prediction goes through q
            online = self.predictor_q(pred_states.reshape(-1, self.cfg.dz)).reshape(pred_states.shape)
        raw = {"pred_latent": pred_loss(online, target, self.collapse.pred_metric)}
        w = {"pred_latent": self.collapse.lambda_pred}
        if self.collapse.has_reg:
            z = self.enc(obs_seq).reshape(-1, self.cfg.dz)          # (B*T, dz) latents for the reg
            if self.expander is not None:                          # VICReg: var/cov on the expander embedding
                z = self.expander(z)
            raw["reg"] = self.collapse.reg_loss(z)
            w["reg"] = self.cfg.lambda_reg
        return raw, w

    def on_optimizer_step(self) -> None:
        self.collapse.on_optimizer_step(self)

    @torch.no_grad()
    def collapse_diagnostics(self, obs_seq: Tensor) -> dict:
        # force fp32: under bf16-mixed autocast the matmul below would be bf16, and eigvalsh has no
        # bf16-CUDA kernel (autocast downcasts matmuls even when the inputs are float()).
        with torch.autocast(device_type=obs_seq.device.type, enabled=False):
            z = self.enc(obs_seq).reshape(-1, self.cfg.dz).float()
            zc = z - z.mean(0, keepdim=True)
            cov = (zc.t() @ zc) / max(1, zc.shape[0] - 1)
            eig = torch.linalg.eigvalsh(cov).clamp_min(0.0)
            eff_rank = (eig.sum() ** 2) / (eig.pow(2).sum() + 1e-12)   # participation ratio in [1, dz]
            d = cov.diag().clamp_min(1e-12).sqrt()
            corr = cov / (d[:, None] * d[None, :])
            n = cov.shape[0]
            offdiag = (corr.abs().sum() - n) / (n * (n - 1))          # mean |off-diagonal correlation|
            return {"effective_rank": eff_rank, "per_dim_std_mean": zc.std(0).mean(), "offdiag_corr": offdiag,
                    "latent_norm": z.norm(dim=-1).mean(),            # mean |z| per sample (magnitude drift)
                    "latent_abs_max": z.abs().max()}                 # worst-case dim magnitude (blow-up watch)
