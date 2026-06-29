"""Collapse-prevention strategies for LSAR (design/models/latent_space_autoregressor.md).

A strategy is a small object the LatentSpaceAR holds (composition, not subclassing). It declares the
knobs the model reads:
  - target_encoder  : where the prediction target comes from (online `enc`, or an EMA copy)
  - pred_metric     : the paper's prediction-loss metric (mse / normed_mse / cosine)
  - lambda_pred     : weight on the latent prediction term (VICReg's invariance weight)
  - reg_loss        : optional distributional penalty on the (optionally expander-projected) latents
  - has_reg         : whether reg_loss contributes a term
  - needs_ema       : register an EMA copy of enc as model.ema_enc
  - needs_expander  : register a VICReg-style expander head (model.expander), used ONLY for reg
  - obs_grounds_encoder : True  -> loss_pred_obs is a real loss shaping the encoder (Dreamer/recon)
                          False -> loss_pred_obs is a DETACHED readout probe (faithful JEPA);
                                   the decoder is trained but never shapes the representation.

The losses themselves live in LatentSpaceAR.loss_terms (+ the unified loss_pred_obs in the
LightningModule). Each variant's training objective matches its paper (VICReg arXiv 2105.04906,
LeJEPA/SIGReg arXiv 2511.08544, BYOL/I-JEPA for EMA).
"""

from __future__ import annotations

import torch
from torch import Tensor


def latent_diagnostics(z: Tensor) -> dict:
    """Collapse diagnostics for a batch of latents z (N, dz), shared by every latent model (LSAR +
    diffusion) so the `collapse/*` logs are computed identically. Magnitude stats are always returned;
    the rank/correlation stats are wrapped in try/except (a degenerate batch must never crash training).
    Caller is responsible for fp32 + autocast-off (the linalg below has no bf16-CUDA kernel)."""
    zc = z - z.mean(0, keepdim=True)
    out = {"per_dim_std_mean": zc.std(0).mean(),
           "latent_norm": z.norm(dim=-1).mean(),          # mean |z| per sample (magnitude drift)
           "latent_abs_max": z.abs().max()}               # worst-case dim magnitude (blow-up watch)
    try:
        # eff_rank from SINGULAR values of the centered latents (participation ratio in [1, dz]):
        # robust where eigvalsh on the covariance fails to converge on ill-conditioned/degenerate
        # latents (e.g. the physical-loss runs drive repeated eigenvalues -> eigvalsh crashed val).
        ev = torch.linalg.svdvals(zc) ** 2                # = (N-1)*eigenvalues; scale cancels in PR
        out["effective_rank"] = (ev.sum() ** 2) / (ev.pow(2).sum() + 1e-12)
        cov = (zc.t() @ zc) / max(1, zc.shape[0] - 1)
        d = cov.diag().clamp_min(1e-12).sqrt()
        corr = cov / (d[:, None] * d[None, :])
        n = cov.shape[0]
        out["offdiag_corr"] = (corr.abs().sum() - n) / (n * (n - 1))   # mean |off-diagonal corr|
    except Exception:   # a degenerate batch must never crash training — just skip these two
        pass
    return out


class CollapseStrategy:
    has_reg: bool = False
    needs_ema: bool = False
    needs_expander: bool = False
    needs_predictor: bool = False       # BYOL's online-only predictor q (the asymmetry that blocks collapse)
    obs_grounds_encoder: bool = False  # False -> loss_pred_obs is a detached eval-only probe
    pred_metric: str = "mse"
    lambda_pred: float = 1.0

    def target_encoder(self, model):
        """Encoder used to build the prediction target. Online `enc` by default; EMA overrides."""
        return model.enc

    def reg_loss(self, z: Tensor) -> Tensor:
        """Distributional penalty on the batch of (expander-projected) latents (0 unless SIGReg/VICReg)."""
        return z.new_zeros(())

    def on_optimizer_step(self, model) -> None:
        """Post-step hook (EMA target update); no-op otherwise."""
        pass


class Naked(CollapseStrategy):
    """Negative control: latent prediction ONLY (loss_pred_latent), nothing prevents collapse. The
    decoder is a detached eval probe; expect the latent to collapse and obs metrics to be garbage."""


class Reconstruction(CollapseStrategy):
    """Dreamer-style: obs reconstruction of the rollout (loss_pred_obs) flows into the encoder and is
    the anti-collapse force. The ONLY variant where loss_pred_obs shapes the representation."""
    obs_grounds_encoder = True


class EMA(CollapseStrategy):
    """BYOL / I-JEPA: target = a slow EMA copy of enc (stop-grad). Collapse blocked by the fast/slow
    asymmetry + the online-only PREDICTOR q (the paper: remove q and the representation collapses).
    Prediction is BYOL's normalized MSE."""
    needs_ema = True
    needs_predictor = True
    pred_metric = "normed_mse"

    def __init__(self, tau: float = 0.99):
        self.tau = tau

    def target_encoder(self, model):
        return model.ema_enc

    @torch.no_grad()
    def on_optimizer_step(self, model) -> None:
        for pe, p in zip(model.ema_enc.parameters(), model.enc.parameters()):
            pe.mul_(self.tau).add_(p.detach(), alpha=1.0 - self.tau)
        for be, b in zip(model.ema_enc.buffers(), model.enc.buffers()):
            be.copy_(b)


class SIGReg(CollapseStrategy):
    """LeJEPA (arXiv 2511.08544): push the latent distribution toward an isotropic Gaussian via random
    1-D sketches scored with the Epps-Pulley characteristic-function normality test (BHEP). For each
    random unit direction, project the latents and integrate |phi_emp(t) - phi_N(t)|^2 weighted by
    phi_N(t) over a small frequency grid; raw (un-standardized) projections so it enforces mean 0,
    variance 1, AND Gaussian shape jointly. Linear in batch size."""
    has_reg = True

    def __init__(self, n_sketches: int = 64, n_grid: int = 17, t_max: float = 5.0):
        self.n_sketches, self.n_grid, self.t_max = n_sketches, n_grid, t_max

    def reg_loss(self, z: Tensor) -> Tensor:
        # fp32 for the characteristic-function integral (bf16 trig/trapz is too coarse); grad still flows.
        with torch.autocast(device_type=z.device.type, enabled=False):
            z = z.reshape(-1, z.shape[-1]).float()
            d = z.shape[-1]
            u = torch.randn(d, self.n_sketches, device=z.device, dtype=z.dtype)
            u = u / u.norm(dim=0, keepdim=True)               # M random unit directions
            p = z @ u                                          # (N, M) raw projections, tested vs N(0,1)
            t = torch.linspace(-self.t_max, self.t_max, self.n_grid, device=z.device, dtype=z.dtype)
            phi_g = torch.exp(-0.5 * t * t)                    # (K,) standard-normal CF (= weight w(t))
            tp = p.unsqueeze(-1) * t                           # (N, M, K)
            phi_re = tp.cos().mean(0)                          # (M, K) empirical CF, real
            phi_im = tp.sin().mean(0)                          # (M, K) empirical CF, imag
            diff = (phi_re - phi_g).pow(2) + phi_im.pow(2)     # |phi_emp - phi_N|^2
            Tm = torch.trapz(phi_g * diff, t, dim=-1)          # (M,) Epps-Pulley integral per sketch
            return Tm.mean()                                   # mean over sketches; O(1), batch-size-free.
            # NB: the paper's ×N (sample-size) test-statistic scaling is folded into lambda_reg here —
            # keeping it in the term made the loss ~N (18k) and swamped prediction (degenerate sigreg).


class VICReg(CollapseStrategy):
    """VICReg (arXiv 2105.04906): variance hinge (per-dim std floor) + covariance decorrelation. Applied
    DIRECTLY to the dz latent, NOT an expander: in a world model the latent IS the state, and an
    expander lets the latent collapse (eff_rank ~3/16) while the high-dim embedding stays full-rank,
    hiding the collapse. Binding var/cov to the latent forces the state itself to use all dz dims. The
    invariance term is loss_pred_latent, weighted lambda_pred=25 (paper's 25/25/1 balance)."""
    has_reg = True
    needs_expander = False  # constrain the latent directly (expander would mask latent collapse)
    lambda_pred = 25.0

    def __init__(self, var_w: float = 25.0, cov_w: float = 1.0, gamma: float = 1.0):
        self.var_w, self.cov_w, self.gamma = var_w, cov_w, gamma

    def reg_loss(self, z: Tensor) -> Tensor:
        z = z.reshape(-1, z.shape[-1])
        zc = z - z.mean(0, keepdim=True)
        std = (zc.var(0) + 1e-4).sqrt()
        var_loss = torch.relu(self.gamma - std).mean()
        n, d = zc.shape
        cov = (zc.t() @ zc) / max(1, n - 1)
        off = cov - torch.diag(torch.diag(cov))
        cov_loss = off.pow(2).sum() / d
        return self.var_w * var_loss + self.cov_w * cov_loss


def make_collapse(cfg) -> CollapseStrategy:
    """Build a strategy from a `collapse` config node (has `.name` + per-mechanism fields)."""
    name = str(cfg.name)
    if name == "naked":
        return Naked()
    if name == "reconstruction":
        return Reconstruction()
    if name == "ema":
        return EMA(tau=float(cfg.tau))
    if name == "sigreg":
        return SIGReg(n_sketches=int(cfg.n_sketches), n_grid=int(cfg.get("n_grid", 17)),
                      t_max=float(cfg.get("t_max", 5.0)))
    if name == "vicreg":
        return VICReg(var_w=float(cfg.var_w), cov_w=float(cfg.cov_w), gamma=float(cfg.gamma))
    raise ValueError(f"unknown collapse mechanism: {name!r}")
