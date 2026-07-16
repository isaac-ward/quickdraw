"""Transport heads — rectified-flow generative heads, reused for the dynamics AND the modality decoders.

`TransportHead` owns the shared algorithm (rectified flow-matching loss, K-step Euler sampling, optional
shortcut self-consistency for K=1 — Frans et al. 2024) over an INJECTED velocity net. Concrete heads
supply `velocity`:
  - `FlowField`     — a small MLP velocity (the dynamics head over the token bag; also the proprio decoder).
  - `ImageFlowHead` — a ViT velocity denoiser over an image, conditioned on the predicted latent tokens
                      (the generative image decoder — sharp, samples a mode, vs the MSE decoder's blur).

`event_dims` = how many TRAILING dims form one sample that shares a noise level (leading dims each get an
independent tau): 1 for the per-token dynamics/proprio heads, 3 (H,W,C) for the image head. eps=0 (the
noise mean) at sample time gives the deterministic, reproducible committed prediction used for metrics.
See design/models/flow_heads.md (the reuse refactor) and design/models/diffusion.md (the dynamics head).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _time_features(t: Tensor, freqs: Tensor) -> Tensor:
    """Scalar t in [0,1] (any leading shape, last dim 1) -> [sin, cos] Fourier features (..., 2*F)."""
    ang = t * freqs
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


class TransportHead(nn.Module):
    """Shared rectified-flow algorithm over an injected `velocity(x, temb, cond, demb)` net. Subclass and
    implement `velocity`. The base owns the time (and shortcut step-size) embeddings + loss/sample/consistency."""

    def __init__(self, *, param: str = "v", shortcut: bool = False, event_dims: int = 1,
                 n_freq: int = 16, time_dim: int = 32):
        super().__init__()
        # param = what the net predicts / how we sample:
        #   "v"  (velocity, rectified flow): net predicts the velocity u = eps - x0; sample INTEGRATES the ODE
        #        noise->data over K Euler steps. Good for low-dim targets (latent tokens, proprio); for
        #        high-dim images the integration accumulates error + overshoots range -> poor reconstruction.
        #   "x0" (data / consistency-style): net predicts the CLEAN target x0 DIRECTLY from a noised input.
        #        sample = predict x0 (1 step from noise) -> precise + in-range; K>1 refines by renoising
        #        (consistency multi-step). Deterministic (eps=0) x0 = the conditional mean (~MSE); stochastic
        #        eps + multi-step gives sharp samples. This is the IWS decode approach; use it for image decode.
        assert param in ("v", "x0")
        if param == "x0":
            shortcut = False                          # x0 is a 1-step predictor; shortcut self-consistency is a v-only trick
        self.param, self.shortcut, self.event_dims, self.time_dim = param, shortcut, event_dims, time_dim
        # fixed log-spaced frequencies (deterministic -> reproducible embeddings / golden-testable)
        self.register_buffer("freqs", 2.0 * math.pi * torch.logspace(0.0, 2.0, n_freq), persistent=False)
        self.tau_mlp = nn.Sequential(nn.Linear(2 * n_freq, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim))
        self.d_mlp = (nn.Sequential(nn.Linear(2 * n_freq, time_dim), nn.GELU(),
                                    nn.Linear(time_dim, time_dim)) if shortcut else None)

    def _temb(self, tau: Tensor) -> Tensor:
        return self.tau_mlp(_time_features(tau, self.freqs))

    def _demb(self, d: Tensor) -> Tensor:
        return self.d_mlp(_time_features(d, self.freqs))

    def velocity(self, x: Tensor, temb: Tensor, cond: Tensor, demb: Tensor | None = None) -> Tensor:
        raise NotImplementedError

    # ---- shapes: leading dims get independent taus; trailing `event_dims` share one ----
    def _tau_shape(self, target: Tensor) -> tuple[int, ...]:
        return target.shape[: target.ndim - self.event_dims] + (1,) * self.event_dims

    def _sample_time(self, shape, device, dtype, time_sampling: str) -> Tensor:
        if time_sampling == "logit_normal":          # SD3-style: weight mid-noise levels more
            return torch.sigmoid(torch.randn(shape, device=device, dtype=dtype))
        return torch.rand(shape, device=device, dtype=dtype)   # uniform (default)

    # ---- training ----
    def loss(self, cond: Tensor, target: Tensor, *, time_sampling: str = "uniform") -> tuple[Tensor, Tensor | None]:
        """param="v": rectified flow-matching ||net - (eps-target)||^2 (+ shortcut self-consistency).
        param="x0": ||net(x_tau,tau) - target||^2 — predict the clean target directly. Returns (L_main, L_shortcut|None)."""
        ts = self._tau_shape(target)
        tau = self._sample_time(ts, target.device, target.dtype, time_sampling)
        eps = torch.randn_like(target)
        x_tau = (1.0 - tau) * target + tau * eps      # straight (rectified) path
        if self.param == "x0":                        # net predicts the CLEAN target directly
            x0_hat = self.velocity(x_tau, self._temb(tau), cond, None)
            return F.mse_loss(x0_hat, target), None
        u = eps - target                              # velocity along the straight path (regression target)
        demb = self._demb(torch.zeros_like(tau)) if self.shortcut else None   # flow-matching = the d->0 field
        v = self.velocity(x_tau, self._temb(tau), cond, demb)
        l_flow = F.mse_loss(v, u)
        return (l_flow, self._consistency(cond, target)) if self.shortcut else (l_flow, None)

    def _consistency(self, cond: Tensor, target: Tensor) -> Tensor:
        """Shortcut self-consistency: one step of size 2d must equal two chained steps of size d (bootstrap
        target stop-gradded) -> accurate large/single steps, so K=1 sampling works."""
        ts = self._tau_shape(target)
        k = torch.randint(1, 4, ts, device=target.device)                   # 1..3
        d = (0.5 ** k.float())                                              # 1/2, 1/4, 1/8
        tau = 2.0 * d + (1.0 - 2.0 * d) * torch.rand(ts, device=target.device, dtype=target.dtype)
        eps = torch.randn_like(target)
        x = (1.0 - tau) * target + tau * eps
        with torch.no_grad():                                              # bootstrap target (stop-grad)
            v1 = self.velocity(x, self._temb(tau), cond, self._demb(d))
            x2 = x - v1 * d
            v2 = self.velocity(x2, self._temb(tau - d), cond, self._demb(d))
            s_target = 0.5 * (v1 + v2)
        v_2d = self.velocity(x, self._temb(tau), cond, self._demb(2.0 * d))  # the large step, with grad
        return F.mse_loss(v_2d, s_target)

    # ---- inference: integrate the ODE ----
    def _sample(self, cond: Tensor, *, event_shape, lead, steps: int, deterministic: bool,
                eps: Tensor | None = None, record_path: bool = False):
        """Euler-integrate dx/dtau = v from tau=1 (x=eps) -> tau=0. eps=0 (deterministic) -> reproducible
        committed prediction. `event_shape`/`lead` let heads with different target shapes reuse this."""
        if eps is None:
            shp = tuple(lead) + tuple(event_shape)
            eps = cond.new_zeros(shp) if deterministic else torch.randn(shp, device=cond.device, dtype=cond.dtype)
        x = eps
        path = [x]
        ts = tuple(lead) + (1,) * self.event_dims
        if self.param == "x0":                        # consistency-style: predict x0, optionally renoise + refine
            for k in range(steps):
                tau = cond.new_full(ts, 1.0 - k / steps)
                x0_hat = self.velocity(x, self._temb(tau), cond, None)      # direct clean-target prediction
                if record_path:
                    path.append(x0_hat)
                if k < steps - 1:                     # renoise to a lower level and refine (0 noise if deterministic)
                    tn = 1.0 - (k + 1) / steps
                    noise = torch.zeros_like(x0_hat) if deterministic else torch.randn_like(x0_hat)
                    x = (1.0 - tn) * x0_hat + tn * noise
            return (x0_hat, path) if record_path else x0_hat
        d = cond.new_full(ts, 1.0 / steps) if self.shortcut else None       # step-size conditioning
        demb = self._demb(d) if self.shortcut else None
        for k in range(steps):                        # param="v": Euler-integrate dx/dtau = v, tau=1 -> 0
            tau = cond.new_full(ts, 1.0 - k / steps)
            x = x - self.velocity(x, self._temb(tau), cond, demb) * (1.0 / steps)
            if record_path:
                path.append(x)
        return (x, path) if record_path else x


class FlowField(TransportHead):
    """MLP velocity `v([x || emb(tau) || cond (|| emb(d))]) -> dz`. The dynamics head over the token bag
    (cond = backbone context h, dz = d) AND the proprio decoder (cond = the proprio token, dz = 6)."""

    def __init__(self, dz: int, h_dim: int, hidden: int, *, cond: str = "concat", param: str = "v",
                 shortcut: bool = False, n_freq: int = 16, time_dim: int = 32):
        if cond != "concat":
            raise NotImplementedError(f"FlowField cond={cond!r} not implemented; use 'concat' (adaln is a future upgrade).")
        super().__init__(param=param, shortcut=shortcut, event_dims=1, n_freq=n_freq, time_dim=time_dim)
        self.dz = dz
        in_dim = dz + time_dim + h_dim + (time_dim if self.shortcut else 0)   # self.shortcut (x0 forces it off)
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, dz))

    def velocity(self, x: Tensor, temb: Tensor, cond: Tensor, demb: Tensor | None = None) -> Tensor:
        parts = [x, temb, cond] + ([demb] if self.shortcut else [])
        return self.net(torch.cat(parts, dim=-1))

    def sample(self, h: Tensor, *, steps: int, deterministic: bool, eps: Tensor | None = None,
               record_path: bool = False):
        """Preserves the original signature (cond=h, dz-shaped output) so the dynamics call sites are unchanged."""
        return self._sample(h, event_shape=(self.dz,), lead=h.shape[:-1], steps=steps,
                            deterministic=deterministic, eps=eps, record_path=record_path)


class ImageFlowHead(TransportHead):
    """ViT velocity denoiser over an image, conditioned on the predicted latent tokens: a GENERATIVE image
    decoder. v(noised_img (M,H,W,C), tau, latent_tokens (M,num_tokens,d)[, d_step]) -> velocity (M,H,W,C).
    Reuses vision.ViTBlock/CrossAttn + linear (de)patchify (own weights, separate from the AE)."""

    def __init__(self, ae_cfg, *, depth: int = 4, param: str = "v", shortcut: bool = False, n_freq: int = 16, time_dim: int = 32):
        super().__init__(param=param, shortcut=shortcut, event_dims=3, n_freq=n_freq, time_dim=time_dim)
        from .vision import CrossAttn, ViTBlock
        c = ae_cfg
        self.cfg = c
        self.gp = c.img_size // c.patch
        self.np = self.gp * self.gp
        pdim = c.patch * c.patch * c.channels
        d = c.d
        self.patch_embed = nn.Linear(pdim, d)
        self.pos = nn.Parameter(torch.zeros(1, self.np, d))
        self.t_proj = nn.Linear(time_dim, d)
        self.d_proj = nn.Linear(time_dim, d) if self.shortcut else None      # self.shortcut (x0 forces it off)
        self.from_latent = CrossAttn(d, c.heads)                      # patches attend to the conditioning tokens
        self.blocks = nn.ModuleList([ViTBlock(d, c.heads, c.mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        self.unpatch = nn.Linear(d, pdim)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def _patchify(self, img):                                          # (M,H,W,C) -> (M, np, patch*patch*C)
        p, gp, C = self.cfg.patch, self.gp, self.cfg.channels
        M = img.shape[0]
        return img.reshape(M, gp, p, gp, p, C).permute(0, 1, 3, 2, 4, 5).reshape(M, gp * gp, p * p * C)

    def _unpatchify(self, x):                                          # (M, np, patch*patch*C) -> (M,H,W,C)
        p, gp, C = self.cfg.patch, self.gp, self.cfg.channels
        x = x.reshape(x.shape[0], gp, gp, p, p, C).permute(0, 1, 3, 2, 4, 5)
        return x.reshape(x.shape[0], gp * p, gp * p, C)

    def velocity(self, x: Tensor, temb: Tensor, cond: Tensor, demb: Tensor | None = None) -> Tensor:
        M = x.shape[0]
        h = self.patch_embed(self._patchify(x)) + self.pos            # (M, np, d)
        h = h + self.t_proj(temb.reshape(M, -1)).unsqueeze(1)         # broadcast one tau-embed over patches
        if self.shortcut and demb is not None:
            h = h + self.d_proj(demb.reshape(M, -1)).unsqueeze(1)
        h = self.from_latent(h, cond)                                 # condition on the predicted latent tokens
        for blk in self.blocks:
            h = blk(h)
        return self._unpatchify(self.unpatch(self.norm(h)))           # (M, H, W, C)

    def sample(self, cond: Tensor, *, steps: int, deterministic: bool, eps: Tensor | None = None,
               record_path: bool = False):
        c = self.cfg
        return self._sample(cond, event_shape=(c.img_size, c.img_size, c.channels), lead=cond.shape[:-2],
                            steps=steps, deterministic=deterministic, eps=eps, record_path=record_path)


class ImageUNetFlowHead(TransportHead):
    """CNN U-Net velocity denoiser over an image (the conv alternative to ImageFlowHead's ViT), conditioned on
    the predicted latent tokens. Same TransportHead contract + `sample()` signature — so it drops in wherever
    ImageFlowHead does. No patch grid -> smooth color fields don't block (see the ep24 ViT-decode blocking)."""

    def __init__(self, ae_cfg, *, base: int = 32, param: str = "v", shortcut: bool = False,
                 n_freq: int = 16, time_dim: int = 32):
        super().__init__(param=param, shortcut=shortcut, event_dims=3, n_freq=n_freq, time_dim=time_dim)
        from .vision import ConditionalUNet
        self.cfg = ae_cfg
        self.unet = ConditionalUNet(ae_cfg, base=base, time_dim=time_dim)

    def velocity(self, x: Tensor, temb: Tensor, cond: Tensor, demb: Tensor | None = None) -> Tensor:
        return self.unet.velocity(x, temb, cond, demb)

    def sample(self, cond: Tensor, *, steps: int, deterministic: bool, eps: Tensor | None = None,
               record_path: bool = False):
        c = self.cfg
        return self._sample(cond, event_shape=(c.img_size, c.img_size, c.channels), lead=cond.shape[:-2],
                            steps=steps, deterministic=deterministic, eps=eps, record_path=record_path)
